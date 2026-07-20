import init, { WasmTrainingSession, init_panic_hook, init_webgpu } from './pkg/burn_nerf_guided_splats.js';

// Redirect Console output to HTML developer console
const developerConsole = document.getElementById('developer-console');

function logToTerminal(message, type = 'info') {
    if (!developerConsole) return;
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    developerConsole.appendChild(line);
    developerConsole.scrollTop = developerConsole.scrollHeight;
}

const orgLog = console.log;
const orgWarn = console.warn;
const orgError = console.error;

console.log = function(...args) {
    orgLog.apply(console, args);
    logToTerminal(args.join(' '), 'info');
};

console.warn = function(...args) {
    orgWarn.apply(console, args);
    logToTerminal(args.join(' '), 'warn');
};

console.error = function(...args) {
    orgError.apply(console, args);
    logToTerminal(args.join(' '), 'error');
};

let session = null;
let isTraining = false;
let width = 128;
let height = 128;
let targetRgb = new Uint8Array(width * height * 3);
let lossHistoryGaussian = [];
let lossHistoryNerf = [];

// DOM Elements
const selectImg = document.getElementById('image-select');
const uploadInput = document.getElementById('image-upload');
const numGaussiansInput = document.getElementById('num-gaussians');
const lrGaussianInput = document.getElementById('lr-gaussian');
const lrNerfInput = document.getElementById('lr-nerf');
const btnTrain = document.getElementById('btn-train');
const btnReset = document.getElementById('btn-reset');
const btnPretrain = document.getElementById('btn-nerf-pretrain');
const btnSeed = document.getElementById('btn-seed');

const canvasTarget = document.getElementById('canvas-target');
const canvasGaussian = document.getElementById('canvas-gaussian');
const canvasNerf = document.getElementById('canvas-nerf');
const canvasChart = document.getElementById('canvas-chart');
const canvasBlend = document.getElementById('canvas-blend');
const blendSlider = document.getElementById('blend-slider');

const labelLossGaussian = document.getElementById('loss-gaussian');
const labelLossNerf = document.getElementById('loss-nerf');

// Initialize WASM
async function start() {
    await init();
    init_panic_hook();
    
    // Initialize WebGPU context asynchronously first
    await init_webgpu();
    
    // Set up default synthetic target image
    generateSyntheticTarget();
    resetSession();
    
    // Wire up events
    selectImg.addEventListener('change', handleImageSelect);
    uploadInput.addEventListener('change', handleImageUpload);
    btnTrain.addEventListener('click', toggleTraining);
    btnReset.addEventListener('click', resetSession);
    blendSlider.addEventListener('input', updateBlendCanvas);
    btnPretrain.addEventListener('click', runNeRFPretraining);
    btnSeed.addEventListener('click', seedGaussiansFromEdges);
}

// Generate default target image: red circle on dark blue background
function generateSyntheticTarget() {
    const ctx = canvasTarget.getContext('2d');
    ctx.fillStyle = '#000080'; // Blue
    ctx.fillRect(0, 0, width, height);

    ctx.beginPath();
    ctx.arc(width / 2, height / 2, width * 0.35, 0, 2 * Math.PI);
    ctx.fillStyle = '#ff0000'; // Red
    ctx.fill();

    // Cache target pixels
    const imgData = ctx.getImageData(0, 0, width, height);
    let idx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        targetRgb[idx++] = imgData.data[i];     // R
        targetRgb[idx++] = imgData.data[i + 1]; // G
        targetRgb[idx++] = imgData.data[i + 2]; // B
    }
}

// Handle image selection dropdown
function handleImageSelect() {
    if (selectImg.value === 'synthetic') {
        uploadInput.classList.add('hidden');
        generateSyntheticTarget();
        resetSession();
    } else if (selectImg.value === 'upload') {
        uploadInput.classList.remove('hidden');
        uploadInput.click();
    }
}

// Handle local custom image upload
function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        const ctx = canvasTarget.getContext('2d');
        // Clear and draw resized custom image
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);

        // Read pixels
        const imgData = ctx.getImageData(0, 0, width, height);
        let idx = 0;
        for (let i = 0; i < imgData.data.length; i += 4) {
            targetRgb[idx++] = imgData.data[i];
            targetRgb[idx++] = imgData.data[i + 1];
            targetRgb[idx++] = imgData.data[i + 2];
        }
        resetSession();
    };
    img.src = URL.createObjectURL(file);
}

// Reset models and clear graphs
function resetSession() {
    isTraining = false;
    btnTrain.textContent = 'Start Training';
    btnTrain.classList.remove('btn-stop');

    btnPretrain.disabled = false;
    btnPretrain.textContent = '1. Pre-train NeRF (50 Steps)';
    btnSeed.disabled = true;
    btnSeed.textContent = '2. Seed Gaussians from NeRF Edges';

    const numGaussians = parseInt(numGaussiansInput.value) || 500;
    
    // Create new session in Rust WASM
    session = new WasmTrainingSession(width, height, numGaussians, targetRgb);
    console.log(`[System] Initialized new session with ${numGaussians} Gaussians.`);

    lossHistoryGaussian = [];
    lossHistoryNerf = [];
    
    // Clear views
    clearCanvas(canvasGaussian);
    clearCanvas(canvasNerf);
    clearCanvas(canvasBlend);
    drawLossChart();

    labelLossGaussian.textContent = 'Loss: --';
    labelLossNerf.textContent = 'Loss: --';
}

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#f1f5f9';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Training toggle
function toggleTraining() {
    if (isTraining) {
        isTraining = false;
        btnTrain.textContent = 'Start Training';
        btnTrain.classList.remove('btn-stop');
        console.log(`[System] Training paused at Step ${lossHistoryGaussian.length}.`);
    } else {
        isTraining = true;
        btnTrain.textContent = 'Pause Training';
        btnTrain.classList.add('btn-stop');
        const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
        const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
        console.log(`[System] Starting training loop. LR: Explicit GS = ${lrGaussian}, Implicit NeRF = ${lrNerf}`);
        requestAnimationFrame(trainingLoop);
    }
}

// Core animation and optimization loop
async function trainingLoop() {
    if (!isTraining || !session) return;

    const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;

    try {
        // 1. Step the Gaussian Splatting model
        const lossG = await session.step_gaussian(lrGaussian);
        lossHistoryGaussian.push(lossG);
        labelLossGaussian.textContent = `Loss: ${lossG.toFixed(5)}`;

        // 2. Step the NeRF MLP model
        const lossN = await session.step_nerf(lrNerf);
        lossHistoryNerf.push(lossN);
        labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;

        // Render results
        renderModelOutput(canvasGaussian, await session.get_gaussian_render());
        renderModelOutput(canvasNerf, await session.get_nerf_render());

        // Update blended view & line chart
        updateBlendCanvas();
        drawLossChart();

        // Periodically log progress to developer console
        if (lossHistoryGaussian.length % 50 === 0) {
            console.log(`[Step ${lossHistoryGaussian.length}] GS Loss: ${lossG.toFixed(5)} | NeRF Loss: ${lossN.toFixed(5)}`);
        }
    } catch (e) {
        console.error("Error during training step:", e);
        isTraining = false;
        btnTrain.textContent = 'Start Training';
        btnTrain.classList.remove('btn-stop');
        return;
    }

    // Loop
    if (isTraining) {
        requestAnimationFrame(() => trainingLoop());
    }
}

// Pre-train NeRF to capture coarse edges
async function runNeRFPretraining() {
    isTraining = false;
    btnTrain.textContent = 'Start Training';
    btnTrain.classList.remove('btn-stop');
    
    btnPretrain.disabled = true;
    btnPretrain.textContent = 'Training NeRF...';
    
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
    console.log("[System] Starting NeRF pre-training for 50 steps to extract spatial gradient importance map...");
    
    try {
        let finalLoss = 0.0;
        // Train for 50 steps
        for (let step = 1; step <= 50; step++) {
            const lossN = await session.step_nerf(lrNerf);
            lossHistoryNerf.push(lossN);
            labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;
            finalLoss = lossN;
            
            if (step % 5 === 0 || step === 50) {
                renderModelOutput(canvasNerf, await session.get_nerf_render());
                drawLossChart();
                // Yield to main thread to keep browser UI responsive
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }
        
        btnPretrain.textContent = 'NeRF Pre-trained!';
        btnPretrain.disabled = true;
        console.log(`[System] NeRF pre-training complete. Final NeRF Loss: ${finalLoss.toFixed(5)}`);
        
        // Fetch and draw importance map to the Blend canvas so the user can see it!
        console.log("[System] Extracting spatial gradient importance map from implicit network...");
        const importanceData = await session.get_nerf_importance_map();
        // The importance map dimension is (width - 1) x (height - 1) = 127 x 127
        renderGrayscaleOutput(canvasBlend, importanceData, width - 1, height - 1);
        
        btnSeed.disabled = false;
    } catch (e) {
        console.error("Error during NeRF pre-training:", e);
        btnPretrain.disabled = false;
        btnPretrain.textContent = '1. Pre-train NeRF (50 Steps)';
    }
}

// Seed Gaussians proportional to NeRF's spatial gradient
async function seedGaussiansFromEdges() {
    btnSeed.disabled = true;
    btnSeed.textContent = 'Seeding Gaussians...';
    console.log("[System] Seeding Gaussians on high-frequency edges...");
    try {
        await session.seed_from_nerf();
        // Render the initial state of the seeded Gaussians (should align to edges!)
        renderModelOutput(canvasGaussian, await session.get_gaussian_render());
        btnSeed.textContent = 'Gaussians Seeded!';
        btnSeed.disabled = true;
        console.log("[System] Guided seeding complete! Re-initialized 500 Gaussians directly along circle boundaries.");

        // Reset loss histories for fresh chart tracking
        lossHistoryGaussian = [];
        lossHistoryNerf = [];
        drawLossChart();
    } catch (e) {
        console.error("Error during guided seeding:", e);
        btnSeed.disabled = false;
        btnSeed.textContent = '2. Seed Gaussians from NeRF Edges';
    }
}

// Render raw RGB data from Rust onto Web Canvas
function renderModelOutput(canvas, rgbData) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    
    let srcIdx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] = rgbData[srcIdx++];     // R
        imgData.data[i + 1] = rgbData[srcIdx++]; // G
        imgData.data[i + 2] = rgbData[srcIdx++]; // B
        imgData.data[i + 3] = 255;               // A
    }
    ctx.putImageData(imgData, 0, 0);
}

// Render grayscale importance/edge data onto Web Canvas
function renderGrayscaleOutput(canvas, rgbData, w, h) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const imgData = ctx.createImageData(w, h);
    
    let srcIdx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] = rgbData[srcIdx++];     // R
        imgData.data[i + 1] = rgbData[srcIdx++]; // G
        imgData.data[i + 2] = rgbData[srcIdx++]; // B
        imgData.data[i + 3] = 255;               // A
    }
    // Draw centered
    ctx.putImageData(imgData, (canvas.width - w) / 2, (canvas.height - h) / 2);
}

// Update the blended/cross-fade viewer canvas
function updateBlendCanvas() {
    const alpha = parseFloat(blendSlider.value);
    const ctxBlend = canvasBlend.getContext('2d');

    const ctxG = canvasGaussian.getContext('2d');
    const ctxN = canvasNerf.getContext('2d');

    const dataG = ctxG.getImageData(0, 0, width, height).data;
    const dataN = ctxN.getImageData(0, 0, width, height).data;

    const imgDataBlend = ctxBlend.createImageData(width, height);
    for (let i = 0; i < imgDataBlend.data.length; i += 4) {
        imgDataBlend.data[i] = (1 - alpha) * dataG[i] + alpha * dataN[i];
        imgDataBlend.data[i + 1] = (1 - alpha) * dataG[i + 1] + alpha * dataN[i + 1];
        imgDataBlend.data[i + 2] = (1 - alpha) * dataG[i + 2] + alpha * dataN[i + 2];
        imgDataBlend.data[i + 3] = 255;
    }
    ctxBlend.putImageData(imgDataBlend, 0, 0);
}

// Render dynamic line chart of loss history on Canvas
function drawLossChart() {
    const ctx = canvasChart.getContext('2d');
    const w = canvasChart.width;
    const h = canvasChart.height;

    // Clear background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Draw borders & grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(35, 10);
    ctx.lineTo(w - 10, 10);
    ctx.lineTo(w - 10, h - 25);
    ctx.lineTo(35, h - 25);
    ctx.closePath();
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '9px Outfit';
    ctx.fillText('0.05', 5, 20);
    ctx.fillText('0.00', 5, h - 22);
    ctx.fillText('Steps', w / 2 - 15, h - 8);

    if (lossHistoryGaussian.length === 0 && lossHistoryNerf.length === 0) return;

    const maxSteps = Math.max(lossHistoryGaussian.length, lossHistoryNerf.length, 100);
    const maxVal = 0.05; // clamp peak display loss

    // Helper to draw a single line
    function drawLine(history, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let i = 0; i < history.length; i++) {
            const x = 35 + (i / maxSteps) * (w - 45);
            const val = Math.min(history[i], maxVal);
            const y = h - 25 - (val / maxVal) * (h - 35);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Draw Gaussian Splatting loss in Cyan
    drawLine(lossHistoryGaussian, '#06b6d4');
    // Draw NeRF MLP loss in Purple
    drawLine(lossHistoryNerf, '#a78bfa');
}

start();
