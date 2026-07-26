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
let targetRgb0 = new Uint8Array(width * height * 3);
let targetRgb1 = new Uint8Array(width * height * 3);
let currentViewV = 0.5; // View interpolation parameter in [0.0, 1.0]

let lossHistoryGaussian = [];
let lossHistoryNerf = [];

// DOM Elements
const selectImg = document.getElementById('image-select');
const customUploadContainer = document.getElementById('custom-upload-container');
const uploadInput0 = document.getElementById('image-upload-0');
const uploadInput1 = document.getElementById('image-upload-1');

const numGaussiansInput = document.getElementById('num-gaussians');
const lrGaussianInput = document.getElementById('lr-gaussian');
const lrNerfInput = document.getElementById('lr-nerf');
const btnTrain = document.getElementById('btn-train');
const btnReset = document.getElementById('btn-reset');
const btnPretrain = document.getElementById('btn-nerf-pretrain');
const btnSeed = document.getElementById('btn-seed');

const viewSlider = document.getElementById('view-slider');
const viewAngleText = document.getElementById('view-angle-text');

const canvasTarget = document.getElementById('canvas-target');
const canvasGaussian = document.getElementById('canvas-gaussian');
const canvasNerf = document.getElementById('canvas-nerf');
const canvasChart = document.getElementById('canvas-chart');
const canvasBlend = document.getElementById('canvas-blend');
const blendSlider = document.getElementById('blend-slider');

const labelLossGaussian = document.getElementById('loss-gaussian');
const labelLossNerf = document.getElementById('loss-nerf');

// Pipeline Step References
const step1Card = document.getElementById('step-1-card');
const step1Status = document.getElementById('step-1-status');
const step2Card = document.getElementById('step-2-card');
const step2Status = document.getElementById('step-2-status');

// Initialize WASM
async function start() {
    await init();
    init_panic_hook();
    
    // Initialize WebGPU context asynchronously first
    await init_webgpu();
    
    // Set up default synthetic target views
    generateSyntheticTargets();
    resetSession();
    
    // Wire up events
    selectImg.addEventListener('change', handleImageSelect);
    if (uploadInput0) uploadInput0.addEventListener('change', () => handleCustomUpload(0));
    if (uploadInput1) uploadInput1.addEventListener('change', () => handleCustomUpload(1));

    btnTrain.addEventListener('click', toggleTraining);
    btnReset.addEventListener('click', resetSession);
    viewSlider.addEventListener('input', handleViewSliderInput);
    blendSlider.addEventListener('input', updateBlendCanvas);
    btnPretrain.addEventListener('click', runNeRFPretraining);
    btnSeed.addEventListener('click', seedGaussiansFromEdges);
}

// Generate default synthetic targets: View 0 (Circle at 0°) and View 1 (Rotated/Displaced at 30°)
function generateSyntheticTargets() {
    // View 0 (0 deg): Red circle in center
    const canvas0 = document.createElement('canvas');
    canvas0.width = width;
    canvas0.height = height;
    const ctx0 = canvas0.getContext('2d');
    ctx0.fillStyle = '#000080'; // Dark Blue background
    ctx0.fillRect(0, 0, width, height);
    ctx0.beginPath();
    ctx0.arc(width * 0.5, height * 0.5, width * 0.35, 0, 2 * Math.PI);
    ctx0.fillStyle = '#ff0000'; // Red
    ctx0.fill();

    const imgData0 = ctx0.getImageData(0, 0, width, height);
    let idx = 0;
    for (let i = 0; i < imgData0.data.length; i += 4) {
        targetRgb0[idx++] = imgData0.data[i];
        targetRgb0[idx++] = imgData0.data[i + 1];
        targetRgb0[idx++] = imgData0.data[i + 2];
    }

    // View 1 (30 deg): Skewed/Shifted Red circle (simulating 30 degree rotated camera view)
    const canvas1 = document.createElement('canvas');
    canvas1.width = width;
    canvas1.height = height;
    const ctx1 = canvas1.getContext('2d');
    ctx1.fillStyle = '#000080';
    ctx1.fillRect(0, 0, width, height);
    ctx1.beginPath();
    ctx1.ellipse(width * 0.58, height * 0.5, width * 0.28, width * 0.35, Math.PI / 12, 0, 2 * Math.PI);
    ctx1.fillStyle = '#ff0000';
    ctx1.fill();

    const imgData1 = ctx1.getImageData(0, 0, width, height);
    idx = 0;
    for (let i = 0; i < imgData1.data.length; i += 4) {
        targetRgb1[idx++] = imgData1.data[i];
        targetRgb1[idx++] = imgData1.data[i + 1];
        targetRgb1[idx++] = imgData1.data[i + 2];
    }

    renderTargetViewBlend();
}

// Render blended Ground Truth view based on current view slider parameter
function renderTargetViewBlend() {
    const ctx = canvasTarget.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    const v = currentViewV;

    let srcIdx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] = (1 - v) * targetRgb0[srcIdx] + v * targetRgb1[srcIdx];
        imgData.data[i + 1] = (1 - v) * targetRgb0[srcIdx + 1] + v * targetRgb1[srcIdx + 1];
        imgData.data[i + 2] = (1 - v) * targetRgb0[srcIdx + 2] + v * targetRgb1[srcIdx + 2];
        imgData.data[i + 3] = 255;
        srcIdx += 3;
    }
    ctx.putImageData(imgData, 0, 0);
}

// Handle image selection dropdown
function handleImageSelect() {
    if (selectImg.value === 'synthetic') {
        if (customUploadContainer) customUploadContainer.classList.add('hidden');
        generateSyntheticTargets();
        resetSession();
    } else if (selectImg.value === 'upload') {
        if (customUploadContainer) customUploadContainer.classList.remove('hidden');
    }
}

// Handle custom dual-view file upload
function handleCustomUpload(viewIndex) {
    const fileInput = viewIndex === 0 ? uploadInput0 : uploadInput1;
    const file = fileInput.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        const imgData = ctx.getImageData(0, 0, width, height);
        const targetArr = viewIndex === 0 ? targetRgb0 : targetRgb1;
        let idx = 0;
        for (let i = 0; i < imgData.data.length; i += 4) {
            targetArr[idx++] = imgData.data[i];
            targetArr[idx++] = imgData.data[i + 1];
            targetArr[idx++] = imgData.data[i + 2];
        }
        renderTargetViewBlend();
        resetSession();
    };
    img.src = URL.createObjectURL(file);
}

// Handle view interpolation slider movement
async function handleViewSliderInput() {
    currentViewV = parseFloat(viewSlider.value);
    const angleDeg = (currentViewV * 30).toFixed(0);
    viewAngleText.textContent = `Angle: ${angleDeg}° (v = ${currentViewV.toFixed(2)})`;

    renderTargetViewBlend();

    if (session) {
        renderModelOutput(canvasGaussian, await session.get_gaussian_render_view(currentViewV));
        renderModelOutput(canvasNerf, await session.get_nerf_render_view(currentViewV));
        updateBlendCanvas();
    }
}

// Reset models and clear graphs
function resetSession() {
    isTraining = false;
    btnTrain.textContent = 'Start Multi-View Fitting';
    btnTrain.classList.remove('btn-stop');

    btnPretrain.disabled = false;
    btnPretrain.textContent = '1. Pre-train NeRF (50 Steps)';
    btnSeed.disabled = true;
    btnSeed.textContent = '2. Initialize 3D GS on Edges';

    // Reset pipeline step cards UI
    step1Card.className = 'pipeline-step active';
    step1Status.textContent = '⚪';
    step2Card.className = 'pipeline-step';
    step2Status.textContent = '⚪';

    const numGaussians = parseInt(numGaussiansInput.value) || 500;
    
    // Create new multi-view session in Rust WASM
    session = new WasmTrainingSession(width, height, numGaussians, targetRgb0, targetRgb1);
    console.log(`[System] Initialized new Multi-View session with ${numGaussians} 3D Gaussians.`);

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
        btnTrain.textContent = 'Start Multi-View Fitting';
        btnTrain.classList.remove('btn-stop');
        console.log(`[System] Multi-View Training paused at Step ${lossHistoryGaussian.length}.`);
    } else {
        isTraining = true;
        btnTrain.textContent = 'Pause Training';
        btnTrain.classList.add('btn-stop');
        const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
        const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
        console.log(`[System] Starting Multi-View optimization. LR: Explicit GS = ${lrGaussian}, Implicit NeRF = ${lrNerf}`);
        requestAnimationFrame(trainingLoop);
    }
}

// Core animation and optimization loop
async function trainingLoop() {
    if (!isTraining || !session) return;

    const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;

    try {
        // 1. Step the Gaussian Splatting model jointly over View 0 and View 1
        const lossG = await session.step_gaussian(lrGaussian);
        lossHistoryGaussian.push(lossG);
        labelLossGaussian.textContent = `Loss: ${lossG.toFixed(5)}`;

        // 2. Step the view-conditioned NeRF MLP model jointly over View 0 and View 1
        const lossN = await session.step_nerf(lrNerf);
        lossHistoryNerf.push(lossN);
        labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;

        // Render results at current view interpolation parameter v
        renderModelOutput(canvasGaussian, await session.get_gaussian_render_view(currentViewV));
        renderModelOutput(canvasNerf, await session.get_nerf_render_view(currentViewV));

        // Update blended view & line chart
        updateBlendCanvas();
        drawLossChart();

        // Periodically log progress to developer console
        if (lossHistoryGaussian.length % 50 === 0) {
            console.log(`[Step ${lossHistoryGaussian.length}] GS Multi-View Loss: ${lossG.toFixed(5)} | NeRF Multi-View Loss: ${lossN.toFixed(5)}`);
        }
    } catch (e) {
        console.error("Error during training step:", e);
        isTraining = false;
        btnTrain.textContent = 'Start Multi-View Fitting';
        btnTrain.classList.remove('btn-stop');
        return;
    }

    // Loop
    if (isTraining) {
        requestAnimationFrame(() => trainingLoop());
    }
}

// Pre-train NeRF to capture coarse multi-view edges
async function runNeRFPretraining() {
    isTraining = false;
    btnTrain.textContent = 'Start Multi-View Fitting';
    btnTrain.classList.remove('btn-stop');
    
    btnPretrain.disabled = true;
    btnPretrain.textContent = 'Training Multi-View NeRF...';
    
    step1Status.textContent = '⏳';
    
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
    console.log("[System] Starting Multi-View NeRF pre-training for 50 steps...");
    
    try {
        let finalLoss = 0.0;
        // Train for 50 steps across both views
        for (let step = 1; step <= 50; step++) {
            const lossN = await session.step_nerf(lrNerf);
            lossHistoryNerf.push(lossN);
            labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;
            finalLoss = lossN;
            
            if (step % 5 === 0 || step === 50) {
                renderModelOutput(canvasNerf, await session.get_nerf_render_view(currentViewV));
                drawLossChart();
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }
        
        btnPretrain.textContent = 'NeRF Pre-trained!';
        btnPretrain.disabled = true;
        
        step1Card.className = 'pipeline-step completed';
        step1Status.textContent = '✅';
        step2Card.className = 'pipeline-step active';
        
        console.log(`[System] Multi-view NeRF pre-training complete. Final Loss: ${finalLoss.toFixed(5)}`);
        
        console.log("[System] Extracting multi-view spatial gradient importance map...");
        const importanceData = await session.get_nerf_importance_map();
        renderGrayscaleOutput(canvasBlend, importanceData, width - 1, height - 1);
        
        btnSeed.disabled = false;
    } catch (e) {
        console.error("Error during NeRF pre-training:", e);
        btnPretrain.disabled = false;
        btnPretrain.textContent = '1. Pre-train NeRF (50 Steps)';
        step1Status.textContent = '❌';
    }
}

// Seed 3D Gaussians proportional to NeRF's spatial gradient across views
async function seedGaussiansFromEdges() {
    btnSeed.disabled = true;
    btnSeed.textContent = 'Seeding...';
    step2Status.textContent = '⏳';
    console.log("[System] Seeding 3D Gaussians on multi-view high-frequency edges...");
    try {
        await session.seed_from_nerf();
        renderModelOutput(canvasGaussian, await session.get_gaussian_render_view(currentViewV));
        btnSeed.textContent = 'Gaussians Seeded!';
        btnSeed.disabled = true;
        
        step2Card.className = 'pipeline-step completed';
        step2Status.textContent = '✅';
        
        console.log("[System] Guided seeding complete! Initialized 3D Gaussians directly along multi-view boundaries.");

        lossHistoryGaussian = [];
        lossHistoryNerf = [];
        drawLossChart();
    } catch (e) {
        console.error("Error during guided seeding:", e);
        btnSeed.disabled = false;
        btnSeed.textContent = '2. Initialize 3D GS on Edges';
        step2Status.textContent = '❌';
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
