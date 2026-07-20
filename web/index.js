import init, { WasmTrainingSession, init_panic_hook } from './pkg/burn_nerf_guided_splats.js';

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
    
    // Set up default synthetic target image
    generateSyntheticTarget();
    resetSession();
    
    // Wire up events
    selectImg.addEventListener('change', handleImageSelect);
    uploadInput.addEventListener('change', handleImageUpload);
    btnTrain.addEventListener('click', toggleTraining);
    btnReset.addEventListener('click', resetSession);
    blendSlider.addEventListener('input', updateBlendCanvas);
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

    const numGaussians = parseInt(numGaussiansInput.value) || 500;
    
    // Create new session in Rust WASM
    session = new WasmTrainingSession(width, height, numGaussians, targetRgb);

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
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Training toggle
function toggleTraining() {
    if (isTraining) {
        isTraining = false;
        btnTrain.textContent = 'Start Training';
        btnTrain.classList.remove('btn-stop');
    } else {
        isTraining = true;
        btnTrain.textContent = 'Pause Training';
        btnTrain.classList.add('btn-stop');
        requestAnimationFrame(trainingLoop);
    }
}

// Core animation and optimization loop
function trainingLoop() {
    if (!isTraining || !session) return;

    const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;

    // 1. Step the Gaussian Splatting model
    const lossG = session.step_gaussian(lrGaussian);
    lossHistoryGaussian.push(lossG);
    labelLossGaussian.textContent = `Loss: ${lossG.toFixed(5)}`;

    // 2. Step the NeRF MLP model
    const lossN = session.step_nerf(lrNerf);
    lossHistoryNerf.push(lossN);
    labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;

    // Render results
    renderModelOutput(canvasGaussian, session.get_gaussian_render());
    renderModelOutput(canvasNerf, session.get_nerf_render());

    // Update blended view & line chart
    updateBlendCanvas();
    drawLossChart();

    // Loop
    requestAnimationFrame(trainingLoop);
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
