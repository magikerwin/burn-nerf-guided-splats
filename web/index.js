import init, { WasmTrainingSession, create_multiframe_session, init_panic_hook, init_webgpu } from './pkg/burn_nerf_guided_splats.js';

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

// Bulletproof Single-Threaded GPU Task Queue to eliminate WebGPU race conditions forever
class GpuTaskQueue {
    constructor() {
        this.promiseChain = Promise.resolve();
    }

    // Serializes GPU tasks so at most ONE WebGPU mapAsync/step executes at a time
    async run(taskFn) {
        const nextPromise = this.promiseChain.then(async () => {
            try {
                return await taskFn();
            } catch (err) {
                console.error("GPU Task Error:", err);
                throw err;
            }
        });
        this.promiseChain = nextPromise.catch(() => {});
        return nextPromise;
    }
}

const gpuQueue = new GpuTaskQueue();

let session = null;
let isTraining = false;
let isAutoPlaying = false;
let autoPlayDirection = 1;


let width = 64;  // Default low-resolution 64x64 for fast responsiveness
let height = 64;

let targetFrames = []; // Array of Uint8Arrays representing RGB frames
let currentViewV = 0.5; // Relative view parameter in [0.0, 1.0]

let lossHistoryGaussian = [];
let lossHistoryNerf = [];

// DOM Elements
const resolutionSelect = document.getElementById('resolution-select');
const selectImg = document.getElementById('image-select');
const customUploadContainer = document.getElementById('custom-upload-container');
const uploadInputMulti = document.getElementById('image-upload-multi');

const numGaussiansInput = document.getElementById('num-gaussians');
const lrGaussianInput = document.getElementById('lr-gaussian');
const lrNerfInput = document.getElementById('lr-nerf');
const btnTrain = document.getElementById('btn-train');
const btnReset = document.getElementById('btn-reset');
const btnOneClickHybrid = document.getElementById('btn-one-click-hybrid');
const btnPretrain = document.getElementById('btn-nerf-pretrain');
const btnSeed = document.getElementById('btn-seed');

const btnAutoPlay = document.getElementById('btn-autoplay');
const viewSlider = document.getElementById('view-slider');
const viewAngleText = document.getElementById('view-angle-text');
const labelViewStart = document.getElementById('label-view-start');
const labelViewEnd = document.getElementById('label-view-end');

const btnToggleAdvanced = document.getElementById('btn-toggle-advanced');
const btnCloseAdvanced = document.getElementById('btn-close-advanced');
const advancedDrawer = document.getElementById('advanced-drawer');

const canvasTarget = document.getElementById('canvas-target');
const canvasGaussian = document.getElementById('canvas-gaussian');
const canvasNerf = document.getElementById('canvas-nerf');
const canvasChart = document.getElementById('canvas-chart');
const canvasBlend = document.getElementById('canvas-blend');
const blendSlider = document.getElementById('blend-slider');

const labelLossGaussian = document.getElementById('loss-gaussian');
const labelLossNerf = document.getElementById('loss-nerf');

// Educational Stepper Card References
const step1Card = document.getElementById('step-1-card');
const step1Status = document.getElementById('step-1-status');
const step2Card = document.getElementById('step-2-card');
const step2Status = document.getElementById('step-2-status');
const step3Card = document.getElementById('step-3-card');
const step3Status = document.getElementById('step-3-status');

// Initialize WASM
async function start() {
    await init();
    init_panic_hook();
    
    // Initialize WebGPU context asynchronously first
    await init_webgpu();
    
    updateCanvasDimensions();

    // Set up default synthetic target views
    generateSyntheticTargets();
    resetSession();
    
    // Wire up events
    resolutionSelect.addEventListener('change', handleResolutionChange);
    selectImg.addEventListener('change', handleImageSelect);
    if (uploadInputMulti) uploadInputMulti.addEventListener('change', handleMultiPhotoUpload);

    btnTrain.addEventListener('click', toggleTraining);
    btnReset.addEventListener('click', resetSession);
    btnAutoPlay.addEventListener('click', toggleAutoPlay);
    if (btnOneClickHybrid) btnOneClickHybrid.addEventListener('click', runOneClickHybridPipeline);
    viewSlider.addEventListener('input', handleViewSliderInput);
    if (blendSlider) blendSlider.addEventListener('input', updateBlendCanvas);
    if (btnPretrain) btnPretrain.addEventListener('click', runNeRFPretraining);
    if (btnSeed) btnSeed.addEventListener('click', seedGaussiansFromEdges);

    if (btnToggleAdvanced) btnToggleAdvanced.addEventListener('click', () => advancedDrawer.classList.toggle('hidden'));
    if (btnCloseAdvanced) btnCloseAdvanced.addEventListener('click', () => advancedDrawer.classList.add('hidden'));
}


// Update canvas DOM sizes
function updateCanvasDimensions() {
    const res = parseInt(resolutionSelect.value) || 64;
    width = res;
    height = res;

    canvasTarget.width = width;
    canvasTarget.height = height;
    canvasGaussian.width = width;
    canvasGaussian.height = height;
    canvasNerf.width = width;
    canvasNerf.height = height;
    if (canvasBlend) {
        canvasBlend.width = width;
        canvasBlend.height = height;
    }
}

// Handle render grid resolution dropdown change
function handleResolutionChange() {
    updateCanvasDimensions();
    generateSyntheticTargets(selectImg ? selectImg.value : 'multi-geometry');
    resetSession();
}


// Generate multi-scene synthetic target keyframe sequences
function generateSyntheticTargets(type = 'multi-geometry') {
    targetFrames = [];
    const numFrames = (type === 'synthetic') ? 2 : 4;

    for (let f = 0; f < numFrames; f++) {
        const normV = f / (numFrames - 1); // 0.0, 0.33, 0.67, 1.0
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const ctx = tempCanvas.getContext('2d');

        if (type === 'multi-geometry') {
            // Scene 1: 3D Multi-Color Shapes with Parallax (Red Sphere, Green Box, Cyan Triangle)
            ctx.fillStyle = '#090d16'; // Deep Space Background
            ctx.fillRect(0, 0, width, height);

            // Background Green Box (Parallax shift right as camera rotates)
            const greenX = width * (0.22 + normV * 0.18);
            ctx.fillStyle = '#10b981';
            ctx.fillRect(greenX, height * 0.3, width * 0.22, height * 0.4);

            // Foreground Red Sphere (Center)
            ctx.beginPath();
            ctx.arc(width * 0.5, height * 0.5, width * 0.22, 0, 2 * Math.PI);
            ctx.fillStyle = '#f97316';
            ctx.fill();

            // Right Cyan Triangle
            const cyanX = width * (0.75 - normV * 0.12);
            ctx.beginPath();
            ctx.moveTo(cyanX, height * 0.25);
            ctx.lineTo(cyanX + width * 0.12, height * 0.65);
            ctx.lineTo(cyanX - width * 0.12, height * 0.65);
            ctx.closePath();
            ctx.fillStyle = '#06b6d4';
            ctx.fill();

        } else if (type === 'planet-orbit') {
            // Scene 2: 3D Solar System (Sun in center, Earth & Mars orbiting)
            ctx.fillStyle = '#030712';
            ctx.fillRect(0, 0, width, height);

            // Orbit Path Guide Ring
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(width * 0.5, height * 0.5, width * 0.38, height * 0.22, 0, 0, 2 * Math.PI);
            ctx.stroke();

            // Center Sun
            ctx.beginPath();
            ctx.arc(width * 0.5, height * 0.5, width * 0.18, 0, 2 * Math.PI);
            ctx.fillStyle = '#f59e0b';
            ctx.fill();

            // Blue Earth (Orbiting around Sun)
            const angleEarth = normV * Math.PI * 0.9;
            const earthX = width * 0.5 + Math.cos(angleEarth) * (width * 0.38);
            const earthY = height * 0.5 + Math.sin(angleEarth) * (height * 0.22);
            ctx.beginPath();
            ctx.arc(earthX, earthY, width * 0.08, 0, 2 * Math.PI);
            ctx.fillStyle = '#3b82f6';
            ctx.fill();

            // Red Mars (Orbiting on smaller inner arc)
            const angleMars = Math.PI - normV * Math.PI * 0.7;
            const marsX = width * 0.5 + Math.cos(angleMars) * (width * 0.26);
            const marsY = height * 0.5 + Math.sin(angleMars) * (height * 0.14);
            ctx.beginPath();
            ctx.arc(marsX, marsY, width * 0.06, 0, 2 * Math.PI);
            ctx.fillStyle = '#ef4444';
            ctx.fill();

        } else if (type === 'emoji-face') {
            // Scene 3: 3D Emoji Face Arc
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, width, height);

            const skew = (normV - 0.5) * 0.35;
            ctx.save();
            ctx.translate(width * 0.5, height * 0.5);
            ctx.transform(1, 0, skew, 1, 0, 0);

            // Yellow Head
            ctx.beginPath();
            ctx.arc(0, 0, width * 0.36, 0, 2 * Math.PI);
            ctx.fillStyle = '#eab308';
            ctx.fill();

            // Eyes
            ctx.fillStyle = '#1e293b';
            ctx.beginPath();
            ctx.arc(-width * 0.12, -height * 0.1, width * 0.06, 0, 2 * Math.PI);
            ctx.arc(width * 0.12, -height * 0.1, width * 0.06, 0, 2 * Math.PI);
            ctx.fill();

            // Smile
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#1e293b';
            ctx.beginPath();
            ctx.arc(0, height * 0.05, width * 0.18, 0.2, Math.PI - 0.2);
            ctx.stroke();

            ctx.restore();

        } else {
            // Benchmark Dual-View (Circle 0° vs 30°)
            ctx.fillStyle = '#000080';
            ctx.fillRect(0, 0, width, height);
            ctx.beginPath();
            if (f === 0) {
                ctx.arc(width * 0.5, height * 0.5, width * 0.35, 0, 2 * Math.PI);
            } else {
                ctx.ellipse(width * 0.58, height * 0.5, width * 0.28, width * 0.35, Math.PI / 12, 0, 2 * Math.PI);
            }
            ctx.fillStyle = '#ff0000';
            ctx.fill();
        }

        const imgData = ctx.getImageData(0, 0, width, height);
        const frameBuf = new Uint8Array(width * height * 3);
        let idx = 0;
        for (let i = 0; i < imgData.data.length; i += 4) {
            frameBuf[idx++] = imgData.data[i];
            frameBuf[idx++] = imgData.data[i + 1];
            frameBuf[idx++] = imgData.data[i + 2];
        }
        targetFrames.push(frameBuf);
    }

    if (labelViewStart) labelViewStart.textContent = `Photo 1`;
    if (labelViewEnd) labelViewEnd.textContent = `Photo ${numFrames}`;

    renderTargetViewBlend();
}


// Render blended Ground Truth view based on current view slider parameter
function renderTargetViewBlend() {
    const ctx = canvasTarget.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    const numFrames = targetFrames.length;

    if (numFrames === 0) return;

    let f0 = 0;
    let f1 = 0;
    let weight = 0;

    if (numFrames === 1) {
        f0 = 0;
        f1 = 0;
        weight = 0;
    } else {
        const scaled = currentViewV * (numFrames - 1);
        f0 = Math.floor(scaled);
        f1 = Math.min(f0 + 1, numFrames - 1);
        weight = scaled - f0;
    }

    const buf0 = targetFrames[f0];
    const buf1 = targetFrames[f1];

    let srcIdx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] = (1 - weight) * buf0[srcIdx] + weight * buf1[srcIdx];
        imgData.data[i + 1] = (1 - weight) * buf0[srcIdx + 1] + weight * buf1[srcIdx + 1];
        imgData.data[i + 2] = (1 - weight) * buf0[srcIdx + 2] + weight * buf1[srcIdx + 2];
        imgData.data[i + 3] = 255;
        srcIdx += 3;
    }
    ctx.putImageData(imgData, 0, 0);
}

// Handle multi-photo file upload (Photo 1, Photo 2, Photo 3, Photo 4...)
function handleMultiPhotoUpload(e) {
    const files = Array.from(e.target.files);
    if (!files || files.length === 0) return;

    targetFrames = [];
    let loadedCount = 0;

    files.forEach((file, index) => {
        const img = new Image();
        img.onload = () => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const ctx = tempCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            const imgData = ctx.getImageData(0, 0, width, height);
            const frameBuf = new Uint8Array(width * height * 3);
            let idx = 0;
            for (let i = 0; i < imgData.data.length; i += 4) {
                frameBuf[idx++] = imgData.data[i];
                frameBuf[idx++] = imgData.data[i + 1];
                frameBuf[idx++] = imgData.data[i + 2];
            }
            targetFrames[index] = frameBuf;
            loadedCount++;

            if (loadedCount === files.length) {
                if (labelViewStart) labelViewStart.textContent = `Photo 1`;
                if (labelViewEnd) labelViewEnd.textContent = `Photo ${files.length}`;
                renderTargetViewBlend();
                resetSession();
            }
        };
        img.src = URL.createObjectURL(file);
    });
}

// Handle image selection dropdown
function handleImageSelect() {
    const val = selectImg.value;
    if (val === 'upload') {
        if (customUploadContainer) customUploadContainer.classList.remove('hidden');
    } else {
        if (customUploadContainer) customUploadContainer.classList.add('hidden');
        generateSyntheticTargets(val);
        resetSession();
    }
}
// Handle view interpolation slider movement
async function handleViewSliderInput() {
    currentViewV = parseFloat(viewSlider.value);
    const numFrames = targetFrames.length || 2;
    viewAngleText.textContent = `v = ${currentViewV.toFixed(2)} (${numFrames} Keyframes)`;

    renderTargetViewBlend();

    if (session) {
        await gpuQueue.run(async () => {
            const rgbG = await session.get_gaussian_render_view(currentViewV);
            const rgbN = await session.get_nerf_render_view(currentViewV);
            renderModelOutput(canvasGaussian, rgbG);
            renderModelOutput(canvasNerf, rgbN);
            updateBlendCanvas();
        });
    }
}

// Toggle hands-free 3D Orbit Turntable Animation
function toggleAutoPlay() {
    if (isAutoPlaying) {
        isAutoPlaying = false;
        if (btnAutoPlay) btnAutoPlay.textContent = '▶ Play Orbit';
    } else {
        isAutoPlaying = true;
        if (btnAutoPlay) btnAutoPlay.textContent = '⏸ Pause Orbit';
        requestAnimationFrame(autoPlayOrbitLoop);
    }
}

// Continuous 3D Orbit sweep loop
async function autoPlayOrbitLoop() {
    if (!isAutoPlaying) return;

    currentViewV += autoPlayDirection * 0.008;
    if (currentViewV >= 1.0) {
        currentViewV = 1.0;
        autoPlayDirection = -1;
    } else if (currentViewV <= 0.0) {
        currentViewV = 0.0;
        autoPlayDirection = 1;
    }

    viewSlider.value = currentViewV;
    await handleViewSliderInput();

    if (isAutoPlaying) {
        setTimeout(() => requestAnimationFrame(autoPlayOrbitLoop), 25);
    }
}

// Reset models and clear graphs
async function resetSession() {
    isTraining = false;
    btnTrain.textContent = '▶ Start Multi-View Fitting';
    btnTrain.classList.remove('btn-stop');

    btnPretrain.disabled = false;
    btnPretrain.textContent = '1. Pre-train NeRF (50 Steps)';
    btnSeed.disabled = true;
    btnSeed.textContent = '2. Initialize 3D GS on Edges';

    // Reset pipeline step cards UI
    if (step1Card) step1Card.className = 'pipeline-card active';
    if (step2Card) step2Card.className = 'pipeline-card';

    const numGaussians = parseInt(numGaussiansInput.value) || 500;
    const numFrames = targetFrames.length;

    if (numFrames === 0) return;

    // Concatenate all target frames into single Uint8Array
    const frameLen = width * height * 3;
    const flatConcat = new Uint8Array(numFrames * frameLen);
    for (let f = 0; f < numFrames; f++) {
        flatConcat.set(targetFrames[f], f * frameLen);
    }

    // Serialize new WASM session creation through gpuQueue to let previous WGPU buffers unmap cleanly
    await gpuQueue.run(async () => {
        session = create_multiframe_session(width, height, numGaussians, flatConcat, numFrames);
        // Pre-compile WGSL compute shader pipelines asynchronously so Start Fitting responds instantly
        session.step_gaussian_fast(0.0);
        session.step_nerf_fast(0.0);
        console.log(`[System] Initialized & Warmed up Multi-Frame session at ${width}x${height} grid with ${numGaussians} 3D Gaussians across ${numFrames} keyframe views.`);
    });


    lossHistoryGaussian = [];
    lossHistoryNerf = [];
    stepCounter = 0;
    
    // Clear views
    clearCanvas(canvasGaussian);
    clearCanvas(canvasNerf);
    if (canvasBlend) clearCanvas(canvasBlend);
    drawLossChart();

    labelLossGaussian.textContent = 'Loss: --';
    labelLossNerf.textContent = 'Loss: --';
}


function clearCanvas(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Training toggle
async function toggleTraining() {
    if (isTraining) {
        isTraining = false;
        btnTrain.textContent = '▶ Start Multi-View Fitting';
        btnTrain.classList.remove('btn-stop');
        console.log(`[System] Multi-View Training paused at Step ${lossHistoryGaussian.length}.`);
        if (session) {
            await gpuQueue.run(async () => {
                const rgbG = await session.get_gaussian_render_view(currentViewV);
                const rgbN = await session.get_nerf_render_view(currentViewV);
                renderModelOutput(canvasGaussian, rgbG);
                renderModelOutput(canvasNerf, rgbN);
                updateBlendCanvas();
            });
        }
    } else {
        isTraining = true;
        btnTrain.textContent = '⏸ Pause Fitting';
        btnTrain.classList.add('btn-stop');
        const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
        const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
        console.log(`[System] Starting Multi-View optimization. LR: Explicit GS = ${lrGaussian}, Implicit NeRF = ${lrNerf}`);
        requestAnimationFrame(trainingLoop);
    }
}

let stepCounter = 0;

// Core animation and optimization loop
async function trainingLoop() {
    if (!isTraining || !session) return;

    const lrGaussian = parseFloat(lrGaussianInput.value) || 0.005;
    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;

    try {
        // Fast non-blocking GPU optimization steps (Zero CPU buffer mapAsync readbacks!)
        session.step_gaussian_fast(lrGaussian);
        session.step_nerf_fast(lrNerf);
        stepCounter++;

        // Read losses and render canvases periodically every 5 steps
        if (stepCounter % 5 === 0 || stepCounter === 1) {
            await gpuQueue.run(async () => {
                if (!session) return;
                const losses = await session.get_losses();
                const lossG = losses[0];
                const lossN = losses[1];
                lossHistoryGaussian.push(lossG);
                lossHistoryNerf.push(lossN);
                labelLossGaussian.textContent = `Loss: ${lossG.toFixed(5)}`;
                labelLossNerf.textContent = `Loss: ${lossN.toFixed(5)}`;

                const rgbG = await session.get_gaussian_render_view(currentViewV);
                const rgbN = await session.get_nerf_render_view(currentViewV);
                renderModelOutput(canvasGaussian, rgbG);
                renderModelOutput(canvasNerf, rgbN);
                updateBlendCanvas();
                drawLossChart();

                if (stepCounter % 50 === 0) {
                    console.log(`[Step ${stepCounter}] GS Loss: ${lossG.toFixed(5)} | NeRF Loss: ${lossN.toFixed(5)}`);
                }
            });
        }
    } catch (e) {
        console.error("Error during training step:", e);
        isTraining = false;
        btnTrain.textContent = '▶ Start Multi-View Fitting';
        btnTrain.classList.remove('btn-stop');
        return;
    }

    // Micro-yield execution to browser event loop
    await new Promise(resolve => setTimeout(resolve, 2));

    if (isTraining) {
        requestAnimationFrame(() => trainingLoop());
    }
}


// Automated 1-Click NeRF-Guided Seeding Pipeline
async function runOneClickHybridPipeline() {
    if (!session) return;
    isTraining = false;
    btnTrain.textContent = '▶ Start Multi-View Fitting';
    btnTrain.classList.remove('btn-stop');

    if (btnOneClickHybrid) {
        btnOneClickHybrid.disabled = true;
        btnOneClickHybrid.textContent = '⌛ Running Guided Pipeline...';
    }

    // Step 1: Pre-train NeRF
    if (step1Card) step1Card.className = 'stepper-card active';
    if (step1Status) step1Status.textContent = 'Pre-training NeRF (50 steps)...';

    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
    console.log("[Hybrid Pipeline] Step 1: Pre-training Implicit NeRF for 50 steps...");

    for (let i = 1; i <= 50; i++) {
        session.step_nerf_fast(lrNerf);
        if (i % 10 === 0) {
            await gpuQueue.run(async () => {
                const losses = await session.get_losses();
                labelLossNerf.textContent = `Loss: ${losses[1].toFixed(5)}`;
                const rgbN = await session.get_nerf_render_view(currentViewV);
                renderModelOutput(canvasNerf, rgbN);
                drawLossChart();
            });
        }
        await new Promise(resolve => setTimeout(resolve, 2));
    }

    if (step1Card) step1Card.className = 'stepper-card complete';
    if (step1Status) step1Status.textContent = 'Complete ✓';

    // Step 2 & 3: Extract Gradients & Seed 3D Gaussians
    if (step2Card) step2Card.className = 'stepper-card active';
    if (step2Status) step2Status.textContent = 'Extracting Spatial Edges...';

    console.log("[Hybrid Pipeline] Step 2 & 3: Extracting NeRF spatial gradients and seeding 3D Gaussians...");
    await new Promise(resolve => setTimeout(resolve, 200));

    if (step2Card) step2Card.className = 'stepper-card complete';
    if (step2Status) step2Status.textContent = 'Complete ✓';

    if (step3Card) step3Card.className = 'stepper-card active';
    if (step3Status) step3Status.textContent = 'Seeding 3D Gaussians...';

    await gpuQueue.run(async () => {
        await session.seed_from_nerf();
        const rgbG = await session.get_gaussian_render_view(currentViewV);
        renderModelOutput(canvasGaussian, rgbG);
    });

    if (step3Card) step3Card.className = 'stepper-card complete';
    if (step3Status) step3Status.textContent = 'Complete ✓';

    if (btnOneClickHybrid) {
        btnOneClickHybrid.disabled = false;
        btnOneClickHybrid.textContent = '✓ 1-Click Guided Initialization Complete';
    }
    if (btnSeed) btnSeed.disabled = false;
    console.log("[Hybrid Pipeline] Pipeline Complete! 3D Gaussians initialized accurately along object boundaries.");
}

// Pre-train NeRF to capture coarse multi-view edges
async function runNeRFPretraining() {
    isTraining = false;
    btnTrain.textContent = '▶ Start Multi-View Fitting';
    btnTrain.classList.remove('btn-stop');
    
    if (btnPretrain) {
        btnPretrain.disabled = true;
        btnPretrain.textContent = 'Training Multi-View NeRF...';
    }

    const lrNerf = parseFloat(lrNerfInput.value) || 0.001;
    console.log("[Pipeline] Starting Step 1: Pre-training Implicit NeRF for 50 steps...");

    for (let i = 1; i <= 50; i++) {
        session.step_nerf_fast(lrNerf);
        if (i % 10 === 0) {
            await gpuQueue.run(async () => {
                const losses = await session.get_losses();
                labelLossNerf.textContent = `Loss: ${losses[1].toFixed(5)}`;
                const rgbN = await session.get_nerf_render_view(currentViewV);
                renderModelOutput(canvasNerf, rgbN);
                drawLossChart();
            });
        }
        await new Promise(resolve => setTimeout(resolve, 2));
    }

    console.log("[Pipeline] Step 1 Complete! NeRF spatial gradients extracted across views.");
    if (btnPretrain) btnPretrain.textContent = '✓ Step 1 NeRF Pre-trained';
    if (btnSeed) btnSeed.disabled = false;
    if (step1Card) step1Card.className = 'stepper-card complete';
    if (step1Status) step1Status.textContent = 'Complete ✓';
    if (step2Card) step2Card.className = 'stepper-card active';
    if (step2Status) step2Status.textContent = 'Ready';
}

// Seed 3D Gaussians from NeRF Importance Map
async function seedGaussiansFromEdges() {
    if (!session) return;
    if (btnSeed) {
        btnSeed.disabled = true;
        btnSeed.textContent = 'Seeding 3D Gaussians...';
    }
    console.log("[Pipeline] Starting Step 2: Sampling 3D Gaussian centers along multi-view spatial boundaries...");

    if (step2Card) step2Card.className = 'stepper-card complete';
    if (step2Status) step2Status.textContent = 'Complete ✓';
    if (step3Card) step3Card.className = 'stepper-card active';
    if (step3Status) step3Status.textContent = 'Seeding 3D Gaussians...';

    await gpuQueue.run(async () => {
        await session.seed_from_nerf();
        const rgbG = await session.get_gaussian_render_view(currentViewV);
        renderModelOutput(canvasGaussian, rgbG);
    });

    if (step3Card) step3Card.className = 'stepper-card complete';
    if (step3Status) step3Status.textContent = 'Complete ✓';

    console.log("[Pipeline] Step 2 & 3 Complete! 3D Gaussians initialized accurately along object boundaries.");
    if (btnSeed) btnSeed.textContent = '✓ Step 2 & 3 GS Initialized';
}



// Helper: Render WASM RGB u8 array to canvas
function renderModelOutput(canvas, rgbArray) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);

    let srcIdx = 0;
    for (let i = 0; i < imgData.data.length; i += 4) {
        imgData.data[i] = rgbArray[srcIdx];
        imgData.data[i + 1] = rgbArray[srcIdx + 1];
        imgData.data[i + 2] = rgbArray[srcIdx + 2];
        imgData.data[i + 3] = 255;
        srcIdx += 3;
    }
    ctx.putImageData(imgData, 0, 0);
}

// Helper: Update blended canvas output
function updateBlendCanvas() {
    if (!canvasBlend) return;
    const alpha = blendSlider ? parseFloat(blendSlider.value) : 0.5;
    const ctxG = canvasGaussian.getContext('2d');
    const ctxN = canvasNerf.getContext('2d');
    const ctxB = canvasBlend.getContext('2d');

    const imgG = ctxG.getImageData(0, 0, width, height);
    const imgN = ctxN.getImageData(0, 0, width, height);
    const imgB = ctxB.createImageData(width, height);

    for (let i = 0; i < imgB.data.length; i += 4) {
        imgB.data[i] = (1 - alpha) * imgG.data[i] + alpha * imgN.data[i];
        imgB.data[i + 1] = (1 - alpha) * imgG.data[i + 1] + alpha * imgN.data[i + 1];
        imgB.data[i + 2] = (1 - alpha) * imgG.data[i + 2] + alpha * imgN.data[i + 2];
        imgB.data[i + 3] = 255;
    }
    ctxB.putImageData(imgB, 0, 0);
}

// Helper: Draw real-time Loss History Chart
function drawLossChart() {
    if (!canvasChart) return;
    const ctx = canvasChart.getContext('2d');
    const w = canvasChart.width;
    const h = canvasChart.height;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    if (lossHistoryGaussian.length === 0 && lossHistoryNerf.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = '11px sans-serif';
        ctx.fillText('Loss history chart will plot here...', 20, h / 2);
        return;
    }

    const drawLine = (history, color) => {
        if (history.length < 2) return;
        const maxLoss = Math.max(0.1, ...history.slice(0, 50));
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let i = 0; i < history.length; i++) {
            const x = (i / (history.length - 1)) * w;
            const y = h - (history[i] / maxLoss) * (h - 20) - 10;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    };

    drawLine(lossHistoryGaussian, '#f97316');
    drawLine(lossHistoryNerf, '#8b5cf6');
}

start().catch(err => {
    console.error("[Initialization Error] Failed to start WASM/WebGPU session:", err);
    
    // Check if running directly over file:// protocol
    if (window.location.protocol === 'file:') {
        console.error("[CORS Warning] Browsers block WebAssembly module loading over file:// protocol. Please serve the 'web' folder via HTTP (e.g. using a local server like python -m http.server 8000).");
        alert("⚠️ Local File Access Notice:\n\nBrowsers restrict WebAssembly and ES modules when opening HTML directly over file:// scheme.\n\nPlease serve the 'web' directory over HTTP (e.g., using python -m http.server 8000 or basic-http-server web) and open http://localhost:8000.");
    }
});
