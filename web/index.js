import init, {
    create_turntable_3d_session,
    init_panic_hook,
    init_webgpu
} from './pkg/burn_nerf_guided_splats.js';

class GpuTaskQueue {
    constructor() { this.chain = Promise.resolve(); }
    async run(fn) {
        const next = this.chain.then(fn);
        this.chain = next.catch(err => console.error('[GpuTaskQueue]', err));
        return next;
    }
}

const gpuQueue = new GpuTaskQueue();
const W = 44, H = 44, NW = 28, NH = 28, OV = 176, WARMUP = 35;

// 11-Part True 3D Saturn System inside [0, 1]^3: Golden Core + 8 Tilted 3D Ring Nodes + 2 Moons
const SPHERES = (() => {
    const s = [[0.50, 0.50, 0.50, 0.095, [250, 199, 51]]];
    for (let k = 0; k < 8; k++) {
        const a = (k * Math.PI * 2) / 8;
        const col = k % 2 === 0 ? [46, 219, 250] : [250, 112, 66];
        s.push([0.50 + 0.27 * Math.cos(a), 0.50 + 0.12 * Math.sin(a), 0.50 + 0.27 * Math.sin(a), 0.056, col]);
    }
    s.push([0.22, 0.23, 0.70, 0.062, [61, 240, 143]]);
    s.push([0.78, 0.77, 0.30, 0.062, [209, 117, 250]]);
    return s;
})();

const ALL_POSES = [
    [-0.75, 0.26], [0.75, -0.22], [2.20, 0.26], [-2.20, -0.22],
    [0.00, 0.32], [1.57, -0.28], [3.14, 0.20], [-1.57, -0.24]
];

let session = null;
let isRacing = false;
let loopBusy = false;
let seeded = false;
let step = 0;
let numViews = 4;
let numSplats = 26;
let yaw = 0.55;
let pitch = 0.26;
let showEllipses = true;
let userAdjustingCamera = false;

const backendPill = document.getElementById('backend-pill');
const stepPill = document.getElementById('step-pill');
const btnRace = document.getElementById('btn-race');
const btnReset = document.getElementById('btn-reset');
const yawSlider = document.getElementById('yaw-slider');
const pitchSlider = document.getElementById('pitch-slider');
const chkEllipses = document.getElementById('chk-ellipses');
const logsBox = document.getElementById('systemLogsBox');

function logLine(msg) {
    if (!logsBox) return;
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    logsBox.textContent += `\n[${ts}] ${msg}`;
    logsBox.scrollTop = logsBox.scrollHeight;
}

function project3d(x, y, z, yw, pt) {
    const cy = Math.cos(yw), sy = Math.sin(yw), cp = Math.cos(pt), sp = Math.sin(pt);
    const x0 = x - 0.5, y0 = y - 0.5, z0 = z - 0.5;
    const x1 = x0 * cy - z0 * sy, z1 = x0 * sy + z0 * cy;
    return [x1 + 0.5, y0 * cp - z1 * sp + 0.5, y0 * sp + z1 * cp + 0.5];
}

function renderTrue3dFrame(w, h, yw, pt) {
    const out = new Uint8Array(w * h * 3);
    const sorted = SPHERES.map(s => {
        const [u, v, d] = project3d(s[0], s[1], s[2], yw, pt);
        return { u, v, d, r: s[3], c: s[4] };
    }).sort((a, b) => a.d - b.d);
    const aa = 0.85 / w;

    for (let r = 0; r < h; r++) {
        const py = (r + 0.5) / h;
        for (let c = 0; c < w; c++) {
            const px = (c + 0.5) / w, idx = (r * w + c) * 3;
            let T = 1.0, ar = 0, ag = 0, ab = 0;
            for (const s of sorted) {
                const dist = Math.hypot(px - s.u, py - s.v);
                if (dist >= s.r + aa) continue;
                const cov = Math.min(1, Math.max(0, (s.r + aa - dist) / (2 * aa)));
                const shade = 0.72 + 0.28 * Math.sqrt(Math.max(0, 1 - Math.min(1, (dist / s.r) ** 2)));
                const wt = T * cov;
                ar += wt * s.c[0] * shade; ag += wt * s.c[1] * shade; ab += wt * s.c[2] * shade;
                T *= (1 - cov);
                if (T < 0.01) break;
            }
            out[idx] = Math.round(ar); out[idx + 1] = Math.round(ag); out[idx + 2] = Math.round(ab);
        }
    }
    return out;
}

function paintRgb(id, w, h, rgb) {
    if (!rgb || rgb.length < w * h * 3) return;
    const cv = document.getElementById(id), ctx = cv.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0, j = 0; i < img.data.length; i += 4, j += 3) {
        img.data[i] = rgb[j];
        img.data[i + 1] = rgb[j + 1];
        img.data[i + 2] = rgb[j + 2];
        img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

function computePsnrFromU8(predU8, gtU8) {
    let sq = 0;
    for (let i = 0; i < gtU8.length; i++) {
        const diff = (predU8[i] - gtU8[i]) / 255.0;
        sq += diff * diff;
    }
    const mse = Math.max(sq / gtU8.length, 1e-8);
    return -10 * Math.log10(mse);
}

function drawOverlay(ovId, splatDesc) {
    const cv = document.getElementById(ovId);
    if (!cv) return;
    const c = cv.getContext('2d');
    c.clearRect(0, 0, OV, OV);

    const lo = 0.12, hi = 0.88;
    const v = [
        [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
        [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi]
    ].map(p => project3d(p[0], p[1], p[2], yaw, pitch));

    c.strokeStyle = 'rgba(148, 163, 184, 0.30)';
    c.lineWidth = 1.1;
    c.beginPath();
    [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
        c.moveTo(v[a][0] * OV, v[a][1] * OV);
        c.lineTo(v[b][0] * OV, v[b][1] * OV);
    });
    c.stroke();

    if (!showEllipses || !splatDesc) return;
    const count = Math.floor(splatDesc.length / 10);
    for (let i = 0; i < count; i++) {
        const o = i * 10;
        const alpha = splatDesc[o + 9];
        if (alpha < 0.12) continue;
        const ux = splatDesc[o] * OV;
        const vy = splatDesc[o + 1] * OV;
        const rx = Math.min(36, Math.max(2.2, splatDesc[o + 3] * OV));
        const ry = Math.min(36, Math.max(2.2, splatDesc[o + 4] * OV));
        const theta = splatDesc[o + 5];
        const r = Math.round(splatDesc[o + 6] * 255);
        const g = Math.round(splatDesc[o + 7] * 255);
        const b = Math.round(splatDesc[o + 8] * 255);

        c.save();
        c.translate(ux, vy);
        c.rotate(theta);
        c.beginPath();
        c.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
        c.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.88)`;
        c.lineWidth = 1.15;
        c.stroke();
        c.restore();
    }
}

async function renderAllStages() {
    const gtRgb = renderTrue3dFrame(W, H, yaw, pitch);
    paintRgb('cv-gt', W, H, gtRgb);
    drawOverlay('ov-gt', null);

    if (!session) {
        drawOverlay('ov-nerf', null);
        drawOverlay('ov-blind', null);
        drawOverlay('ov-guided', null);
        return;
    }

    const nRgb = await session.get_vol_nerf_turntable_render(NW, NH, yaw, pitch);
    const bRgb = await session.get_blind_turntable_render(yaw, pitch);
    const bSplats = showEllipses ? await session.get_blind_turntable_splats(yaw, pitch) : null;

    paintRgb('cv-nerf', NW, NH, nRgb);
    paintRgb('cv-blind', W, H, bRgb);
    drawOverlay('ov-nerf', null);
    drawOverlay('ov-blind', bSplats);

    const dbB = computePsnrFromU8(bRgb, gtRgb);
    document.getElementById('score-blind').textContent = `${dbB.toFixed(1)} dB`;
    document.getElementById('bar-blind').style.width = `${Math.min(100, Math.max(8, (dbB - 14) * 8.5))}%`;

    if (seeded) {
        const gRgb = await session.get_guided_turntable_render(yaw, pitch);
        const gSplats = showEllipses ? await session.get_guided_turntable_splats(yaw, pitch) : null;
        paintRgb('cv-guided', W, H, gRgb);
        drawOverlay('ov-guided', gSplats);

        const dbG = computePsnrFromU8(gRgb, gtRgb);
        document.getElementById('score-guided').textContent = `${dbG.toFixed(1)} dB`;
        document.getElementById('bar-guided').style.width = `${Math.min(100, Math.max(8, (dbG - 14) * 8.5))}%`;
    } else {
        drawOverlay('ov-guided', null);
    }
}

async function resetSession() {
    isRacing = false;
    seeded = false;
    step = 0;
    btnRace.textContent = '▶ Start Fair 3D Race';
    btnRace.classList.remove('running');
    stepPill.textContent = 'Step 0';
    document.getElementById('pill-gt').textContent = `K=${numViews} · 3D`;
    document.getElementById('status-nerf').textContent = `0/${WARMUP}`;
    document.getElementById('bar-nerf').style.width = '0%';
    document.getElementById('score-blind').textContent = '-- dB';
    document.getElementById('score-guided').textContent = 'Wait';
    document.getElementById('bar-blind').style.width = '0%';
    document.getElementById('bar-guided').style.width = '0%';

    const gCtx = document.getElementById('cv-guided').getContext('2d');
    gCtx.fillStyle = '#06070b';
    gCtx.fillRect(0, 0, W, H);

    const activePoses = ALL_POSES.slice(0, numViews);
    const frameLen = W * H * 3;
    const flatRgb = new Uint8Array(numViews * frameLen);
    const flatPoses = new Float32Array(numViews * 2);

    for (let k = 0; k < numViews; k++) {
        const [yw, pt] = activePoses[k];
        flatPoses[k * 2] = yw;
        flatPoses[k * 2 + 1] = pt;
        flatRgb.set(renderTrue3dFrame(W, H, yw, pt), k * frameLen);
    }

    await gpuQueue.run(async () => {
        session = create_turntable_3d_session(W, H, numSplats, flatRgb, flatPoses);
        await renderAllStages();
    });

    logLine(`Session ready: K=${numViews} turntable views, N=${numSplats} 3D Gaussians (${session.get_backend_name()}).`);
}

async function raceLoop() {
    if (loopBusy || !isRacing || !session) return;
    loopBusy = true;

    await gpuQueue.run(async () => {
        if (!isRacing || !session) return;
        step++;
        stepPill.textContent = `Step ${step}`;

        if (!userAdjustingCamera) {
            yaw = ((yaw + 0.025 + Math.PI) % (2 * Math.PI)) - Math.PI;
            yawSlider.value = yaw.toFixed(2);
        }

        if (step <= WARMUP) {
            session.step_3d_warmup_fair(0.075, 0.026);
            document.getElementById('status-nerf').textContent = `Rays ${step}/${WARMUP}`;
            document.getElementById('bar-nerf').style.width = `${Math.round((step / WARMUP) * 100)}%`;

            if (step === WARMUP) {
                const voxels = await session.seed_from_3d_vol_nerf();
                seeded = true;
                document.getElementById('status-nerf').textContent = '✓ σ Seeded';
                logLine(`Step ${step}: Extracted ${(voxels.length / 7) | 0} 3D voxels & seeded N=${numSplats} Gaussians via farthest-point selection.`);
            }
        } else {
            session.step_3d_race_fast(0.075);
        }

        if (step % 2 === 0 || step === WARMUP) {
            await renderAllStages();
        }
    });

    loopBusy = false;
    if (isRacing) {
        await new Promise(r => setTimeout(r, 8));
        requestAnimationFrame(raceLoop);
    }
}

async function main() {
    btnRace.onclick = () => {
        isRacing = !isRacing;
        btnRace.textContent = isRacing ? '⏸ Pause 3D Race' : (step > 0 ? '▶ Resume 3D Race' : '▶ Start Fair 3D Race');
        btnRace.classList.toggle('running', isRacing);
        if (isRacing) requestAnimationFrame(raceLoop);
    };

    btnReset.onclick = () => {
        userAdjustingCamera = false;
        resetSession();
    };

    document.querySelectorAll('#view-tabs .tab-btn').forEach(btn => {
        btn.onclick = async () => {
            numViews = parseInt(btn.dataset.views, 10);
            document.querySelectorAll('#view-tabs .tab-btn').forEach(b => b.classList.toggle('active', b === btn));
            await resetSession();
        };
    });

    document.querySelectorAll('#budget-tabs .tab-btn').forEach(btn => {
        btn.onclick = async () => {
            numSplats = parseInt(btn.dataset.n, 10);
            document.querySelectorAll('#budget-tabs .tab-btn').forEach(b => b.classList.toggle('active', b === btn));
            await resetSession();
        };
    });

    yawSlider.addEventListener('input', e => {
        userAdjustingCamera = true;
        yaw = parseFloat(e.target.value);
        if (!isRacing) gpuQueue.run(renderAllStages);
    });
    yawSlider.addEventListener('change', () => { userAdjustingCamera = false; });

    pitchSlider.addEventListener('input', e => {
        userAdjustingCamera = true;
        pitch = parseFloat(e.target.value);
        if (!isRacing) gpuQueue.run(renderAllStages);
    });
    pitchSlider.addEventListener('change', () => { userAdjustingCamera = false; });

    chkEllipses.addEventListener('change', e => {
        showEllipses = e.target.checked;
        if (!isRacing) gpuQueue.run(renderAllStages);
    });

    await init();
    init_panic_hook();
    const hasGpu = await init_webgpu();
    backendPill.textContent = hasGpu ? '⚡ Burn WebGPU' : '🖥️ Burn CPU WASM';
    logLine(`Burn runtime initialized (${hasGpu ? 'WebGPU hardware acceleration' : 'Flex CPU WASM fallback'}).`);
    await resetSession();
}

main();
