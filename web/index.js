import init, {
    create_turntable_3d_session,
    init_panic_hook,
    init_webgpu
} from './pkg/burn_nerf_guided_splats.js';

class GpuTaskQueue {
    constructor() { this.chain = Promise.resolve(); }
    async run(fn) {
        const next = this.chain.then(fn);
        this.chain = next.catch(() => {});
        return next;
    }
}

const gpuQueue = new GpuTaskQueue();
const W = 44, H = 44, NW = 28, NH = 28, WARMUP = 35;

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
const TRAIN_POSES = [-0.75, 0.26, 0.75, -0.22, 2.20, 0.26, -2.20, -0.22];

let session = null;
let isRacing = false;
let loopBusy = false;
let seeded = false;
let step = 0;
let numSplats = 26;
let yaw = 0.55, pitch = 0.26;
let dragging = false, lastX = 0, lastY = 0;

const backendPill = document.getElementById('backend-pill');
const stepPill = document.getElementById('step-pill');
const btnRace = document.getElementById('btn-race');
const btnReset = document.getElementById('btn-reset');

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
    if (!rgb) return;
    const cv = document.getElementById(id), ctx = cv.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0, j = 0; i < img.data.length; i += 4, j += 3) {
        img.data[i] = rgb[j]; img.data[i + 1] = rgb[j + 1]; img.data[i + 2] = rgb[j + 2]; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

function drawWireCubes() {
    ['ov-nerf', 'ov-blind', 'ov-guided'].forEach(id => {
        const cv = document.getElementById(id), c = cv.getContext('2d');
        const S = cv.width;
        c.clearRect(0, 0, S, S);
        const lo = 0.12, hi = 0.88;
        const v = [[lo,lo,lo],[hi,lo,lo],[hi,hi,lo],[lo,hi,lo],[lo,lo,hi],[hi,lo,hi],[hi,hi,hi],[lo,hi,hi]]
            .map(p => project3d(p[0], p[1], p[2], yaw, pitch));
        c.strokeStyle = 'rgba(148, 163, 184, 0.35)';
        c.lineWidth = 1.2;
        c.beginPath();
        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
            c.moveTo(v[a][0] * S, v[a][1] * S); c.lineTo(v[b][0] * S, v[b][1] * S);
        });
        c.stroke();
    });
}

async function refreshViews() {
    drawWireCubes();
    if (!session) return;
    await gpuQueue.run(async () => {
        const nRgb = await session.get_vol_nerf_turntable_render(yaw, pitch);
        const bRgb = await session.get_blind_turntable_render(yaw, pitch);
        paintRgb('cv-nerf', NW, NH, nRgb);
        paintRgb('cv-blind', W, H, bRgb);
        if (seeded) {
            const gRgb = await session.get_guided_turntable_render(yaw, pitch);
            paintRgb('cv-guided', W, H, gRgb);
        }
    });
}

async function resetSession() {
    isRacing = false; seeded = false; step = 0;
    btnRace.textContent = '▶ Start Fair 3D Race';
    btnRace.classList.remove('running');
    stepPill.textContent = 'Step 0';
    document.getElementById('status-nerf').textContent = `Warmup 0/${WARMUP}`;
    document.getElementById('bar-nerf').style.width = '0%';
    document.getElementById('score-blind').textContent = 'PSNR: -- dB';
    document.getElementById('score-guided').textContent = 'Waiting...';
    document.getElementById('bar-blind').style.width = '0%';
    document.getElementById('bar-guided').style.width = '0%';
    document.getElementById('cv-guided').getContext('2d').fillRect(0, 0, W, H);

    const frameLen = W * H * 3;
    const numViews = TRAIN_POSES.length / 2;
    const flat = new Uint8Array(numViews * frameLen);
    for (let k = 0; k < numViews; k++) {
        flat.set(renderTrue3dFrame(W, H, TRAIN_POSES[k * 2], TRAIN_POSES[k * 2 + 1]), k * frameLen);
    }

    await gpuQueue.run(async () => {
        session = create_turntable_3d_session(W, H, numSplats, flat, new Float32Array(TRAIN_POSES));
    });
    await refreshViews();
}

async function raceLoop() {
    if (loopBusy || !isRacing || !session) return;
    loopBusy = true;

    await gpuQueue.run(async () => {
        if (!isRacing || !session) return;
        step++;
        stepPill.textContent = `Step ${step}`;
        if (!dragging) yaw += 0.02;

        if (step <= WARMUP) {
            session.step_3d_warmup_fair(0.075, 0.025);
            document.getElementById('status-nerf').textContent = `3D Rays ${step}/${WARMUP}`;
            document.getElementById('bar-nerf').style.width = `${Math.round((step / WARMUP) * 100)}%`;
            if (step === WARMUP) {
                await session.seed_from_3d_vol_nerf();
                seeded = true;
                document.getElementById('status-nerf').textContent = '✓ 3D σ Seeded!';
            }
        } else {
            session.step_3d_race_fast(0.075);
        }

        if (step % 3 === 0 || step === WARMUP) {
            const losses = await session.get_turntable_losses();
            const dbB = -10 * Math.log10(Math.max(losses[0], 1e-6));
            document.getElementById('score-blind').textContent = `${dbB.toFixed(1)} dB`;
            document.getElementById('bar-blind').style.width = `${Math.min(100, Math.max(8, (dbB - 14) * 8))}%`;
            if (seeded) {
                const dbG = -10 * Math.log10(Math.max(losses[1], 1e-6));
                document.getElementById('score-guided').textContent = `${dbG.toFixed(1)} dB`;
                document.getElementById('bar-guided').style.width = `${Math.min(100, Math.max(8, (dbG - 14) * 8))}%`;
            }
            const nRgb = await session.get_vol_nerf_turntable_render(yaw, pitch);
            const bRgb = await session.get_blind_turntable_render(yaw, pitch);
            paintRgb('cv-nerf', NW, NH, nRgb);
            paintRgb('cv-blind', W, H, bRgb);
            if (seeded) {
                const gRgb = await session.get_guided_turntable_render(yaw, pitch);
                paintRgb('cv-guided', W, H, gRgb);
            }
            drawWireCubes();
        }
    });

    loopBusy = false;
    if (isRacing) {
        await new Promise(r => setTimeout(r, 6));
        requestAnimationFrame(raceLoop);
    }
}

async function start() {
    btnRace.onclick = () => {
        isRacing = !isRacing;
        btnRace.textContent = isRacing ? '⏸ Pause 3D Race' : '▶ Continue 3D Race';
        btnRace.classList.toggle('running', isRacing);
        if (isRacing) requestAnimationFrame(raceLoop);
    };
    btnReset.onclick = resetSession;
    document.querySelectorAll('.budget-btn').forEach(btn => {
        btn.onclick = async () => {
            numSplats = parseInt(btn.dataset.n, 10);
            document.querySelectorAll('.budget-btn').forEach(b => b.classList.toggle('active', b === btn));
            await resetSession();
        };
    });

    const yawSlider = document.getElementById('yaw-slider');
    if (yawSlider) {
        yawSlider.addEventListener('input', async e => {
            dragging = true;
            yaw = parseFloat(e.target.value);
            if (!isRacing) await refreshViews(); else drawWireCubes();
        });
        yawSlider.addEventListener('change', () => { dragging = false; });
    }

    await init();
    init_panic_hook();
    const hasGpu = await init_webgpu();
    backendPill.textContent = hasGpu ? '⚡ WebGPU Ready' : '🖥️ CPU WASM Ready';
    await resetSession();
}

start();
