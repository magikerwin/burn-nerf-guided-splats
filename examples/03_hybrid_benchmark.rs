//! Example 03: Multi-Topology 3D Volumetric NeRF -> 3D Gaussian Splatting Benchmark
//!
//! Evaluates Blind 3DGS vs. 3D Volumetric Ray-Marched NeRF-Guided 3DGS across 3 distinct
//! 3D scene topologies inside `[0, 1]^3` (4 training views + 4 unseen 360° novel test views):
//!   1. `Saturn`      — Multi-scale core + 8 tilted 3D ring nodes + 2 outer moons (front/back occlusion)
//!   2. `DNA Helix`   — Two intertwined 3D spirals (continuous 360° depth-crossing parallax)
//!   3. `Hollow Cage` — 8 outer corners + 6 shell nodes with a 100% empty interior
//!
//! Run with: `cargo run --release --example 03_hybrid_benchmark`

use std::time::Instant;
use burn::backend::{Autodiff, Flex};
use burn::module::Param;
use burn::optim::AdamConfig;
use burn::tensor::{Tensor, TensorData};
use burn_nerf_guided_splats::hybrid::seed_gaussians_from_3d_volume;
use burn_nerf_guided_splats::model::gaussian::GaussianModel;
use burn_nerf_guided_splats::model::nerf::VolumetricNerfModel;
use burn_nerf_guided_splats::training::{
    train_step_gaussian_turntable, train_step_volumetric_turntable,
};

type B = Autodiff<Flex>;

#[derive(Clone, Copy)]
enum Scene3D {
    Saturn,
    DnaHelix,
    HollowCage,
}

impl Scene3D {
    fn name(self) -> &'static str {
        match self {
            Self::Saturn => "Saturn Ring",
            Self::DnaHelix => "3D DNA Helix",
            Self::HollowCage => "Hollow 3D Cage",
        }
    }

    fn spheres(self) -> Vec<(f32, f32, f32, f32, [f32; 3])> {
        match self {
            Self::Saturn => {
                let mut s = Vec::with_capacity(11);
                s.push((0.50, 0.50, 0.50, 0.095, [0.98, 0.78, 0.20]));
                for k in 0..8 {
                    let ang = (k as f32) * std::f32::consts::TAU / 8.0;
                    let col = if k % 2 == 0 {
                        [0.18, 0.86, 0.98]
                    } else {
                        [0.98, 0.44, 0.26]
                    };
                    s.push((
                        0.50 + 0.27 * ang.cos(),
                        0.50 + 0.12 * ang.sin(),
                        0.50 + 0.27 * ang.sin(),
                        0.056,
                        col,
                    ));
                }
                s.push((0.22, 0.23, 0.70, 0.062, [0.24, 0.94, 0.56]));
                s.push((0.78, 0.77, 0.30, 0.062, [0.82, 0.46, 0.98]));
                s
            }
            Self::DnaHelix => {
                let mut s = Vec::with_capacity(14);
                for k in 0..7 {
                    let t = (k as f32) / 6.0;
                    let y = 0.18 + 0.64 * t;
                    let ang = t * 3.8;
                    // Strand A (Signal Vermilion)
                    s.push((
                        0.50 + 0.23 * ang.cos(),
                        y,
                        0.50 + 0.23 * ang.sin(),
                        0.058,
                        [0.95, 0.34, 0.18],
                    ));
                    // Strand B (Ocean Cyan, opposite phase)
                    let ang_b = ang + std::f32::consts::PI;
                    s.push((
                        0.50 + 0.23 * ang_b.cos(),
                        y,
                        0.50 + 0.23 * ang_b.sin(),
                        0.058,
                        [0.18, 0.82, 0.96],
                    ));
                }
                s
            }
            Self::HollowCage => {
                let mut s = Vec::with_capacity(14);
                // 8 outer cube corners (100% hollow center!)
                for &x in &[0.27_f32, 0.73] {
                    for &y in &[0.27_f32, 0.73] {
                        for &z in &[0.27_f32, 0.73] {
                            s.push((x, y, z, 0.058, [0.98, 0.78, 0.22]));
                        }
                    }
                }
                // 6 outer face-center nodes
                let faces = [
                    (0.50, 0.23, 0.50),
                    (0.50, 0.77, 0.50),
                    (0.23, 0.50, 0.50),
                    (0.77, 0.50, 0.50),
                    (0.50, 0.50, 0.23),
                    (0.50, 0.50, 0.77),
                ];
                for (x, y, z) in faces {
                    s.push((x, y, z, 0.054, [0.16, 0.86, 0.78]));
                }
                s
            }
        }
    }
}

fn render_scene_3d<Bk: burn::tensor::backend::Backend>(
    scene: Scene3D,
    w: usize,
    h: usize,
    yaw: f32,
    pitch: f32,
    device: &Bk::Device,
) -> Tensor<Bk, 3> {
    let spheres = scene.spheres();
    let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());

    let mut proj: Vec<(f32, f32, f32, f32, [f32; 3])> = spheres
        .iter()
        .map(|&(x, y, z, rad, rgb)| {
            let (xw, yw, zw) = (x - 0.5, y - 0.5, z - 0.5);
            let x1 = xw * cy - zw * sy;
            let z1 = xw * sy + zw * cy;
            (yw * sp + z1 * cp + 0.5, x1 + 0.5, yw * cp - z1 * sp + 0.5, rad, rgb)
        })
        .collect();
    proj.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let aa = 0.85 / (w as f32);
    let mut data = vec![0.0_f32; h * w * 3];
    for r in 0..h {
        let py = (r as f32 + 0.5) / (h as f32);
        for c in 0..w {
            let px = (c as f32 + 0.5) / (w as f32);
            let idx = (r * w + c) * 3;
            let (mut t_rem, mut ar, mut ag, mut ab) = (1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for &(_, u, v, rad, rgb) in &proj {
                let dist = ((px - u).powi(2) + (py - v).powi(2)).sqrt();
                if dist >= rad + aa {
                    continue;
                }
                let cov = ((rad + aa - dist) / (2.0 * aa)).clamp(0.0, 1.0);
                let shade = 0.72 + 0.28 * (1.0 - (dist / rad).min(1.0).powi(2)).max(0.0).sqrt();
                let wt = t_rem * cov;
                ar += wt * rgb[0] * shade;
                ag += wt * rgb[1] * shade;
                ab += wt * rgb[2] * shade;
                t_rem *= 1.0 - cov;
                if t_rem < 0.01 {
                    break;
                }
            }
            data[idx] = ar;
            data[idx + 1] = ag;
            data[idx + 2] = ab;
        }
    }
    Tensor::<Bk, 3>::from_data(TensorData::new(data, [h, w, 3]), device)
}

fn eval_novel_psnr(model: &GaussianModel<B>, novel_views: &[(f32, f32, Tensor<B, 3>)]) -> f32 {
    let mse = model
        .forward_loss_turntable(novel_views)
        .inner()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    -10.0 * mse.max(1e-8).log10()
}

fn main() {
    let device = Default::default();
    let (w, h, cw, ch, n_splats) = (36, 36, 16, 16, 28);
    let train_poses = [(-0.75_f32, 0.25_f32), (0.75, -0.22), (2.20, 0.25), (-2.20, -0.22)];
    let novel_poses = [(0.0_f32, 0.30_f32), (1.45, 0.15), (-1.45, -0.25), (3.10, -0.18)];

    println!(
        "=== Multi-Topology 3D Volumetric NeRF -> 3DGS Benchmark (4 Train + 4 Novel 360° Views, N={}) ===",
        n_splats
    );

    for scene in [Scene3D::Saturn, Scene3D::DnaHelix, Scene3D::HollowCage] {
        let full_train: Vec<(f32, f32, Tensor<B, 3>)> = train_poses
            .iter()
            .map(|&(y, p)| (y, p, render_scene_3d::<B>(scene, w, h, y, p, &device)))
            .collect();
        let coarse_train: Vec<(f32, f32, Tensor<B, 3>)> = train_poses
            .iter()
            .map(|&(y, p)| (y, p, render_scene_3d::<B>(scene, cw, ch, y, p, &device)))
            .collect();
        let novel_test: Vec<(f32, f32, Tensor<B, 3>)> = novel_poses
            .iter()
            .map(|&(y, p)| (y, p, render_scene_3d::<B>(scene, w, h, y, p, &device)))
            .collect();

        // Racer A: Blind 3DGS (Kerbl-scheduled LRs, 75 full-res steps)
        let t0 = Instant::now();
        let mut blind = GaussianModel::<B>::new(n_splats, &device);
        blind.set_alpha_compositing(true);
        blind.scales = Param::from_tensor(Tensor::<B, 2>::full([n_splats, 2], -3.05, &device));
        let mut opt_b = AdamConfig::new().init();
        for _ in 1..=75 {
            let (m, _) = train_step_gaussian_turntable(blind, &mut opt_b, &full_train, 0.075);
            blind = m;
        }
        let ms_b = t0.elapsed().as_millis();
        let novel_b = eval_novel_psnr(&blind, &novel_test);

        // Racer B: Coarse 3D Volumetric NeRF (45 steps) -> 3D Farthest-Point Seeding -> 50 3DGS steps
        let t1 = Instant::now();
        let mut vol_nerf = VolumetricNerfModel::<B>::new(4, 32, 14, &device);
        let mut opt_n = AdamConfig::new().init();
        for _ in 1..=45 {
            let (m, _) = train_step_volumetric_turntable(vol_nerf, &mut opt_n, &coarse_train, 2.6e-2);
            vol_nerf = m;
        }
        let grid_vec = futures_lite_block_on(vol_nerf.extract_3d_density_grid_async(15));
        let mut guided = seed_gaussians_from_3d_volume::<B>(&grid_vec, 0.01, n_splats, &device);
        let mut opt_g = AdamConfig::new().init();
        for _ in 1..=50 {
            let (m, _) = train_step_gaussian_turntable(guided, &mut opt_g, &full_train, 0.075);
            guided = m;
        }
        let ms_g = t1.elapsed().as_millis();
        let novel_g = eval_novel_psnr(&guided, &novel_test);

        println!(
            "  [{:<14}] Blind 3DGS ({:4} ms): {:5.2} dB  |  3D-Guided ({:4} ms): {:5.2} dB  ({:+.2} dB!)",
            scene.name(),
            ms_b,
            novel_b,
            ms_g,
            novel_g,
            novel_g - novel_b
        );
    }
}

fn futures_lite_block_on<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker { dummy_raw_waker() }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(val) = unsafe { Pin::new_unchecked(&mut fut) }.poll(&mut cx) {
            return val;
        }
    }
}
