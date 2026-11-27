use std::time::Instant;
use burn::backend::{Autodiff, Flex, Wgpu};
use burn::module::Param;
use burn::optim::AdamConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use burn_nerf_guided_splats::hybrid::seed_gaussians_from_3d_volume;
use burn_nerf_guided_splats::model::gaussian::GaussianModel;
use burn_nerf_guided_splats::model::nerf::VolumetricNerfModel;
use burn_nerf_guided_splats::training::{
    train_step_gaussian_turntable, train_step_volumetric_turntable,
};
use burn_nerf_guided_splats::utils::tensor_to_image;

fn saturn_3d_spheres() -> Vec<(f32, f32, f32, f32, [f32; 3])> {
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

fn render_saturn_3d<B: burn::tensor::backend::Backend>(
    w: usize,
    h: usize,
    yaw: f32,
    pitch: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    let spheres = saturn_3d_spheres();
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
    Tensor::<B, 3>::from_data(TensorData::new(data, [h, w, 3]), device)
}

fn run_pipeline<B: AutodiffBackend>(backend_label: &str, device: B::Device) {
    println!("==================================================================");
    println!("  🪐 3D Volumetric NeRF-Guided Gaussian Splatting — Burn ({})  ", backend_label);
    println!("==================================================================");

    let (w, h, cw, ch, n_splats) = (36, 36, 16, 16, 26);
    let out_res = 128;
    let train_poses = [(-0.75_f32, 0.25_f32), (0.75, -0.22), (2.20, 0.25), (-2.20, -0.22)];
    let novel_poses = [(0.0_f32, 0.30_f32), (1.45, 0.15), (-1.45, -0.25), (3.10, -0.18)];

    let full_train: Vec<(f32, f32, Tensor<B, 3>)> = train_poses
        .iter()
        .map(|&(y, p)| (y, p, render_saturn_3d::<B>(w, h, y, p, &device)))
        .collect();
    let coarse_train: Vec<(f32, f32, Tensor<B, 3>)> = train_poses
        .iter()
        .map(|&(y, p)| (y, p, render_saturn_3d::<B>(cw, ch, y, p, &device)))
        .collect();
    let novel_test: Vec<(f32, f32, Tensor<B, 3>)> = novel_poses
        .iter()
        .map(|&(y, p)| (y, p, render_saturn_3d::<B>(w, h, y, p, &device)))
        .collect();

    let psnr = |m: f32| -10.0 * m.max(1e-8).log10();

    // 1. Train Blind 3DGS Baseline
    println!("\n[1/3] Training Blind 3D Gaussian Splatting (N={}, 75 steps)...", n_splats);
    let t0 = Instant::now();
    let mut blind = GaussianModel::<B>::new(n_splats, &device);
    blind.set_alpha_compositing(true);
    blind.scales = Param::from_tensor(Tensor::<B, 2>::full([n_splats, 2], -3.05, &device));
    let mut opt_b = AdamConfig::new().init();
    for step in 1..=75 {
        let (m, l) = train_step_gaussian_turntable(blind, &mut opt_b, &full_train, 0.075);
        blind = m;
        if step == 1 || step % 25 == 0 {
            let mse = l.into_data().into_vec::<f32>().unwrap()[0];
            println!("  Step {:2}/75 - Train PSNR: {:5.2} dB", step, psnr(mse));
        }
    }
    let ms_blind = t0.elapsed().as_millis();
    let novel_blind_mse = blind.forward_loss_turntable(&novel_test).inner().into_data().into_vec::<f32>().unwrap()[0];

    // 2. Train Coarse 3D Volumetric NeRF & Farthest-Point Seed 3D Gaussians
    println!("\n[2/3] Ray-Marching Coarse 3D Volumetric NeRF (45 warmup steps)...");
    let t1 = Instant::now();
    let mut vol_nerf = VolumetricNerfModel::<B>::new(4, 32, 14, &device);
    let mut opt_n = AdamConfig::new().init();
    for step in 1..=45 {
        let (m, _) = train_step_volumetric_turntable(vol_nerf, &mut opt_n, &coarse_train, 2.6e-2);
        vol_nerf = m;
        if step == 1 || step == 25 || step == 45 {
            println!("  Warmup {:2}/45 - Learning 3D density field σ(x, y, z)...", step);
        }
    }

    let voxels = block_on_ready(vol_nerf.extract_3d_density_grid_async(15));
    println!(
        "  -> Extracted {} 3D voxels & seeded N={} Gaussians via farthest-point selection",
        voxels.len() / 7,
        n_splats
    );

    // 3. Refine 3D Volumetric-Guided Gaussians with Kerbl-Scheduled LRs
    println!("\n[3/3] Refining 3D Volumetric-Guided Gaussians (50 Kerbl-scheduled steps)...");
    let mut guided = seed_gaussians_from_3d_volume::<B>(&voxels, 0.01, n_splats, &device);
    let mut opt_g = AdamConfig::new().init();
    for step in 1..=50 {
        let (m, l) = train_step_gaussian_turntable(guided, &mut opt_g, &full_train, 0.075);
        guided = m;
        if step == 1 || step % 25 == 0 {
            let mse = l.into_data().into_vec::<f32>().unwrap()[0];
            println!("  Step {:2}/50 - Train PSNR: {:5.2} dB", step, psnr(mse));
        }
    }
    let ms_guided = t1.elapsed().as_millis();
    let novel_guided_mse = guided.forward_loss_turntable(&novel_test).inner().into_data().into_vec::<f32>().unwrap()[0];

    // Export high-res 128x128 novel-view PNG comparisons
    let (show_yaw, show_pitch) = (0.55_f32, 0.26_f32);
    tensor_to_image(render_saturn_3d::<B::InnerBackend>(out_res, out_res, show_yaw, show_pitch, &Default::default()))
        .save("target.png")
        .expect("Failed to save target.png");
    tensor_to_image(vol_nerf.render_turntable(out_res, out_res, show_yaw, show_pitch).inner())
        .save("output_nerf.png")
        .expect("Failed to save output_nerf.png");
    tensor_to_image(blind.render_turntable(out_res, out_res, show_yaw, show_pitch).inner())
        .save("output_blind_3dgs.png")
        .expect("Failed to save output_blind_3dgs.png");
    tensor_to_image(guided.render_turntable(out_res, out_res, show_yaw, show_pitch).inner())
        .save("output_guided_3dgs.png")
        .expect("Failed to save output_guided_3dgs.png");

    println!("\n==================================================================");
    println!(
        "  Blind 3DGS           [{:5} ms] -> Novel 360° PSNR: {:5.2} dB",
        ms_blind,
        psnr(novel_blind_mse)
    );
    println!(
        "  3D-Guided 3DGS       [{:5} ms] -> Novel 360° PSNR: {:5.2} dB ({:+.2} dB!)",
        ms_guided,
        psnr(novel_guided_mse),
        psnr(novel_guided_mse) - psnr(novel_blind_mse)
    );
    println!("  Saved 128×128 renders: target.png, output_nerf.png, output_blind_3dgs.png, output_guided_3dgs.png");
    println!("==================================================================");
}

fn main() {
    let use_gpu = std::env::args().any(|a| a == "--gpu");
    if use_gpu {
        run_pipeline::<Autodiff<Wgpu>>("Wgpu GPU", Default::default());
    } else {
        run_pipeline::<Autodiff<Flex>>("Flex CPU", Default::default());
    }
}

fn block_on_ready<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker { raw_waker() }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = unsafe { Pin::new_unchecked(&mut fut) }.poll(&mut cx) {
            return v;
        }
    }
}
