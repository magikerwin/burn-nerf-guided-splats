//! Example 02: Spectral Bias — Why Coordinate MLPs Need Fourier Positional Encoding
//! Compares L = 0 (raw x,y,v coords) vs L = 8 (Fourier features) head-to-head.
//! Run with: `cargo run --example 02_spectral_bias`

use burn::backend::{Autodiff, Flex};
use burn::optim::AdamConfig;
use burn_nerf_guided_splats::model::nerf::NerfModel;
use burn_nerf_guided_splats::training::train_step;
use burn_nerf_guided_splats::utils::{generate_synthetic_target, image_to_tensor};

fn run_nerf_experiment(num_frequencies: usize) -> (f32, f32, usize) {
    type B = Autodiff<Flex>;
    let device = Default::default();
    let target = image_to_tensor::<B>(&generate_synthetic_target(32, 32), &device);

    let mut model = NerfModel::<B>::new(num_frequencies, 48, &device);
    let params = model.num_parameters();
    let mut optim = AdamConfig::new().init();
    let mut last_mse = 1.0;

    for _ in 1..=120 {
        let (updated, loss_t) = train_step(model, &mut optim, &target, 3e-3);
        model = updated;
        last_mse = loss_t.into_data().into_vec::<f32>().unwrap()[0];
    }
    let psnr = -10.0 * last_mse.max(1e-8).log10();
    (last_mse, psnr, params)
}

fn main() {
    println!("=== Example 02: Spectral Bias Lab (L=0 Raw Coords vs L=8 Fourier Encoding) ===");
    let (mse_l0, psnr_l0, p0) = run_nerf_experiment(0);
    let (mse_l8, psnr_l8, p8) = run_nerf_experiment(8);

    println!("  BEFORE (L = 0, No Positional Encoding | {:5} params): MSE = {:.5} | PSNR = {:5.2} dB (Blurry blob!)", p0, mse_l0, psnr_l0);
    println!("  AFTER  (L = 8, Fourier Features       | {:5} params): MSE = {:.5} | PSNR = {:5.2} dB (+{:.2} dB sharper!)", p8, mse_l8, psnr_l8, psnr_l8 - psnr_l0);
}
