//! Example 01: Minimal Autodiff 2D/3D Gaussian Splatting in Burn (~45 lines)
//! Run with: `cargo run --example 01_single_splat`

use burn::backend::{Autodiff, Flex};
use burn::optim::AdamConfig;
use burn_nerf_guided_splats::model::gaussian::GaussianModel;
use burn_nerf_guided_splats::training::train_step;
use burn_nerf_guided_splats::utils::{generate_synthetic_target, image_to_tensor};

fn main() {
    type B = Autodiff<Flex>;
    let device = Default::default();

    println!("=== Example 01: Minimal Burn Autodiff Gaussian Splatting ===");
    let target_img = generate_synthetic_target(32, 32);
    let target = image_to_tensor::<B>(&target_img, &device);

    let mut model = GaussianModel::<B>::new(40, &device);
    model.set_alpha_compositing(true);
    let mut optim = AdamConfig::new().init();

    for step in 1..=150 {
        let (updated, loss_t) = train_step(model, &mut optim, &target, 0.02);
        model = updated;
        if step == 1 || step % 50 == 0 {
            let mse = loss_t.into_data().into_vec::<f32>().unwrap()[0];
            let psnr = -10.0 * mse.max(1e-8).log10();
            println!("  Step {:3}/150 | MSE: {:.5} | PSNR: {:5.2} dB", step, mse, psnr);
        }
    }
}
