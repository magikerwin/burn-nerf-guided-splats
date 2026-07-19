use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use burn_nerf_guided_splats::model::gaussian::GaussianModel;
use burn_nerf_guided_splats::model::nerf::NerfModel;
use burn_nerf_guided_splats::model::ImageFitter;
use burn_nerf_guided_splats::training::train_step;
use burn_nerf_guided_splats::utils::{generate_synthetic_target, image_to_tensor, tensor_to_image};

fn main() {
    // We use the Wgpu backend wrapped in Autodiff for training
    type B = Autodiff<Wgpu>;
    let device = Default::default();

    println!("=======================================================");
    println!("    NeRF-Guided 2D Gaussian Splatting in Burn CLI      ");
    println!("=======================================================");

    // 1. Generate and save synthetic target circle
    let width = 128;
    let height = 128;
    let target_img = generate_synthetic_target(width, height);
    target_img.save("target.png").expect("Failed to save target.png");
    println!("[1/3] Generated synthetic target circle -> target.png");

    let target_tensor = image_to_tensor::<B>(&target_img, &device);

    // 2. Train Gaussian Splatting Model (Explicit representation)
    println!("\n[2/3] Training 2D Gaussian Splatting Model (Explicit)...");
    let mut gaussian_model = GaussianModel::<B>::new(500, &device);
    let mut gaussian_optim = AdamConfig::new().init();
    let gaussian_lr = 5e-3;

    for step in 1..=1000 {
        let (updated, loss) = train_step(gaussian_model, &mut gaussian_optim, &target_tensor, gaussian_lr);
        gaussian_model = updated;
        if step % 100 == 0 || step == 1 {
            println!("  Step {:4}/1000 - Loss: {:.6}", step, loss);
        }
    }

    // Save final Gaussian splat rendering
    let gaussian_render = gaussian_model.render(width as usize, height as usize);
    let gaussian_out_img = tensor_to_image(gaussian_render);
    gaussian_out_img.save("output_gaussian.png").expect("Failed to save output_gaussian.png");
    println!("  -> Saved Gaussian render to output_gaussian.png");

    // 3. Train Coordinate MLP / NeRF Model (Implicit representation)
    println!("\n[3/3] Training Coordinate MLP (NeRF) Model (Implicit)...");
    let mut nerf_model = NerfModel::<B>::new(8, 64, &device);
    let mut nerf_optim = AdamConfig::new().init();
    let nerf_lr = 1e-3;

    for step in 1..=1000 {
        let (updated, loss) = train_step(nerf_model, &mut nerf_optim, &target_tensor, nerf_lr);
        nerf_model = updated;
        if step % 100 == 0 || step == 1 {
            println!("  Step {:4}/1000 - Loss: {:.6}", step, loss);
        }
    }

    // Save final NeRF coordinate MLP rendering
    let nerf_render = nerf_model.render(width as usize, height as usize);
    let nerf_out_img = tensor_to_image(nerf_render);
    nerf_out_img.save("output_nerf.png").expect("Failed to save output_nerf.png");
    println!("  -> Saved NeRF render to output_nerf.png");

    println!("\n=======================================================");
    println!("Training complete! You can inspect files in project root:");
    println!("  - target.png: The target circle");
    println!("  - output_gaussian.png: 2D Gaussian Splat reconstruction");
    println!("  - output_nerf.png: 2D NeRF coordinate MLP reconstruction");
    println!("=======================================================");
}
