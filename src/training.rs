use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use crate::model::{ImageFitter, MultiViewFitter};

/// Executes a single optimization step for any model implementing `ImageFitter` and `AutodiffModule`.
/// Returns the updated model and the loss tensor.
pub fn train_step<B: AutodiffBackend, M, O>(
    model: M,
    optimizer: &mut O,
    target_image: &Tensor<B, 3>,
    lr: f64,
) -> (M, Tensor<B::InnerBackend, 1>)
where
    M: ImageFitter<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    // 1. Forward pass: compute the reconstruction loss
    let loss = model.forward_loss(target_image);
    
    // Extract the inner tensor (non-autodiff) before backward consumes it
    let loss_inner = loss.clone().inner();

    // 2. Backward pass: compute gradients
    let grads = loss.backward();

    // 3. Map gradients to parameters
    let grads = GradientsParams::from_grads(grads, &model);

    // 4. Update the model parameters via the optimizer
    let updated_model = optimizer.step(lr, model, grads);

    (updated_model, loss_inner)
}

/// Executes a single optimization step evaluating joint multi-view loss across multiple views.
pub fn train_step_multiview<B: AutodiffBackend, M, O>(
    model: M,
    optimizer: &mut O,
    target_v0: &Tensor<B, 3>,
    target_v1: &Tensor<B, 3>,
    lr: f64,
) -> (M, Tensor<B::InnerBackend, 1>)
where
    M: MultiViewFitter<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    train_step_multiframe(model, optimizer, &[target_v0.clone(), target_v1.clone()], lr)
}

/// Executes a single optimization step evaluating joint multi-frame loss across an arbitrary slice of keyframe targets.
pub fn train_step_multiframe<B: AutodiffBackend, M, O>(
    model: M,
    optimizer: &mut O,
    targets: &[Tensor<B, 3>],
    lr: f64,
) -> (M, Tensor<B::InnerBackend, 1>)
where
    M: MultiViewFitter<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    // 1. Forward pass: compute multi-frame reconstruction loss
    let loss = model.forward_loss_multiframe(targets);
    
    // Extract inner tensor before backward consumes it
    let loss_inner = loss.clone().inner();

    // 2. Backward pass: compute gradients
    let grads = loss.backward();

    // 3. Map gradients to parameters
    let grads = GradientsParams::from_grads(grads, &model);

    // 4. Update model parameters via optimizer
    let updated_model = optimizer.step(lr, model, grads);

    (updated_model, loss_inner)
}

/// Per-parameter-group learning rates for 3D Gaussian Splatting (following Kerbl et al. 2023):
/// Position `means` require ~10x smaller Adam steps than color/opacity logits so newly seeded
/// 3D Gaussians fine-tune smoothly on surface geometry instead of jumping `+-lr` off the seed!
#[derive(Clone, Copy, Debug)]
pub struct GaussianLrSchedule {
    pub lr_means: f64,
    pub lr_scales: f64,
    pub lr_appearance: f64,
}

impl GaussianLrSchedule {
    /// Balanced 3DGS ratios scaled from a base appearance learning rate `base_lr` (e.g. `0.045`).
    pub fn from_base(base_lr: f64) -> Self {
        Self {
            lr_means: base_lr * 0.10,
            lr_scales: base_lr * 0.30,
            lr_appearance: base_lr,
        }
    }
}

/// Executes a single 3D turntable optimization step for `VolumetricNerfModel`.
pub fn train_step_volumetric_turntable<B: AutodiffBackend, O>(
    model: crate::model::nerf::VolumetricNerfModel<B>,
    optimizer: &mut O,
    views: &[(f32, f32, Tensor<B, 3>)],
    lr: f64,
) -> (crate::model::nerf::VolumetricNerfModel<B>, Tensor<B::InnerBackend, 1>)
where
    O: Optimizer<crate::model::nerf::VolumetricNerfModel<B>, B>,
{
    let loss = model.forward_loss_turntable(views);
    let loss_inner = loss.clone().inner();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    (optimizer.step(lr, model, grads), loss_inner)
}

/// Executes a single 3D turntable optimization step for `GaussianModel` using Kerbl-style
/// decoupled parameter-group learning rates (`lr_means << lr_scales < lr_appearance`).
pub fn train_step_gaussian_turntable<B: AutodiffBackend, O>(
    model: crate::model::gaussian::GaussianModel<B>,
    optimizer: &mut O,
    views: &[(f32, f32, Tensor<B, 3>)],
    base_lr: f64,
) -> (crate::model::gaussian::GaussianModel<B>, Tensor<B::InnerBackend, 1>)
where
    O: Optimizer<crate::model::gaussian::GaussianModel<B>, B>,
{
    train_step_gaussian_turntable_scheduled(
        model,
        optimizer,
        views,
        GaussianLrSchedule::from_base(base_lr),
    )
}

/// Executes a single 3D turntable optimization step with explicit per-group learning rates.
pub fn train_step_gaussian_turntable_scheduled<B: AutodiffBackend, O>(
    model: crate::model::gaussian::GaussianModel<B>,
    optimizer: &mut O,
    views: &[(f32, f32, Tensor<B, 3>)],
    schedule: GaussianLrSchedule,
) -> (crate::model::gaussian::GaussianModel<B>, Tensor<B::InnerBackend, 1>)
where
    O: Optimizer<crate::model::gaussian::GaussianModel<B>, B>,
{
    let loss = model.forward_loss_turntable(views);
    let loss_inner = loss.clone().inner();
    let mut raw_grads = loss.backward();

    let grads_means = GradientsParams::from_params(&mut raw_grads, &model, &[model.means.id]);
    let grads_scales = GradientsParams::from_params(&mut raw_grads, &model, &[model.scales.id]);
    let grads_appearance = GradientsParams::from_module(&mut raw_grads, &model);

    let model = optimizer.step(schedule.lr_means, model, grads_means);
    let model = optimizer.step(schedule.lr_scales, model, grads_scales);
    let model = optimizer.step(schedule.lr_appearance, model, grads_appearance);

    (model, loss_inner)
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::autodiff::Autodiff;
    use burn::backend::Flex;
    use burn::optim::AdamConfig;
    use crate::model::gaussian::GaussianModel;

    #[test]
    fn test_train_step_gaussian() {
        type B = Autodiff<Flex>;
        let device = Default::default();

        // 1. Instantiate the model under autodiff backend wrapper
        let model = GaussianModel::<B>::new(5, &device);

        // 2. Create target images and optimizer
        let target0 = Tensor::<B, 3>::zeros([8, 8, 3], &device);
        let target1 = Tensor::<B, 3>::zeros([8, 8, 3], &device);
        let mut optimizer = AdamConfig::new().init();

        // 3. Perform single training step
        let (model, loss_tensor) = train_step(model, &mut optimizer, &target0, 1e-3);
        let loss_val = loss_tensor.into_data().into_vec::<f32>().unwrap()[0];
        assert!(loss_val >= 0.0);

        let (_updated_model, mv_loss_tensor) = train_step_multiview(model, &mut optimizer, &target0, &target1, 1e-3);
        let mv_loss_val = mv_loss_tensor.into_data().into_vec::<f32>().unwrap()[0];
        assert!(mv_loss_val >= 0.0);
    }
}

