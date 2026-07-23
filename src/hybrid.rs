use burn::tensor::{backend::Backend, Tensor};

/// Computes the spatial gradient magnitude of a rendered image tensor of shape [H, W, C].
/// Returns a tensor of shape [H - 1, W - 1, 1] representing the importance/gradient map.
pub fn compute_importance_map<B: Backend>(image: Tensor<B, 3>) -> Tensor<B, 3> {
    let shape = image.shape();
    let dims = shape.dims::<3>();
    let h = dims[0];
    let w = dims[1];

    // dx = image[0:H-1, 1:W] - image[0:H-1, 0:W-1]
    let right = image.clone().narrow(0, 0, h - 1).narrow(1, 1, w - 1);
    let left = image.clone().narrow(0, 0, h - 1).narrow(1, 0, w - 1);
    let dx = right.sub(left);

    // dy = image[1:H, 0:W-1] - image[0:H-1, 0:W-1]
    let bottom = image.clone().narrow(0, 1, h - 1).narrow(1, 0, w - 1);
    let top = image.clone().narrow(0, 0, h - 1).narrow(1, 0, w - 1);
    let dy = bottom.sub(top);

    // Magnitude squared: dx^2 + dy^2
    let mag_sq = dx.powf_scalar(2.0).add(dy.powf_scalar(2.0));

    // Sum over channel dimension and take square root: shape [H - 1, W - 1, 1]
    mag_sq.sum_dim(2).sqrt()
}

use rand::Rng;
use crate::model::gaussian::GaussianModel;

/// Educational Hybrid Seeding Function:
/// Bridge between Implicit (NeRF) continuous representations and Explicit (Gaussian Splats) primitive parameters.
///
/// ### Hybrid Seeding Rationale:
/// 1. **70% Edge Importance Sampling:** High-frequency spatial gradients extracted from NeRF indicate
///    object boundaries and complex textures. Concentrating 70% of splats here accelerates convergence.
/// 2. **30% Uniform Background Filling:** If 100% of splats are placed on edges, flat background areas receive
///    zero splats, leaving unassigned black gaps and inflating overall MSE loss.
/// 3. **Color & Scale Transfer:**
///    - **Colors:** Extracted from NeRF predictions at $(x, y)$ and transformed into pre-sigmoid logits via
///      $\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$.
///    - **Scales:** Edge splats are assigned tighter variance ($\text{scale} = e^{-3.0} \approx 0.05$) to model fine details,
///      while background splats receive broader variance ($\text{scale} = e^{-2.0} \approx 0.135$) for smooth coverage.
pub fn seed_gaussians_from_importance<B: Backend>(
    importance_data: &[f32],
    nerf_render_rgb: &[f32], // [H * W * 3] float vector in [0, 1]
    h: usize,
    w: usize,
    num_gaussians: usize,
    device: &B::Device,
) -> GaussianModel<B> {
    let sum: f32 = importance_data.iter().sum();
    let mut rng = rand::thread_rng();
    
    let mut sampled_means = Vec::with_capacity(num_gaussians * 2);
    let mut sampled_colors = Vec::with_capacity(num_gaussians * 3);
    let mut sampled_scales = Vec::with_capacity(num_gaussians * 2);

    let use_cdf = sum >= 1e-5;
    let cdf = if use_cdf {
        let mut cdf_vec = Vec::with_capacity(importance_data.len());
        let mut cumulative = 0.0;
        for &val in importance_data.iter() {
            cumulative += val / sum;
            cdf_vec.push(cumulative);
        }
        cdf_vec
    } else {
        Vec::new()
    };

    // 70% importance sampled from NeRF edges, 30% uniform background filling
    let num_edge = if use_cdf { (num_gaussians as f32 * 0.70) as usize } else { 0 };

    for i in 0..num_gaussians {
        let (row, col, x, y) = if i < num_edge {
            // Edge-guided CDF sampling
            let r: f32 = rng.r#gen();
            let idx = match cdf.binary_search_by(|val| val.partial_cmp(&r).unwrap()) {
                Ok(index) => index,
                Err(index) => index.min(cdf.len() - 1),
            };
            let row = idx / w;
            let col = idx % w;
            let y = (row as f32 + 0.5) / (h as f32);
            let x = (col as f32 + 0.5) / (w as f32);
            (row, col, x, y)
        } else {
            // Uniform background coverage sampling
            let x: f32 = rng.r#gen();
            let y: f32 = rng.r#gen();
            let row = ((y * h as f32).floor() as usize).min(h - 1);
            let col = ((x * w as f32).floor() as usize).min(w - 1);
            (row, col, x, y)
        };

        sampled_means.push(x);
        sampled_means.push(y);

        // Extract RGB color at pixel coordinate from NeRF rendered image
        let pix_idx = (row * w + col) * 3;
        let r_val = nerf_render_rgb.get(pix_idx).copied().unwrap_or(0.5);
        let g_val = nerf_render_rgb.get(pix_idx + 1).copied().unwrap_or(0.5);
        let b_val = nerf_render_rgb.get(pix_idx + 2).copied().unwrap_or(0.5);

        // Inverse sigmoid logit transformation for pre-sigmoid colors: logit(p) = ln(p / (1 - p))
        // Ensures that when Gaussian model renders sigmoid(logit), it reproduces NeRF's RGB color.
        let logit = |p: f32| {
            let p_clamped = p.clamp(0.01, 0.99);
            (p_clamped / (1.0 - p_clamped)).ln()
        };

        sampled_colors.push(logit(r_val));
        sampled_colors.push(logit(g_val));
        sampled_colors.push(logit(b_val));

        // Sample initial scales: edge splats get smaller scale (-3.0), background splats get wider scale (-2.0)
        if i < num_edge {
            sampled_scales.push(-3.0);
            sampled_scales.push(-3.0);
        } else {
            sampled_scales.push(-2.0);
            sampled_scales.push(-2.0);
        }
    }

    let mut model = GaussianModel::<B>::new(num_gaussians, device);

    // Overwrite parameters with sampled initial values
    let means_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_means, [num_gaussians, 2]), device);
    let colors_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_colors, [num_gaussians, 3]), device);
    let scales_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_scales, [num_gaussians, 2]), device);

    model.means = burn::module::Param::from_tensor(means_tensor);
    model.colors = burn::module::Param::from_tensor(colors_tensor);
    model.scales = burn::module::Param::from_tensor(scales_tensor);

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;

    #[test]
    fn test_compute_importance_map() {
        let device = Default::default();
        // Create a 4x4 image with a sharp boundary (R channel step change)
        let data = vec![
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
        ];
        let tensor = Tensor::<Flex, 3>::from_data(TensorData::new(data, [4, 4, 3]), &device);
        let importance = compute_importance_map(tensor);

        assert_eq!(importance.shape().dims::<3>(), [3, 3, 1]);
        
        // Check that the middle column (index 1) has high gradient magnitude due to the transition from 0 to 1
        let vec = importance.into_data().into_vec::<f32>().unwrap();
        assert!(vec[1] > 0.9);
        assert!(vec[0] < 0.1);
    }

    #[test]
    fn test_seed_gaussians_from_importance() {
        let device = Default::default();
        // Create a simple 2x2 gradient map where only the bottom-right pixel has a gradient
        let importance_data = vec![
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        
        let nerf_render_rgb = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ];
        let model = seed_gaussians_from_importance::<Flex>(&importance_data, &nerf_render_rgb, 2, 2, 100, &device);
        assert_eq!(model.num_gaussians, 100);

        let means = model.means.val().into_data().into_vec::<f32>().unwrap();
        let colors = model.colors.val().into_data().into_vec::<f32>().unwrap();

        // The first 70 Gaussians are edge-sampled (index 3: bottom-right, x=0.75, y=0.75)
        for i in 0..70 {
            let x = means[i * 2];
            let y = means[i * 2 + 1];
            assert!((x - 0.75).abs() < 1e-4);
            assert!((y - 0.75).abs() < 1e-4);

            // Red channel pre-sigmoid logit of 1.0 (clamped to 0.99) should be positive
            let r_logit = colors[i * 3];
            assert!(r_logit > 0.0);
        }
    }
}
