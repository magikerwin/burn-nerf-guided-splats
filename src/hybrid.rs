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

/// Seeds Gaussian model means by sampling coordinates proportional to the importance map.
pub fn seed_gaussians_from_importance<B: Backend>(
    importance_map: Tensor<B, 3>,
    num_gaussians: usize,
    device: &B::Device,
) -> GaussianModel<B> {
    let shape = importance_map.shape();
    let dims = shape.dims::<3>();
    let h = dims[0];
    let w = dims[1];

    let mag = importance_map.into_data().into_vec::<f32>().expect("Failed to get importance map data");
    
    // Compute sum for normalization
    let sum: f32 = mag.iter().sum();
    
    let mut rng = rand::thread_rng();
    let mut sampled_means = Vec::with_capacity(num_gaussians * 2);

    if sum < 1e-5 {
        // If importance map is uniform/empty, fallback to random uniform seeding
        for _ in 0..num_gaussians {
            sampled_means.push(rng.r#gen::<f32>());
            sampled_means.push(rng.r#gen::<f32>());
        }
    } else {
        // Compute cumulative distribution function (CDF)
        let mut cdf = Vec::with_capacity(mag.len());
        let mut cumulative = 0.0;
        for &val in mag.iter() {
            cumulative += val / sum;
            cdf.push(cumulative);
        }

        for _ in 0..num_gaussians {
            let r: f32 = rng.r#gen();
            // Binary search to find the pixel index
            let idx = match cdf.binary_search_by(|val| val.partial_cmp(&r).unwrap()) {
                Ok(index) => index,
                Err(index) => index.min(cdf.len() - 1),
            };

            // Map flat index to 2D pixel coordinates (row, col) normalized to [0.0, 1.0]
            let row = idx / w;
            let col = idx % w;

            let y = (row as f32 + 0.5) / (h as f32);
            let x = (col as f32 + 0.5) / (w as f32);

            sampled_means.push(x);
            sampled_means.push(y);
        }
    }

    // Create a new Gaussian model and overwrite its means parameter
    let mut model = GaussianModel::<B>::new(num_gaussians, device);
    
    // Construct means tensor of shape [N, 2]
    let means_data = burn::tensor::TensorData::new(sampled_means, [num_gaussians, 2]);
    let means_tensor = Tensor::<B, 2>::from_data(means_data, device);
    model.means = burn::module::Param::from_tensor(means_tensor);

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
        let importance_tensor = Tensor::<Flex, 3>::from_data(TensorData::new(importance_data, [2, 2, 1]), &device);
        
        // Seed 100 Gaussians. They should all be placed in the bottom-right pixel (index 3).
        // Coordinates for index 3: row = 1, col = 1 -> y = 1.5/2 = 0.75, x = 1.5/2 = 0.75
        let model = seed_gaussians_from_importance(importance_tensor, 100, &device);
        assert_eq!(model.num_gaussians, 100);

        let means = model.means.val().into_data().into_vec::<f32>().unwrap();
        for i in 0..100 {
            let x = means[i * 2];
            let y = means[i * 2 + 1];
            assert!((x - 0.75).abs() < 1e-4);
            assert!((y - 0.75).abs() < 1e-4);
        }
    }
}
