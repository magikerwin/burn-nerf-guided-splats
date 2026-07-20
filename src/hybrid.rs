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
        // Since H=3, W=3:
        // row 0: [0, 1, 0]
        // row 1: [0, 1, 0]
        // row 2: [0, 1, 0]
        assert!(vec[1] > 0.9);
        assert!(vec[0] < 0.1);
    }
}
