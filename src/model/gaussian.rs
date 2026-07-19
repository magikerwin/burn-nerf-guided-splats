use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct GaussianModel<B: Backend> {
    pub means: Param<Tensor<B, 2>>,     // [N, 2]
    pub scales: Param<Tensor<B, 2>>,    // [N, 2]
    pub rotations: Param<Tensor<B, 2>>, // [N, 1]
    pub colors: Param<Tensor<B, 2>>,    // [N, 3]
    pub opacities: Param<Tensor<B, 2>>, // [N, 1]
    pub num_gaussians: usize,
}

impl<B: Backend> GaussianModel<B> {
    /// Initializes a new GaussianModel with random parameters.
    pub fn new(num_gaussians: usize, device: &B::Device) -> Self {
        // Initialize means uniformly in [0, 1]
        let means = Tensor::<B, 2>::random(
            [num_gaussians, 2],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        );

        // Initialize scales log-uniformly around small values (e.g. -2.0 to -1.0 in log space)
        let scales = Tensor::<B, 2>::random(
            [num_gaussians, 2],
            burn::tensor::Distribution::Uniform(-3.0, -2.0),
            device,
        );

        // Initialize rotations uniformly in [0, 2*pi]
        let rotations = Tensor::<B, 2>::random(
            [num_gaussians, 1],
            burn::tensor::Distribution::Uniform(0.0, 2.0 * std::f64::consts::PI),
            device,
        );

        // Initialize colors uniformly in [0, 1] (or random pre-sigmoid logits around 0.0)
        let colors = Tensor::<B, 2>::random(
            [num_gaussians, 3],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            device,
        );

        // Initialize opacities log-odds around 0.5 probability (pre-sigmoid logits around 0.0)
        let opacities = Tensor::<B, 2>::random(
            [num_gaussians, 1],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            device,
        );

        Self {
            means: Param::from_tensor(means),
            scales: Param::from_tensor(scales),
            rotations: Param::from_tensor(rotations),
            colors: Param::from_tensor(colors),
            opacities: Param::from_tensor(opacities),
            num_gaussians,
        }
    }

    /// Computes the components of the inverse covariance matrix for all Gaussians.
    /// Returns (inv_cov_00, inv_cov_01, inv_cov_11) each of shape [N, 1].
    pub fn compute_inverse_covariance(&self) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // scales has shape [N, 2]. Ensure scales are positive.
        let scales = self.scales.val().exp();
        let sx = scales.clone().narrow(1, 0, 1); // [N, 1]
        let sy = scales.narrow(1, 1, 1);        // [N, 1]

        let sx2 = sx.clone().powf_scalar(2.0);
        let sy2 = sy.clone().powf_scalar(2.0);

        // rotations has shape [N, 1]
        let theta = self.rotations.val();
        let cos = theta.clone().cos();
        let sin = theta.sin();

        let cos2 = cos.clone().powf_scalar(2.0);
        let sin2 = sin.clone().powf_scalar(2.0);
        let cos_sin = cos.mul(sin);

        // Compute covariance components:
        // a = sx^2 * cos^2 + sy^2 * sin^2
        // b = (sx^2 - sy^2) * cos * sin
        // d = sx^2 * sin^2 + sy^2 * cos^2
        let a = sx2.clone().mul(cos2.clone()).add(sy2.clone().mul(sin2.clone()));
        let b = sx2.clone().sub(sy2.clone()).mul(cos_sin);
        let d = sx2.mul(sin2).add(sy2.mul(cos2));

        // Determinant of Sigma: det = sx^2 * sy^2.
        // We add epsilon to prevent division by zero.
        let det = sx.mul(sy).powf_scalar(2.0).add_scalar(1e-6);
        let inv_det = det.recip(); // [N, 1]

        // Inverse Covariance components:
        // Sigma^-1 = 1/det * [ d, -b ]
        //                    [ -b, a ]
        let inv_cov_00 = d.mul(inv_det.clone());
        let inv_cov_01 = b.mul_scalar(-1.0).mul(inv_det.clone());
        let inv_cov_11 = a.mul(inv_det);

        (inv_cov_00, inv_cov_01, inv_cov_11)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn test_compute_inverse_covariance_shape() {
        let device = Default::default();
        let model = GaussianModel::<Flex>::new(10, &device);
        let (c00, c01, c11) = model.compute_inverse_covariance();

        assert_eq!(c00.shape().dims::<2>(), [10, 1]);
        assert_eq!(c01.shape().dims::<2>(), [10, 1]);
        assert_eq!(c11.shape().dims::<2>(), [10, 1]);
    }
}
