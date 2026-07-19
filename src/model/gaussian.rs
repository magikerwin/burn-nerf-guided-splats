use burn::module::{Module, Param};
use burn::tensor::{activation::sigmoid, backend::Backend, Tensor};

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

    /// Renders the Gaussians given a pre-computed coordinate grid of shape [H, W, 2].
    /// Returns a rendered image tensor of shape [H, W, 3].
    pub fn render_with_coords(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let n = self.means.val().shape().dims::<2>()[0];

        // 1. Compute inverse covariance elements
        let (c00, c01, c11) = self.compute_inverse_covariance(); // each is [N, 1]

        // Reshape inverse covariance elements to [N, 1, 1] for broadcasting
        let c00_3d = c00.unsqueeze_dim::<3>(2);
        let c01_3d = c01.unsqueeze_dim::<3>(2);
        let c11_3d = c11.unsqueeze_dim::<3>(2);

        // 2. Compute differences: pixel_coords - means
        // coords has shape [H, W, 2]. Reshape to [1, H, W, 2]
        let coords_4d = coords.unsqueeze_dim::<4>(0);
        // means has shape [N, 2]. Reshape to [N, 1, 1, 2]
        let means_4d = self.means.val().reshape([n, 1, 1, 2]);

        // Broadcasted subtraction: [N, H, W, 2]
        let diff = coords_4d.sub(means_4d);

        // Extract dx and dy of shape [N, H, W]
        let dx = diff.clone().narrow(3, 0, 1).squeeze_dim(3);
        let dy = diff.narrow(3, 1, 1).squeeze_dim(3);

        // 3. Compute power exponent for each Gaussian at each pixel:
        // power = -0.5 * (c00 * dx^2 + 2 * c01 * dx * dy + c11 * dy^2)
        let dx2 = dx.clone().powf_scalar(2.0);
        let dy2 = dy.clone().powf_scalar(2.0);
        let dx_dy = dx.mul(dy);

        let term1 = c00_3d.mul(dx2);
        let term2 = c01_3d.mul(dx_dy).mul_scalar(2.0);
        let term3 = c11_3d.mul(dy2);

        let power = term1.add(term2).add(term3).mul_scalar(-0.5);

        // Gaussian density: G = exp(power) -> shape [N, H, W]
        let g = power.exp();

        // 4. Multiply by opacities and colors and sum
        // opacities has shape [N, 1]. Apply sigmoid and reshape to [N, 1, 1, 1]
        let opac = sigmoid(self.opacities.val()).reshape([n, 1, 1, 1]);
        // colors has shape [N, 3]. Apply sigmoid and reshape to [N, 1, 1, 3]
        let col = sigmoid(self.colors.val()).reshape([n, 1, 1, 3]);

        // Reshape G to [N, H, W, 1]
        let g_4d = g.unsqueeze_dim::<4>(3);

        // Contribution of each Gaussian: [N, H, W, 3]
        let contribution = g_4d.mul(opac).mul(col);

        // Sum contributions along dimension 0 (Gaussian dim): [1, H, W, 3]
        let rendered_sum = contribution.sum_dim(0);

        // Squeeze first dimension to return [H, W, 3]
        rendered_sum.squeeze_dim(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::Int;

    #[test]
    fn test_compute_inverse_covariance_shape() {
        let device = Default::default();
        let model = GaussianModel::<Flex>::new(10, &device);
        let (c00, c01, c11) = model.compute_inverse_covariance();

        assert_eq!(c00.shape().dims::<2>(), [10, 1]);
        assert_eq!(c01.shape().dims::<2>(), [10, 1]);
        assert_eq!(c11.shape().dims::<2>(), [10, 1]);
    }

    #[test]
    fn test_render_with_coords_shape() {
        let device = Default::default();
        let model = GaussianModel::<Flex>::new(5, &device);

        // Create a dummy coordinate grid of shape [8, 8, 2]
        let x = Tensor::<Flex, 1, Int>::arange(0..8, &device).float().div_scalar(8.0);
        let y = Tensor::<Flex, 1, Int>::arange(0..8, &device).float().div_scalar(8.0);
        let x_2d = x.unsqueeze_dim::<2>(0).repeat(&[8, 1]); // [8, 8]
        let y_2d = y.unsqueeze_dim::<2>(1).repeat(&[1, 8]); // [8, 8]

        let coords = Tensor::cat(
            vec![x_2d.unsqueeze_dim::<3>(2), y_2d.unsqueeze_dim::<3>(2)],
            2,
        );

        let rendered = model.render_with_coords(coords);
        assert_eq!(rendered.shape().dims::<3>(), [8, 8, 3]);
    }
}
