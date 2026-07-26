use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Int, Tensor};

#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    pub num_frequencies: usize,
}

impl PositionalEncoding {
    pub fn new(num_frequencies: usize) -> Self {
        Self { num_frequencies }
    }

    /// Maps coords of shape [H, W, C] to higher-frequency features of shape [H, W, C * 2 * L].
    pub fn forward<B: Backend>(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = coords.shape();
        let dims = shape.dims::<3>();
        let height = dims[0];
        let width = dims[1];
        let num_channels = dims[2];
        let device = coords.device();

        // 1. Generate frequency scale factors: pi * 2.0^0, pi * 2.0^1, ..., pi * 2.0^(L-1)
        let mut freqs_val = Vec::with_capacity(self.num_frequencies);
        let pi = std::f32::consts::PI;
        for i in 0..self.num_frequencies {
            freqs_val.push(pi * 2.0f32.powi(i as i32));
        }

        // Create constant frequency scale tensor of shape [1, 1, 1, L]
        let freqs_data = burn::tensor::TensorData::new(freqs_val, [self.num_frequencies]);
        let freqs_1d = Tensor::<B, 1>::from_data(freqs_data, &device);
        let freqs_4d = freqs_1d.reshape([1, 1, 1, self.num_frequencies]);

        // 2. Unsqueeze coords of shape [H, W, C] to [H, W, C, 1]
        let coords_4d = coords.unsqueeze_dim::<4>(3);

        // 3. Multiply coordinates by frequency scales: [H, W, C, L]
        let scaled = coords_4d.mul(freqs_4d);

        // 4. Compute sine and cosine: [H, W, C, L]
        let sin_enc = scaled.clone().sin();
        let cos_enc = scaled.cos();

        // 5. Concatenate along dimension 3: [H, W, C, 2 * L]
        let enc = Tensor::cat(vec![sin_enc, cos_enc], 3);

        // 6. Reshape to flat 3D representation: [H, W, C * 2 * L]
        let out_channels = num_channels * 2 * self.num_frequencies;
        enc.reshape([height, width, out_channels])
    }
}

#[derive(Module, Debug)]
pub struct NerfModel<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub linear3: Linear<B>,
    pub linear4: Linear<B>,
    pub num_frequencies: usize,
}

impl<B: Backend> NerfModel<B> {
    /// Initializes a new NerfModel with given configuration.
    pub fn new(num_frequencies: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let input_dim = 3 * 2 * num_frequencies; // (3 coords: x, y, view_v) * (sin + cos) * L

        let linear1 = LinearConfig::new(input_dim, hidden_dim).init(device);
        let linear2 = LinearConfig::new(hidden_dim, hidden_dim).init(device);
        let linear3 = LinearConfig::new(hidden_dim, hidden_dim).init(device);
        let linear4 = LinearConfig::new(hidden_dim, 3).init(device); // Outputs RGB

        Self {
            linear1,
            linear2,
            linear3,
            linear4,
            num_frequencies,
        }
    }

    /// Evaluates the coordinate MLP on positional encoded features.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        use burn::tensor::activation::{relu, sigmoid};

        let x = self.linear1.forward(input);
        let x = relu(x);
        let x = self.linear2.forward(x);
        let x = relu(x);
        let x = self.linear3.forward(x);
        let x = relu(x);
        let x = self.linear4.forward(x);
        sigmoid(x)
    }
}

impl<B: Backend> crate::model::MultiViewFitter<B> for NerfModel<B> {
    fn render_view(&self, width: usize, height: usize, view_v: f32) -> Tensor<B, 3> {
        let device = self.linear1.weight.val().device();

        // 1. Generate coordinates grid (bounds expected as i64)
        let x = Tensor::<B, 1, Int>::arange(0..(width as i64), &device).float().div_scalar(width as f32);
        let y = Tensor::<B, 1, Int>::arange(0..(height as i64), &device).float().div_scalar(height as f32);
        let v = Tensor::<B, 1>::full([1], view_v, &device);

        let x_2d = x.unsqueeze_dim::<2>(0).repeat(&[height, 1]); // [height, width]
        let y_2d = y.unsqueeze_dim::<2>(1).repeat(&[1, width]);  // [height, width]
        let v_2d = v.unsqueeze_dim::<2>(0).repeat(&[height, width]); // [height, width]

        let coords = Tensor::cat(
            vec![
                x_2d.unsqueeze_dim::<3>(2),
                y_2d.unsqueeze_dim::<3>(2),
                v_2d.unsqueeze_dim::<3>(2),
            ],
            2,
        );

        // 2. Map coordinates through PositionalEncoding
        let pe = PositionalEncoding::new(self.num_frequencies);
        let encoded = pe.forward(coords); // [height, width, 6 * L]

        // 3. Forward pass through MLP to output RGB [height, width, 3]
        self.forward(encoded)
    }

    fn forward_loss_multiview(&self, target_v0: &Tensor<B, 3>, target_v1: &Tensor<B, 3>) -> Tensor<B, 1> {
        let shape = target_v0.shape();
        let dims = shape.dims::<3>();
        let height = dims[0];
        let width = dims[1];

        let render_v0 = self.render_view(width, height, 0.0);
        let render_v1 = self.render_view(width, height, 1.0);

        let diff0 = render_v0.sub(target_v0.clone()).powf_scalar(2.0).mean();
        let diff1 = render_v1.sub(target_v1.clone()).powf_scalar(2.0).mean();

        diff0.add(diff1).mul_scalar(0.5)
    }
}

impl<B: Backend> crate::model::ImageFitter<B> for NerfModel<B> {
    fn render(&self, width: usize, height: usize) -> Tensor<B, 3> {
        use crate::model::MultiViewFitter;
        self.render_view(width, height, 0.5)
    }

    fn forward_loss(&self, target_image: &Tensor<B, 3>) -> Tensor<B, 1> {
        use crate::model::MultiViewFitter;
        let shape = target_image.shape();
        let dims = shape.dims::<3>();
        let height = dims[0];
        let width = dims[1];

        let rendered = self.render_view(width, height, 0.0);
        let diff = rendered.sub(target_image.clone());
        diff.powf_scalar(2.0).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageFitter, MultiViewFitter};
    use burn::backend::Flex;

    #[test]
    fn test_positional_encoding_shape() {
        let device = Default::default();
        let pe = PositionalEncoding::new(10); // L = 10 -> out_channels = 3 * 2 * 10 = 60

        // Create dummy coords: shape [4, 4, 3]
        let coords = Tensor::<Flex, 3>::zeros([4, 4, 3], &device);
        let encoded = pe.forward(coords);

        assert_eq!(encoded.shape().dims::<3>(), [4, 4, 60]);
    }

    #[test]
    fn test_nerf_mlp_forward_shape() {
        let device = Default::default();
        let model = NerfModel::<Flex>::new(8, 16, &device); // input dim = 48

        // Create dummy encoded features: shape [4, 4, 48]
        let input = Tensor::<Flex, 3>::zeros([4, 4, 48], &device);
        let output = model.forward(input);

        assert_eq!(output.shape().dims::<3>(), [4, 4, 3]);
    }

    #[test]
    fn test_nerf_image_fitter_implementation() {
        let device = Default::default();
        let model = NerfModel::<Flex>::new(4, 8, &device);

        // Create dummy target: shape [6, 6, 3]
        let target = Tensor::<Flex, 3>::zeros([6, 6, 3], &device);

        let rendered = model.render(6, 6);
        assert_eq!(rendered.shape().dims::<3>(), [6, 6, 3]);

        let loss = model.forward_loss(&target);
        assert_eq!(loss.shape().dims::<1>(), [1]);

        let render_view_res = model.render_view(6, 6, 0.5);
        assert_eq!(render_view_res.shape().dims::<3>(), [6, 6, 3]);
    }
}

