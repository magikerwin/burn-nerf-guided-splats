use burn::tensor::{backend::Backend, Tensor};

#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    pub num_frequencies: usize,
}

impl PositionalEncoding {
    pub fn new(num_frequencies: usize) -> Self {
        Self { num_frequencies }
    }

    /// Maps coords of shape [H, W, 2] to higher-frequency features of shape [H, W, 4 * L].
    pub fn forward<B: Backend>(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = coords.shape();
        let dims = shape.dims::<3>();
        let height = dims[0];
        let width = dims[1];
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

        // 2. Unsqueeze coords of shape [H, W, 2] to [H, W, 2, 1]
        let coords_4d = coords.unsqueeze_dim::<4>(3);

        // 3. Multiply coordinates by frequency scales: [H, W, 2, L]
        let scaled = coords_4d.mul(freqs_4d);

        // 4. Compute sine and cosine: [H, W, 2, L]
        let sin_enc = scaled.clone().sin();
        let cos_enc = scaled.cos();

        // 5. Concatenate along dimension 3: [H, W, 2, 2 * L]
        let enc = Tensor::cat(vec![sin_enc, cos_enc], 3);

        // 6. Reshape to flat 3D representation: [H, W, 4 * L]
        let out_channels = 4 * self.num_frequencies;
        enc.reshape([height, width, out_channels])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn test_positional_encoding_shape() {
        let device = Default::default();
        let pe = PositionalEncoding::new(10); // L = 10 -> out_channels = 40

        // Create dummy coords: shape [4, 4, 2]
        let coords = Tensor::<Flex, 3>::zeros([4, 4, 2], &device);
        let encoded = pe.forward(coords);

        assert_eq!(encoded.shape().dims::<3>(), [4, 4, 40]);
    }
}
