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

    /// Maps coords of shape [H, W, C] to higher-frequency features of shape [H, W, C * 2 * L],
    /// or returns raw coords [H, W, C] when `num_frequencies == 0` (demonstrating Spectral Bias).
    pub fn forward<B: Backend>(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.num_frequencies == 0 {
            return coords;
        }

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
    pub hidden_dim: usize,
}

impl<B: Backend> NerfModel<B> {
    /// Initializes a new NerfModel with given configuration.
    /// When `num_frequencies == 0`, raw `(x, y, v)` coordinates are fed directly into the MLP.
    pub fn new(num_frequencies: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let input_dim = if num_frequencies == 0 {
            3
        } else {
            3 * 2 * num_frequencies // (3 coords: x, y, view_v) * (sin + cos) * L
        };

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
            hidden_dim,
        }
    }

    /// Returns total number of trainable scalar parameters in the Coordinate MLP.
    pub fn num_parameters(&self) -> usize {
        let input_dim = if self.num_frequencies == 0 {
            3
        } else {
            3 * 2 * self.num_frequencies
        };
        let h = self.hidden_dim;
        (input_dim * h + h) + 2 * (h * h + h) + (h * 3 + 3)
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
        self.forward_loss_multiframe(&[target_v0.clone(), target_v1.clone()])
    }

    fn forward_loss_multiframe(&self, targets: &[Tensor<B, 3>]) -> Tensor<B, 1> {
        use crate::model::ImageFitter;
        let k = targets.len();
        if k == 0 {
            let device = self.linear1.weight.val().device();
            return Tensor::<B, 1>::zeros([1], &device);
        }
        if k == 1 {
            return self.forward_loss(&targets[0]);
        }


        let shape = targets[0].shape();
        let dims = shape.dims::<3>();
        let height = dims[0];
        let width = dims[1];

        let mut total_loss = self.render_view(width, height, 0.0)
            .sub(targets[0].clone())
            .powf_scalar(2.0)
            .mean();

        for i in 1..k {
            let v = i as f32 / ((k - 1) as f32);
            let loss_i = self.render_view(width, height, v)
                .sub(targets[i].clone())
                .powf_scalar(2.0)
                .mean();
            total_loss = total_loss.add(loss_i);
        }

        total_loss.div_scalar(k as f32)
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

/// True 3D Volumetric Radiance Field: maps 3D world coordinates `(x, y, z) in [0, 1]^3`
/// to `(RGB, sigma)` and renders views via differentiable 3D ray marching:
/// `C(r) = sum_{k=1..S} T_k * (1 - exp(-sigma_k * dt)) * c_k`, where `T_k = exp(-dt * sum_{j<k} sigma_j)`.
#[derive(Module, Debug)]
pub struct VolumetricNerfModel<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub linear3: Linear<B>,
    pub num_frequencies: usize,
    pub num_ray_samples: usize,
}

impl<B: Backend> VolumetricNerfModel<B> {
    pub fn new(
        num_frequencies: usize,
        hidden_dim: usize,
        num_ray_samples: usize,
        device: &B::Device,
    ) -> Self {
        let input_dim = if num_frequencies == 0 {
            3
        } else {
            3 * 2 * num_frequencies
        };
        Self {
            linear1: LinearConfig::new(input_dim, hidden_dim).init(device),
            linear2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            linear3: LinearConfig::new(hidden_dim, 4).init(device), // [R, G, B, sigma]
            num_frequencies,
            num_ray_samples: num_ray_samples.max(4),
        }
    }

    /// Evaluates `(x, y, z)` tensor of shape `[Rays, Samples, 3]` -> `(rgb [Rays, Samples, 3], sigma [Rays, Samples])`.
    pub fn query_points_3d(&self, pts_world: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        use burn::tensor::activation::{relu, sigmoid, softplus};

        let pe = PositionalEncoding::new(self.num_frequencies);
        let encoded = pe.forward(pts_world);
        let h = relu(self.linear1.forward(encoded));
        let h = relu(self.linear2.forward(h));
        let raw = self.linear3.forward(h); // [R, S, 4]

        let rgb = sigmoid(raw.clone().narrow(2, 0, 3));
        // Calibrated initial density bias (-1.3): starts translucent while keeping sigmoid(-1.3)=0.21
        // gradient strong enough for thin 3D structures to solidify within ~25-40 coarse steps!
        let sigma = softplus(raw.narrow(2, 3, 1).squeeze_dim(2).sub_scalar(1.3), 1.0);
        (rgb, sigma)
    }

    /// Marches `S` 3D sample points per pixel ray from turntable camera `(yaw, pitch)` through `[0, 1]^3`.
    pub fn render_turntable(&self, width: usize, height: usize, yaw: f32, pitch: f32) -> Tensor<B, 3> {
        let device = self.linear1.weight.val().device();
        let r_count = height * width;
        let s = self.num_ray_samples;
        // Unit-cube optical extinction scale (8.0x) so thin 3D features (~0.05 radius) solidify cleanly
        let dt = (0.90_f32 / (s as f32)) * 8.0;

        // 1. Camera-space ray grid (u_cam, v_cam) in [-0.5, +0.5]^2 and depth steps d_cam in [-0.45, +0.45]
        let u_1d = Tensor::<B, 1, Int>::arange(0..(width as i64), &device)
            .float()
            .add_scalar(0.5)
            .div_scalar(width as f32)
            .sub_scalar(0.5);
        let v_1d = Tensor::<B, 1, Int>::arange(0..(height as i64), &device)
            .float()
            .add_scalar(0.5)
            .div_scalar(height as f32)
            .sub_scalar(0.5);
        let d_1d = Tensor::<B, 1, Int>::arange(0..(s as i64), &device)
            .float()
            .add_scalar(0.5)
            .div_scalar(s as f32)
            .mul_scalar(0.90)
            .sub_scalar(0.45);

        let u_cam = u_1d
            .unsqueeze_dim::<2>(0)
            .repeat(&[height, 1])
            .reshape([r_count, 1])
            .repeat(&[1, s]); // [R, S]
        let v_cam = v_1d
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, width])
            .reshape([r_count, 1])
            .repeat(&[1, s]); // [R, S]
        let d_cam = d_1d.unsqueeze_dim::<2>(0).repeat(&[r_count, 1]); // [R, S]

        // 2. Inverse turntable rotation R_y(-yaw) * R_x(-pitch) into world [0, 1]^3
        let cy = yaw.cos();
        let sy = yaw.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();

        let x1 = u_cam;
        let y1 = v_cam.clone().mul_scalar(cp).add(d_cam.clone().mul_scalar(sp));
        let z1 = v_cam.mul_scalar(-sp).add(d_cam.mul_scalar(cp));

        let xw = x1.clone().mul_scalar(cy).add(z1.clone().mul_scalar(sy)).add_scalar(0.5);
        let yw = y1.add_scalar(0.5);
        let zw = x1.mul_scalar(-sy).add(z1.mul_scalar(cy)).add_scalar(0.5);

        let pts_world = Tensor::cat(
            vec![
                xw.unsqueeze_dim::<3>(2),
                yw.unsqueeze_dim::<3>(2),
                zw.unsqueeze_dim::<3>(2),
            ],
            2,
        ); // [R, S, 3]

        // 3. Query 3D MLP and integrate volume rendering equation along each ray!
        let (rgb, sigma) = self.query_points_3d(pts_world); // [R, S, 3], [R, S]
        let sigma_dt = sigma.mul_scalar(dt); // [R, S]
        let alpha = sigma_dt.clone().mul_scalar(-1.0).exp().mul_scalar(-1.0).add_scalar(1.0); // 1 - exp(-sigma*dt)

        // Exact log-transmittance: ln(T_k) = -sum_{j < k} sigma_j * dt via [R, S] x [S, S] strict upper-triangular matmul
        let j_idx = Tensor::<B, 1, Int>::arange(0..(s as i64), &device)
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, s]);
        let k_idx = Tensor::<B, 1, Int>::arange(0..(s as i64), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[s, 1]);
        let strict_upper = k_idx.greater(j_idx).float().mul_scalar(-1.0); // -1.0 where j < k

        let transmittance = sigma_dt.matmul(strict_upper).exp(); // [R, S]
        let weights = transmittance.mul(alpha).unsqueeze_dim::<3>(2); // [R, S, 1]

        weights
            .mul(rgb)
            .sum_dim(1)
            .reshape([height, width, 3])
    }

    /// Computes foreground-balanced multi-view 3D ray-marched MSE loss across `(yaw, pitch, target_image)` poses.
    /// In sparse 3D scenes (~93% empty sky), weighting non-empty target rays (`1 + 6 * tgt`) mirrors
    /// Instant-NGP's 50/50 foreground/sky ray sampling so empty-space rays don't drown out thin 3D rings!
    pub fn forward_loss_turntable(&self, views: &[(f32, f32, Tensor<B, 3>)]) -> Tensor<B, 1> {
        let k = views.len();
        let device = self.linear1.weight.val().device();
        if k == 0 {
            return Tensor::<B, 1>::zeros([1], &device);
        }
        let dims = views[0].2.shape().dims::<3>();
        let (h, w) = (dims[0], dims[1]);

        let ray_loss = |yaw: f32, pitch: f32, tgt: &Tensor<B, 3>| {
            let sq_err = self
                .render_turntable(w, h, yaw, pitch)
                .sub(tgt.clone())
                .powf_scalar(2.0);
            let fg_weight = tgt.clone().mul_scalar(6.0).add_scalar(1.0);
            sq_err.mul(fg_weight).mean()
        };

        let mut total = ray_loss(views[0].0, views[0].1, &views[0].2);
        for &(yaw, pitch, ref tgt) in &views[1..] {
            total = total.add(ray_loss(yaw, pitch, tgt));
        }
        total.div_scalar(k as f32)
    }

    /// Samples the learned 3D volume on a regular `res × res × res` 3D lattice in `[0.12, 0.88]^3`.
    /// Returns flat `[x, y, z, r, g, b, voxel_alpha]` (`7 * res^3` floats) for direct 3D Gaussian seeding!
    pub async fn extract_3d_density_grid_async(&self, res: usize) -> Vec<f32> {
        let device = self.linear1.weight.val().device();
        let total = res * res * res;
        let mut coords = Vec::with_capacity(total * 3);
        for iz in 0..res {
            let z = 0.12 + 0.76 * ((iz as f32 + 0.5) / (res as f32));
            for iy in 0..res {
                let y = 0.12 + 0.76 * ((iy as f32 + 0.5) / (res as f32));
                for ix in 0..res {
                    let x = 0.12 + 0.76 * ((ix as f32 + 0.5) / (res as f32));
                    coords.extend_from_slice(&[x, y, z]);
                }
            }
        }

        let pts = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(coords.clone(), [total, 1, 3]),
            &device,
        );
        let (rgb_t, sigma_t) = self.query_points_3d(pts);
        let dt = (0.84 / (res as f32)) * 8.0;
        let alpha_t = sigma_t.mul_scalar(-dt).exp().mul_scalar(-1.0).add_scalar(1.0);

        let rgb_v = rgb_t.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let alpha_v = alpha_t.into_data_async().await.unwrap().into_vec::<f32>().unwrap();

        let mut out = Vec::with_capacity(total * 7);
        for i in 0..total {
            out.push(coords[i * 3]);
            out.push(coords[i * 3 + 1]);
            out.push(coords[i * 3 + 2]);
            out.push(rgb_v[i * 3]);
            out.push(rgb_v[i * 3 + 1]);
            out.push(rgb_v[i * 3 + 2]);
            out.push(alpha_v[i]);
        }
        out
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
        let pe = PositionalEncoding::new(10);
        let coords = Tensor::<Flex, 3>::zeros([4, 4, 3], &device);
        let encoded = pe.forward(coords);
        assert_eq!(encoded.shape().dims::<3>(), [4, 4, 60]);
    }

    #[test]
    fn test_nerf_mlp_forward_shape() {
        let device = Default::default();
        let model = NerfModel::<Flex>::new(8, 16, &device);
        let input = Tensor::<Flex, 3>::zeros([4, 4, 48], &device);
        let output = model.forward(input);
        assert_eq!(output.shape().dims::<3>(), [4, 4, 3]);
    }

    #[test]
    fn test_nerf_image_fitter_implementation() {
        let device = Default::default();
        let model = NerfModel::<Flex>::new(4, 8, &device);
        let target = Tensor::<Flex, 3>::zeros([6, 6, 3], &device);
        let rendered = model.render(6, 6);
        assert_eq!(rendered.shape().dims::<3>(), [6, 6, 3]);
        let loss = model.forward_loss(&target);
        assert_eq!(loss.shape().dims::<1>(), [1]);
        let render_view_res = model.render_view(6, 6, 0.5);
        assert_eq!(render_view_res.shape().dims::<3>(), [6, 6, 3]);
    }

    #[test]
    fn test_volumetric_3d_nerf_ray_marching_and_autodiff() {
        use burn::backend::autodiff::Autodiff;
        type AD = Autodiff<Flex>;
        let device = Default::default();
        let vol_nerf = VolumetricNerfModel::<AD>::new(4, 16, 8, &device);

        let img = vol_nerf.render_turntable(6, 6, 0.4, -0.2);
        assert_eq!(img.shape().dims::<3>(), [6, 6, 3]);

        let tgt = Tensor::<AD, 3>::zeros([6, 6, 3], &device);
        let loss = vol_nerf.forward_loss_turntable(&[(0.0, 0.0, tgt)]);
        let _grads = loss.backward();
    }
}


