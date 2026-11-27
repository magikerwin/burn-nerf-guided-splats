use burn::module::{Module, Param};
use burn::tensor::{activation::sigmoid, backend::Backend, Int, Tensor};

#[derive(Module, Debug)]
pub struct GaussianModel<B: Backend> {
    pub means: Param<Tensor<B, 2>>,     // [N, 3] (3D positions x, y, z)
    pub scales: Param<Tensor<B, 2>>,    // [N, 2]
    pub rotations: Param<Tensor<B, 2>>, // [N, 1]
    pub colors: Param<Tensor<B, 2>>,    // [N, 3]
    pub opacities: Param<Tensor<B, 2>>, // [N, 1]
    pub num_gaussians: usize,
    /// 0 = Additive summation (legacy), 1 = Depth-sorted front-to-back alpha compositing
    pub compositing_mode: usize,
}

impl<B: Backend> GaussianModel<B> {
    /// Initializes a new GaussianModel with random parameters.
    pub fn new(num_gaussians: usize, device: &B::Device) -> Self {
        // Initialize means uniformly in [0, 1] for 3D coordinates (x, y, z)
        let means = Tensor::<B, 2>::random(
            [num_gaussians, 3],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        );

        // Initialize scales log-uniformly around small values (e.g. -3.0 to -2.0 in log space)
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

        // Initialize colors uniformly in [-1, 1] pre-sigmoid logits
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
            compositing_mode: 0,
        }
    }

    /// Enables or disables front-to-back depth-sorted alpha compositing (`sum T_i * alpha_i * C_i`).
    pub fn set_alpha_compositing(&mut self, enabled: bool) {
        self.compositing_mode = if enabled { 1 } else { 0 };
    }

    pub fn is_alpha_compositing(&self) -> bool {
        self.compositing_mode == 1
    }

    /// Returns total number of trainable scalar parameters (`10 * N`).
    pub fn num_parameters(&self) -> usize {
        self.num_gaussians * 10
    }

    /// Projects 3D Gaussian means (x, y, z) under a full 3D turntable camera `(yaw, pitch)` in radians
    /// around scene center `(0.5, 0.5, 0.5)`. Returns `(means_2d [N, 2], depth_cam [N])`.
    pub fn compute_turntable_means_and_depths(&self, yaw: f32, pitch: f32) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let cy = yaw.cos();
        let sy = yaw.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();

        let means = self.means.val();
        let dims = means.shape().dims::<2>();
        let n = dims[0];
        let num_dims = dims[1];
        let device = means.device();

        if num_dims == 3 {
            let xw = means.clone().narrow(1, 0, 1).sub_scalar(0.5);
            let yw = means.clone().narrow(1, 1, 1).sub_scalar(0.5);
            let zw = means.narrow(1, 2, 1).sub_scalar(0.5);

            // 1. Yaw around Y-axis
            let x1 = xw.clone().mul_scalar(cy).sub(zw.clone().mul_scalar(sy));
            let z1 = xw.mul_scalar(sy).add(zw.mul_scalar(cy));

            // 2. Pitch around X-axis
            let u_proj = x1.add_scalar(0.5);
            let v_proj = yw.clone().mul_scalar(cp).sub(z1.clone().mul_scalar(sp)).add_scalar(0.5);
            let d_cam = yw.mul_scalar(sp).add(z1.mul_scalar(cp)).add_scalar(0.5).reshape([n]);

            (Tensor::cat(vec![u_proj, v_proj], 1), d_cam)
        } else {
            (means, Tensor::<B, 1>::zeros([n], &device))
        }
    }

    /// Renders 3D Gaussians from an arbitrary 3D `(yaw, pitch)` turntable camera in radians.
    pub fn render_turntable(&self, width: usize, height: usize, yaw: f32, pitch: f32) -> Tensor<B, 3> {
        let device = self.means.val().device();
        let x = Tensor::<B, 1, Int>::arange(0..(width as i64), &device)
            .float()
            .add_scalar(0.5)
            .div_scalar(width as f32);
        let y = Tensor::<B, 1, Int>::arange(0..(height as i64), &device)
            .float()
            .add_scalar(0.5)
            .div_scalar(height as f32);

        let x_2d = x.unsqueeze_dim::<2>(0).repeat(&[height, 1]);
        let y_2d = y.unsqueeze_dim::<2>(1).repeat(&[1, width]);
        let coords = Tensor::cat(vec![x_2d.unsqueeze_dim::<3>(2), y_2d.unsqueeze_dim::<3>(2)], 2);

        let (means_2d, depths) = self.compute_turntable_means_and_depths(yaw, pitch);
        self.render_with_coords_means_and_depths(coords, means_2d, Some(depths))
    }

    /// Computes multi-view 3D MSE loss across arbitrary `(yaw, pitch, target_image)` camera poses.
    pub fn forward_loss_turntable(&self, views: &[(f32, f32, Tensor<B, 3>)]) -> Tensor<B, 1> {
        let k = views.len();
        let device = self.means.val().device();
        if k == 0 {
            return Tensor::<B, 1>::zeros([1], &device);
        }
        let dims = views[0].2.shape().dims::<3>();
        let (h, w) = (dims[0], dims[1]);

        let mut total = self
            .render_turntable(w, h, views[0].0, views[0].1)
            .sub(views[0].2.clone())
            .powf_scalar(2.0)
            .mean();
        for &(yaw, pitch, ref tgt) in &views[1..] {
            let l = self
                .render_turntable(w, h, yaw, pitch)
                .sub(tgt.clone())
                .powf_scalar(2.0)
                .mean();
            total = total.add(l);
        }
        total.div_scalar(k as f32)
    }

    /// Projects 3D Gaussian means (x, y, z) to 2D image plane (x', y') AND camera-space depth z'
    /// given camera view parameter v in [0, 1].
    pub fn compute_projected_means_and_depths(&self, view_v: f32) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let theta = (view_v - 0.5) * 0.6;
        self.compute_turntable_means_and_depths(theta, 0.0)
    }

    /// Projects 3D Gaussian means (x, y, z) to 2D image plane (x', y') given camera view parameter v in [0, 1].
    pub fn compute_projected_means(&self, view_v: f32) -> Tensor<B, 2> {
        self.compute_projected_means_and_depths(view_v).0
    }

    /// Computes the components of the inverse covariance matrix for all Gaussians.
    /// Returns (inv_cov_00, inv_cov_01, inv_cov_11) each of shape [N, 1].
    pub fn compute_inverse_covariance(&self) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let scales = self.scales.val().exp();
        let sx = scales.clone().narrow(1, 0, 1); // [N, 1]
        let sy = scales.narrow(1, 1, 1);        // [N, 1]

        let sx2 = sx.clone().powf_scalar(2.0);
        let sy2 = sy.clone().powf_scalar(2.0);

        let theta = self.rotations.val();
        let cos = theta.clone().cos();
        let sin = theta.sin();

        let cos2 = cos.clone().powf_scalar(2.0);
        let sin2 = sin.clone().powf_scalar(2.0);
        let cos_sin = cos.mul(sin);

        let a = sx2.clone().mul(cos2.clone()).add(sy2.clone().mul(sin2.clone()));
        let b = sx2.clone().sub(sy2.clone()).mul(cos_sin);
        let d = sx2.mul(sin2).add(sy2.mul(cos2));

        let det = sx.mul(sy).powf_scalar(2.0).add_scalar(1e-6);
        let inv_det = det.recip(); // [N, 1]

        let inv_cov_00 = d.mul(inv_det.clone());
        let inv_cov_01 = b.mul_scalar(-1.0).mul(inv_det.clone());
        let inv_cov_11 = a.mul(inv_det);

        (inv_cov_00, inv_cov_01, inv_cov_11)
    }

    /// Renders the Gaussians with explicit 2D center positions, coordinate grid, and optional depth sorting.
    pub fn render_with_coords_means_and_depths(
        &self,
        coords: Tensor<B, 3>,
        means_2d: Tensor<B, 2>,
        depths: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 3> {
        let n = self.num_gaussians;
        let coord_dims = coords.shape().dims::<3>();
        let h = coord_dims[0];
        let w = coord_dims[1];
        let device = coords.device();

        // 1. Compute inverse covariance elements
        let (c00, c01, c11) = self.compute_inverse_covariance(); // each is [N, 1]
        let c00_3d = c00.unsqueeze_dim::<3>(2);
        let c01_3d = c01.unsqueeze_dim::<3>(2);
        let c11_3d = c11.unsqueeze_dim::<3>(2);

        // 2. Compute differences: pixel_coords - means
        let coords_4d = coords.unsqueeze_dim::<4>(0);
        let means_4d = means_2d.reshape([n, 1, 1, 2]);
        let diff = coords_4d.sub(means_4d);

        let dx = diff.clone().narrow(3, 0, 1).squeeze_dim(3);
        let dy = diff.narrow(3, 1, 1).squeeze_dim(3);

        // 3. Compute Gaussian evaluation G_i(x, y) = exp(-0.5 * (x - mu)^T Sigma^-1 (x - mu))
        let dx2 = dx.clone().powf_scalar(2.0);
        let dy2 = dy.clone().powf_scalar(2.0);
        let dx_dy = dx.mul(dy);

        let term1 = c00_3d.mul(dx2);
        let term2 = c01_3d.mul(dx_dy).mul_scalar(2.0);
        let term3 = c11_3d.mul(dy2);
        let power = term1.add(term2).add(term3).mul_scalar(-0.5);
        let g = power.exp(); // [N, H, W]

        if self.is_alpha_compositing() {
            // Front-to-back depth-sorted Alpha Compositing:
            // C(x) = sum_i T_i(x) * alpha_i(x) * c_i, where T_i(x) = prod_{j < i} (1 - alpha_j(x))
            let sorted_idx = if let Some(z_cam) = depths {
                z_cam.argsort(0)
            } else {
                Tensor::<B, 1, Int>::arange(0..(n as i64), &device)
            };

            let g_sorted = g.select(0, sorted_idx.clone()); // [N, H, W]
            let opac_sorted = sigmoid(self.opacities.val())
                .select(0, sorted_idx.clone())
                .reshape([n, 1, 1]);
            let col_sorted = sigmoid(self.colors.val())
                .select(0, sorted_idx)
                .reshape([n, 1, 1, 3]);

            let eff_alpha = g_sorted.mul(opac_sorted).clamp(0.0, 0.99); // [N, H, W]

            // Fused GPU transmittance via log-space strictly lower-triangular matrix multiply:
            // ln(T_i) = sum_{j < i} ln(1 - alpha_j)
            let log_one_minus_a = eff_alpha
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .clamp(1e-4, 1.0)
                .log()
                .reshape([n, h * w]);

            let rows = Tensor::<B, 1, Int>::arange(0..(n as i64), &device)
                .unsqueeze_dim::<2>(1)
                .repeat(&[1, n]);
            let cols = Tensor::<B, 1, Int>::arange(0..(n as i64), &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[n, 1]);
            let strict_lower = rows.greater(cols).float(); // [N, N] with 1.0 where j < i

            let transmittance = strict_lower
                .matmul(log_one_minus_a)
                .exp()
                .reshape([n, h, w, 1]);

            let weights = transmittance.mul(eff_alpha.unsqueeze_dim::<4>(3));
            weights.mul(col_sorted).sum_dim(0).squeeze_dim(0)
        } else {
            // Legacy Additive Splatting: sum_i alpha_i * G_i(x) * c_i
            let opac = sigmoid(self.opacities.val()).reshape([n, 1, 1, 1]);
            let col = sigmoid(self.colors.val()).reshape([n, 1, 1, 3]);
            let g_4d = g.unsqueeze_dim::<4>(3);
            let contribution = g_4d.mul(opac).mul(col);
            contribution.sum_dim(0).squeeze_dim(0)
        }
    }

    /// Renders the Gaussians with explicit 2D center positions and coordinate grid.
    pub fn render_with_coords_and_means(&self, coords: Tensor<B, 3>, means_2d: Tensor<B, 2>) -> Tensor<B, 3> {
        self.render_with_coords_means_and_depths(coords, means_2d, None)
    }

    /// Renders the Gaussians given a pre-computed coordinate grid of shape [H, W, 2].
    pub fn render_with_coords(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let (projected_means, depths) = self.compute_projected_means_and_depths(0.5);
        self.render_with_coords_means_and_depths(coords, projected_means, Some(depths))
    }

    /// Exports flat `[x_proj, y_proj, z_cam, sx, sy, theta, r, g, b, alpha]` (10 floats per Gaussian)
    /// for glass-box 2D ellipse & depth visualization in the UI.
    pub async fn extract_splat_descriptors_async(&self, view_v: f32) -> Vec<f32> {
        let (proj_means, z_cam) = self.compute_projected_means_and_depths(view_v);
        let scales = self.scales.val().exp();
        let rots = self.rotations.val();
        let cols = sigmoid(self.colors.val());
        let opacs = sigmoid(self.opacities.val());

        let xy_vec = proj_means.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let z_vec = z_cam.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let s_vec = scales.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let r_vec = rots.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let c_vec = cols.into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let o_vec = opacs.into_data_async().await.unwrap().into_vec::<f32>().unwrap();

        let n = self.num_gaussians;
        let mut out = Vec::with_capacity(n * 10);
        for i in 0..n {
            out.push(xy_vec[i * 2]);
            out.push(xy_vec[i * 2 + 1]);
            out.push(z_vec[i]);
            out.push(s_vec[i * 2]);
            out.push(s_vec[i * 2 + 1]);
            out.push(r_vec[i]);
            out.push(c_vec[i * 3]);
            out.push(c_vec[i * 3 + 1]);
            out.push(c_vec[i * 3 + 2]);
            out.push(o_vec[i]);
        }
        out
    }

    /// Educational Adaptive Density Control (Kerbl et al. 3DGS style):
    /// 1. Prunes transparent "dead" splats (`alpha < min_opacity`) or oversized floaters (`scale > 0.35`).
    /// 2. Splits the largest high-opacity splats along their major axis into two sharper child splats,
    ///    preserving total splat count `N` while concentrating detail onto active structures.
    /// Returns `(updated_model, pruned_count, split_count)`.
    pub async fn prune_and_densify_async(&self, min_opacity: f32, device: &B::Device) -> (Self, usize, usize) {
        let means_v = self.means.val().into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let scales_v = self.scales.val().into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let rots_v = self.rotations.val().into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let cols_v = self.colors.val().into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let opacs_v = self.opacities.val().into_data_async().await.unwrap().into_vec::<f32>().unwrap();

        let n = self.num_gaussians;
        let sigmoid_fn = |x: f32| 1.0 / (1.0 + (-x).exp());

        let mut survivors: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let alpha = sigmoid_fn(opacs_v[i]);
            let max_s = scales_v[i * 2].exp().max(scales_v[i * 2 + 1].exp());
            if alpha >= min_opacity && max_s <= 0.38 {
                survivors.push(i);
            }
        }

        // Ensure we have at least a handful of survivors to split from
        if survivors.is_empty() {
            survivors.extend(0..n.min(4));
        }

        let mut pruned_count = n.saturating_sub(survivors.len());

        // Even if few splats were transparent, force-prune the weakest ~15% if pruned_count == 0
        // so clicking "Prune & Split" always visibly demonstrates mitosis splitting on coarse splats!
        if pruned_count == 0 && survivors.len() >= 8 {
            survivors.sort_by(|&a, &b| opacs_v[b].partial_cmp(&opacs_v[a]).unwrap());
            let target_keep = ((n as f32) * 0.85).round() as usize;
            pruned_count = survivors.len().saturating_sub(target_keep);
            survivors.truncate(target_keep);
        }

        // Rank survivors by visual footprint: alpha * max(sx, sy) descending
        let mut ranked = survivors.clone();
        ranked.sort_by(|&a, &b| {
            let score_a = sigmoid_fn(opacs_v[a]) * scales_v[a * 2].exp().max(scales_v[a * 2 + 1].exp());
            let score_b = sigmoid_fn(opacs_v[b]) * scales_v[b * 2].exp().max(scales_v[b * 2 + 1].exp());
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_means = Vec::with_capacity(n * 3);
        let mut new_scales = Vec::with_capacity(n * 2);
        let mut new_rots = Vec::with_capacity(n);
        let mut new_cols = Vec::with_capacity(n * 3);
        let mut new_opacs = Vec::with_capacity(n);

        // Copy survivors; for the ones we split, shrink their parent scale too!
        let split_count = pruned_count;
        let mut split_set = vec![false; n];
        for k in 0..split_count.min(ranked.len()) {
            split_set[ranked[k]] = true;
        }

        for &idx in &survivors {
            let mut mx = means_v[idx * 3];
            let mut my = means_v[idx * 3 + 1];
            let mz = means_v[idx * 3 + 2];
            let mut log_sx = scales_v[idx * 2];
            let mut log_sy = scales_v[idx * 2 + 1];
            let theta = rots_v[idx];

            if split_set[idx] {
                let sx = log_sx.exp();
                let dx = 0.45 * sx * theta.cos();
                let dy = 0.45 * sx * theta.sin();
                mx = (mx - dx).clamp(0.02, 0.98);
                my = (my - dy).clamp(0.02, 0.98);
                log_sx -= 0.35; // shrink by ~1.42x
                log_sy -= 0.35;
            }

            new_means.extend_from_slice(&[mx, my, mz]);
            new_scales.extend_from_slice(&[log_sx, log_sy]);
            new_rots.push(theta);
            new_cols.extend_from_slice(&cols_v[idx * 3..idx * 3 + 3]);
            new_opacs.push(opacs_v[idx]);
        }

        // Spawn child splats on the opposite side of each split parent's major axis
        for k in 0..split_count {
            let idx = ranked[k % ranked.len()];
            let sx = scales_v[idx * 2].exp();
            let theta = rots_v[idx];
            let dx = 0.45 * sx * theta.cos();
            let dy = 0.45 * sx * theta.sin();
            let mx = (means_v[idx * 3] + dx).clamp(0.02, 0.98);
            let my = (means_v[idx * 3 + 1] + dy).clamp(0.02, 0.98);
            let mz = means_v[idx * 3 + 2];

            new_means.extend_from_slice(&[mx, my, mz]);
            new_scales.extend_from_slice(&[scales_v[idx * 2] - 0.35, scales_v[idx * 2 + 1] - 0.35]);
            new_rots.push(theta);
            new_cols.extend_from_slice(&cols_v[idx * 3..idx * 3 + 3]);
            new_opacs.push(opacs_v[idx]);
        }

        let updated = Self {
            means: Param::from_tensor(Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(new_means, [n, 3]),
                device,
            )),
            scales: Param::from_tensor(Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(new_scales, [n, 2]),
                device,
            )),
            rotations: Param::from_tensor(Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(new_rots, [n, 1]),
                device,
            )),
            colors: Param::from_tensor(Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(new_cols, [n, 3]),
                device,
            )),
            opacities: Param::from_tensor(Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(new_opacs, [n, 1]),
                device,
            )),
            num_gaussians: n,
            compositing_mode: self.compositing_mode,
        };

        (updated, pruned_count, split_count)
    }
}

impl<B: Backend> crate::model::MultiViewFitter<B> for GaussianModel<B> {
    fn render_view(&self, width: usize, height: usize, view_v: f32) -> Tensor<B, 3> {
        let device = self.means.val().device();

        let x = Tensor::<B, 1, Int>::arange(0..(width as i64), &device).float().div_scalar(width as f32);
        let y = Tensor::<B, 1, Int>::arange(0..(height as i64), &device).float().div_scalar(height as f32);

        let x_2d = x.unsqueeze_dim::<2>(0).repeat(&[height, 1]);
        let y_2d = y.unsqueeze_dim::<2>(1).repeat(&[1, width]);

        let coords = Tensor::cat(
            vec![x_2d.unsqueeze_dim::<3>(2), y_2d.unsqueeze_dim::<3>(2)],
            2,
        );

        let (projected_means, depths) = self.compute_projected_means_and_depths(view_v);
        self.render_with_coords_means_and_depths(coords, projected_means, Some(depths))
    }

    fn forward_loss_multiview(&self, target_v0: &Tensor<B, 3>, target_v1: &Tensor<B, 3>) -> Tensor<B, 1> {
        self.forward_loss_multiframe(&[target_v0.clone(), target_v1.clone()])
    }

    fn forward_loss_multiframe(&self, targets: &[Tensor<B, 3>]) -> Tensor<B, 1> {
        use crate::model::ImageFitter;
        let k = targets.len();
        if k == 0 {
            let device = self.means.val().device();
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

impl<B: Backend> crate::model::ImageFitter<B> for GaussianModel<B> {
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

        let view_render = model.render_view(8, 8, 0.0);
        assert_eq!(view_render.shape().dims::<3>(), [8, 8, 3]);
    }

    #[test]
    fn test_image_fitter_implementation() {
        let device = Default::default();
        let model = GaussianModel::<Flex>::new(3, &device);

        // Create a dummy target image of shape [4, 4, 3]
        let target = Tensor::<Flex, 3>::zeros([4, 4, 3], &device);

        let rendered = model.render(4, 4);
        assert_eq!(rendered.shape().dims::<3>(), [4, 4, 3]);

        let loss = model.forward_loss(&target);
        assert_eq!(loss.shape().dims::<1>(), [1]);
    }

    #[test]
    fn test_alpha_compositing_occlusion_and_autodiff() {
        use burn::backend::autodiff::Autodiff;
        use burn::tensor::TensorData;

        type AD = Autodiff<Flex>;
        let device = Default::default();
        let mut model = GaussianModel::<AD>::new(2, &device);

        // Splat 0 (far background z=0.9, pure green) and Splat 1 (near foreground z=0.1, pure red)
        model.means = Param::from_tensor(Tensor::<AD, 2>::from_data(
            TensorData::new(vec![0.5, 0.5, 0.9, 0.5, 0.5, 0.1], [2, 3]),
            &device,
        ));
        model.scales = Param::from_tensor(Tensor::<AD, 2>::from_data(
            TensorData::new(vec![-0.5, -0.5, -0.5, -0.5], [2, 2]),
            &device,
        ));
        model.rotations = Param::from_tensor(Tensor::<AD, 2>::zeros([2, 1], &device));
        model.colors = Param::from_tensor(Tensor::<AD, 2>::from_data(
            TensorData::new(vec![-8.0, 8.0, -8.0, 8.0, -8.0, -8.0], [2, 3]),
            &device,
        ));
        model.opacities = Param::from_tensor(Tensor::<AD, 2>::from_data(
            TensorData::new(vec![8.0, 8.0], [2, 1]),
            &device,
        ));

        // 1. Legacy additive mode sums both colors -> center pixel has BOTH high R and high G
        model.set_alpha_compositing(false);
        let add_img = model.render_view(5, 5, 0.5).into_data().into_vec::<f32>().unwrap();
        let center_idx = (2 * 5 + 2) * 3;
        assert!(add_img[center_idx] > 0.8 && add_img[center_idx + 1] > 0.8);

        // 2. Alpha compositing sorts near splat (Red, z=0.1) in front of far splat (Green, z=0.9) -> Green is occluded!
        model.set_alpha_compositing(true);
        let comp_tensor = model.render_view(5, 5, 0.5);
        let comp_img = comp_tensor.clone().into_data().into_vec::<f32>().unwrap();
        assert!(comp_img[center_idx] > 0.85, "Foreground red should dominate");
        assert!(comp_img[center_idx + 1] < 0.15, "Background green should be occluded");

        // 3. Verify backward pass succeeds cleanly through depth-sorted alpha compositing
        let target = Tensor::<AD, 3>::zeros([5, 5, 3], &device);
        let loss = model.forward_loss(&target);
        let _grads = loss.backward();
    }
}


