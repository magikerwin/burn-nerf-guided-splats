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
    
    let mut sampled_means = Vec::with_capacity(num_gaussians * 3);
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

        let z: f32 = rng.r#gen(); // Initial depth z in [0, 1]

        sampled_means.push(x);
        sampled_means.push(y);
        sampled_means.push(z);

        // Extract RGB color at pixel coordinate from NeRF rendered image
        let pix_idx = (row * w + col) * 3;
        let r_val = nerf_render_rgb.get(pix_idx).copied().unwrap_or(0.5);
        let g_val = nerf_render_rgb.get(pix_idx + 1).copied().unwrap_or(0.5);
        let b_val = nerf_render_rgb.get(pix_idx + 2).copied().unwrap_or(0.5);

        // Inverse sigmoid logit transformation for pre-sigmoid colors: logit(p) = ln(p / (1 - p))
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
    let means_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_means, [num_gaussians, 3]), device);
    let colors_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_colors, [num_gaussians, 3]), device);
    let scales_tensor = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(sampled_scales, [num_gaussians, 2]), device);

    model.means = burn::module::Param::from_tensor(means_tensor);
    model.colors = burn::module::Param::from_tensor(colors_tensor);
    model.scales = burn::module::Param::from_tensor(scales_tensor);

    model
}

/// Bilinearly samples RGB from a single horizontal scanline `row` at normalized `x in [0, 1]`.
fn sample_scanline_rgb(img: &[f32], w: usize, h: usize, row: usize, x: f32) -> [f32; 3] {
    let r = row.min(h.saturating_sub(1));
    let fx = (x.clamp(0.0, 0.9999) * (w as f32) - 0.5).clamp(0.0, (w.saturating_sub(1)) as f32);
    let c0 = fx.floor() as usize;
    let c1 = (c0 + 1).min(w.saturating_sub(1));
    let t = fx - (c0 as f32);

    let i0 = (r * w + c0) * 3;
    let i1 = (r * w + c1) * 3;
    [
        img[i0] * (1.0 - t) + img[i1] * t,
        img[i0 + 1] * (1.0 - t) + img[i1 + 1] * t,
        img[i0 + 2] * (1.0 - t) + img[i1 + 2] * t,
    ]
}

/// Estimates 3D depth `z in [0.05, 0.95]` at `(x, y)` by minimizing multi-view photometric
/// reprojection error across NeRF renders at `v = 0.0`, `v = 0.5`, and `v = 1.0`.
pub fn triangulate_depth_from_views(
    nerf_v0: &[f32],
    nerf_vmid: &[f32],
    nerf_v1: &[f32],
    w: usize,
    h: usize,
    x: f32,
    row: usize,
) -> f32 {
    let theta = 0.3_f32; // Matches GaussianModel: (1.0 - 0.5) * 0.6 = 0.3 rad
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let step_x = 1.0 / (w.max(1) as f32);

    let mut best_z = 0.5_f32;
    let mut best_err = f32::MAX;

    for k in 0..19 {
        let z = 0.05 + 0.90 * ((k as f32) / 18.0);
        let x0 = (x - 0.5) * cos_t + (z - 0.5) * sin_t + 0.5;
        let x1 = (x - 0.5) * cos_t - (z - 0.5) * sin_t + 0.5;

        let mut err = 0.0_f32;
        for delta in [-1.0_f32, 0.0, 1.0] {
            let dx = delta * step_x;
            let c_mid = sample_scanline_rgb(nerf_vmid, w, h, row, x + dx);
            let c_0 = sample_scanline_rgb(nerf_v0, w, h, row, x0 + dx);
            let c_1 = sample_scanline_rgb(nerf_v1, w, h, row, x1 + dx);

            for ch in 0..3 {
                let d0 = c_0[ch] - c_mid[ch];
                let d1 = c_1[ch] - c_mid[ch];
                err += d0 * d0 + d1 * d1;
            }
        }
        // Mild prior toward z = 0.5 breaks ties cleanly in flat untextured regions
        err += 0.002 * (z - 0.5) * (z - 0.5);

        if err < best_err {
            best_err = err;
            best_z = z;
        }
    }
    best_z
}

/// Upgraded Multi-View NeRF-Guided Seeding:
/// Fixes the three classic pitfalls of naive 2D edge seeding:
/// 1. **Stereo 3D Depth Triangulation:** Replaces blind `z = rng.gen()` with photometric triangulation
///    across NeRF's `v = 0.0`, `0.5`, `1.0` predictions so splats start at true 3D parallax depths.
/// 2. **Balanced Interior + Edge Allocation (45% Edge / 45% Foreground Interior / 10% Background):**
///    Prevents "hollow ring" artifacts inside solid shapes.
/// 3. **Inward Color Nudge:** Picks the brightest foreground neighbor in a 3×3 window for edge splats
///    so boundary splats inherit pure object colors rather than muddy 50%-background border pixels.
pub fn seed_gaussians_multiview<B: Backend>(
    nerf_v0: &[f32],
    nerf_vmid: &[f32],
    nerf_v1: &[f32],
    h: usize,
    w: usize,
    num_gaussians: usize,
    device: &B::Device,
) -> GaussianModel<B> {
    let mut rng = rand::thread_rng();

    // 1. Estimate background RGB from the 4 corners of nerf_vmid
    let corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)];
    let mut bg = [0.0_f32; 3];
    for &(r, c) in &corners {
        let idx = (r * w + c) * 3;
        bg[0] += nerf_vmid[idx] * 0.25;
        bg[1] += nerf_vmid[idx + 1] * 0.25;
        bg[2] += nerf_vmid[idx + 2] * 0.25;
    }

    // 2. Compute exact H×W Edge Magnitude & Foreground Salience maps
    let mut edge_map = vec![0.0_f32; h * w];
    let mut fg_map = vec![0.0_f32; h * w];

    for r in 0..h {
        for c in 0..w {
            let idx = (r * w + c) * 3;
            let dr_bg = nerf_vmid[idx] - bg[0];
            let dg_bg = nerf_vmid[idx + 1] - bg[1];
            let db_bg = nerf_vmid[idx + 2] - bg[2];
            // Square salience to sharpen soft early-NeRF blobs onto true object peaks
            fg_map[r * w + c] = dr_bg * dr_bg + dg_bg * dg_bg + db_bg * db_bg;

            let r_next = (r + 1).min(h - 1);
            let c_next = (c + 1).min(w - 1);
            let ir = (r * w + c_next) * 3;
            let ib = (r_next * w + c) * 3;

            let mut grad_sq = 0.0_f32;
            for ch in 0..3 {
                let dx = nerf_vmid[ir + ch] - nerf_vmid[idx + ch];
                let dy = nerf_vmid[ib + ch] - nerf_vmid[idx + ch];
                grad_sq += dx * dx + dy * dy;
            }
            edge_map[r * w + c] = grad_sq;
        }
    }

    let build_cdf = |weights: &[f32]| -> Option<Vec<f32>> {
        let sum: f32 = weights.iter().sum();
        if sum < 1e-6 {
            return None;
        }
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &v in weights {
            acc += v / sum;
            cdf.push(acc);
        }
        Some(cdf)
    };

    let sample_cdf = |cdf: &[f32], rng: &mut rand::rngs::ThreadRng| -> (usize, usize, f32, f32) {
        let u: f32 = rng.r#gen();
        let idx = match cdf.binary_search_by(|val| val.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(cdf.len() - 1),
        };
        let row = idx / w;
        let col = idx % w;
        (
            row,
            col,
            (col as f32 + 0.5) / (w as f32),
            (row as f32 + 0.5) / (h as f32),
        )
    };

    let edge_cdf = build_cdf(&edge_map);
    let fg_cdf = build_cdf(&fg_map);

    let num_edge = if edge_cdf.is_some() { (num_gaussians as f32 * 0.45) as usize } else { 0 };
    let num_fg = if fg_cdf.is_some() { (num_gaussians as f32 * 0.45) as usize } else { 0 };

    let mut sampled_means = Vec::with_capacity(num_gaussians * 3);
    let mut sampled_colors = Vec::with_capacity(num_gaussians * 3);
    let mut sampled_scales = Vec::with_capacity(num_gaussians * 2);
    let mut sampled_opacities = Vec::with_capacity(num_gaussians);

    let logit = |p: f32| {
        let cl = p.clamp(0.02, 0.98);
        (cl / (1.0 - cl)).ln()
    };

    for i in 0..num_gaussians {
        // Use semi-transparent initial opacity (-0.2 -> alpha ~ 0.45) so overlapping splats
        // never block gradients (T_i > 0) to splats behind them during early optimization!
        let (row, col, x, y, log_s, opac_logit, nudge_inward) = if i < num_edge {
            let (r, c, x, y) = sample_cdf(edge_cdf.as_ref().unwrap(), &mut rng);
            (r, c, x, y, -3.1_f32, -0.1_f32, true)
        } else if i < num_edge + num_fg {
            let (r, c, x, y) = sample_cdf(fg_cdf.as_ref().unwrap(), &mut rng);
            (r, c, x, y, -2.55_f32, 0.0_f32, false)
        } else {
            let x: f32 = rng.r#gen();
            let y: f32 = rng.r#gen();
            let r = ((y * h as f32).floor() as usize).min(h - 1);
            let c = ((x * w as f32).floor() as usize).min(w - 1);
            (r, c, x, y, -1.8_f32, -0.8_f32, false)
        };

        // Triangulate 3D parallax depth z* across v = 0.0, 0.5, 1.0 instead of blind random z!
        let z = triangulate_depth_from_views(nerf_v0, nerf_vmid, nerf_v1, w, h, x, row);
        sampled_means.extend_from_slice(&[x, y, z]);

        // For edge splats, pick the highest-salience pixel in a 3×3 neighborhood to avoid muddy border colors
        let (mut cr, mut cc) = (row, col);
        if nudge_inward {
            let mut best_fg = fg_map[row * w + col];
            for dr in -1_isize..=1 {
                for dc in -1_isize..=1 {
                    let nr = (row as isize + dr).clamp(0, (h - 1) as isize) as usize;
                    let nc = (col as isize + dc).clamp(0, (w - 1) as isize) as usize;
                    if fg_map[nr * w + nc] > best_fg {
                        best_fg = fg_map[nr * w + nc];
                        cr = nr;
                        cc = nc;
                    }
                }
            }
        }

        let pix_idx = (cr * w + cc) * 3;
        sampled_colors.push(logit(nerf_vmid[pix_idx]));
        sampled_colors.push(logit(nerf_vmid[pix_idx + 1]));
        sampled_colors.push(logit(nerf_vmid[pix_idx + 2]));

        sampled_scales.extend_from_slice(&[log_s, log_s]);
        sampled_opacities.push(opac_logit);
    }

    let mut model = GaussianModel::<B>::new(num_gaussians, device);
    model.set_alpha_compositing(true);
    model.means = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_means, [num_gaussians, 3]),
        device,
    ));
    model.colors = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_colors, [num_gaussians, 3]),
        device,
    ));
    model.scales = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_scales, [num_gaussians, 2]),
        device,
    ));
    model.opacities = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_opacities, [num_gaussians, 1]),
        device,
    ));
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
        let data = vec![
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
        ];
        let tensor = Tensor::<Flex, 3>::from_data(TensorData::new(data, [4, 4, 3]), &device);
        let importance = compute_importance_map(tensor);

        assert_eq!(importance.shape().dims::<3>(), [3, 3, 1]);
        let vec = importance.into_data().into_vec::<f32>().unwrap();
        assert!(vec[1] > 0.9);
        assert!(vec[0] < 0.1);
    }

    #[test]
    fn test_seed_gaussians_from_importance() {
        let device = Default::default();
        let importance_data = vec![0.0, 0.0, 0.0, 1.0];
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

        for i in 0..70 {
            let x = means[i * 3];
            let y = means[i * 3 + 1];
            assert!((x - 0.75).abs() < 1e-4);
            assert!((y - 0.75).abs() < 1e-4);
            assert!(colors[i * 3] > 0.0);
        }
    }

    #[test]
    fn test_stereo_depth_triangulation_recovers_near_and_far() {
        let (w, h) = (32, 8);
        let theta = 0.3_f32;
        let render_dot = |z_true: f32, v: f32| -> Vec<f32> {
            let ang = (v - 0.5) * 0.6;
            let x_proj = 0.0 * ang.cos() - (z_true - 0.5) * ang.sin() + 0.5;
            let mut img = vec![0.0_f32; w * h * 3];
            for c in 0..w {
                let px = (c as f32 + 0.5) / (w as f32);
                let val = (-((px - x_proj) * 18.0).powi(2)).exp();
                for r in 0..h {
                    img[(r * w + c) * 3] = val;
                }
            }
            img
        };

        // Object A at near depth z = 0.15
        let near_z = triangulate_depth_from_views(
            &render_dot(0.15, 0.0),
            &render_dot(0.15, 0.5),
            &render_dot(0.15, 1.0),
            w,
            h,
            0.5,
            4,
        );
        // Object B at far depth z = 0.85
        let far_z = triangulate_depth_from_views(
            &render_dot(0.85, 0.0),
            &render_dot(0.85, 0.5),
            &render_dot(0.85, 1.0),
            w,
            h,
            0.5,
            4,
        );
        let _ = theta;
        assert!(near_z < 0.30, "Expected near depth < 0.30, got {}", near_z);
        assert!(far_z > 0.70, "Expected far depth > 0.70, got {}", far_z);
    }

    #[test]
    fn test_seed_gaussians_from_3d_volume() {
        let device = Default::default();
        // 2 voxels: Voxel 0 is empty air (alpha = 0.001), Voxel 1 at (0.7, 0.3, 0.2) is dense solid cyan (alpha = 0.95)
        let voxels = vec![
            0.1, 0.1, 0.1,  0.0, 0.0, 0.0,  0.001,
            0.7, 0.3, 0.2,  0.1, 0.8, 0.9,  0.950,
        ];
        let gs = seed_gaussians_from_3d_volume::<Flex>(&voxels, 0.01, 20, &device);
        let means = gs.means.val().into_data().into_vec::<f32>().unwrap();
        for i in 0..20 {
            assert!((means[i * 3] - 0.7).abs() < 0.02);
            assert!((means[i * 3 + 1] - 0.3).abs() < 0.02);
            assert!((means[i * 3 + 2] - 0.2).abs() < 0.02);
        }
    }
}

/// True 3D Volumetric NeRF -> 3D Gaussian Splatting Seeding:
/// Filters active solid 3D voxels `(voxel_alpha >= cutoff)` from `VolumetricNerfModel` and applies
/// **density-weighted 3D farthest-point selection** (`min_dist^2 * (0.4 + alpha)`) so every 3D part
/// (central planet, thin tilted ring arcs, distant moons) receives spatially balanced 3D Gaussians!
pub fn seed_gaussians_from_3d_volume<B: Backend>(
    volume_voxels: &[f32], // flat [x, y, z, r, g, b, voxel_alpha] * M
    voxel_jitter: f32,
    num_gaussians: usize,
    device: &B::Device,
) -> GaussianModel<B> {
    let m = volume_voxels.len() / 7;
    let mut rng = rand::thread_rng();

    // Rank all 3D voxels by learned density alpha descending
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        volume_voxels[b * 7 + 6]
            .partial_cmp(&volume_voxels[a * 7 + 6])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_a = order.first().map(|&i| volume_voxels[i * 7 + 6]).unwrap_or(0.0);
    let min_a = order.last().map(|&i| volume_voxels[i * 7 + 6]).unwrap_or(0.0);
    let cutoff = min_a + 0.35 * (max_a - min_a);
    let active: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| volume_voxels[i * 7 + 6] >= cutoff && max_a > min_a + 1e-4)
        .collect();

    let candidates = if active.is_empty() {
        &order[..order.len().min(num_gaussians.max(1))]
    } else {
        &active[..]
    };

    // Density-weighted 3D farthest-point selection across active voxels
    let mut chosen: Vec<usize> = Vec::with_capacity(num_gaussians);
    if !candidates.is_empty() {
        chosen.push(candidates[0]);
        while chosen.len() < num_gaussians {
            let mut best_idx = candidates[chosen.len() % candidates.len()];
            let mut best_score = -1.0_f32;
            for &cand in candidates {
                let cb = cand * 7;
                let mut min_d2 = f32::MAX;
                for &prev in &chosen {
                    let pb = prev * 7;
                    let d2 = (volume_voxels[cb] - volume_voxels[pb]).powi(2)
                        + (volume_voxels[cb + 1] - volume_voxels[pb + 1]).powi(2)
                        + (volume_voxels[cb + 2] - volume_voxels[pb + 2]).powi(2);
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                let score = min_d2 * (0.4 + volume_voxels[cb + 6]);
                if score > best_score {
                    best_score = score;
                    best_idx = cand;
                }
            }
            chosen.push(best_idx);
        }
    }

    let logit = |p: f32| {
        let cl = p.clamp(0.02, 0.98);
        (cl / (1.0 - cl)).ln()
    };

    let mut sampled_means = Vec::with_capacity(num_gaussians * 3);
    let mut sampled_colors = Vec::with_capacity(num_gaussians * 3);
    let mut sampled_scales = Vec::with_capacity(num_gaussians * 2);
    let mut sampled_opacities = Vec::with_capacity(num_gaussians);

    for k in 0..num_gaussians {
        let idx = chosen.get(k).copied().unwrap_or(0);
        let base = idx * 7;

        let jx = (rng.r#gen::<f32>() - 0.5) * voxel_jitter;
        let jy = (rng.r#gen::<f32>() - 0.5) * voxel_jitter;
        let jz = (rng.r#gen::<f32>() - 0.5) * voxel_jitter;

        sampled_means.push((volume_voxels[base] + jx).clamp(0.04, 0.96));
        sampled_means.push((volume_voxels[base + 1] + jy).clamp(0.04, 0.96));
        sampled_means.push((volume_voxels[base + 2] + jz).clamp(0.04, 0.96));

        sampled_colors.push(logit(volume_voxels[base + 3]));
        sampled_colors.push(logit(volume_voxels[base + 4]));
        sampled_colors.push(logit(volume_voxels[base + 5]));

        sampled_scales.extend_from_slice(&[-3.05_f32, -3.05_f32]);
        sampled_opacities.push(1.2_f32);
    }

    let mut model = GaussianModel::<B>::new(num_gaussians, device);
    model.set_alpha_compositing(true);
    model.means = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_means, [num_gaussians, 3]),
        device,
    ));
    model.colors = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_colors, [num_gaussians, 3]),
        device,
    ));
    model.scales = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_scales, [num_gaussians, 2]),
        device,
    ));
    model.opacities = burn::module::Param::from_tensor(Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(sampled_opacities, [num_gaussians, 1]),
        device,
    ));
    model
}



