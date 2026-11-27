use wasm_bindgen::prelude::*;
use burn::backend::{Autodiff, Flex};
use burn::module::Param;
use burn::optim::{Adam, AdamConfig};
use burn::tensor::{Tensor, TensorData};
use crate::model::gaussian::GaussianModel;
use crate::model::nerf::{NerfModel, VolumetricNerfModel};
use crate::model::MultiViewFitter;
use crate::training::{
    train_step_gaussian_turntable, train_step_multiframe, train_step_volumetric_turntable,
};

type WasmB = Autodiff<Flex>;

#[wasm_bindgen]
pub struct WasmTrainingSession {
    width: usize,
    height: usize,
    target_tensors: Vec<Tensor<WasmB, 3>>,
    turntable_views: Vec<(f32, f32, Tensor<WasmB, 3>)>,
    coarse_turntable_views: Vec<(f32, f32, Tensor<WasmB, 3>)>,
    blind_gaussian_model: GaussianModel<WasmB>,
    blind_gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<WasmB>, WasmB>,
    gaussian_model: GaussianModel<WasmB>,
    gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<WasmB>, WasmB>,
    nerf_model: NerfModel<WasmB>,
    nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, NerfModel<WasmB>, WasmB>,
    vol_nerf_model: VolumetricNerfModel<WasmB>,
    vol_nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, VolumetricNerfModel<WasmB>, WasmB>,
}

impl WasmTrainingSession {
    fn from_slices(
        width: usize,
        height: usize,
        num_gaussians: usize,
        rgb_slices: &[&[u8]],
        poses: Option<&[(f32, f32)]>,
    ) -> Self {
        let device = Default::default();
        let shape = [height, width, 3];
        let cw = 16.min(width);
        let ch = 16.min(height);

        let mut target_tensors = Vec::with_capacity(rgb_slices.len());
        let mut turntable_views = Vec::with_capacity(rgb_slices.len());
        let mut coarse_turntable_views = Vec::with_capacity(rgb_slices.len());

        for (k, slice) in rgb_slices.iter().enumerate() {
            let float_data: Vec<f32> = slice.iter().map(|&val| val as f32 / 255.0).collect();
            let full_t = Tensor::<WasmB, 3>::from_data(TensorData::new(float_data, shape), &device);
            target_tensors.push(full_t.clone());

            let (yaw, pitch) = if let Some(p) = poses.and_then(|arr| arr.get(k)) {
                *p
            } else {
                let v = if rgb_slices.len() > 1 {
                    k as f32 / ((rgb_slices.len() - 1) as f32)
                } else {
                    0.5
                };
                ((v - 0.5) * 0.6, 0.0)
            };
            turntable_views.push((yaw, pitch, full_t));

            let mut coarse_data = Vec::with_capacity(ch * cw * 3);
            for r in 0..ch {
                let sr = ((r * height) / ch).min(height - 1);
                for c in 0..cw {
                    let sc = ((c * width) / cw).min(width - 1);
                    let idx = (sr * width + sc) * 3;
                    coarse_data.push(slice[idx] as f32 / 255.0);
                    coarse_data.push(slice[idx + 1] as f32 / 255.0);
                    coarse_data.push(slice[idx + 2] as f32 / 255.0);
                }
            }
            let coarse_t = Tensor::<WasmB, 3>::from_data(TensorData::new(coarse_data, [ch, cw, 3]), &device);
            coarse_turntable_views.push((yaw, pitch, coarse_t));
        }

        let mut blind_gaussian_model = GaussianModel::<WasmB>::new(num_gaussians, &device);
        blind_gaussian_model.set_alpha_compositing(true);
        blind_gaussian_model.scales =
            Param::from_tensor(Tensor::<WasmB, 2>::full([num_gaussians, 2], -3.05, &device));
        let blind_gaussian_optim = AdamConfig::new().init();

        let mut gaussian_model = GaussianModel::<WasmB>::new(num_gaussians, &device);
        gaussian_model.set_alpha_compositing(true);
        let gaussian_optim = AdamConfig::new().init();

        let nerf_model = NerfModel::<WasmB>::new(8, 64, &device);
        let nerf_optim = AdamConfig::new().init();

        let vol_nerf_model = VolumetricNerfModel::<WasmB>::new(4, 32, 14, &device);
        let vol_nerf_optim = AdamConfig::new().init();

        Self {
            width,
            height,
            target_tensors,
            turntable_views,
            coarse_turntable_views,
            blind_gaussian_model,
            blind_gaussian_optim,
            gaussian_model,
            gaussian_optim,
            nerf_model,
            nerf_optim,
            vol_nerf_model,
            vol_nerf_optim,
        }
    }

    fn tensor_to_rgb_u8(rendered: Tensor<WasmB, 3>) -> Vec<u8> {
        rendered
            .into_data()
            .into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }
}

#[wasm_bindgen]
impl WasmTrainingSession {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, num_gaussians: usize, target_rgb_0: &[u8], target_rgb_1: &[u8]) -> Self {
        Self::from_slices(width, height, num_gaussians, &[target_rgb_0, target_rgb_1], None)
    }

    pub fn get_backend_name(&self) -> String {
        "Burn Flex WASM".to_string()
    }

    pub fn step_3d_warmup_fair(&mut self, lr_blind: f64, lr_vol_nerf: f64) {
        let (b_up, _) = train_step_gaussian_turntable(
            self.blind_gaussian_model.clone(),
            &mut self.blind_gaussian_optim,
            &self.turntable_views,
            lr_blind,
        );
        self.blind_gaussian_model = b_up;

        let (n_up, _) = train_step_volumetric_turntable(
            self.vol_nerf_model.clone(),
            &mut self.vol_nerf_optim,
            &self.coarse_turntable_views,
            lr_vol_nerf,
        );
        self.vol_nerf_model = n_up;
    }

    pub fn step_3d_race_fast(&mut self, lr: f64) {
        let (b_up, _) = train_step_gaussian_turntable(
            self.blind_gaussian_model.clone(),
            &mut self.blind_gaussian_optim,
            &self.turntable_views,
            lr,
        );
        self.blind_gaussian_model = b_up;

        let (g_up, _) = train_step_gaussian_turntable(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.turntable_views,
            lr,
        );
        self.gaussian_model = g_up;
    }

    pub async fn seed_from_3d_vol_nerf(&mut self) -> Vec<f32> {
        let voxels = self.vol_nerf_model.extract_3d_density_grid_async(15).await;
        let num_gaussians = self.gaussian_model.num_gaussians;
        self.gaussian_model = crate::hybrid::seed_gaussians_from_3d_volume::<WasmB>(
            &voxels,
            0.015,
            num_gaussians,
            &Default::default(),
        );
        self.gaussian_optim = AdamConfig::new().init();
        voxels
    }

    pub async fn get_turntable_losses(&self) -> Vec<f32> {
        let lb = self.blind_gaussian_model.forward_loss_turntable(&self.turntable_views);
        let lg = self.gaussian_model.forward_loss_turntable(&self.turntable_views);
        let db = lb.into_data().into_vec::<f32>().unwrap()[0];
        let dg = lg.into_data().into_vec::<f32>().unwrap()[0];
        vec![db, dg]
    }

    pub async fn get_vol_nerf_turntable_render(&self, nw: usize, nh: usize, yaw: f32, pitch: f32) -> Vec<u8> {
        Self::tensor_to_rgb_u8(self.vol_nerf_model.render_turntable(nw, nh, yaw, pitch))
    }

    pub async fn get_blind_turntable_render(&self, yaw: f32, pitch: f32) -> Vec<u8> {
        Self::tensor_to_rgb_u8(self.blind_gaussian_model.render_turntable(self.width, self.height, yaw, pitch))
    }

    pub async fn get_guided_turntable_render(&self, yaw: f32, pitch: f32) -> Vec<u8> {
        Self::tensor_to_rgb_u8(self.gaussian_model.render_turntable(self.width, self.height, yaw, pitch))
    }

    pub async fn get_blind_turntable_splats(&self, yaw: f32, pitch: f32) -> Vec<f32> {
        self.blind_gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await
    }

    pub async fn get_guided_turntable_splats(&self, yaw: f32, pitch: f32) -> Vec<f32> {
        self.gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await
    }

    pub fn step_gaussian_fast(&mut self, lr: f64) {
        let (up, _) = train_step_multiframe(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.target_tensors,
            lr,
        );
        self.gaussian_model = up;
    }

    pub fn step_nerf_fast(&mut self, lr: f64) {
        let (up, _) = train_step_multiframe(
            self.nerf_model.clone(),
            &mut self.nerf_optim,
            &self.target_tensors,
            lr,
        );
        self.nerf_model = up;
    }

    pub async fn get_losses(&self) -> Vec<f32> {
        let lg = self.gaussian_model.forward_loss_multiframe(&self.target_tensors).into_data().into_vec::<f32>().unwrap()[0];
        let ln = self.nerf_model.forward_loss_multiframe(&self.target_tensors).into_data().into_vec::<f32>().unwrap()[0];
        vec![lg, ln]
    }

    pub async fn get_gaussian_render_view(&self, view_v: f32) -> Vec<u8> {
        Self::tensor_to_rgb_u8(self.gaussian_model.render_view(self.width, self.height, view_v))
    }

    pub async fn get_nerf_render_view(&self, view_v: f32) -> Vec<u8> {
        Self::tensor_to_rgb_u8(self.nerf_model.render_view(self.width, self.height, view_v))
    }
}

#[wasm_bindgen]
pub fn create_turntable_3d_session(
    width: usize,
    height: usize,
    num_gaussians: usize,
    flat_rgb_concatenated: &[u8],
    yaw_pitch_pairs: &[f32],
) -> WasmTrainingSession {
    let num_frames = yaw_pitch_pairs.len() / 2;
    let frame_len = width * height * 3;
    let mut slices = Vec::with_capacity(num_frames);
    let mut poses = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * frame_len;
        let end = start + frame_len;
        if end <= flat_rgb_concatenated.len() {
            slices.push(&flat_rgb_concatenated[start..end]);
            poses.push((yaw_pitch_pairs[i * 2], yaw_pitch_pairs[i * 2 + 1]));
        }
    }
    WasmTrainingSession::from_slices(width, height, num_gaussians, &slices, Some(&poses))
}

#[wasm_bindgen]
pub fn create_multiframe_session(
    width: usize,
    height: usize,
    num_gaussians: usize,
    flat_rgb_concatenated: &[u8],
    num_frames: usize,
) -> WasmTrainingSession {
    let frame_len = width * height * 3;
    let mut slices = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * frame_len;
        let end = start + frame_len;
        if end <= flat_rgb_concatenated.len() {
            slices.push(&flat_rgb_concatenated[start..end]);
        }
    }
    WasmTrainingSession::from_slices(width, height, num_gaussians, &slices, None)
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn init_webgpu() -> bool {
    false
}
