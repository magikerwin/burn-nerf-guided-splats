use std::cell::Cell;
use wasm_bindgen::prelude::*;
use burn::backend::{Autodiff, Flex, Wgpu};
use burn::module::Param;
use burn::optim::{Adam, AdamConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use crate::model::gaussian::GaussianModel;
use crate::model::nerf::{NerfModel, VolumetricNerfModel};
use crate::model::MultiViewFitter;
use crate::training::{
    train_step_gaussian_turntable, train_step_multiframe, train_step_volumetric_turntable,
};

type WgpuAd = Autodiff<Wgpu>;
type CpuAd = Autodiff<Flex>;

thread_local! {
    static WEBGPU_READY: Cell<bool> = const { Cell::new(false) };
}

#[wasm_bindgen(inline_js = "
export async function probe_webgpu_support() {
    try {
        if (typeof navigator === 'undefined' || !navigator.gpu) return false;
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch (_) {
        return false;
    }
}
")]
extern "C" {
    async fn probe_webgpu_support() -> JsValue;
}

struct GenericSession<B: AutodiffBackend> {
    width: usize,
    height: usize,
    target_tensors: Vec<Tensor<B, 3>>,
    turntable_views: Vec<(f32, f32, Tensor<B, 3>)>,
    coarse_turntable_views: Vec<(f32, f32, Tensor<B, 3>)>,
    /// Blind (Randomly Initialized) 3D Gaussian Splatting baseline
    blind_gaussian_model: GaussianModel<B>,
    blind_gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<B>, B>,
    /// Primary / 3D Volumetric NeRF-Guided 3D Gaussian Splatting model
    gaussian_model: GaussianModel<B>,
    gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<B>, B>,
    /// Legacy 2D+view Coordinate MLP (kept for before.html compatibility)
    nerf_model: NerfModel<B>,
    nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, NerfModel<B>, B>,
    /// True 3D Ray-Marched Volumetric NeRF: (x, y, z) -> (RGB, sigma)
    vol_nerf_model: VolumetricNerfModel<B>,
    vol_nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, VolumetricNerfModel<B>, B>,
    device: B::Device,
}

impl<B: AutodiffBackend> GenericSession<B> {
    fn new(
        width: usize,
        height: usize,
        num_gaussians: usize,
        rgb_slices: &[&[u8]],
        poses: Option<&[(f32, f32)]>,
    ) -> Self {
        let device = B::Device::default();
        let shape = [height, width, 3];
        let cw = 16.min(width);
        let ch = 16.min(height);

        let mut target_tensors = Vec::with_capacity(rgb_slices.len());
        let mut turntable_views = Vec::with_capacity(rgb_slices.len());
        let mut coarse_turntable_views = Vec::with_capacity(rgb_slices.len());

        for (k, slice) in rgb_slices.iter().enumerate() {
            let mut float_data = Vec::with_capacity(slice.len());
            for &val in slice.iter() {
                float_data.push(val as f32 / 255.0);
            }
            let full_t = Tensor::<B, 3>::from_data(TensorData::new(float_data, shape), &device);
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
            let coarse_t = Tensor::<B, 3>::from_data(TensorData::new(coarse_data, [ch, cw, 3]), &device);
            coarse_turntable_views.push((yaw, pitch, coarse_t));
        }

        let mut blind_gaussian_model = GaussianModel::<B>::new(num_gaussians, &device);
        blind_gaussian_model.set_alpha_compositing(true);
        blind_gaussian_model.scales =
            Param::from_tensor(Tensor::<B, 2>::full([num_gaussians, 2], -3.05, &device));
        let blind_gaussian_optim = AdamConfig::new().init();

        let mut gaussian_model = GaussianModel::<B>::new(num_gaussians, &device);
        gaussian_model.set_alpha_compositing(true);
        let gaussian_optim = AdamConfig::new().init();

        let nerf_model = NerfModel::<B>::new(8, 64, &device);
        let nerf_optim = AdamConfig::new().init();

        let vol_nerf_model = VolumetricNerfModel::<B>::new(4, 32, 14, &device);
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
            device,
        }
    }

    fn step_blind_turntable_fast(&mut self, lr: f64) {
        let (updated, _) = train_step_gaussian_turntable(
            self.blind_gaussian_model.clone(),
            &mut self.blind_gaussian_optim,
            &self.turntable_views,
            lr,
        );
        self.blind_gaussian_model = updated;
    }

    fn step_guided_turntable_fast(&mut self, lr: f64) {
        let (updated, _) = train_step_gaussian_turntable(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.turntable_views,
            lr,
        );
        self.gaussian_model = updated;
    }

    fn step_vol_nerf_coarse_fast(&mut self, lr: f64) {
        let (updated, _) = train_step_volumetric_turntable(
            self.vol_nerf_model.clone(),
            &mut self.vol_nerf_optim,
            &self.coarse_turntable_views,
            lr,
        );
        self.vol_nerf_model = updated;
    }

    fn step_gaussian_fast(&mut self, lr: f64) {
        let (updated, _) = train_step_multiframe(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.target_tensors,
            lr,
        );
        self.gaussian_model = updated;
    }

    fn step_nerf_fast(&mut self, lr: f64) {
        let (updated, _) = train_step_multiframe(
            self.nerf_model.clone(),
            &mut self.nerf_optim,
            &self.target_tensors,
            lr,
        );
        self.nerf_model = updated;
    }

    async fn get_losses(&self) -> Vec<f32> {
        let loss_g = self.gaussian_model.forward_loss_multiframe(&self.target_tensors);
        let loss_n = self.nerf_model.forward_loss_multiframe(&self.target_tensors);
        let data_g = loss_g.into_data_async().await.unwrap();
        let data_n = loss_n.into_data_async().await.unwrap();
        vec![data_g.as_slice::<f32>().unwrap()[0], data_n.as_slice::<f32>().unwrap()[0]]
    }

    async fn get_turntable_losses(&self) -> Vec<f32> {
        let lb = self.blind_gaussian_model.forward_loss_turntable(&self.turntable_views);
        let lg = self.gaussian_model.forward_loss_turntable(&self.turntable_views);
        let db = lb.into_data_async().await.unwrap();
        let dg = lg.into_data_async().await.unwrap();
        vec![db.as_slice::<f32>().unwrap()[0], dg.as_slice::<f32>().unwrap()[0]]
    }

    async fn tensor_to_rgb_u8(rendered: Tensor<B, 3>) -> Vec<u8> {
        let data = rendered
            .into_data_async()
            .await
            .unwrap()
            .into_vec::<f32>()
            .unwrap();
        data.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    async fn seed_from_3d_vol_nerf(&mut self) -> Vec<f32> {
        let voxels = self.vol_nerf_model.extract_3d_density_grid_async(15).await;
        let num_gaussians = self.gaussian_model.num_gaussians;
        self.gaussian_model = crate::hybrid::seed_gaussians_from_3d_volume::<B>(
            &voxels,
            0.015,
            num_gaussians,
            &self.device,
        );
        self.gaussian_optim = AdamConfig::new().init();
        voxels
    }

    async fn seed_from_nerf_legacy(&mut self) {
        let v0 = self.nerf_model.render_view(self.width, self.height, 0.0).into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let vm = self.nerf_model.render_view(self.width, self.height, 0.5).into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let v1 = self.nerf_model.render_view(self.width, self.height, 1.0).into_data_async().await.unwrap().into_vec::<f32>().unwrap();
        let n = self.gaussian_model.num_gaussians;
        self.gaussian_model = crate::hybrid::seed_gaussians_multiview::<B>(&v0, &vm, &v1, self.height, self.width, n, &self.device);
        self.gaussian_optim = AdamConfig::new().init();
    }
}

enum SessionBackend {
    Wgpu(GenericSession<WgpuAd>),
    Cpu(GenericSession<CpuAd>),
}

#[wasm_bindgen]
pub struct WasmTrainingSession {
    inner: SessionBackend,
}

fn create_session_internal(
    width: usize,
    height: usize,
    num_gaussians: usize,
    rgb_slices: &[&[u8]],
    poses: Option<&[(f32, f32)]>,
) -> WasmTrainingSession {
    let use_wgpu = WEBGPU_READY.with(|flag| flag.get());
    let inner = if use_wgpu {
        SessionBackend::Wgpu(GenericSession::<WgpuAd>::new(width, height, num_gaussians, rgb_slices, poses))
    } else {
        SessionBackend::Cpu(GenericSession::<CpuAd>::new(width, height, num_gaussians, rgb_slices, poses))
    };
    WasmTrainingSession { inner }
}

#[wasm_bindgen]
impl WasmTrainingSession {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, num_gaussians: usize, target_rgb_0: &[u8], target_rgb_1: &[u8]) -> Self {
        create_session_internal(width, height, num_gaussians, &[target_rgb_0, target_rgb_1], None)
    }

    pub fn get_backend_name(&self) -> String {
        match &self.inner {
            SessionBackend::Wgpu(_) => "WebGPU".to_string(),
            SessionBackend::Cpu(_) => "CPU WASM".to_string(),
        }
    }

    /// Fair t=0 3D clock warmup: Blind 3DGS trains on full views WHILE Volumetric NeRF marches coarse 3D rays!
    pub fn step_3d_warmup_fair(&mut self, lr_blind: f64, lr_vol_nerf: f64) {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => {
                s.step_blind_turntable_fast(lr_blind);
                s.step_vol_nerf_coarse_fast(lr_vol_nerf);
            }
            SessionBackend::Cpu(s) => {
                s.step_blind_turntable_fast(lr_blind);
                s.step_vol_nerf_coarse_fast(lr_vol_nerf);
            }
        }
    }

    /// Post-warmup 3D turntable race step for both Blind and 3D-Volumetric-Guided Gaussians.
    pub fn step_3d_race_fast(&mut self, lr: f64) {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => {
                s.step_blind_turntable_fast(lr);
                s.step_guided_turntable_fast(lr);
            }
            SessionBackend::Cpu(s) => {
                s.step_blind_turntable_fast(lr);
                s.step_guided_turntable_fast(lr);
            }
        }
    }

    pub async fn seed_from_3d_vol_nerf(&mut self) -> Vec<f32> {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => s.seed_from_3d_vol_nerf().await,
            SessionBackend::Cpu(s) => s.seed_from_3d_vol_nerf().await,
        }
    }

    pub async fn get_turntable_losses(&self) -> Vec<f32> {
        match &self.inner {
            SessionBackend::Wgpu(s) => s.get_turntable_losses().await,
            SessionBackend::Cpu(s) => s.get_turntable_losses().await,
        }
    }

    pub async fn get_vol_nerf_turntable_render(&self, nw: usize, nh: usize, yaw: f32, pitch: f32) -> Vec<u8> {
        match &self.inner {
            SessionBackend::Wgpu(s) => GenericSession::tensor_to_rgb_u8(s.vol_nerf_model.render_turntable(nw, nh, yaw, pitch)).await,
            SessionBackend::Cpu(s) => GenericSession::tensor_to_rgb_u8(s.vol_nerf_model.render_turntable(nw, nh, yaw, pitch)).await,
        }
    }

    pub async fn get_blind_turntable_render(&self, yaw: f32, pitch: f32) -> Vec<u8> {
        match &self.inner {
            SessionBackend::Wgpu(s) => GenericSession::tensor_to_rgb_u8(s.blind_gaussian_model.render_turntable(s.width, s.height, yaw, pitch)).await,
            SessionBackend::Cpu(s) => GenericSession::tensor_to_rgb_u8(s.blind_gaussian_model.render_turntable(s.width, s.height, yaw, pitch)).await,
        }
    }

    pub async fn get_guided_turntable_render(&self, yaw: f32, pitch: f32) -> Vec<u8> {
        match &self.inner {
            SessionBackend::Wgpu(s) => GenericSession::tensor_to_rgb_u8(s.gaussian_model.render_turntable(s.width, s.height, yaw, pitch)).await,
            SessionBackend::Cpu(s) => GenericSession::tensor_to_rgb_u8(s.gaussian_model.render_turntable(s.width, s.height, yaw, pitch)).await,
        }
    }

    pub async fn get_blind_turntable_splats(&self, yaw: f32, pitch: f32) -> Vec<f32> {
        match &self.inner {
            SessionBackend::Wgpu(s) => s.blind_gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await,
            SessionBackend::Cpu(s) => s.blind_gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await,
        }
    }

    pub async fn get_guided_turntable_splats(&self, yaw: f32, pitch: f32) -> Vec<f32> {
        match &self.inner {
            SessionBackend::Wgpu(s) => s.gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await,
            SessionBackend::Cpu(s) => s.gaussian_model.extract_turntable_splat_descriptors_async(yaw, pitch).await,
        }
    }

    // Legacy 1D methods preserved for web/before.html
    pub fn step_gaussian_fast(&mut self, lr: f64) {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => s.step_gaussian_fast(lr),
            SessionBackend::Cpu(s) => s.step_gaussian_fast(lr),
        }
    }

    pub fn step_nerf_fast(&mut self, lr: f64) {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => s.step_nerf_fast(lr),
            SessionBackend::Cpu(s) => s.step_nerf_fast(lr),
        }
    }

    pub async fn get_losses(&self) -> Vec<f32> {
        match &self.inner {
            SessionBackend::Wgpu(s) => s.get_losses().await,
            SessionBackend::Cpu(s) => s.get_losses().await,
        }
    }

    pub async fn get_gaussian_render_view(&self, view_v: f32) -> Vec<u8> {
        match &self.inner {
            SessionBackend::Wgpu(s) => GenericSession::tensor_to_rgb_u8(s.gaussian_model.render_view(s.width, s.height, view_v)).await,
            SessionBackend::Cpu(s) => GenericSession::tensor_to_rgb_u8(s.gaussian_model.render_view(s.width, s.height, view_v)).await,
        }
    }

    pub async fn get_nerf_render_view(&self, view_v: f32) -> Vec<u8> {
        match &self.inner {
            SessionBackend::Wgpu(s) => GenericSession::tensor_to_rgb_u8(s.nerf_model.render_view(s.width, s.height, view_v)).await,
            SessionBackend::Cpu(s) => GenericSession::tensor_to_rgb_u8(s.nerf_model.render_view(s.width, s.height, view_v)).await,
        }
    }

    pub async fn seed_from_nerf(&mut self) {
        match &mut self.inner {
            SessionBackend::Wgpu(s) => s.seed_from_nerf_legacy().await,
            SessionBackend::Cpu(s) => s.seed_from_nerf_legacy().await,
        }
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
    create_session_internal(width, height, num_gaussians, &slices, Some(&poses))
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
    create_session_internal(width, height, num_gaussians, &slices, None)
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn init_webgpu() -> bool {
    let supported = probe_webgpu_support().await.as_bool().unwrap_or(false);
    if supported {
        let device = Default::default();
        burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
            &device,
            Default::default(),
        )
        .await;
        WEBGPU_READY.with(|flag| flag.set(true));
        true
    } else {
        WEBGPU_READY.with(|flag| flag.set(false));
        false
    }
}
