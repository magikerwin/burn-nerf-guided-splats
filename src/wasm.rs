use wasm_bindgen::prelude::*;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::{Tensor, TensorData};
use crate::model::gaussian::GaussianModel;
use crate::model::nerf::NerfModel;
use crate::model::MultiViewFitter;
use crate::training::train_step_multiframe;

type B = Autodiff<Wgpu>;

#[wasm_bindgen]
pub struct WasmTrainingSession {
    width: usize,
    height: usize,
    target_tensors: Vec<Tensor<B, 3>>,
    gaussian_model: GaussianModel<B>,
    gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<B>, B>,
    nerf_model: NerfModel<B>,
    nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, NerfModel<B>, B>,
    device: <B as burn::tensor::backend::BackendTypes>::Device,
}

fn create_multiframe_session_internal(
    width: usize,
    height: usize,
    num_gaussians: usize,
    rgb_slices: &[&[u8]],
) -> WasmTrainingSession {
    let device = Default::default();
    let shape = [height, width, 3];
    let mut target_tensors = Vec::with_capacity(rgb_slices.len());

    for slice in rgb_slices {
        let mut float_data = Vec::with_capacity(slice.len());
        for &val in slice.iter() {
            float_data.push(val as f32 / 255.0);
        }
        let tensor_data = TensorData::new(float_data, shape);
        target_tensors.push(Tensor::<B, 3>::from_data(tensor_data, &device));
    }

    let gaussian_model = GaussianModel::<B>::new(num_gaussians, &device);
    let gaussian_optim = AdamConfig::new().init();

    let nerf_model = NerfModel::<B>::new(8, 64, &device);
    let nerf_optim = AdamConfig::new().init();

    WasmTrainingSession {
        width,
        height,
        target_tensors,
        gaussian_model,
        gaussian_optim,
        nerf_model,
        nerf_optim,
        device,
    }
}

#[wasm_bindgen]
impl WasmTrainingSession {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        num_gaussians: usize,
        target_rgb_0: &[u8],
        target_rgb_1: &[u8],
    ) -> Self {
        create_multiframe_session_internal(width, height, num_gaussians, &[target_rgb_0, target_rgb_1])
    }


    /// Fast non-blocking GPU optimization step for 3D Gaussian Splatting (Zero CPU buffer readback).
    pub fn step_gaussian_fast(&mut self, lr: f64) {
        let (updated_model, _loss_tensor) = train_step_multiframe(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.target_tensors,
            lr,
        );
        self.gaussian_model = updated_model;
    }

    /// Fast non-blocking GPU optimization step for Implicit NeRF (Zero CPU buffer readback).
    pub fn step_nerf_fast(&mut self, lr: f64) {
        let (updated_model, _loss_tensor) = train_step_multiframe(
            self.nerf_model.clone(),
            &mut self.nerf_optim,
            &self.target_tensors,
            lr,
        );
        self.nerf_model = updated_model;
    }

    /// Asynchronously fetches current loss values for Gaussian Splatting and NeRF.
    pub async fn get_losses(&self) -> Vec<f32> {
        let loss_g = self.gaussian_model.forward_loss_multiframe(&self.target_tensors);
        let loss_n = self.nerf_model.forward_loss_multiframe(&self.target_tensors);
        
        let data_g = loss_g.into_data_async().await.expect("Failed to read loss G");
        let data_n = loss_n.into_data_async().await.expect("Failed to read loss N");

        vec![
            data_g.as_slice::<f32>().unwrap()[0],
            data_n.as_slice::<f32>().unwrap()[0],
        ]
    }

    pub async fn step_gaussian(&mut self, lr: f64) -> f32 {
        self.step_gaussian_fast(lr);
        let loss_g = self.gaussian_model.forward_loss_multiframe(&self.target_tensors);
        let data = loss_g.into_data_async().await.expect("Failed to read loss data");
        data.as_slice::<f32>().unwrap()[0]
    }

    pub async fn step_nerf(&mut self, lr: f64) -> f32 {
        self.step_nerf_fast(lr);
        let loss_n = self.nerf_model.forward_loss_multiframe(&self.target_tensors);
        let data = loss_n.into_data_async().await.expect("Failed to read loss data");
        data.as_slice::<f32>().unwrap()[0]
    }


    pub async fn get_gaussian_render_view(&self, view_v: f32) -> Vec<u8> {
        let rendered = self.gaussian_model.render_view(self.width, self.height, view_v);
        let data = rendered.into_data_async().await.expect("Failed to read render data").into_vec::<f32>().expect("Failed to get tensor data");
        let mut rgb = Vec::with_capacity(data.len());
        for &val in data.iter() {
            rgb.push((val.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgb
    }

    pub async fn get_nerf_render_view(&self, view_v: f32) -> Vec<u8> {
        let rendered = self.nerf_model.render_view(self.width, self.height, view_v);
        let data = rendered.into_data_async().await.expect("Failed to read render data").into_vec::<f32>().expect("Failed to get tensor data");
        let mut rgb = Vec::with_capacity(data.len());
        for &val in data.iter() {
            rgb.push((val.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgb
    }

    pub async fn get_gaussian_render(&self) -> Vec<u8> {
        self.get_gaussian_render_view(0.5).await
    }

    pub async fn get_nerf_render(&self) -> Vec<u8> {
        self.get_nerf_render_view(0.5).await
    }

    pub async fn seed_from_nerf(&mut self) {
        let nerf_render_0 = self.nerf_model.render_view(self.width, self.height, 0.5);
        let nerf_render_vec = nerf_render_0
            .into_data_async()
            .await
            .expect("Failed to read NeRF render data")
            .into_vec::<f32>()
            .expect("Failed to get NeRF render float data");

        let h = self.height;
        let w = self.width;
        let mut importance_data = Vec::with_capacity((h - 1) * (w - 1));

        for r in 0..(h - 1) {
            for c in 0..(w - 1) {
                let idx_curr = (r * w + c) * 3;
                let idx_right = (r * w + (c + 1)) * 3;
                let idx_bottom = ((r + 1) * w + c) * 3;

                let dx_r = nerf_render_vec[idx_right] - nerf_render_vec[idx_curr];
                let dx_g = nerf_render_vec[idx_right + 1] - nerf_render_vec[idx_curr + 1];
                let dx_b = nerf_render_vec[idx_right + 2] - nerf_render_vec[idx_curr + 2];
                let dx2 = dx_r * dx_r + dx_g * dx_g + dx_b * dx_b;

                let dy_r = nerf_render_vec[idx_bottom] - nerf_render_vec[idx_curr];
                let dy_g = nerf_render_vec[idx_bottom + 1] - nerf_render_vec[idx_curr + 1];
                let dy_b = nerf_render_vec[idx_bottom + 2] - nerf_render_vec[idx_curr + 2];
                let dy2 = dy_r * dy_r + dy_g * dy_g + dy_b * dy_b;

                importance_data.push((dx2 + dy2).sqrt());
            }
        }

        let num_gaussians = self.gaussian_model.num_gaussians;
        let device = Default::default();

        let new_gaussian_model = crate::hybrid::seed_gaussians_from_importance::<B>(
            &importance_data,
            &nerf_render_vec,
            h - 1,
            w - 1,
            num_gaussians,
            &device,
        );

        self.gaussian_model = new_gaussian_model;
        self.gaussian_optim = AdamConfig::new().init();
    }

    pub async fn get_nerf_importance_map(&self) -> Vec<u8> {
        let nerf_render = self.nerf_model.render_view(self.width, self.height, 0.0);
        let importance_tensor = crate::hybrid::compute_importance_map(nerf_render);
        
        let data = importance_tensor
            .into_data_async()
            .await
            .expect("Failed to read importance data")
            .into_vec::<f32>()
            .expect("Failed to get tensor data");
            
        let mut rgb = Vec::with_capacity(data.len() * 3);
        for &val in data.iter() {
            let v = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
        rgb
    }
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
    create_multiframe_session_internal(width, height, num_gaussians, &slices)

}


#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Asynchronously initializes Burn WGPU WebGPU device context for WASM.
#[wasm_bindgen]
pub async fn init_webgpu() {
    let device = Default::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        &device,
        Default::default(),
    )
    .await;
}



