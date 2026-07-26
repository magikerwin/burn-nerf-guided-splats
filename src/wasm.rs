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
        Self::new_multiframe(width, height, num_gaussians, &[target_rgb_0, target_rgb_1])
    }

    pub fn new_multiframe(
        width: usize,
        height: usize,
        num_gaussians: usize,
        rgb_slices: &[&[u8]],
    ) -> Self {
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

        Self {
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

    pub async fn step_gaussian(&mut self, lr: f64) -> f32 {
        let (updated_model, loss_tensor) = train_step_multiframe(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.target_tensors,
            lr,
        );
        self.gaussian_model = updated_model;
        let data = loss_tensor.into_data_async().await.expect("Failed to read loss data");
        data.as_slice::<f32>().unwrap()[0]
    }

    pub async fn step_nerf(&mut self, lr: f64) -> f32 {
        let (updated_model, loss_tensor) = train_step_multiframe(
            self.nerf_model.clone(),
            &mut self.nerf_optim,
            &self.target_tensors,
            lr,
        );
        self.nerf_model = updated_model;
        let data = loss_tensor.into_data_async().await.expect("Failed to read loss data");
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
        let nerf_render_0 = self.nerf_model.render_view(self.width, self.height, 0.0);
        let nerf_render_1 = self.nerf_model.render_view(self.width, self.height, 1.0);

        let imp_0 = crate::hybrid::compute_importance_map(nerf_render_0.clone());
        let imp_1 = crate::hybrid::compute_importance_map(nerf_render_1.clone());
        let importance_tensor = imp_0.add(imp_1).mul_scalar(0.5);
        
        let dims = importance_tensor.shape().dims::<3>();
        let h = dims[0];
        let w = dims[1];
        
        let importance_vec = importance_tensor
            .into_data_async()
            .await
            .expect("Failed to read importance data")
            .into_vec::<f32>()
            .expect("Failed to get importance map data");

        let nerf_render_vec = nerf_render_0
            .into_data_async()
            .await
            .expect("Failed to read NeRF render data")
            .into_vec::<f32>()
            .expect("Failed to get NeRF render float data");

        let num_gaussians = self.gaussian_model.num_gaussians;
        let seeded_model = crate::hybrid::seed_gaussians_from_importance(
            &importance_vec,
            &nerf_render_vec,
            h,
            w,
            num_gaussians,
            &self.device,
        );
        self.gaussian_model = seeded_model;
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
    WasmTrainingSession::new_multiframe(width, height, num_gaussians, &slices)
}


#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn init_webgpu() {
    let device = Default::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        &device,
        Default::default(),
    )
    .await;
}

