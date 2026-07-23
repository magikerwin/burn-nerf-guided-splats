use wasm_bindgen::prelude::*;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::{Tensor, TensorData};
use crate::model::gaussian::GaussianModel;
use crate::model::nerf::NerfModel;
use crate::model::ImageFitter;
use crate::training::train_step;

type B = Autodiff<Wgpu>;

#[wasm_bindgen]
pub struct WasmTrainingSession {
    width: usize,
    height: usize,
    target_tensor: Tensor<B, 3>,
    gaussian_model: GaussianModel<B>,
    gaussian_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, GaussianModel<B>, B>,
    nerf_model: NerfModel<B>,
    nerf_optim: burn::optim::adaptor::OptimizerAdaptor<Adam, NerfModel<B>, B>,
    device: <B as burn::tensor::backend::BackendTypes>::Device,
}

#[wasm_bindgen]
impl WasmTrainingSession {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, num_gaussians: usize, target_rgb: &[u8]) -> Self {
        let device = Default::default();

        // Convert the target_rgb &[u8] slice to a Burn Tensor.
        // target_rgb contains flat RGB values: [r0, g0, b0, r1, g1, b1, ...]
        let mut float_data = Vec::with_capacity(target_rgb.len());
        for &val in target_rgb.iter() {
            float_data.push(val as f32 / 255.0);
        }

        let shape = [height, width, 3];
        let tensor_data = TensorData::new(float_data, shape);
        let target_tensor = Tensor::<B, 3>::from_data(tensor_data, &device);

        let gaussian_model = GaussianModel::<B>::new(num_gaussians, &device);
        let gaussian_optim = AdamConfig::new().init();

        let nerf_model = NerfModel::<B>::new(8, 64, &device);
        let nerf_optim = AdamConfig::new().init();

        Self {
            width,
            height,
            target_tensor,
            gaussian_model,
            gaussian_optim,
            nerf_model,
            nerf_optim,
            device,
        }
    }

    pub async fn step_gaussian(&mut self, lr: f64) -> f32 {
        let (updated_model, loss_tensor) = train_step(
            self.gaussian_model.clone(),
            &mut self.gaussian_optim,
            &self.target_tensor,
            lr,
        );
        self.gaussian_model = updated_model;
        let data = loss_tensor.into_data_async().await.expect("Failed to read loss data");
        data.as_slice::<f32>().unwrap()[0]
    }

    pub async fn step_nerf(&mut self, lr: f64) -> f32 {
        let (updated_model, loss_tensor) = train_step(
            self.nerf_model.clone(),
            &mut self.nerf_optim,
            &self.target_tensor,
            lr,
        );
        self.nerf_model = updated_model;
        let data = loss_tensor.into_data_async().await.expect("Failed to read loss data");
        data.as_slice::<f32>().unwrap()[0]
    }

    pub async fn get_gaussian_render(&self) -> Vec<u8> {
        let rendered = self.gaussian_model.render(self.width, self.height);
        // Transfer to host/CPU asynchronously and map to [0, 255] u8
        let data = rendered.into_data_async().await.expect("Failed to read render data").into_vec::<f32>().expect("Failed to get tensor data");
        let mut rgb = Vec::with_capacity(data.len());
        for &val in data.iter() {
            rgb.push((val.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgb
    }

    pub async fn get_nerf_render(&self) -> Vec<u8> {
        let rendered = self.nerf_model.render(self.width, self.height);
        // Transfer to host/CPU asynchronously and map to [0, 255] u8
        let data = rendered.into_data_async().await.expect("Failed to read render data").into_vec::<f32>().expect("Failed to get tensor data");
        let mut rgb = Vec::with_capacity(data.len());
        for &val in data.iter() {
            rgb.push((val.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgb
    }

    pub async fn seed_from_nerf(&mut self) {
        let nerf_render = self.nerf_model.render(self.width, self.height);
        let importance_tensor = crate::hybrid::compute_importance_map(nerf_render.clone());
        
        let dims = importance_tensor.shape().dims::<3>();
        let h = dims[0];
        let w = dims[1];
        
        // Fetch importance map values to CPU asynchronously
        let importance_vec = importance_tensor
            .into_data_async()
            .await
            .expect("Failed to read importance data")
            .into_vec::<f32>()
            .expect("Failed to get importance map data");

        // Fetch NeRF render RGB values to CPU asynchronously
        let nerf_render_vec = nerf_render
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
        let nerf_render = self.nerf_model.render(self.width, self.height);
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
