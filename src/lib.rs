// NeRF-Guided 2D Gaussian Splatting in Burn library root
pub mod model;
pub mod utils;
pub mod training;
pub mod hybrid;

#[cfg(target_arch = "wasm32")]
pub mod wasm;


