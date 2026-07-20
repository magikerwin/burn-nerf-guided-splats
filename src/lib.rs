// NeRF-Guided 2D Gaussian Splatting in Burn library root
pub mod model;
pub mod utils;
pub mod training;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

