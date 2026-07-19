use burn::tensor::{backend::Backend, Tensor};

pub mod gaussian;

pub trait ImageFitter<B: Backend> {
    /// Renders the current state to an image tensor of shape [height, width, 3]
    fn render(&self, width: usize, height: usize) -> Tensor<B, 3>;

    /// Computes the MSE loss between the rendered image and the target image
    fn forward_loss(&self, target_image: &Tensor<B, 3>) -> Tensor<B, 1>;
}
