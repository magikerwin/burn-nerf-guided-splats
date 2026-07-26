//! # Model Abstractions: Single-View and Multi-View Representations
//!
//! This module defines the common traits shared by Explicit 3D Gaussian Splatting (`GaussianModel`)
//! and Implicit Coordinate MLP (`NerfModel`).
//!
//! ### Core Concepts:
//! - **Single-View Fitting (`ImageFitter`)**: Fits a single 2D target image from a fixed viewpoint.
//! - **Multi-View Synthesis (`MultiViewFitter`)**: Fits multiple target images captured from different
//!   camera angles $v \in [0.0, 1.0]$. Enables continuous novel view interpolation in-between frames.

use burn::tensor::{backend::Backend, Tensor};

pub mod gaussian;
pub mod nerf;

/// Interface for models capable of rendering a 2D image and evaluating single-view loss.
pub trait ImageFitter<B: Backend> {
    /// Renders the current representation state into an RGB image tensor of shape `[height, width, 3]`.
    fn render(&self, width: usize, height: usize) -> Tensor<B, 3>;

    /// Computes Mean Squared Error (MSE) loss against a target ground truth image.
    fn forward_loss(&self, target_image: &Tensor<B, 3>) -> Tensor<B, 1>;
}

/// Interface for models supporting view-conditioned rendering and multi-view loss evaluation.
pub trait MultiViewFitter<B: Backend>: ImageFitter<B> {
    /// Renders the scene from an arbitrary camera view parameter $v \in [0.0, 1.0]$.
    ///
    /// - $v = 0.0$: Angle 1 ($0^\circ$)
    /// - $v = 1.0$: Angle 2 ($30^\circ$)
    /// - $v \in (0, 1)$: Synthesized in-between view frame
    fn render_view(&self, width: usize, height: usize, view_v: f32) -> Tensor<B, 3>;

    /// Computes joint multi-view MSE loss across View 0 and View 1:
    /// $\mathcal{L}_{\text{multiview}} = \frac{1}{2} \left( \text{MSE}(I_{\text{pred}}(v=0), I_{\text{gt0}}) + \text{MSE}(I_{\text{pred}}(v=1), I_{\text{gt1}}) \right)$
    fn forward_loss_multiview(&self, target_v0: &Tensor<B, 3>, target_v1: &Tensor<B, 3>) -> Tensor<B, 1>;
}


