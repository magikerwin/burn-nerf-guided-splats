# BurnSplat: NeRF-Guided 2D Gaussian Splatting

An educational 2D rendering engine implemented in Rust using the **Burn** deep learning framework. This project demonstrates how explicit and implicit neural representations can be combined into a state-of-the-art hybrid pipeline:

1. **Explicit Representation (2D Gaussian Splatting):** Optimizing the physical parameters (Position, Scale, Rotation, Color, Opacity) of 2D Gaussians. Excellent at capturing sharp, high-frequency details and rendering at very high frame rates.
2. **Implicit Representation (2D NeRF / Coordinate MLP):** Training a Multi-Layer Perceptron (MLP) with Positional Encoding to map $(x, y)$ coordinate grids to RGB colors. Excellent at learning continuous coordinate representations, smooth gradients, and global structures.
3. **Cooperative Hybrid Representation (NeRF-Guided GS):** Seeding the initial locations of the explicit 2D Gaussians based on spatial derivatives (variance and edges) extracted from the partially trained implicit NeRF MLP, rather than a uniform random distribution. This directly mirrors state-of-the-art hybrid 3D reconstruction pipelines.

Both models compile to **WebAssembly (WASM)** and run locally in the web browser accelerated by **WebGPU** using Burn's WGPU backend.

---

## 1. Mathematical Specifications

### 2D Gaussian Splatting (Explicit)
Each of the $N$ Gaussians is parameterized by:
*   **Means ($\mu$):** Shape `[N, 2]` representing $(x, y)$ coordinates in $[0, 1]$.
*   **Scales ($S$):** Shape `[N, 2]` representing standard deviations along major and minor axes.
*   **Rotations ($R$):** Shape `[N, 1]` storing rotation angle $\theta$.
*   **Colors ($C$):** Shape `[N, 3]` representing RGB channels (clamped to $[0, 1]$ using a sigmoid activation).
*   **Opacities ($\alpha$):** Shape `[N, 1]` representing density contribution (clamped to $[0, 1]$ using a sigmoid activation).

The 2D covariance matrix $\Sigma$ is computed analytically from scale $S$ and rotation $R$:
$$\Sigma = R S S^T R^T$$

The unnormalized Gaussian value at a canvas pixel coordinate $x$ is:
$$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

To render the canvas, the contributions of all $N$ Gaussians are accumulated:
$$\text{Color}(x) = \sum_{i=1}^N \alpha_i \cdot G_i(x) \cdot C_i$$

### 2D NeRF / Coordinate MLP (Implicit)
Input $(x, y)$ coordinates are mapped to a higher-dimensional space using Fourier features (sine and cosine frequencies):
$$\gamma(p) = \Big(\sin(2^k \pi p), \cos(2^k \pi p)\Big)_{k=0}^{L-1}$$
For two coordinates ($x$ and $y$), this yields a $4L$-dimensional input vector.

This vector is passed through a coordinate network (MLP) consisting of:
*   3-4 Linear layers with ReLU activations.
*   A final Linear layer with a Sigmoid activation outputting $(r, g, b)$ colors for each pixel.

---

## 2. Project Layout

```
├── Cargo.toml          # Cargo dependencies and WASM/Native targets
├── README.md           # Project documentation and math
├── index.html          # Web frontend layout
├── style.css           # Modern dark-mode UI stylesheet
├── index.js            # Frontend animation and canvas manager
└── src
    ├── main.rs         # Native CLI training & image exporter
    ├── lib.rs          # WASM entrypoints and bindings
    ├── utils.rs        # Image/Tensor conversions & synthetic targets
    └── model
        ├── mod.rs      # ImageFitter trait declaration
        ├── gaussian.rs # 2D Gaussian Splatting implementation
        └── nerf.rs     # Coordinate MLP implementation
```
## 2.1 Model Implementation Details

### Gaussian Splatting (`src/model/gaussian.rs`)
*   **Struct:** `GaussianModel<B: Backend>`
*   **Initialization:** `GaussianModel::new(num_gaussians, device)` initializes physical parameters (means, scales, rotations, colors, opacities) log-uniformly or uniformly on the target device.
*   **Covariance Math:** `compute_inverse_covariance()` calculates the inverse of the 2D covariance matrices $\Sigma^{-1}$ using tensor operations.
*   **Rendering:** `render_with_coords(coords)` computes unnormalized Gaussian density values broadcasted over a coordinates grid of shape `[H, W, 2]`, multiplies by opacities and colors, and sums them.
*   **Trait Integration:** Implements `ImageFitter<B>` supporting `render(width, height)` and `forward_loss(target_image)` (MSE loss).
*   **Unit Tests:** Local unit tests run on the `Flex` CPU backend to verify:
    1. Inverse covariance output tensor shape dimensions (`[N, 1]`).
    2. Rendering coordinates output tensor shape dimensions (`[H, W, 3]`).
    3. Trait rendering and MSE loss convergence/scalar shapes (`[1]`).

### Coordinate MLP (`src/model/nerf.rs`)
*   **Struct:** `NerfModel<B: Backend>`
*   **Positional Encoding:** `PositionalEncoding::forward(coords)` maps low-dimensional $(x,y)$ coordinates to higher-frequency features using Fourier sinusoidal mapping.
*   **Initialization:** `NerfModel::new(num_frequencies, hidden_dim, device)` creates a multi-layer linear projection network with ReLU activations and a final sigmoid output mapping to RGB.
*   **Trait Integration:** Implements `ImageFitter<B>` supporting `render(width, height)` and `forward_loss(target_image)` (MSE loss).
*   **Unit Tests:** Local unit tests verify:
    1. Fourier positional mapping output dimensions (`[H, W, 4L]`).
    2. Coordinate MLP feedforward dims (`[H, W, 3]`).
    3. Trait rendering and MSE loss convergence/scalar shapes (`[1]`).

---

## 3. Getting Started

### Prerequisites
Ensure you have the latest Rust toolchain installed:
```bash
rustup update
```

### Native CLI (Local Debugging)
To run local verification and train the models on a programmatically generated target circle:
```bash
cargo run --release
```

### WebAssembly compilation (WebUI)
To compile the WebAssembly library and launch the WebGPU frontend:
1. Install `wasm-pack`:
   ```bash
   cargo install wasm-pack
   ```
2. Build the project for the web target:
   ```bash
   wasm-pack build --target web
   ```
3. Host the folder using a local server (e.g. using Python or Node.js) and open the page in a WebGPU-enabled browser:
   ```bash
   python -m http.server 8080
   ```
