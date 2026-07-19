# BurnSplat (Splat-vs-NeRF)

An educational 2D rendering engine implemented in Rust using the **Burn** deep learning framework. This project demonstrates and compares two competing machine learning paradigms side-by-side:

1. **Explicit Representation (2D Gaussian Splatting):** Optimizing the physical parameters (Position, Scale, Rotation, Color, Opacity) of a set of 2D Gaussians.
2. **Implicit Representation (2D NeRF / Coordinate MLP):** Training a Multi-Layer Perceptron (MLP) with Positional Encoding to map $(x, y)$ coordinate grids to $(r, g, b)$ colors.
3. **Hybrid Representation (NeRF-Assisted GS):** Seeding the initial locations of 2D Gaussians based on the spatial gradients/edges learned by a partially trained Coordinate MLP.

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
