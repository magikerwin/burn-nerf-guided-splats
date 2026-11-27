# 🪐 3D NeRF-Guided Gaussian Splatting — Burn (Rust)

> An interactive, browser-based 3D differentiable rendering engine combining **Volumetric Ray-Marched Neural Radiance Fields (NeRF)** and **3D Gaussian Splatting (3DGS)** built with the [Burn](https://burn.dev/) deep learning framework in Rust. Watch a coarse 3D NeRF discover volumetric density $\sigma(x, y, z)$ to guide explicit 3D Gaussian Splatting right in your browser!
>
> 🚀 **[Try the Live WebAssembly Demo!](https://magikerwin.github.io/burn-nerf-guided-splats/)**

![WebAssembly Demo](assets/web_demo.webp)

---

## 📑 Table of Contents

- [Features](#-features)
- [Model Architecture](#️-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Run Benchmarks & Progressive Examples](#run-benchmarks--progressive-examples)
  - [Run Tests](#run-tests)
  - [Client-Side WebAssembly App (WASM)](#client-side-webassembly-app-wasm)
- [3D Saturn Scene & Hybrid Tuning Details](#-3d-saturn-scene--hybrid-tuning-details)
- [References](#-references)
- [License](#-license)

---

## ✨ Features

- **True 3D Volumetric Ray-Marched NeRF (`VolumetricNerfModel`)** — Evaluates Fourier-encoded 3D world coordinates $(x, y, z) \in [0, 1]^3 \to (\text{RGB}, \sigma)$ along camera rays with Mildenhall volume rendering ($T_k = \exp(-\Delta t \sum_{j < k} \sigma_j)$) accelerated via a fused strict-upper-triangular `matmul`.
- **Depth-Sorted 3D Gaussian Splatting (`GaussianModel`)** — Projects explicit 3D Gaussians (3D Means, Log-Scales, Rotation, RGB Logits, Opacity Logit) under arbitrary $360^\circ$ `(yaw, pitch)` turntable cameras with exact front-to-back $\alpha$-compositing (`argsort` + strict-lower-triangular transmittance `matmul`).
- **Farthest-Point 3D Voxel Seeding** — Samples active 3D NeRF density voxels ($\alpha \ge \tau$) using density-weighted 3D farthest-point selection ($\min d^2 \cdot (0.4 + \alpha)$) so thin 3D rings, planet cores, and distant moons all receive balanced 3D Gaussian coverage.
- **Kerbl-Style Decoupled Parameter-Group Learning Rates** — Uses Burn's `GradientsParams::from_params` to step 3D positions (`lr_means = 0.10 × base_lr`) and scales (`0.30 × base_lr`) more gently than color/opacity logits (`1.0 × base_lr`), preventing Adam from kicking newly seeded splats off 3D surfaces.
- **Instant-NGP Foreground/Sky Ray Balancing** — Balances foreground object rays against empty-sky carving rays (`1 + 6 · target_rgb`) with an $8\times$ unit-cube optical extinction scale so sparse 3D structures solidify within ~35 warmup steps.
- **Dual WebGPU + CPU WASM Execution** — Runs hardware-accelerated client-side training via `Autodiff<Wgpu>` when WebGPU is available, and automatically falls back to `Autodiff<Flex>` CPU WebAssembly on browsers without WebGPU.
- **Interactive Educational Tooltips & Developer Console** — Inline `?` parameter tooltips, live 3D Gaussian ellipse overlays, configurable camera views (`K = 2, 4, 8`) and splat budgets (`N = 26, 48`), and real-time Burn telemetry.
- **Fully in Rust** — Differentiable ray marcher, depth-sorted 3D splat rasterizer, hybrid voxel bridge, CLI benchmarks, and WebAssembly bindings in a single crate.

---

## 🏗️ Model Architecture

The hybrid pipeline couples an implicit volumetric coordinate MLP (Stage 1) with an explicit differentiable 3D Gaussian rasterizer (Stage 3) through a 3D density-lattice seeding bridge (Stage 2):

```
Stage 1 — Coarse 3D Volumetric NeRF Discovery (35–45 Warmup Steps)
  Camera Rays (u, v, yaw, pitch) × S=14 Depth Samples ──► World Points (x, y, z) [R × 14 × 3]
    → Fourier Positional Encoding (L=4 octaves, sin/cos)                        [R × 14 × 24]
    → Linear(24 → 32) → ReLU → Linear(32 → 32) → ReLU → Linear(32 → 4)          [R × 14 × 4]
    → Sigmoid(RGB), Softplus(σ - 1.3) × 8.0Δt Extinction
    → Fused Upper-Triangular Log-Transmittance Matmul ──► Rendered Coarse Views [H_c × W_c × 3]

Stage 2 — 3D Density Lattice Extraction & Farthest-Point Seeding
  15×15×15 Lattice Query in [0.12, 0.88]³ ──► 3,375 Voxels (x, y, z, r, g, b, α_voxel)
    → Active Voxel Filter: α_voxel ≥ α_min + 0.35(α_max - α_min)
    → 3D Farthest-Point Selection: argmax min_dist² · (0.4 + α_voxel) ──► N Seeded 3D Gaussians

Stage 3 — Explicit Depth-Sorted 3D Gaussian Splatting Refinement
  Trainable Params: means [N, 3], scales [N, 2], rotations [N, 1], colors [N, 3], opacities [N, 1]
    → 3D Turntable Projection R_x(pitch)·R_y(yaw) ──► 2D Centers [N, 2] & Depths d_cam [N]
    → Front-to-Back Depth Argsort + 2D Anisotropic Covariance Σ = R S Sᵀ Rᵀ
    → Fused Strict-Lower-Triangular Transmittance T_i = ∏_{j<i}(1 - α_j G_j)
    → Kerbl Decoupled Adam Step (lr_means = 0.1×, lr_scales = 0.3×, lr_appearance = 1.0×)
```

### Mathematical Formulation

1. **Volumetric Ray Marching (`VolumetricNerfModel`):**
   $$\hat{C}(\mathbf{r}) = \sum_{k=1}^{S} T_k \left(1 - \exp(-\sigma_k \delta_k)\right) \mathbf{c}_k, \quad T_k = \exp\!\left(-\sum_{j < k} \sigma_j \delta_j\right)$$

2. **Depth-Sorted 3D Gaussian $\alpha$-Compositing (`GaussianModel`):**
   $$\hat{C}(\mathbf{p}) = \sum_{i=1}^{N} \mathbf{c}_{\pi(i)} \, \alpha_{\pi(i)}(\mathbf{p}) \prod_{j < i} \left(1 - \alpha_{\pi(j)}(\mathbf{p})\right), \quad \alpha_i(\mathbf{p}) = o_i \exp\!\left(-\tfrac{1}{2}(\mathbf{p} - \boldsymbol{\mu}'_i)^\top \boldsymbol{\Sigma}_i^{-1} (\mathbf{p} - \boldsymbol{\mu}'_i)\right)$$

---

## 📁 Project Structure

```
burn-nerf-guided-splats/
├── src/
│   ├── lib.rs              # Crate root exporting model, training, hybrid, utils, & wasm modules
│   ├── main.rs             # Native CLI image-fitting runner & PNG exporter
│   ├── model/
│   │   ├── mod.rs          # ImageFitter & MultiViewFitter trait definitions
│   │   ├── gaussian.rs     # Depth-sorted 3D Gaussian Splatting & turntable projection
│   │   └── nerf.rs         # Fourier Positional Encoding, VolumetricNerfModel, & NerfModel
│   ├── hybrid.rs           # 3D active-voxel farthest-point seeding & multi-view importance bridge
│   ├── training.rs         # Kerbl-scheduled per-parameter-group Burn autodiff optimizers
│   ├── utils.rs            # Image <-> Tensor conversions & synthetic target generators
│   └── wasm.rs             # WebGPU + Flex CPU WASM session bindings & async buffer getters
├── examples/
│   ├── 01_single_splat.rs      # Minimal ~40-line Burn autodiff Gaussian Splatting intro
│   ├── 02_spectral_bias.rs     # Spectral Bias Lab: L=0 (raw coords) vs L=8 (Fourier PE)
│   └── 03_hybrid_benchmark.rs  # Fair 3D Saturn benchmark: Blind 3DGS vs Volumetric-Guided 3DGS
├── web/                    # Static WebAssembly frontend (served by GitHub Pages)
│   ├── index.html          # 4-stage 3D turntable race UI, tooltips, & Developer Console
│   ├── style.css           # Responsive dark Burn theme (desktop 4-up / mobile 2×2 scroll-safe)
│   ├── index.js            # GpuTaskQueue mutex, live 3D ellipse overlay, & race loop
│   └── pkg/                # Compiled WASM output (gitignored, built automatically in CI)
└── assets/
    └── web_demo.webp       # Live WebAssembly showcase animation
```

---

## 🚀 Getting Started

### Run Benchmarks & Progressive Examples

Step through the three self-contained educational examples (uses Burn's fast `Autodiff<Flex>` CPU backend out of the box—no GPU setup required):

```sh
# 1. Minimal Burn Autodiff Gaussian Splatting intro
cargo run --release --example 01_single_splat

# 2. Spectral Bias Lab: L=0 (raw coords) vs L=8 (Fourier Positional Encoding)
cargo run --release --example 02_spectral_bias

# 3. True 3D Saturn Benchmark: Blind 3DGS vs Volumetric Ray-Marched NeRF-Guided 3DGS
cargo run --release --example 03_hybrid_benchmark
```

To run the full native multi-view exporter (`target.png`, `output_gaussian.png`, `output_nerf.png`):

```sh
cargo run --release
```

#### 📊 Results

Evaluated on the **11-part 3D Saturn + Tilted Ring + 2 Moons** scene inside $[0, 1]^3$ (`N = 26` 3D Gaussians, 4 training views + 4 unseen $360^\circ$ novel test views):

| Pipeline | Wall Time | Train PSNR | Unseen $360^\circ$ Novel-View PSNR |
| :--- | :---: | :---: | :---: |
| **Blind 3DGS (Uniform Adam LR)** | `6.39 s` | `21.24 dB` | `20.26 dB` |
| **Blind 3DGS (Kerbl Decoupled LR)** | `6.56 s` | `21.53 dB` | `19.54 dB` |
| **3D Volumetric-Guided 3DGS (Ours)** | **`6.11 s`** | **`23.55 dB`** | **`23.45 dB` (`+3.18 dB` faster!)** |

<details>
<summary>📝 Example Terminal Output (03_hybrid_benchmark & 02_spectral_bias)</summary>

```text
$ cargo run --release --example 03_hybrid_benchmark
=== True 3D Saturn Benchmark (13-Part 3D Ringed Planet, 4 Train + 4 Novel 360° Views, N=26) ===
  1. Blind 3DGS (Uniform LR)   [ 6392 ms]: Train PSNR = 21.24 dB | Novel 360° PSNR = 20.26 dB
  2. Blind 3DGS (Kerbl LR)     [ 6556 ms]: Train PSNR = 21.53 dB | Novel 360° PSNR = 19.54 dB
  3. 3D Volumetric-Guided 3DGS [ 6114 ms]: Train PSNR = 23.55 dB | Novel 360° PSNR = 23.45 dB (+3.18 dB vs best Blind!)

$ cargo run --release --example 02_spectral_bias
=== Example 02: Neural Spectral Bias (120 Adam Steps on High-Frequency Checkerboard) ===
  L = 0 (Raw (x,y) only)        -> MSE: 0.02158 | PSNR: 16.66 dB (Blurry low-freq wash!)
  L = 8 (Fourier sin/cos 2^k π) -> MSE: 0.00069 | PSNR: 31.64 dB (Crisp checkerboard!)
  Fourier Positional Encoding Gain: +14.98 dB
```

</details>

---

### Run Tests

Run all 15 unit tests verifying covariance projection, front-to-back occlusion autodiff, Fourier positional encoding, volumetric 3D ray marching, stereo depth triangulation, and 3D voxel seeding:

```sh
cargo test --release
```

---

### Client-Side WebAssembly App (WASM)

Both `VolumetricNerfModel` and `GaussianModel` compile directly to WebAssembly for 100% client-side in-browser training and real-time $360^\circ$ rendering.

#### 1. Build & preview the WASM bundle locally

1. **Install `wasm-pack`**:
   ```sh
   cargo install wasm-pack
   ```

2. **Build the WebAssembly package**:
   ```sh
   wasm-pack build --target web --out-dir web/pkg
   ```

3. **Serve `web/` over HTTP**:
   ```sh
   python3 -m http.server 8000 --directory web
   ```
   Navigate to **[http://localhost:8000](http://localhost:8000)**.

#### 2. Automatic Deployments (GitHub Pages)

The CI workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) automatically compiles the Rust crate to WebAssembly via `wasm-pack` and deploys `./web` to **[https://magikerwin.github.io/burn-nerf-guided-splats/](https://magikerwin.github.io/burn-nerf-guided-splats/)** on every push to `main`.

---

## 🪐 3D Saturn Scene & Hybrid Tuning Details

To make the difference between **Blind** and **NeRF-Guided** 3D Gaussian Splatting crystal clear even at interactive browser resolutions (`44 × 44`), the default benchmark reconstructs an **11-body 3D Saturn System** inside $[0, 1]^3$:

| Component | 3D Geometry in $[0, 1]^3$ | Purpose in Benchmark |
| :--- | :--- | :--- |
| **Golden Planet Core** | Center `(0.50, 0.50, 0.50)`, $r = 0.095$ | Tests front/back depth occlusion as the tilted ring passes in front & behind |
| **8 Tilted 3D Ring Nodes** | Orbit radius $R = 0.27$, 3D tilt $\Delta y = \pm 0.12$, $r = 0.056$ | Thin 3D arc occupying `< 1%` volume where random 3D splats miss in empty air |
| **2 Outer 3D Moons** | `(0.22, 0.23, 0.70)` & `(0.78, 0.77, 0.30)`, $r = 0.062$ | Tests spatial coverage of farthest-point 3D voxel seeding vs. peak clumping |

### Key Tuning Insights

1. **Why Uniform Adam LR Hurts Guided 3DGS**: Because Adam normalizes first-step updates to $\pm \text{lr}$, applying a large appearance LR (`0.05+`) to `means` violently kicks seeded Gaussians off the 3D ring on Step 1. Scaling `lr_means = 0.10 × lr_appearance` (mirroring Kerbl et al. 2023) preserves the 3D NeRF initialization while colors and opacities converge rapidly.
2. **Why Empty-Sky Rays Must Be Balanced**: Because `98.6%` of $[0, 1]^3$ is empty air, unweighted pixel MSE lets empty background rays drown out thin 3D ring density during short warmups. Foreground ray weighting (`1 + 6 · target_rgb`) + $8\times$ unit-cube extinction carves crisp 3D density peaks in 35 steps.
3. **Why Multi-View Camera Count `K` Matters**: Switching the live demo from `4 Views` to `2 Views` demonstrates **epipolar ambiguity**—two cameras cannot uniquely triangulate an 11-body 3D ring without ghost intersections, whereas `4` or `8` views lock onto true 3D $(x, y, z)$ coordinates with `< 0.1 dB` drop on unseen $360^\circ$ angles.

---

## 📚 References

- [Burn — Deep Learning Framework for Rust](https://burn.dev/)
- [The Burn Book](https://burn.dev/book/)
- [Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (ECCV 2020)](https://arxiv.org/abs/2003.08934)
- [Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Müller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (SIGGRAPH 2022)](https://nvlabs.github.io/instant-ngp/)
- [Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020)](https://arxiv.org/abs/2006.10739)
- [wasm-pack — Rust WebAssembly Packager](https://rustwasm.github.io/wasm-pack/)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
