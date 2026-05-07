# Multimodal Low-Altitude Near-Field Dataset Generator

This repository provides a complete pipeline for generating, integrating, and analyzing a large-scale multimodal dataset for low-altitude near-field wireless communications. The generated dataset, **Multimodal-NF**, contains synchronized near-field wireless channels, RGB images, LiDAR point clouds, and trajectory-level information for low-altitude UAV communication scenarios.

## 🛠️ Environment Requirements

The project is built using **Python 3.10.18** and is optimized for CUDA 12.1. Ensure your system has the following dependencies installed:

**Core Simulation & Ray Tracing:**

* NVIDIA Sionna (`sionna-rt==1.1.0`)
* Mitsuba 3

**Deep Learning & GPU:**

* `tensorflow==2.13.1`
* `torch==2.4.1+cu121`
* `torchaudio==2.4.1+cu121`
* `torchvision==0.19.1+cu121`
* `nvidia-cublas-cu12==12.1.3.1`

**Data Processing & Vision:**

* `open3d==0.19.0`
* `opencv-python==4.12.0.88`
* `osmnx`
* `geopandas`
* PyVista and h5py are also required for 3D visualization and data packing.

## 🚀 Workflow

### 1. Scene Generation

**Run:** `OSM_to_SionnaScene.ipynb`

**Description:** Generate and customize the 3D communication scene and UAV trajectory. Environment boundaries, layouts, base-station position, and trajectory settings can be defined in this notebook.

### 2. Channel & Multimodal Data Synthesis

**Run:**

```bash
python channel_generation.py
```

**Description:** Synthesize near-field wireless channels and generate synchronized multimodal data, including images and LiDAR point clouds.

**Customization:** Modify this script to configure antenna arrays, carrier frequencies, UAV trajectories, and ray-tracing parameters.

This step utilizes the provided beam codebooks:

* `upa64x64_NF_codebook.pkl`
* `upa64x64_NF_codebook_small.pkl`

### 3. Data Integration

**Run:**

```bash
python pick_image_to_h5.py
python pick_lidar_to_h5.py
```

**Description:** Process and pack the generated image and LiDAR point cloud data into efficient `.h5` files for easier storage, handling, and model training.

### 4. Statistical Analysis

**Run:**

```bash
python stastics_analysis.py
```

**Description:** Analyze dataset statistics and verify data distribution. This script outputs:

* `dataset_comprehensive_stats.csv`
* `dataset_health_dashboard.png`

These files can be used for dataset integrity checking and statistical visualization.

## 📂 Codebooks

Two pre-computed near-field beamforming codebooks are provided:

* `upa64x64_NF_codebook.pkl`: Dense codebook with size `90 × 45 × 16`.
* `upa64x64_NF_codebook_small.pkl`: Compact codebook with size `20 × 20 × 10`, suitable for faster testing and lower memory consumption.

The codebooks are available at:

https://huggingface.co/datasets/lmyxxn/MultimodalNF

Please download them and move them to the main folder before running the channel generation scripts.

## 📊 Dataset Overview

The generated **Multimodal-NF** dataset is designed for near-field low-altitude sensing and communications with a BS equipped with a **UPA 64 × 64** antenna array. The dataset is split by cities into training, validation, and testing sets to evaluate the generalization capability across different urban environments.

All trajectories contain **T = 20 frames** with a sampling interval of **0.1 s**. The UAV altitude ranges from **5 m to 80 m**. The BS UPA is located at **(0, 0, 65) m**. The dataset is split by cities into **Train/Val/Test = 22/4/4**.

Overall, the dataset contains **215,400 samples**, including **201,075 LoS samples** and **14,325 NLoS samples**. The LoS and NLoS ratios are **93.35%** and **6.65%**, respectively, which reflects typical low-altitude communication environments where the air-to-ground link is often dominated by LoS propagation while still containing non-negligible blockage and multipath cases.

### Dataset Splits and Sample Statistics

| Split | Mode | Cities | Trajectories | Total Samples | LoS Samples | NLoS Samples |
|---|---|---:|---:|---:|---:|---:|
| Train | Easy | 22 | 2,614 | 52,280 | 49,923 | 2,357 |
| Train | Hard | 22 | 5,185 | 103,700 | 94,251 | 9,449 |
| Val | Easy | 4 | 488 | 9,760 | 9,374 | 386 |
| Val | Hard | 4 | 988 | 19,760 | 19,139 | 621 |
| Test | Easy | 4 | 494 | 9,880 | 9,381 | 499 |
| Test | Hard | 4 | 1,001 | 20,020 | 19,007 | 1,013 |
| **Total** | -- | **30** | **10,770** | **215,400** | **201,075** | **14,325** |

### Multipath Statistics

The generated channels are characterized by sparse but highly LoS-dominant low-altitude propagation. The average number of paths is **2.53**, while the median number of paths is **2**, indicating that most samples are dominated by a small number of strong propagation components. The RMS delay spread is generally small, with a mean value of **2.17 ns**, while the maximum excess delay can be much larger due to occasional long-delay reflected paths.

| Metric | Mean | Median | 90th Percentile |
|---|---:|---:|---:|
| Number of paths | 2.53 | 2.00 | 4.00 |
| RMS delay spread (ns) | 2.17 | 1.38 | 4.81 |
| Maximum excess delay (ns) | 112.77 | 95.68 | 244.95 |
| PDP T50 (ns) | 1.03 | -- | -- |
| PDP T90 (ns) | 2.28 | -- | -- |
| K-factor (dB) | 52.01 | 59.72 | 75.30 |

The high K-factor further confirms the LoS-dominant nature of the dataset. Meanwhile, the non-zero NLoS ratio and the large 90th-percentile maximum excess delay show that the dataset still includes challenging multipath and blockage conditions.

### Temporal Consistency Analysis

The dataset also preserves temporal continuity along UAV trajectories. Consecutive samples are highly correlated in the spatial channel domain, while the selected near-field beam indices vary smoothly across adjacent frames in the azimuth, zenith, and range dimensions. This makes the dataset suitable not only for snapshot-based tasks, but also for trajectory-level learning, temporal channel prediction, beam tracking, and multimodal sensing-aided communication.

![Temporal consistency analysis](Figs/temporal_analysis1.png)


## 📦 Dataset

The dataset is available at:

https://huggingface.co/datasets/lmyxxn/MultimodalNF

More details and citations can be found at:

https://lmyxxn.github.io/6GXLMIMODatasets/

## 🌟 Welcome & Citation

We highly encourage researchers and developers to explore, experiment, and build upon this dataset and generation toolchain. Whether you are testing new channel estimation algorithms, exploring multimodal foundation models, or analyzing near-field XL-MIMO characteristics, we hope this repository serves as a valuable resource for your work.

If you find our code, dataset, or codebooks useful in your research or projects, please consider citing our paper:

```bibtex
@article{Li2026MultimodalNF,
  author  = {Li, M. and Lu, Q. and Tian, J. and Hu, H. and Han, Y. and Li, X. and Wen, C.-K. and Jin, S.},
  title   = {{Multimodal-NF: A Wireless Dataset for Near-Field Low-Altitude Sensing and Communications}},
  journal = {arXiv preprint arXiv:2603.28280},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.28280}
}
```
