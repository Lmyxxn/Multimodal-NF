# Multimodal Low-Altitude Near-Field Dataset Generator

This repository provides a complete pipeline for generating, integrating, and analyzing a large-scale multimodal dataset for low-altitude near-field wireless communications. 

## 🛠️ Environment Requirements

The project is built using **Python 3.10.18** and is optimized for CUDA 12.1. Ensure your system has the following dependencies installed:

**Core Simulation & Ray Tracing:**
* NVIDIA Sionna (sionna-rt==1.1.0)
* Mitsuba 3

**Deep Learning & GPU:**
* tensorflow==2.13.1
* torch==2.4.1+cu121
* torchaudio==2.4.1+cu121
* torchvision==0.19.1+cu121
* nvidia-cublas-cu12==12.1.3.1

**Data Processing & Vision:**
* open3d==0.19.0
* opencv-python==4.12.0.88
* osmnx
* geopandas
* *(Note: PyVista and h5py are also required for 3D visualization and data packing).*

## 🚀 Workflow

**1. Scene Generation**
* **Run:** `OSM_to_SionnaScene.ipynb`
* **Description:** Generate and customize the 3D communication scene using OpenStreetMap and SUMO data. Define environment boundaries and layouts here.

**2. Channel & Multimodal Data Synthesis**
* **Run:** `python channel_generation.py`
* **Description:** Synthesize near-field channels and generate synchronized multimodal data. 
* **Customization:** Modify this script to configure antenna arrays and operating frequencies. This step utilizes the provided beam codebooks (`upa64x64_NF_codebook.pkl` or `upa64x64_NF_codebook_small.pkl`).

**3. Data Integration**
* **Run:** 
  python pick_image_to_h5.py
  python pick_lidar_to_h5.py

```

* **Description:** Process and pack the generated image and LiDAR point cloud data into efficient `.h5` files for easier handling and model training.

**4. Statistical Analysis**

* **Run:** `python stastics_analysis.py`
* **Description:** View dataset statistics and verify data distribution. This outputs `dataset_comprehensive_stats.csv` and `dataset_health_dashboard.png` for integrity checking.

## 📂 Codebooks

Two pre-computed codebooks are included for near-field beamforming:

* `upa64x64_NF_codebook.pkl`: Dense codebook (90*45*16).
* `upa64x64_NF_codebook_small.pkl`: Compact codebook, lightweight version for faster testing and lower memory footprint (20*20*10).
