# -*- coding: utf-8 -*-
import os
import sys

# ================= 0. GPU Forced Configuration =================
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

# ================= 1. Import Dependencies =================
import json
import re
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import pandas as pd
import math
import pickle
import h5py
import tensorflow as tf

try:
    import sionna
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    import mitsuba as mi 
except ImportError:
    print("❌ Sionna not found.")
    sys.exit(1)

# VRAM Adaptive Allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Active GPU Configuration: ID {gpu_num}")
    except RuntimeError as e: print(e)

# ================= 2. Core Parameter Configuration =================
# 🔥 Dataset Root Directory
INPUT_ROOT = Path("Dataset/OSM_Train_AllModes") 
OUTPUT_ROOT = Path("Dataset/Synthetic_Channel_SmallCodebook_AllModes/TrainSet_FixedLoS")
CODEBOOK_SAVE_PATH = "upa64x64_NF_codebook_small.pkl"

# 🔥🔥🔥 City Whitelist Filtering 🔥🔥🔥
# If the list is empty [], run all cities under INPUT_ROOT
# If populated like ["City_001", "City_002"], only run those two
# TARGET_CITIES = ["City_002", "City_003", "City_004", "City_005", "City_006", 
#                  "City_007", "City_008", "City_009", "City_010"]
# TARGET_CITIES = ["City_011", "City_012", "City_013", "City_014",
#                 "City_015", "City_016", "City_017", "City_018",
#                 "City_019", "City_020"]
# TARGET_CITIES = ["City_021", "City_022","City_023", "City_024", "City_025", "City_026",
#                  "City_027", "City_028", "City_029", "City_030"] 
#TARGET_CITIES = ["City_019", "City_020", "City_021", "City_022", "City_023"] 
# TARGET_CITIES = ["City_032", "City_033", "City_034", "City_035"]
TARGET_CITIES = []

CENTER_FC = 7e9             
SUBCARRIER_SPACING = 30e3   
K = 1024                       
MAX_DEPTH = 3               
USE_SYNTHETIC_ARRAY = False 

# Array definition
UPA_VERTICAL = 64
UPA_HORIZONTAL = 64
NUM_ANT = UPA_VERTICAL * UPA_HORIZONTAL
EL_SPACING = 0.5
AZ_SPACING = 0.5
RX_POLAR = 'V'; TX_POLAR = 'V'
h_receiver = 65.0
BS_POS = [0.0, 0.0, h_receiver]
C = 3e8; LAMBDA = C / CENTER_FC

# Trajectory Mode Mapping Table
MODE_MAPPING = {
    # --- Easy Modes (0-5) ---
    'city_cruise': 0, 
    'fast_transit': 1, 
    'orbit': 2, 
    'scan': 3,      
    'hover': 4,
    'inspect': 5, 
    
    # --- Hard Modes (6-9) ---
    'street_patrol': 6, 
    'Zigzag': 7,     
    'wall_hug': 8, 
    'sudden_turn': 9
}

# ================= 3. Smart Codebook Configuration =================
D2R = np.pi / 180.0
THETA_RANGE = np.linspace(-72 * D2R, 72 * D2R, 20)
PHI_RANGE = np.linspace(60 * D2R,  150* D2R, 20)

def gen_dist_layers(min_d, max_d, num_layers):
    c_start, c_end = 1.0 / max_d, 1.0 / min_d
    return np.round(1.0 / np.linspace(c_end, c_start, num_layers), 2).tolist()
DIST_RANGE = gen_dist_layers(min_d=20, max_d=155, num_layers=10)

print(f"📐 Smart Codebook Config: Theta={len(THETA_RANGE)}, Phi={len(PHI_RANGE)}, Dist={len(DIST_RANGE)}")

# D2R = np.pi / 180.0
# THETA_RANGE = np.linspace(-72 * D2R, 72 * D2R, 90)
# PHI_RANGE = np.linspace(60 * D2R,  150* D2R, 45)

# def gen_dist_layers(min_d, max_d, num_layers):
#     c_start, c_end = 1.0 / max_d, 1.0 / min_d
#     return np.round(1.0 / np.linspace(c_end, c_start, num_layers), 2).tolist()
# DIST_RANGE = gen_dist_layers(min_d=20, max_d=155, num_layers=16)

# print(f"📐 Smart Codebook Config: Theta={len(THETA_RANGE)}, Phi={len(PHI_RANGE)}, Dist={len(DIST_RANGE)}")

# ================= 4. Core Utility Functions =================

def load_codebook_tf():
    """ Load Codebook """
    if not os.path.exists(CODEBOOK_SAVE_PATH):
        print(f"❌ Error: Codebook file not found at {CODEBOOK_SAVE_PATH}")
        sys.exit(1)
    with open(CODEBOOK_SAVE_PATH, 'rb') as f:
        data = pickle.load(f)
    W_matrix_tf = tf.convert_to_tensor(data['matrix'], dtype=tf.complex64)
    if W_matrix_tf.shape[0] != NUM_ANT:
        W_matrix_tf = tf.transpose(W_matrix_tf)
    return W_matrix_tf

def search_top_k_beams(h_vec_complex_tf, W_matrix_tf, k=5):
    """ GPU Beam Search """
    h_in = tf.reshape(h_vec_complex_tf, [1, NUM_ANT])
    beam_response = tf.matmul(h_in, tf.math.conj(W_matrix_tf))
    gains = tf.abs(beam_response) ** 2 
    top_k_values, top_k_indices = tf.math.top_k(gains[0], k=k)
    return top_k_indices, top_k_values

def decode_beam_index(indices_tf):
    """ Decode Beam Index """
    indices = indices_tf.numpy()
    num_phi = len(PHI_RANGE)
    num_dist = len(DIST_RANGE)
    results = []
    for idx in indices:
        i_dist = idx % num_dist
        temp = idx // num_dist
        i_phi = temp % num_phi
        i_theta = temp // num_phi
        results.append([i_theta, i_phi, i_dist])
    return np.array(results, dtype=np.uint16)

@contextmanager
def _pushd(path: Path):
    old = os.getcwd(); os.chdir(str(path))
    try: yield
    finally: os.chdir(old)

def _build_frequencies(fc, K, scs):
    return fc + (np.arange(K) - K/ 2.0) * scs

def _rayleigh_distance_diagonal(lam, M_V, M_H):
    """ 🔥 Calculate Rayleigh Distance Based on Diagonal Aperture """
    D_H = (M_H-1)*lam/2
    D_V = (M_V-1)*lam/2
    D = np.sqrt(D_H**2+D_V**2)
    return 2.0 * (D ** 2) / lam

def _calculate_angles(tx_pos, rx_pos):
    delta = tx_pos - rx_pos
    r = np.linalg.norm(delta)
    if r < 1e-9: return np.array([0.0, 0.0], dtype=np.float32)
    theta = np.arccos(np.clip(delta[2] / r, -1.0, 1.0))
    phi = np.arctan2(delta[1], delta[0])
    return np.array([theta, phi], dtype=np.float32)

def _flatten_H_complex(h_complex_tf) -> np.ndarray:
    if hasattr(h_complex_tf, 'numpy'):
        h = h_complex_tf.numpy()
    else:
        h = h_complex_tf
    if h.ndim == 7: h = h[0] 
    Nr = h.shape[0] * h.shape[1]
    Nt = h.shape[2] * h.shape[3]
    T  = h.shape[4]
    K_dim = h.shape[5]
    h4 = h.reshape(Nr, Nt, T, K_dim)
    h3 = h4.reshape(Nr * Nt, T, K_dim)
    h_real = np.real(h3).transpose(1, 0, 2)
    h_imag = np.imag(h3).transpose(1, 0, 2)
    return np.stack([h_real, h_imag], axis=-1) 

def prepare_static_scene(city_dir: Path):
    sample_xml = next(city_dir.rglob("scene.xml"), None)
    if not sample_xml: return None, None
    try:
        tree = ET.parse(sample_xml); root = tree.getroot()
        for shape in root.findall('shape'):
            if shape.get('id') == 'uav': root.remove(shape)
        temp_xml = sample_xml.parent / "_static_scene_temp.xml"
        tree.write(temp_xml); return temp_xml, sample_xml.parent
    except: return None, None

# ================= 5. Main Processing Logic (process_single_city) =================
def process_single_city(city_dir: Path, save_root: Path):
    city_dir = city_dir.resolve()
    save_root = save_root.resolve()

    try:
        city_idx_val = int(city_dir.name.split('_')[-1])
    except:
        city_idx_val = 0

    print(f"\n========================================")
    print(f"🏙️  Processing City (Strict Align): {city_dir.name}")
    print(f"========================================")
    save_root.mkdir(parents=True, exist_ok=True)
    
    xml_path, context_dir = prepare_static_scene(city_dir)
    if not xml_path:
        print(f"⚠️ No scene file found in {city_dir}")
        return

    W_codebook_tf = load_codebook_tf()
    print(f"✅ Codebook Loaded. Shape: {W_codebook_tf.shape}")
    
    R_RAYLEIGH = _rayleigh_distance_diagonal(LAMBDA, UPA_VERTICAL, UPA_HORIZONTAL)
    print(f"📏 Rayleigh Distance (Diagonal): {R_RAYLEIGH:.2f} m")

    solver = PathSolver()
    freqs = _build_frequencies(CENTER_FC, K, SUBCARRIER_SPACING)
    
    rx_array_cfg = dict(num_rows=UPA_VERTICAL, num_cols=UPA_HORIZONTAL, 
                        vertical_spacing=EL_SPACING, horizontal_spacing=AZ_SPACING, 
                        pattern="tr38901", polarization=RX_POLAR)
    tx_array_cfg = dict(num_rows=1, num_cols=1, 
                        vertical_spacing=0.5, horizontal_spacing=0.5, 
                        pattern="tr38901", polarization=TX_POLAR)

    try:
        with _pushd(context_dir):
            scene = load_scene(xml_path.name)
            scene.frequency = CENTER_FC
            scene.synthetic_array = USE_SYNTHETIC_ARRAY
            scene.rx_array = PlanarArray(**rx_array_cfg)
            scene.tx_array = PlanarArray(**tx_array_cfg)
            
            rx_pos_np = np.array(BS_POS, dtype=float)
            rx_pos_mi = mi.Point3f(float(BS_POS[0]), float(BS_POS[1]), float(BS_POS[2]))
            rx = Receiver("rx", position=rx_pos_mi)
            tx_pos_init = mi.Point3f(0.0, 0.0, 100.0)
            tx = Transmitter("tx", position=tx_pos_init)
            scene.add(rx); scene.add(tx)
            
            # Initialize all data containers
            City_H_list = []; City_Pos_list = []; City_Angle_list = [] 
            City_NF_list = []; City_BeamIdx_list = []; City_BeamPower_list = [] 
            City_Meta_list = []; City_LoS_list = []; City_Mode_list = []
            
            traj_dirs = sorted([d for d in city_dir.iterdir() if d.is_dir() and "Traj" in d.name])
            
            for traj_idx, t_dir in enumerate(traj_dirs):
                try: traj_idx_val = int(t_dir.name.split('_')[-1])
                except: traj_idx_val = 0
                
                print(f"   👉 Traj {traj_idx+1}/{len(traj_dirs)}: {t_dir.name}", end="\r")
                
                csv_path = t_dir / "uav_trajectory.csv"
                if not csv_path.exists(): continue
                
                df = pd.read_csv(csv_path)
                positions = df[['x', 'y', 'z']].values
                mode_names = df['mode'].str.strip().values if 'mode' in df.columns else ["unknown"] * len(positions)
                
                # Initialize trajectory temporary lists
                Traj_H = []; Traj_Pos = []; Traj_Angle = []; Traj_NF = []
                Traj_BeamIdx = []; Traj_BeamPower = []; Traj_Meta = []
                Traj_LoS = []; Traj_Mode = []
                is_traj_valid = True

                for frame_idx, pos in enumerate(positions):
                    tx_xyz = np.array(pos, dtype=float)
                    tx.position = mi.Point3f(float(tx_xyz[0]), float(tx_xyz[1]), float(tx_xyz[2]))
                    tx.look_at(rx)
                    
                    # 🔥 High-Fidelity Ray Tracing: Restore diffraction and refraction
                    paths = solver(scene=scene, max_depth=MAX_DEPTH, los=True,
                                   specular_reflection=True, diffuse_reflection=False,refraction=False,
                                   synthetic_array=USE_SYNTHETIC_ARRAY, seed=42)
                    
                    try:
                        a_cir, tau = paths.cir()
                        num_paths = int(a_cir[0].shape[-2])
                    except: num_paths = 0
                    
                    if num_paths == 0: 
                        is_traj_valid = False; break 
                        
                    # 🔥 Statistics for LoS
                    try:
                        # 1. Get Mask (marks which paths are valid) [batch, rx, tx, max_paths] -> Flatten
                        mask = paths.mask.numpy().flatten()
                        
                        # 2. Get Types (interaction types) [..., max_paths, max_depth] -> [total_paths, max_depth]
                        all_types = paths.types.numpy().reshape(-1, MAX_DEPTH)
                        
                        # 3. Filter out valid path Types
                        valid_types = all_types[mask]
                        
                        # 4. Determine: LoS path characteristics are no interactions, meaning Types are all -1
                        if valid_types.size > 0:
                            # Check if every row is all -1
                            is_los_path = np.all(valid_types == -1, axis=1)
                            # As long as there is one True, it is Has_LoS
                            has_los = 1 if np.any(is_los_path) else 0
                        else:
                            has_los = 0
                    except: has_los = 0
                        
                    # 🔥 Statistics for Mode
                    current_mode_name = mode_names[frame_idx]
                    mode_idx = MODE_MAPPING.get(current_mode_name, -1)

                    h_freq = paths.cfr(freqs-CENTER_FC, normalize_delays=False, out_type="numpy")
                    if np.all(np.abs(h_freq) < 1e-9):
                        is_traj_valid = False; break
                    
                    # Channel processing and beam search
                    H_t_structured = _flatten_H_complex(h_freq) 
                    if H_t_structured.shape[0] != 1: H_t_structured = H_t_structured[:1]
                    H_t_final = H_t_structured[0, :, 0, :] 
                    
                    h_complex = H_t_final[:, 0] + 1j * H_t_final[:, 1]
                    h_complex_tf = tf.convert_to_tensor(h_complex, dtype=tf.complex64)
                    
                    top_5_indices, top_5_values = search_top_k_beams(h_complex_tf, W_codebook_tf, k=5)
                    decoded_indices = decode_beam_index(top_5_indices)
                    
                    H_out = H_t_final[:, np.newaxis, :] 
                    dist = float(np.linalg.norm(tx_xyz - rx_pos_np))
                    is_nf = 1 if dist < R_RAYLEIGH else 0
                    angle = _calculate_angles(tx_xyz, rx_pos_np)
                    meta_record = [city_idx_val, traj_idx_val]
                    
                    Traj_H.append(H_out)
                    Traj_Pos.append(tx_xyz)
                    Traj_Angle.append(angle)
                    Traj_NF.append(is_nf)
                    Traj_BeamIdx.append(decoded_indices)
                    Traj_BeamPower.append(top_5_values.numpy()) 
                    Traj_Meta.append(meta_record)
                    Traj_LoS.append(has_los)
                    Traj_Mode.append(mode_idx)

                # Save valid trajectories
                if is_traj_valid and len(Traj_H) == len(positions):
                    City_H_list.append(np.stack(Traj_H, axis=0))     
                    City_Pos_list.append(np.stack(Traj_Pos, axis=0))
                    City_Angle_list.append(np.stack(Traj_Angle, axis=0))
                    City_NF_list.append(np.array(Traj_NF, dtype=np.uint8))
                    City_BeamIdx_list.append(np.stack(Traj_BeamIdx, axis=0))
                    City_BeamPower_list.append(np.stack(Traj_BeamPower, axis=0))
                    City_Meta_list.append(np.array(Traj_Meta, dtype=np.int16))
                    City_LoS_list.append(np.array(Traj_LoS, dtype=np.uint8))
                    City_Mode_list.append(np.array(Traj_Mode, dtype=np.int8))
            
            # HDF5 Saving
            if City_H_list:
                final_H = np.concatenate(City_H_list, axis=0).astype(np.float32)
                out_h5 = save_root / f"{city_dir.name}_dataset.h5"
                print(f"\n   💾 Saving HDF5: {out_h5}")
                print(f"       Total Samples: {final_H.shape[0]}")
                
                with h5py.File(out_h5, 'w') as f:
                    f.create_dataset('H', data=final_H, compression="gzip", compression_opts=4)
                    f.create_dataset('Pos', data=np.concatenate(City_Pos_list, axis=0), compression="gzip")
                    f.create_dataset('Angle', data=np.concatenate(City_Angle_list, axis=0), compression="gzip")
                    f.create_dataset('Is_NF', data=np.concatenate(City_NF_list, axis=0).astype(np.uint8))
                    f.create_dataset('BeamIdx', data=np.concatenate(City_BeamIdx_list, axis=0).astype(np.uint16), compression="gzip")
                    f.create_dataset('BeamPower', data=np.concatenate(City_BeamPower_list, axis=0).astype(np.float32), compression="gzip")
                    f.create_dataset('Metadata', data=np.concatenate(City_Meta_list, axis=0).astype(np.int16))
                    f.create_dataset('Has_LoS', data=np.concatenate(City_LoS_list, axis=0).astype(np.uint8))
                    f.create_dataset('Mode_Idx', data=np.concatenate(City_Mode_list, axis=0).astype(np.int8))
                    f['Mode_Idx'].attrs['mapping'] = json.dumps(MODE_MAPPING)
            else:
                print(f"\n   ⚠️ Warning: No valid data generated for {city_dir.name}")
                
    except Exception as e:
        print(f"❌ Critical Error processing {city_dir.name}: {e}")
        import traceback; traceback.print_exc()
    finally:
        if xml_path and xml_path.exists():
            try: os.remove(xml_path)
            except: pass

# ================= 6. Batch Processing Entry =================
def run_batch_processing():
    if not INPUT_ROOT.exists(): 
        print(f"❌ Input not found: {INPUT_ROOT}")
        return
    
    # 1. Automatically scan city folders
    sub_city_dirs = sorted([d for d in INPUT_ROOT.iterdir() if d.is_dir() and "City" in d.name])
    target_dirs = sub_city_dirs if sub_city_dirs else [INPUT_ROOT]
    
    # 🔥 2. Apply Whitelist Filtering
    if TARGET_CITIES and len(TARGET_CITIES) > 0:
        print(f"📋 Applying Filter: Processing ONLY {TARGET_CITIES}")
        target_dirs = [d for d in target_dirs if d.name in TARGET_CITIES]
        if not target_dirs:
            print("⚠️  Warning: No cities matched your filter list!")
            return

    print(f"🎯 Final Target List ({len(target_dirs)}): {[d.name for d in target_dirs]}")
    
    # 3. Loop Execution
    for c_dir in target_dirs:
        process_single_city(c_dir, OUTPUT_ROOT)
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_batch_processing()