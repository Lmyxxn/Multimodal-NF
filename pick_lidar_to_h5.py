import os
import re
from pathlib import Path
import numpy as np
import h5py
import open3d as o3d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# ⚙️ Path Configuration (Please double-check)
# ==========================================
LIDAR_ROOT = Path("./Dataset/OSM_Train_AllModes")
POS_ROOT = Path("./Dataset/Synthetic_Channel_SmallCodebook_AllModes/TrainSet_FixedLoS") 
OUT_DIR = Path("./Dataset/Synthetic_Lidar_10000_AllModes/TrainSet")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters
NUM_POINTS = 10000        # Fixed number of points
DTYPE_POINTS = np.float32 # Recommend float32 to ensure precision
BATCH_SIZE = 500          # Number of frames to accumulate before writing to H5
NUM_WORKERS = min(16, os.cpu_count()) # Limit the maximum number of processes to prevent freezing

WEATHER_PROB = 0.05       # 🔥 5% of data simulates atmospheric attenuation (rain/fog)

# ==========================================
# 🛠️ Worker Function (Must be at the top level)
# ==========================================
def process_single_ply(args):
    """
    Worker process: Read -> Sample -> [Optional] Superimpose distance-aware rain/fog noise -> Return Numpy
    """
    ply_path, target_pts, apply_weather_noise = args
    
    # Default empty point cloud
    empty = np.zeros((target_pts, 3), dtype=DTYPE_POINTS)
    
    if ply_path is None: return empty

    try:
        # Suppress verbose Open3D output
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=DTYPE_POINTS)
        
        n = pts.shape[0]
        if n == 0: return empty
        
        # 1. Sampling logic (random sampling or padding)
        if n >= target_pts:
            idx = np.random.choice(n, target_pts, replace=False)
            pts = pts[idx]
        else:
            # Cyclic padding (better than zero-padding, as zero-padding causes accumulation at the origin)
            num_pad = target_pts - n
            pad_idx = np.random.choice(n, num_pad, replace=True)
            pad = pts[pad_idx]
            pts = np.concatenate([pts, pad], axis=0)

        # 2. 🔥 Simulate atmospheric attenuation (rain/fog weather)
        if apply_weather_noise:
            # Calculate distance from point to base station (origin) d = sqrt(x^2 + y^2 + z^2)
            d = np.linalg.norm(pts, axis=1, keepdims=True) 
            
            # Calculate noise standard deviation that grows with the square of the distance
            noise_std = (d ** 2) * 0.0001
            
            # Generate and superimpose Gaussian noise
            noise = np.random.normal(0, 1.0, size=pts.shape).astype(DTYPE_POINTS) * noise_std
            pts = pts + noise
            
        return pts
        
    except Exception:
        return empty

# ==========================================
# 🛠️ Index Building
# ==========================================
def build_lidar_index(city_path):
    print(f"   🔍 Scanning PLY files in {city_path.name}...")
    index_map = {}
    
    # Use rglob to recursively find all ply files
    all_plys = list(city_path.rglob("bs_lidar_*.ply"))
    
    if len(all_plys) == 0:
        print(f"   ⚠️ No PLY files found in {city_path}")
        return index_map

    for p in all_plys:
        # Parse filename: bs_lidar_005.ply
        m_frame = re.search(r"bs_lidar_(\d+)\.ply", p.name)
        # Parse parent folder name: Traj_001
        m_traj = re.search(r"Traj_0*(\d+)", str(p))
        
        if not m_frame or not m_traj: continue
        
        index_map[(int(m_traj.group(1)), int(m_frame.group(1)))] = p
        
    print(f"   ✅ Found {len(index_map)} PLY files.")
    return index_map

# ==========================================
# 🚀 Main Logic
# ==========================================
def pack_aligned_lidar(city_name):
    print(f"\n🚀 Processing {city_name}...")
    
    # 1. Check position data H5
    pos_h5_path = POS_ROOT / f"{city_name}_dataset.h5"
    if not pos_h5_path.exists():
        print(f"❌ Pos H5 Missing: {pos_h5_path}")
        return

    # 2. Check original Lidar folder
    city_path = LIDAR_ROOT / city_name
    if not city_path.exists():
        print(f"❌ Lidar Dir Missing: {city_path}")
        return
    
    # 3. Build index
    lidar_index_map = build_lidar_index(city_path)
    if not lidar_index_map:
        return # Skip if no files

    # 4. Read task list (Alignment)
    try:
        with h5py.File(pos_h5_path, 'r') as f_pos:
            metadata = f_pos['Metadata'][:] 
            N = metadata.shape[0]
            traj_ids = metadata[:, 1]
    except Exception as e:
        print(f"❌ Error reading H5: {e}")
        return
    
    frame_ids = np.zeros(N, dtype=np.int32)
    weather_mask = np.zeros(N, dtype=bool) 
    
    current_traj = -1
    counter = 0
    task_list = [] 

    print(f"📦 Aligning frames ({N} total)...")
    for i, tid in enumerate(traj_ids):
        if tid != current_traj:
            current_traj, counter = tid, 0
        else:
            counter += 1
        frame_ids[i] = counter
        
        # Random weather
        apply_noise = np.random.rand() < WEATHER_PROB
        weather_mask[i] = apply_noise
        
        # Find the corresponding ply path
        p = lidar_index_map.get((int(tid), counter))
        task_list.append((p, NUM_POINTS, apply_noise))

    # 5. Multiprocessing and writing
    out_path = OUT_DIR / f"{city_name}_lidar.h5"
    print(f"🔨 Multiprocessing to {out_path}...")
    
    with h5py.File(out_path, "w") as f_out:
        d_pts = f_out.create_dataset(
            "points", shape=(N, NUM_POINTS, 3), 
            dtype=DTYPE_POINTS, chunks=(min(128, N), NUM_POINTS, 3),
            compression="gzip", compression_opts=4
        )
        f_out.create_dataset("traj_id", data=traj_ids, dtype=np.int16)
        f_out.create_dataset("frame_id", data=frame_ids, dtype=np.int16)
        f_out.create_dataset("is_weather_affected", data=weather_mask, dtype=bool) 
        f_out.attrs["aligned_source"] = str(pos_h5_path.name)
        f_out.attrs["weather_sim_prob"] = WEATHER_PROB

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # chunksize=100 reduces inter-process communication overhead
            results_iter = executor.map(process_single_ply, task_list, chunksize=100)
            
            batch_buffer = []
            start_idx = 0
            
            for pts_arr in tqdm(results_iter, total=N, unit="frame"):
                batch_buffer.append(pts_arr)
                
                # Write once a batch is accumulated
                if len(batch_buffer) >= BATCH_SIZE:
                    end_idx = start_idx + len(batch_buffer)
                    d_pts[start_idx : end_idx] = np.array(batch_buffer)
                    start_idx = end_idx
                    batch_buffer = []
            
            # Write the remaining
            if len(batch_buffer) > 0:
                d_pts[start_idx : start_idx + len(batch_buffer)] = np.array(batch_buffer)

    print(f"✅ {city_name} Lidar Done.\n")

def main():
    print(f"CPU Cores: {NUM_WORKERS}")
    
    # 🔥 Automatically scan which cities are under POS_ROOT
    if not POS_ROOT.exists():
        print(f"❌ Critical Error: POS_ROOT does not exist: {POS_ROOT}")
        exit()
        
    existing_h5s = sorted(list(POS_ROOT.glob("City_*_dataset.h5")))
    
    if not existing_h5s:
        print("❌ No dataset.h5 files found in POS_ROOT.")
    else:
        print(f"📋 Found {len(existing_h5s)} cities to process.")
        
        for h5_path in existing_h5s:
            city_name = h5_path.name.replace("_dataset.h5", "")
            pack_aligned_lidar(city_name)

if __name__ == "__main__":
    main()