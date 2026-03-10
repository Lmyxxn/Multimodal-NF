import os
import re
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

# ================= Path Configuration =================
# 1. Ensure these paths match the actual folder names on your hard drive
IMG_ROOT = Path("./Dataset/OSM_Train_AllModes")
POS_ROOT = Path("./Dataset/Synthetic_Channel_SmallCodebook_AllModes/TrainSet_FixedLoS") # Note: Check if this is TrainSet or TrainSet_FixedLoS
OUT_DIR = Path("./Dataset/Synthetic_Image_AllModes/TrainSet")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Image Parameters
IMG_SIZE = (224, 224)
NIGHT_PROB = 0.05  # 5% probability to simulate night time

def read_and_resize_png(png_path, size_wh):
    try:
        img = Image.open(png_path).convert("RGB")
        img = img.resize(size_wh)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ Read Error {png_path}: {e}")
        return np.zeros((size_wh[1], size_wh[0], 3), dtype=np.uint8)

def pack_aligned_city_fast(city_name):
    print(f"\n🔍 Checking {city_name}...")
    
    # 1. Check if position data exists
    pos_h5_path = POS_ROOT / f"{city_name}_dataset.h5"
    if not pos_h5_path.exists():
        print(f"❌ Missing Position H5: {pos_h5_path}")
        print(f"   (Please check if POS_ROOT is correct)")
        return

    # 2. Check if image folder exists
    city_dir = IMG_ROOT / city_name
    if not city_dir.exists():
        print(f"❌ Missing Image Dir: {city_dir}")
        return

    # 3. Read position metadata
    try:
        with h5py.File(pos_h5_path, 'r') as f_pos:
            metadata = f_pos['Metadata'][:]
            traj_ids = metadata[:, 1]
            
            # Calculate frame_id (since metadata usually doesn't have an explicit frame_id)
            frame_ids = np.zeros(len(traj_ids), dtype=np.int32)
            curr, cnt = -1, 0
            for i, t in enumerate(traj_ids):
                if t != curr: curr, cnt = t, 0
                else: cnt += 1
                frame_ids[i] = cnt
            
            # Create mapping table: (Traj_ID, Frame_ID) -> Global_Index
            needed_map = {(t, f): idx for idx, (t, f) in enumerate(zip(traj_ids, frame_ids))}
            N = len(traj_ids)
    except Exception as e:
        print(f"❌ Error reading H5 {pos_h5_path}: {e}")
        return

    # 4. Scan all images
    # Note: rglob searches recursively, ensure the filename format matches
    all_pngs = sorted(list(city_dir.rglob("bs_rgb_*.png")))
    
    if len(all_pngs) == 0:
        print(f"⚠️ No PNGs found in {city_dir}. Check file naming (bs_rgb_*.png).")
        return

    # 5. Prepare Buffer
    images_buffer = np.zeros((N, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    is_night_mask = np.zeros(N, dtype=bool) 
    filled_cnt = 0
    
    # 6. Start matching
    print(f"🚀 Processing {city_name}: {len(all_pngs)} images found. Target Size: {N}")
    
    for p in tqdm(all_pngs, desc=f"Packing {city_name}", unit="img"):
        try:
            # Parse filename: .../Traj_001/.../bs_rgb_005.png
            # The regex here should match your folder structure
            path_str = str(p).replace('\\', '/') # Windows compatibility
            
            m_t = re.search(r"Traj_0*(\d+)", path_str)
            m_f = re.search(r"bs_rgb_(\d+)\.png", p.name)
            
            if not m_t or not m_f: continue
            
            tid = int(m_t.group(1))
            fid = int(m_f.group(1))
            
            if (tid, fid) in needed_map:
                target_idx = needed_map[(tid, fid)]
                
                # Night time logic
                if np.random.rand() < NIGHT_PROB:
                    images_buffer[target_idx] = 0 
                    is_night_mask[target_idx] = True
                else:
                    images_buffer[target_idx] = read_and_resize_png(p, IMG_SIZE)
                
                filled_cnt += 1
        except: continue

    # 7. Save results
    out_path = OUT_DIR / f"{city_name}_img.h5"
    print(f"💾 Saving to {out_path} (Matched: {filled_cnt}/{N})...")
    
    with h5py.File(out_path, "w") as f_out:
        f_out.create_dataset("image", data=images_buffer, compression="gzip", compression_opts=4)
        f_out.create_dataset("is_night", data=is_night_mask, dtype=bool)
        f_out.create_dataset("traj_id", data=traj_ids, dtype=np.int16)
        f_out.create_dataset("frame_id", data=frame_ids, dtype=np.int16)
    
    print(f"✅ {city_name} Finished.")

if __name__ == "__main__":
    # 🔥 Automatically scan which cities are under POS_ROOT, instead of manually specifying range(33, 36)
    if not POS_ROOT.exists():
        print(f"❌ Critical Error: POS_ROOT does not exist: {POS_ROOT}")
        exit()
        
    existing_h5s = sorted(list(POS_ROOT.glob("City_*_dataset.h5")))
    
    if not existing_h5s:
        print("❌ No dataset.h5 files found in POS_ROOT.")
    else:
        print(f"📋 Found {len(existing_h5s)} cities to process.")
        
        for h5_path in existing_h5s:
            # Extract city name from filename (City_001_dataset.h5 -> City_001)
            city_name = h5_path.name.replace("_dataset.h5", "")
            pack_aligned_city_fast(city_name)