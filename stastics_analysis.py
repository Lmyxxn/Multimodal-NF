# -*- coding: utf-8 -*-
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# ================= ⚙️ Configuration Area =================
# Dataset Directory
DATASET_DIR = Path("multi-modal/dataset_generation/Dataset/Synthetic_Channel_SmallCodebook_AllModes/TrainSet_FixedLoS")

# Trajectory Length (used to calculate trajectory-level blockage)
TRAJ_LEN = 20 

# Default Mode Mapping
DEFAULT_MODE_MAPPING = {
    0: 'City Cruise', 1: 'Fast Transit', 2: 'Orbit', 3: 'Scan', 4: 'Hover',
    5: 'Inspect', 6: 'Street Patrol', 7: 'Zigzag', 8: 'Wall Hug', 9: 'Sudden Turn'
}

def analyze_dataset_comprehensive(data_dir: Path):
    # 1. Check directory
    if not data_dir.exists():
        print(f"❌ Error: Directory not found {data_dir}")
        return

    # 2. Search for files
    print(f"📂 Scanning directory: {data_dir} ...")
    h5_files = sorted(list(data_dir.rglob("*.h5"))) 
    
    if len(h5_files) == 0:
        print(f"❌ Error: No .h5 files found in the directory")
        return
    
    print(f"🔎 Found {len(h5_files)} files, starting comprehensive analysis...\n")

    # --- Global Statistics Counters ---
    global_total_pts = 0
    global_total_trajs = 0
    
    global_los_pts = 0
    global_nlos_pts = 0
    
    global_clean_trajs = 0 # Trajectories without blockage
    global_nlos_trajs = 0  # Trajectories with blockage
    
    global_nf = 0
    global_ff = 0
    global_modes = {k: 0 for k in DEFAULT_MODE_MAPPING.keys()}
    
    # --- Single File Statistics List ---
    file_stats_list = []
    mode_map_loaded = None 

    # --- Iterate through files ---
    for fp in tqdm(h5_files, desc="Analyzing Files"):
        filename = fp.name
        try:
            with h5py.File(fp, 'r') as f:
                # -----------------------
                # 1. Basic Info & Mode Extraction
                # -----------------------
                if 'Pos' not in f: continue
                N = f['Pos'].shape[0]
                num_trajs_in_file = N // TRAJ_LEN

                # Extract Mode
                if 'Mode_Idx' in f:
                    mode_arr = f['Mode_Idx'][:]
                    unique, counts = np.unique(mode_arr, return_counts=True)
                    for m_idx, count in zip(unique, counts):
                        global_modes[m_idx] = global_modes.get(m_idx, 0) + count
                    
                    # Attempt to load map
                    if mode_map_loaded is None and 'mapping' in f['Mode_Idx'].attrs:
                        try:
                            mapping_str = f['Mode_Idx'].attrs['mapping']
                            temp_map = json.loads(mapping_str)
                            mode_map_loaded = {int(v): k for k, v in temp_map.items()}
                        except: pass

                # -----------------------
                # 2. Blockage Statistics (Point & Trajectory)
                # -----------------------
                # Point-level
                if 'Has_LoS' in f:
                    los_arr = f['Has_LoS'][:] # 1=LoS, 0=NLoS
                    n_los = int(np.sum(los_arr))
                    n_nlos = N - n_los
                else:
                    n_los, n_nlos = N, 0 # Default all clear

                # Trajectory-level (Core Fusion Point)
                # Logic: As long as Traj_Is_NLoS exists, count how many "bad trajectories" there are
                if 'Traj_Is_NLoS' in f:
                    traj_labels = f['Traj_Is_NLoS'][:] 
                    # Traj_Is_NLoS is usually flattened at the point level, with the same value every 20 points. Sample by step.
                    traj_flags = traj_labels[::TRAJ_LEN] 
                    n_bad_trajs = int(np.sum(traj_flags)) # 1 = Bad
                    n_good_trajs = num_trajs_in_file - n_bad_trajs
                else:
                    # If there are no pre-computed labels, assume all are good
                    n_bad_trajs = 0
                    n_good_trajs = num_trajs_in_file

                # -----------------------
                # 3. Near-Field Statistics
                # -----------------------
                if 'Is_NF' in f:
                    nf_arr = f['Is_NF'][:]
                    n_nf = int(np.sum(nf_arr))
                    n_ff = N - n_nf
                else:
                    n_nf, n_ff = 0, N

                # --- Update Global Counters ---
                global_total_pts += N
                global_total_trajs += num_trajs_in_file
                
                global_los_pts += n_los
                global_nlos_pts += n_nlos
                
                global_clean_trajs += n_good_trajs
                global_nlos_trajs += n_bad_trajs
                
                global_nf += n_nf
                global_ff += n_ff

                # --- Record single file data (for sorting) ---
                file_stats_list.append({
                    "Filename": filename,
                    "Total_Samples": N,
                    "NLoS_Point_Ratio": (n_nlos / N * 100) if N > 0 else 0,
                    "NLoS_Traj_Ratio": (n_bad_trajs / num_trajs_in_file * 100) if num_trajs_in_file > 0 else 0,
                    "NF_Ratio": (n_nf / N * 100) if N > 0 else 0,
                    "Modes": str(list(np.unique(mode_arr))) if 'Mode_Idx' in f else "Unknown"
                })

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # Use loaded mapping
    final_mode_map = mode_map_loaded if mode_map_loaded else DEFAULT_MODE_MAPPING

    # ==========================================
    # 📊 1. Global Summary Report
    # ==========================================
    print("\n" + "="*60)
    print(f"📊 DATASET HEALTH REPORT (Global Summary)")
    print("="*60)
    print(f"Files Processed      : {len(file_stats_list)}")
    print(f"Total Points         : {global_total_pts:,}")
    print(f"Total Trajectories   : {global_total_trajs:,}")
    print("-" * 60)

    # 1. LoS (Point Level)
    los_pct = global_los_pts / global_total_pts * 100 if global_total_pts > 0 else 0
    nlos_pct = global_nlos_pts / global_total_pts * 100 if global_total_pts > 0 else 0
    print(f"\n📡 [Point-Level] Signal Visibility:")
    print(f"  • Pure LoS Points    : {global_los_pts:,} ({los_pct:.2f}%)")
    print(f"  • NLoS Points (Block): {global_nlos_pts:,} ({nlos_pct:.2f}%)")

    # 2. LoS (Trajectory Level)
    clean_traj_pct = global_clean_trajs / global_total_trajs * 100 if global_total_trajs > 0 else 0
    bad_traj_pct = global_nlos_trajs / global_total_trajs * 100 if global_total_trajs > 0 else 0
    print(f"\n🚀 [Trajectory-Level] Route Reliability:")
    print(f"  • Safe Trajectories  : {global_clean_trajs:,} ({clean_traj_pct:.2f}%)")
    print(f"  • Risky Trajectories : {global_nlos_trajs:,} ({bad_traj_pct:.2f}%) -> Contains at least 1 NLoS point")

    # 3. Near Field
    nf_pct = global_nf / global_total_pts * 100 if global_total_pts > 0 else 0
    print(f"\n📏 Near-Field XL-MIMO Stats:")
    print(f"  • Near-Field Samples : {global_nf:,} ({nf_pct:.2f}%)")
    print(f"  • Far-Field Samples  : {global_ff:,} ({100-nf_pct:.2f}%)")

    # 4. Motion Modes
    print(f"\n🚁 Motion Mode Distribution:")
    mode_data = []
    for k, v in global_modes.items():
        if v > 0:
            mode_name = final_mode_map.get(k, f"Mode {k}")
            pct = v / global_total_pts * 100
            mode_data.append({"Mode": mode_name, "Count": v, "Percent": pct})
    
    df_modes = pd.DataFrame(mode_data).sort_values("Count", ascending=False)
    print(df_modes.to_string(index=False, formatters={'Count': '{:,}'.format, 'Percent': '{:.1f}%'.format}))

    # ==========================================
    # 🔥 2. Hard Sample Mining (Hardest Files)
    # ==========================================
    df_files = pd.DataFrame(file_stats_list)
    if not df_files.empty:
        # Sort by "Trajectory-level NLoS Ratio" from high to low
        df_hard = df_files.sort_values("NLoS_Traj_Ratio", ascending=False).head(15)
        
        print("\n" + "="*60)
        print("🔥 TOP 15 HARDEST SCENES (Highest Trajectory NLoS %)")
        print("="*60)
        print("A high 'NLoS_Traj_Ratio' here indicates that UAVs in this city easily lose connection mid-flight.")
        print(df_hard[['Filename', 'NLoS_Point_Ratio', 'NLoS_Traj_Ratio', 'NF_Ratio']].to_string(index=False, formatters={
            'NLoS_Point_Ratio': '{:.2f}%'.format,
            'NLoS_Traj_Ratio': '{:.2f}%'.format,
            'NF_Ratio': '{:.2f}%'.format
        }))
        
        # Save CSV
        df_files.to_csv("dataset_comprehensive_stats.csv", index=False)
        print(f"\n💾 Comprehensive statistics saved to: dataset_comprehensive_stats.csv")

    # ==========================================
    # 🎨 3. Visualization Plotting
    # ==========================================
    plot_comprehensive_charts(
        los_counts=[global_los_pts, global_nlos_pts],
        traj_counts=[global_clean_trajs, global_nlos_trajs],
        nf_counts=[global_nf, global_ff],
        mode_df=df_modes
    )

def plot_comprehensive_charts(los_counts, traj_counts, nf_counts, mode_df):
    try:
        sns.set_theme(style="whitegrid")
        # 2x2 Layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Point LoS Pie
        axes[0,0].pie(los_counts, labels=['LoS (Visible)', 'NLoS (Blocked)'], autopct='%1.1f%%', 
                      colors=['#66b3ff','#ff9999'], startangle=90, explode=(0, 0.1))
        axes[0,0].set_title("Point-Level Visibility (Instant)", fontsize=14, fontweight='bold')

        # 2. Trajectory LoS Pie (NEW!)
        axes[0,1].pie(traj_counts, labels=['Safe Route', 'Risky Route (Contains NLoS)'], autopct='%1.1f%%', 
                      colors=['#99ff99','#ffcc00'], startangle=90, explode=(0, 0.1))
        axes[0,1].set_title("Trajectory-Level Reliability (Sequence)", fontsize=14, fontweight='bold')

        # 3. Near-Field Pie
        axes[1,0].pie(nf_counts, labels=['Near-Field', 'Far-Field'], autopct='%1.1f%%', 
                      colors=['#c2c2f0','#ffb3e6'], startangle=90)
        axes[1,0].set_title("Near-Field vs Far-Field", fontsize=14, fontweight='bold')

        # 4. Modes Bar Chart
        if not mode_df.empty:
            sns.barplot(x="Count", y="Mode", data=mode_df, ax=axes[1,1], palette="viridis", hue="Mode", legend=False)
            axes[1,1].set_title("Samples per Motion Mode", fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel("Number of Samples")
        
        plt.tight_layout()
        save_path = "dataset_health_dashboard.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n🖼️  Dashboard saved to: {save_path}")

    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

if __name__ == "__main__":
    analyze_dataset_comprehensive(DATASET_DIR)