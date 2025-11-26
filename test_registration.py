import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import register_and_stack_measurements, compute_stability_map

def process_patient_comparison(patient_base):
    """
    Process all 4 measurements of a patient, register them, and create comparison.
    Shows: Single Measurement vs Stacked (stability-filtered)
    """
    measurements = [f"{patient_base}M0{i}" for i in range(1, 5)]
    
    hybrid_maps = []
    masks = []
    
    print(f"\n{'='*50}")
    print(f"Processing {patient_base}")
    print('='*50)
    
    for pid in measurements:
        frames, dims = load_frames_h5(config.H5_DIR, pid, config.MAX_FRAMES)
        if frames is None:
            continue
            
        h, w = frames.shape[1], frames.shape[2]
        mask = load_coco_mask(config.COCO_PATH, pid, h, w)
        
        hybrid_map, weight_map, bioheat_raw = calculate_hybrid_preserving(frames, mask, config)
        hybrid_maps.append(hybrid_map)
        masks.append(mask)
        print(f"  {pid}: Loaded")
    
    if len(hybrid_maps) < 2:
        print("  Need at least 2 measurements")
        return None
    
    # Register and stack (using masks + affine for better alignment)
    stacked, transforms, aligned_maps, included = register_and_stack_measurements(
        hybrid_maps, max_shift=50, masks=masks, use_affine=True
    )
    
    for i, (dx, dy, angle) in enumerate(transforms):
        status = "[OK]" if included[i] else "[EXCLUDED]"
        mag = np.sqrt(dy**2 + dx**2)
        if abs(angle) > 0.01:
            print(f"  M0{i+1}: shift={mag:.1f}px, rot={angle:.2f}deg {status}")
        else:
            print(f"  M0{i+1}: shift={mag:.1f}px {status}")
    
    n_used = sum(included)
    print(f"  Using {n_used}/{len(included)} measurements")
    
    # Compute stability (80th percentile = only strong signals count as "active")
    valid_aligned = [m for m in aligned_maps if m is not None]
    stability = compute_stability_map(valid_aligned, threshold_percentile=80)
    
    # Stability-filtered stacked map (only keep stable perforators)
    # Threshold: must appear in at least 50% of measurements
    stability_threshold = 0.5
    stacked_filtered = stacked * (stability >= stability_threshold)
    
    # Combine masks
    combined_mask = masks[0].copy()
    for m in masks[1:]:
        combined_mask = combined_mask & m
    
    # Segmentation from both
    _, segment_single = extract_structure(hybrid_maps[0], masks[0])  # Single (M01)
    _, segment_stacked = extract_structure(stacked_filtered, combined_mask)  # Stacked+filtered
    
    return {
        'single': hybrid_maps[0],
        'stacked': stacked,
        'stacked_filtered': stacked_filtered,
        'stability': stability,
        'segment_single': segment_single,
        'segment_stacked': segment_stacked,
        'mask': combined_mask,
        'n_used': n_used
    }

def create_comparison_figure(patients_data, patient_names):
    """
    Create a side-by-side comparison figure for multiple patients.
    Columns: Single M01 | Stacked+Filtered | Stability Map | Segmentation Comparison
    """
    n_patients = len(patients_data)
    
    fig, axes = plt.subplots(n_patients, 4, figsize=(20, 5*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    plt.suptitle("Registration & Stability Filtering: Single vs Multi-Measurement", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for i, (data, name) in enumerate(zip(patients_data, patient_names)):
        if data is None:
            continue
            
        mask = data['mask']
        
        # Column 0: Single measurement (M01)
        single = data['single']
        vmax = np.percentile(single[mask & (single > 0)], 99.5) if np.any(single > 0) else 1
        axes[i, 0].imshow(single, cmap='magma', vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"{name}: Single (M01)", fontsize=11)
        axes[i, 0].axis('off')
        
        # Column 1: Stacked + Stability filtered
        filtered = data['stacked_filtered']
        vmax_f = np.percentile(filtered[mask & (filtered > 0)], 99.5) if np.any(filtered > 0) else vmax
        axes[i, 1].imshow(filtered, cmap='magma', vmin=0, vmax=vmax_f)
        axes[i, 1].set_title(f"Stacked+Filtered ({data['n_used']} meas)", fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 2: Stability map
        im = axes[i, 2].imshow(data['stability'], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title("Stability (yellow=consistent)", fontsize=11)
        axes[i, 2].axis('off')
        
        # Column 3: Segmentation comparison (overlay)
        # Green = single only, Blue = stacked only, White = both
        seg_single = data['segment_single'] > 0
        seg_stacked = data['segment_stacked'] > 0
        
        overlay = np.zeros((*seg_single.shape, 3), dtype=np.uint8)
        overlay[seg_single & ~seg_stacked] = [0, 255, 0]    # Green: only in single
        overlay[seg_stacked & ~seg_single] = [0, 100, 255]  # Blue: only in stacked
        overlay[seg_single & seg_stacked] = [255, 255, 255] # White: both
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Seg: Green=single, Blue=stacked, White=both", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "Comparison_Single_vs_Stacked.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparison to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Process multiple patients
    patient_list = ["P15", "P16", "P17", "P23", "P24", "P25"]
    
    results = []
    valid_names = []
    
    for p in patient_list:
        data = process_patient_comparison(p)
        if data is not None:
            results.append(data)
            valid_names.append(p)
    
    # Create comparison figure
    if results:
        create_comparison_figure(results, valid_names)
        print(f"\nProcessed {len(results)} patients successfully!")

