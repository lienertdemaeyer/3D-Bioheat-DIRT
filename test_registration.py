import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import (register_and_stack_measurements, compute_stability_map, 
                               compute_union_tiered_map, compute_smart_consensus_map)

def process_patient_comparison(patient_base):
    """
    Process all 4 measurements of a patient, register them, and create comparison.
    Shows: Single Measurement vs Stacked (stability-filtered)
    
    NEW: Uses thermal frames for registration (more structure), applies transforms to bioheat maps.
    """
    measurements = [f"{patient_base}M0{i}" for i in range(1, 5)]
    
    hybrid_maps = []
    thermal_refs = []  # For registration
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
        
        # Compute hybrid map
        hybrid_map, weight_map, bioheat_raw = calculate_hybrid_preserving(frames, mask, config)
        hybrid_maps.append(hybrid_map)
        masks.append(mask)
        
        # Store thermal reference (mean frame) for registration
        mean_thermal = np.mean(frames[15:45], axis=0)
        thermal_refs.append(mean_thermal)
        
        print(f"  {pid}: Loaded")
    
    if len(hybrid_maps) < 2:
        print("  Need at least 2 measurements")
        return None
    
    # Two-stage registration: 1) Affine for global alignment, 2) Optical flow for local deformations
    from src.registration import estimate_affine_transform, align_image_affine, compute_optical_flow_warp
    from scipy.ndimage import gaussian_filter
    
    # Blur bioheat maps for registration (creates smoother features)
    blur_sigma = 5.0
    blurred_refs = [gaussian_filter(h, sigma=blur_sigma) for h in hybrid_maps]
    
    reference_blurred = blurred_refs[0]
    reference_mask = masks[0]
    aligned_maps = [hybrid_maps[0].copy()]
    transforms = [(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)]  # Added flow magnitude
    included = [True]
    
    for i in range(1, len(blurred_refs)):
        combined_mask = reference_mask & masks[i]
        
        # Stage 1: Affine alignment (global)
        warp_matrix, dx, dy, angle, sx, sy = estimate_affine_transform(
            reference_blurred, blurred_refs[i], mask=combined_mask, use_full_affine=True
        )
        
        shift_magnitude = np.sqrt(dy**2 + dx**2)
        
        if shift_magnitude > 50:
            transforms.append((dx, dy, angle, sx, sy, 0.0))
            aligned_maps.append(None)
            included.append(False)
        else:
            # Apply affine to bioheat map
            affine_aligned = align_image_affine(hybrid_maps[i], warp_matrix)
            
            # Stage 2: Coarse optical flow first
            coarse_aligned, flow_mag_coarse = compute_optical_flow_warp(
                hybrid_maps[0], affine_aligned, mask=combined_mask, fine_mode=False
            )
            
            # Stage 3: Fine optical flow for small features
            final_aligned, flow_mag_fine = compute_optical_flow_warp(
                hybrid_maps[0], coarse_aligned, mask=combined_mask, fine_mode=True
            )
            
            flow_mag = flow_mag_coarse + flow_mag_fine
            
            transforms.append((dx, dy, angle, sx, sy, flow_mag))
            aligned_maps.append(final_aligned)
            included.append(True)
    
    n_used = sum(included)
    
    for i, transform in enumerate(transforms):
        status = "[OK]" if included[i] else "[EXCLUDED]"
        if len(transform) == 6:
            dx, dy, angle, sx, sy, flow_mag = transform
            mag = np.sqrt(dy**2 + dx**2)
            scale_info = f", scale=({sx:.3f},{sy:.3f})" if abs(sx-1) > 0.01 or abs(sy-1) > 0.01 else ""
            rot_info = f", rot={angle:.2f}deg" if abs(angle) > 0.01 else ""
            flow_info = f", flow={flow_mag:.1f}px" if flow_mag > 0.5 else ""
            print(f"  M0{i+1}: shift={mag:.1f}px{rot_info}{scale_info}{flow_info} {status}")
        else:
            dx, dy, angle = transform[:3]
            mag = np.sqrt(dy**2 + dx**2)
            print(f"  M0{i+1}: shift={mag:.1f}px {status}")
    
    print(f"  Using {n_used}/{len(included)} measurements")
    
    # Compute stacked from aligned maps
    valid_aligned = [m for m in aligned_maps if m is not None]
    stacked = np.mean(valid_aligned, axis=0) if valid_aligned else None
    
    # Compute stability (80th percentile = only strong signals count as "active")
    valid_aligned = [m for m in aligned_maps if m is not None]
    stability = compute_stability_map(valid_aligned, threshold_percentile=80)
    
    # Stability-filtered stacked map (only keep stable perforators)
    # Threshold: must appear in at least 50% of measurements
    stability_threshold = 0.5
    stacked_filtered = stacked * (stability >= stability_threshold)
    
    # Combine masks (intersection for overlap approach)
    combined_mask = masks[0].copy()
    for m in masks[1:]:
        combined_mask = combined_mask & m
    
    # Union/Tiered approach (preserves unique detections)
    union_map, confidence_map, tiered_map = compute_union_tiered_map(valid_aligned, masks, threshold_percentile=80)
    
    # NEW: Smart Consensus approach (intensity-validated unique detections)
    # More inclusive: consensus=25% (1/4 measurements), intensity=60th percentile
    smart_map, smart_confidence, smart_stats = compute_smart_consensus_map(
        valid_aligned, masks, 
        consensus_threshold=0.25,  # Only need 1/4 measurements to agree
        intensity_percentile=60,   # More lenient intensity filter
        detection_percentile=80
    )
    print(f"  Smart: {smart_stats['n_consensus']} consensus, {smart_stats['n_strong_unique']} strong unique, {smart_stats['n_weak_excluded']} excluded")
    
    # Union mask (any measurement covers this pixel)
    union_mask = masks[0].copy()
    for m in masks[1:]:
        union_mask = union_mask | m
    
    # Segmentation from all approaches
    _, segment_single = extract_structure(hybrid_maps[0], masks[0])  # Single (M01)
    _, segment_stacked = extract_structure(stacked_filtered, combined_mask)  # Stacked+filtered (overlap)
    _, segment_tiered = extract_structure(tiered_map, union_mask)  # Tiered (union)
    # Smart: pass consensus map so elongated shapes with high consensus are kept
    _, segment_smart = extract_structure(smart_map, union_mask, consensus_map=smart_confidence)
    
    return {
        'single': hybrid_maps[0],
        'stacked': stacked,
        'stacked_filtered': stacked_filtered,
        'stability': stability,
        'tiered_map': tiered_map,
        'smart_map': smart_map,
        'smart_confidence': smart_confidence,
        'confidence_map': confidence_map,
        'segment_single': segment_single,
        'segment_stacked': segment_stacked,
        'segment_tiered': segment_tiered,
        'segment_smart': segment_smart,
        'mask': combined_mask,
        'union_mask': union_mask,
        'n_used': n_used
    }

def create_comparison_figure(patients_data, patient_names):
    """
    Create a side-by-side comparison figure for multiple patients.
    Columns: Smart Consensus | Stacked Seg | Smart Seg | Overlay
    """
    n_patients = len(patients_data)
    
    n_cols = 4
    fig, axes = plt.subplots(n_patients, n_cols, figsize=(4*n_cols, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    for i, (data, name) in enumerate(zip(patients_data, patient_names)):
        if data is None:
            continue
            
        union_mask = data['union_mask']
        mask = data['mask']
        
        # Column 0: Smart Consensus input
        smart = data['smart_map']
        vmax_s = np.percentile(smart[union_mask & (smart > 0)], 99.5) if np.any(smart > 0) else 1
        axes[i, 0].imshow(smart, cmap='magma', vmin=0, vmax=vmax_s)
        axes[i, 0].set_title(f"{name}: Smart Consensus", fontsize=10)
        axes[i, 0].axis('off')
        
        # Column 1: Stacked segmentation (overlap-based)
        seg_stacked = data['segment_stacked'] > 0
        axes[i, 1].imshow(seg_stacked, cmap='gray')
        axes[i, 1].set_title("Stacked Segmentation", fontsize=10)
        axes[i, 1].axis('off')
        
        # Column 2: Smart segmentation (rule-based)
        seg_smart = data['segment_smart'] > 0
        axes[i, 2].imshow(seg_smart, cmap='gray')
        axes[i, 2].set_title("Smart Segmentation", fontsize=10)
        axes[i, 2].axis('off')
        
        # Column 3: Overlay on input
        smart_norm = smart / (vmax_s + 1e-8)
        smart_norm = np.clip(smart_norm, 0, 1)
        overlay = cv2.cvtColor((smart_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Draw green contours for segmentation
        contours, _ = cv2.findContours(seg_smart.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Clinical Overlay", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle("Rule-based Perforator Segmentation", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "Comparison_RuleBased.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparison to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Test with patients (rule-based segmentation)
    patient_list = ["P15", "P16", "P17", "P18", "P19", "P20", "P21"]
    
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

