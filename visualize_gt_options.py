"""
Visualize different Ground Truth options for U-Net training:
1. Overlap (conservative) - only structures in ALL measurements
2. Union (inclusive) - structures in ANY measurement  
3. Intensity-Gated Union (smart) - consensus OR high-intensity unique
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

import config.config as config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.registration import estimate_affine_transform, align_image_affine, compute_optical_flow_warp
from src.segmentation import extract_structure


def compute_gt_options(patient_base):
    """Compute different GT options for a patient."""
    measurements = [f"{patient_base}M0{i}" for i in range(1, 5)]
    
    hybrid_maps = []
    masks = []
    
    for pid in measurements:
        frames, dims = load_frames_h5(config.H5_DIR, pid, config.MAX_FRAMES)
        if frames is None:
            continue
        
        h, w = frames.shape[1], frames.shape[2]
        mask = load_coco_mask(config.COCO_PATH, pid, h, w)
        
        hybrid_map, _, _ = calculate_hybrid_preserving(frames, mask, config)
        hybrid_maps.append(hybrid_map)
        masks.append(mask)
    
    if len(hybrid_maps) < 2:
        return None
    
    # Register measurements
    blur_sigma = 5.0
    blurred_refs = [gaussian_filter(h, sigma=blur_sigma) for h in hybrid_maps]
    
    aligned_maps = [hybrid_maps[0].copy()]
    aligned_masks = [masks[0].copy()]
    
    for i in range(1, len(blurred_refs)):
        combined_mask = masks[0] & masks[i]
        warp_matrix, dx, dy, angle, sx, sy = estimate_affine_transform(
            blurred_refs[0], blurred_refs[i], mask=combined_mask, use_full_affine=True
        )
        
        if np.sqrt(dy**2 + dx**2) <= 50:
            affine_aligned = align_image_affine(hybrid_maps[i], warp_matrix)
            final_aligned, _ = compute_optical_flow_warp(
                hybrid_maps[0], affine_aligned, mask=combined_mask, fine_mode=True
            )
            aligned_maps.append(final_aligned)
            aligned_masks.append(align_image_affine(masks[i].astype(np.float32), warp_matrix) > 0.5)
    
    if len(aligned_maps) < 2:
        return None
    
    # Combined mask
    combined_mask = aligned_masks[0]
    for m in aligned_masks[1:]:
        combined_mask = combined_mask & m
    
    # Stacked map
    stacked = np.mean(aligned_maps, axis=0)
    
    # Binary maps for each measurement
    binary_maps = []
    for m in aligned_maps:
        if np.any(m > 0):
            thresh = np.percentile(m[m > 0], 80)
            binary_maps.append(m > thresh)
        else:
            binary_maps.append(m > 0)
    
    # Stability map (how many measurements detected each pixel)
    stability = np.mean(binary_maps, axis=0)
    
    # --- OPTION 1: Overlap (conservative) ---
    # Only keep pixels detected in ALL measurements (stability = 1.0)
    # Or at least 50% for robustness
    overlap_map = stacked * (stability >= 0.5)
    _, overlap_seg = extract_structure(overlap_map, combined_mask, use_ml_filter=False, apply_shape_filter=True)
    
    # --- OPTION 2: Union (inclusive) ---
    # Keep pixels detected in ANY measurement
    union_binary = np.any(binary_maps, axis=0)
    union_map = stacked * union_binary
    _, union_seg = extract_structure(union_map, combined_mask, use_ml_filter=False, apply_shape_filter=True)
    
    # --- OPTION 3: Intensity-Gated Union (smart) ---
    # Keep if: consensus (stability >= 0.5) OR high intensity (top 20%)
    intensity_threshold = np.percentile(stacked[stacked > 0], 80) if np.any(stacked > 0) else 0
    
    # Consensus regions
    consensus_mask = stability >= 0.5
    
    # High-intensity unique regions (in union but not consensus)
    unique_regions = union_binary & ~consensus_mask
    high_intensity_unique = unique_regions & (stacked >= intensity_threshold)
    
    # Combine
    smart_mask = consensus_mask | high_intensity_unique
    smart_map = stacked * smart_mask
    _, smart_seg = extract_structure(smart_map, combined_mask, use_ml_filter=False, apply_shape_filter=True)
    
    return {
        'stacked': stacked,
        'stability': stability,
        'combined_mask': combined_mask,
        'overlap_seg': overlap_seg,
        'union_seg': union_seg,
        'smart_seg': smart_seg,
        'consensus_mask': consensus_mask,
        'high_intensity_unique': high_intensity_unique
    }


def visualize_gt_comparison(patients=["P15", "P16", "P17", "P18", "P19", "P20", "P21"]):
    """Create comparison figure of GT options."""
    
    results = []
    valid_patients = []
    
    for p in patients:
        print(f"Processing {p}...")
        data = compute_gt_options(p)
        if data is not None:
            results.append(data)
            valid_patients.append(p)
    
    if not results:
        print("No valid results!")
        return
    
    # Create figure: rows = patients, cols = [Stacked, Overlap GT, Union GT, Intensity-Gated GT, Comparison]
    n_patients = len(results)
    fig, axes = plt.subplots(n_patients, 5, figsize=(20, 4 * n_patients))
    
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    for i, (data, pname) in enumerate(zip(results, valid_patients)):
        # Column 0: Stacked map
        axes[i, 0].imshow(data['stacked'], cmap='magma')
        axes[i, 0].set_title(f"{pname}: Stacked Map", fontsize=11)
        axes[i, 0].axis('off')
        
        # Column 1: Overlap (conservative)
        axes[i, 1].imshow(data['overlap_seg'], cmap='gray')
        n_overlap = np.sum(data['overlap_seg'])
        axes[i, 1].set_title(f"Overlap GT ({n_overlap} px)", fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 2: Union
        axes[i, 2].imshow(data['union_seg'], cmap='gray')
        n_union = np.sum(data['union_seg'])
        axes[i, 2].set_title(f"Union GT ({n_union} px)", fontsize=11)
        axes[i, 2].axis('off')
        
        # Column 3: Intensity-Gated
        axes[i, 3].imshow(data['smart_seg'], cmap='gray')
        n_smart = np.sum(data['smart_seg'])
        axes[i, 3].set_title(f"Intensity-Gated GT ({n_smart} px)", fontsize=11)
        axes[i, 3].axis('off')
        
        # Column 4: Comparison overlay
        overlay = np.zeros((*data['overlap_seg'].shape, 3), dtype=np.uint8)
        # Red = only in Union (potential artifacts)
        overlay[data['union_seg'] & ~data['smart_seg']] = [255, 100, 100]
        # Yellow = added by intensity-gating (recovered)
        overlay[data['smart_seg'] & ~data['overlap_seg']] = [255, 255, 0]
        # White = in all (overlap)
        overlay[data['overlap_seg'].astype(bool)] = [255, 255, 255]
        
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title("White=overlap, Yellow=+intensity, Red=excluded", fontsize=9)
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "GT_Options_Comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_file}")
    plt.close()
    
    # Open the image
    os.startfile(output_file)


if __name__ == "__main__":
    visualize_gt_comparison()

