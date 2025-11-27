"""
Compare different ground truth strategies for U-Net training:
1. Overlap (conservative) - only structures in ALL measurements
2. Union (inclusive) - structures in ANY measurement  
3. Intensity-gated Union - smart hybrid approach
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import config.config as config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.registration import estimate_affine_transform, align_image_affine, compute_optical_flow_warp
from src.segmentation import extract_structure


def compute_gt_strategies(patient_base):
    """Compute different GT strategies for a patient."""
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
            aligned_masks.append(masks[i])
    
    if len(aligned_maps) < 2:
        return None
    
    # Combined mask
    combined_mask = aligned_masks[0].copy()
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
            binary_maps.append(np.zeros_like(m, dtype=bool))
    
    # Stability map (how often a pixel is detected)
    stability = np.mean(binary_maps, axis=0)
    
    # === STRATEGY 1: OVERLAP (conservative) ===
    # Only keep structures detected in ALL measurements
    overlap_map = stacked * (stability >= 0.5)  # At least half
    _, overlap_seg = extract_structure(overlap_map, combined_mask, use_ml_filter=False, apply_shape_filter=True)
    
    # === STRATEGY 2: UNION (inclusive, no shape filter) ===
    # Keep structures detected in ANY measurement - NO filtering
    union_binary = np.any(binary_maps, axis=0)
    union_map = stacked * union_binary
    _, union_seg = extract_structure(union_map, combined_mask, use_ml_filter=False, apply_shape_filter=False)
    
    # === STRATEGY 3: INTENSITY-GATED UNION (improved) ===
    # Use MAX instead of MEAN for intensity check (stacking dilutes unique detections)
    max_map = np.max(aligned_maps, axis=0)
    
    # Lower threshold - 70th percentile of max intensities
    intensity_threshold = np.percentile(max_map[max_map > 0], 70) if np.any(max_map > 0) else 0
    
    # Consensus regions (detected in ≥2 measurements) - always keep
    consensus_mask = stability >= 0.5
    
    # Unique regions: use MAX intensity (not diluted by averaging)
    unique_mask = (stability > 0) & (stability < 0.5)
    high_intensity_mask = max_map > intensity_threshold
    
    # Combine: consensus OR (unique AND high intensity in original measurement)
    smart_binary = consensus_mask | (unique_mask & high_intensity_mask)
    smart_map = stacked * smart_binary
    
    # NO shape filtering - let U-Net learn the full picture
    _, smart_seg = extract_structure(smart_map, combined_mask, use_ml_filter=False, apply_shape_filter=False)
    
    return {
        'smart_consensus': stacked * (stability >= 0.25),  # For visualization
        'overlap_seg': overlap_seg,
        'union_seg': union_seg,
        'smart_seg': smart_seg,
        'stacked': stacked,
        'stability': stability,
        'combined_mask': combined_mask
    }


def create_comparison(patients):
    """Create comparison figure for multiple patients."""
    results = []
    valid_names = []
    
    for p in patients:
        print(f"Processing {p}...")
        data = compute_gt_strategies(p)
        if data is not None:
            results.append(data)
            valid_names.append(p)
    
    if not results:
        print("No valid results!")
        return
    
    n_patients = len(results)
    fig, axes = plt.subplots(n_patients, 5, figsize=(20, 4 * n_patients))
    
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    for i, (data, name) in enumerate(zip(results, valid_names)):
        # Column 0: Smart Consensus (input visualization)
        axes[i, 0].imshow(data['smart_consensus'], cmap='magma')
        axes[i, 0].set_title(f"{name}: Smart Consensus", fontsize=11)
        axes[i, 0].axis('off')
        
        # Column 1: Overlap Seg (current GT)
        axes[i, 1].imshow(data['overlap_seg'], cmap='gray')
        axes[i, 1].set_title("Overlap GT (current)", fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 2: Union Seg (all structures)
        axes[i, 2].imshow(data['union_seg'], cmap='gray')
        axes[i, 2].set_title("Union GT (all)", fontsize=11)
        axes[i, 2].axis('off')
        
        # Column 3: Intensity-gated Union
        axes[i, 3].imshow(data['smart_seg'], cmap='gray')
        axes[i, 3].set_title("Intensity-gated GT (smart)", fontsize=11)
        axes[i, 3].axis('off')
        
        # Column 4: Comparison overlay
        overlay = np.zeros((*data['overlap_seg'].shape, 3), dtype=np.uint8)
        # Red = only in union (potential artifacts)
        overlay[data['union_seg'] & ~data['smart_seg']] = [255, 100, 100]
        # Blue = only in overlap (consensus)
        overlay[data['overlap_seg'] & ~data['smart_seg']] = [100, 100, 255]
        # Green = in smart but not overlap (recovered by intensity)
        overlay[data['smart_seg'] & ~data['overlap_seg']] = [100, 255, 100]
        # White = in both overlap and smart
        overlay[data['overlap_seg'] & data['smart_seg']] = [255, 255, 255]
        
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title("White=both, Green=recovered, Red=excluded", fontsize=9)
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "GT_Strategy_Comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Test on patients that had issues (P21 corner region, etc.)
    patients = ["P15", "P18", "P21"]
    create_comparison(patients)

