"""
Test Early Consensus: Apply agreement at the bioheat-gradient level BEFORE stacking.

Approach:
1. Compute bioheat-gradient maps for all measurements
2. Register them
3. Binarize each map (identify "active" pixels above threshold)
4. Vote map: count how many measurements have each pixel as "active"
5. Consensus mask: only pixels with ≥N votes are kept
6. Apply consensus mask to stacked map
7. Segment the filtered stacked map

This should reduce noise because random noise won't agree across measurements.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import estimate_affine_transform, align_image_affine, compute_optical_flow_warp
from scipy.ndimage import gaussian_filter
from scipy import ndimage


def compute_early_consensus(hybrid_maps, masks, min_agreement=2, percentile_threshold=70):
    """
    Compute consensus at the bioheat-gradient level.
    
    Args:
        hybrid_maps: List of aligned bioheat-gradient maps
        masks: List of masks
        min_agreement: Minimum number of measurements that must agree (default: 2)
        percentile_threshold: Percentile to threshold each map for "active" pixels
    
    Returns:
        consensus_map: Stacked map filtered by agreement
        vote_map: Number of measurements where each pixel was "active"
        binary_maps: List of binarized maps
    """
    n_maps = len(hybrid_maps)
    h, w = hybrid_maps[0].shape
    
    # Combined mask (intersection)
    combined_mask = masks[0].copy()
    for m in masks[1:]:
        combined_mask = combined_mask & m
    
    # Binarize each map - identify "active" pixels
    binary_maps = []
    for hmap in hybrid_maps:
        # Get threshold for this map
        valid_pixels = hmap[hmap > 0]
        if len(valid_pixels) > 0:
            thresh = np.percentile(valid_pixels, percentile_threshold)
            binary = (hmap > thresh).astype(np.float32)
        else:
            binary = np.zeros_like(hmap)
        binary_maps.append(binary)
    
    # Vote map: how many measurements have each pixel as "active"
    vote_map = np.sum(binary_maps, axis=0)
    
    # Consensus mask: pixels with >= min_agreement votes
    consensus_mask = (vote_map >= min_agreement).astype(np.float32)
    
    # Stack the maps (mean)
    stacked = np.mean(hybrid_maps, axis=0)
    
    # Apply consensus mask to stacked map
    consensus_map = stacked * consensus_mask * combined_mask
    
    return consensus_map, vote_map, binary_maps, combined_mask


def simple_threshold_segmentation(image, mask, percentile=70):
    """
    Simple intensity-based segmentation without morphological operations.
    Just threshold at percentile - catches everything bright.
    """
    valid_pixels = image[mask & (image > 0)]
    if len(valid_pixels) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    thresh = np.percentile(valid_pixels, percentile)
    binary = ((image > thresh) & mask).astype(np.uint8)
    
    # Light cleanup - remove tiny noise (< 10 pixels)
    from scipy import ndimage as ndi
    labeled, n = ndi.label(binary)
    for i in range(1, n + 1):
        if np.sum(labeled == i) < 10:
            binary[labeled == i] = 0
    
    return binary


def peak_based_segmentation(image, mask, min_distance=10, threshold_rel=0.3):
    """
    Peak-based detection using scipy only (no skimage needed).
    Find local maxima, then dilate to create regions around peaks.
    """
    from scipy import ndimage as ndi
    
    # Normalize image
    masked_image = image * mask
    if masked_image.max() == 0:
        return np.zeros_like(image, dtype=np.uint8), 0
    
    normalized = masked_image / masked_image.max()
    
    # Find local maxima using maximum filter
    # A pixel is a local max if it equals its neighborhood maximum
    neighborhood_size = min_distance * 2 + 1
    local_max = ndi.maximum_filter(normalized, size=neighborhood_size)
    
    # Peaks are where image equals local max AND above threshold
    threshold_abs = threshold_rel * normalized.max()
    peaks = (normalized == local_max) & (normalized > threshold_abs) & mask
    
    # Label connected peak regions
    labeled_peaks, n_peaks = ndi.label(peaks)
    
    if n_peaks == 0:
        return np.zeros_like(image, dtype=np.uint8), 0
    
    # Grow regions around peaks: dilate and intersect with high-intensity areas
    # Use a threshold to define "high intensity" around peaks
    high_intensity = (normalized > threshold_rel * 0.5) & mask
    
    # Dilate peaks to grow regions
    struct = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    dilated = ndi.binary_dilation(peaks, struct, iterations=5)
    
    # Final segmentation: dilated peaks intersected with high intensity
    binary = (dilated & high_intensity).astype(np.uint8)
    
    # Count final regions
    _, n_regions = ndi.label(binary)
    
    return binary, n_peaks


def compute_adaptive_consensus(hybrid_maps, masks, min_agreement=2, n_total=4):
    """
    Compute consensus with ADAPTIVE threshold based on agreement level.
    
    Key insight: More agreement = more confidence = can use lower threshold
    
    Threshold mapping (for 4 measurements):
        ≥1 agree: 80th percentile (strict - need to filter noise)
        ≥2 agree: 70th percentile (moderate)
        ≥3 agree: 55th percentile (lenient - high confidence)
        ≥4 agree: 40th percentile (very lenient - maximum confidence)
    """
    # Adaptive threshold: higher agreement = lower (more lenient) threshold
    # This allows us to detect more when we're more certain
    threshold_map = {
        1: 80,  # Low confidence → strict threshold
        2: 70,  # Medium confidence
        3: 55,  # High confidence → lenient threshold
        4: 40,  # Maximum confidence → very lenient
    }
    
    # Adjust for actual number of measurements
    if n_total == 3:
        threshold_map = {1: 80, 2: 65, 3: 45}
    elif n_total == 2:
        threshold_map = {1: 75, 2: 55}
    
    percentile = threshold_map.get(min_agreement, 70)
    
    return compute_early_consensus(hybrid_maps, masks, min_agreement, percentile), percentile


def process_patient_early_consensus(patient_base):
    """Process a patient with early consensus approach."""
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
        
        hybrid_map, _, _ = calculate_hybrid_preserving(frames, mask, config)
        hybrid_maps.append(hybrid_map)
        masks.append(mask)
        print(f"  {pid}: Loaded")
    
    if len(hybrid_maps) < 2:
        print("  Need at least 2 measurements")
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
        
        shift_magnitude = np.sqrt(dy**2 + dx**2)
        
        if shift_magnitude > 50:
            print(f"  M0{i+1}: shift={shift_magnitude:.1f}px [EXCLUDED]")
            continue
        
        # Apply affine to bioheat map
        affine_aligned = align_image_affine(hybrid_maps[i], warp_matrix)
        
        # Fine optical flow
        final_aligned, flow_mag = compute_optical_flow_warp(
            hybrid_maps[0], affine_aligned, mask=combined_mask, fine_mode=True
        )
        
        aligned_maps.append(final_aligned)
        aligned_masks.append(masks[i])
        print(f"  M0{i+1}: shift={shift_magnitude:.1f}px, flow={flow_mag:.1f}px [OK]")
    
    n_used = len(aligned_maps)
    print(f"  Using {n_used}/{len(hybrid_maps)} measurements")
    
    # ===== EARLY CONSENSUS WITH ADAPTIVE THRESHOLD =====
    # Key insight: More agreement = more confidence = can use lower threshold
    results = {}
    
    print(f"\n  --- Fixed threshold (70th percentile) ---")
    for min_agree in [2, 3, 4]:
        if min_agree > n_used:
            continue
        consensus_map, vote_map, binary_maps, combined_mask = compute_early_consensus(
            aligned_maps, aligned_masks, min_agreement=min_agree, percentile_threshold=70
        )
        _, segmentation = extract_structure(consensus_map, combined_mask)
        labeled, n_regions = ndimage.label(segmentation > 0)
        
        key = f"fixed_{min_agree}"
        results[key] = {
            'consensus_map': consensus_map,
            'segmentation': segmentation,
            'n_regions': n_regions,
            'combined_mask': combined_mask,
            'threshold': 70
        }
        print(f"  ≥{min_agree}/{n_used} @ 70th pct: {n_regions} regions")
    
    print(f"\n  --- Adaptive threshold (more agree = lower threshold) ---")
    for min_agree in [2, 3, 4]:
        if min_agree > n_used:
            continue
        (consensus_map, vote_map, binary_maps, combined_mask), pct = compute_adaptive_consensus(
            aligned_maps, aligned_masks, min_agreement=min_agree, n_total=n_used
        )
        _, segmentation = extract_structure(consensus_map, combined_mask)
        labeled, n_regions = ndimage.label(segmentation > 0)
        
        key = f"adaptive_{min_agree}"
        results[key] = {
            'consensus_map': consensus_map,
            'segmentation': segmentation,
            'n_regions': n_regions,
            'combined_mask': combined_mask,
            'threshold': pct
        }
        print(f"  ≥{min_agree}/{n_used} @ {pct}th pct (adaptive): {n_regions} regions")
    
    # Get combined mask for simple methods
    combined_mask = aligned_masks[0].copy()
    for m in aligned_masks[1:]:
        combined_mask = combined_mask & m
    stacked = np.mean(aligned_maps, axis=0) * combined_mask
    
    print(f"\n  --- Alternative segmentation methods ---")
    
    # Simple threshold at 70th (stricter to reduce noise)
    seg_simple = simple_threshold_segmentation(stacked, combined_mask, percentile=70)
    labeled, n_simple = ndimage.label(seg_simple > 0)
    results['simple_thresh'] = {
        'consensus_map': stacked,
        'segmentation': seg_simple,
        'n_regions': n_simple,
        'combined_mask': combined_mask,
        'threshold': 70
    }
    print(f"  Simple threshold @70th: {n_simple} regions")
    
    # Peak-based detection
    try:
        seg_peaks, n_peaks = peak_based_segmentation(stacked, combined_mask, min_distance=10, threshold_rel=0.25)
        results['peak_based'] = {
            'consensus_map': stacked,
            'segmentation': seg_peaks,
            'n_regions': n_peaks,
            'combined_mask': combined_mask,
            'threshold': 'peaks'
        }
        print(f"  Peak-based detection: {n_peaks} peaks, {ndimage.label(seg_peaks > 0)[1]} regions")
    except Exception as e:
        print(f"  Peak-based failed: {e}")
    
    # Early consensus (≥2) + Simple threshold
    (consensus_map_2, _, _, _) = compute_early_consensus(
        aligned_maps, aligned_masks, min_agreement=2, percentile_threshold=65
    )
    seg_combo_2 = simple_threshold_segmentation(consensus_map_2, combined_mask, percentile=65)
    labeled, n_combo_2 = ndimage.label(seg_combo_2 > 0)
    results['consensus_simple_2'] = {
        'consensus_map': consensus_map_2,
        'segmentation': seg_combo_2,
        'n_regions': n_combo_2,
        'combined_mask': combined_mask,
        'threshold': '≥2 + simple@65th'
    }
    print(f"  ≥2 Agree + Simple @65th: {n_combo_2} regions")
    
    # Early consensus (≥3) + Simple threshold (more conservative)
    if n_used >= 3:
        (consensus_map_3, _, _, _) = compute_early_consensus(
            aligned_maps, aligned_masks, min_agreement=3, percentile_threshold=55
        )
        seg_combo_3 = simple_threshold_segmentation(consensus_map_3, combined_mask, percentile=55)
        labeled, n_combo_3 = ndimage.label(seg_combo_3 > 0)
        results['consensus_simple_3'] = {
            'consensus_map': consensus_map_3,
            'segmentation': seg_combo_3,
            'n_regions': n_combo_3,
            'combined_mask': combined_mask,
            'threshold': '≥3 + simple@55th'
        }
        print(f"  ≥3 Agree + Simple @55th: {n_combo_3} regions")
    
    # Early consensus (≥4) + Simple threshold (most conservative - all agree)
    if n_used >= 4:
        (consensus_map_4, _, _, _) = compute_early_consensus(
            aligned_maps, aligned_masks, min_agreement=4, percentile_threshold=45
        )
        seg_combo_4 = simple_threshold_segmentation(consensus_map_4, combined_mask, percentile=45)
        labeled, n_combo_4 = ndimage.label(seg_combo_4 > 0)
        results['consensus_simple_4'] = {
            'consensus_map': consensus_map_4,
            'segmentation': seg_combo_4,
            'n_regions': n_combo_4,
            'combined_mask': combined_mask,
            'threshold': '≥4 + simple@45th'
        }
        print(f"  ≥4 Agree + Simple @45th: {n_combo_4} regions")
    
    # Also compute the old approach for comparison (no early consensus)
    stacked_no_consensus = np.mean(aligned_maps, axis=0)
    combined_mask = aligned_masks[0].copy()
    for m in aligned_masks[1:]:
        combined_mask = combined_mask & m
    stacked_no_consensus = stacked_no_consensus * combined_mask
    
    _, seg_no_consensus = extract_structure(stacked_no_consensus, combined_mask)
    labeled, n_regions_old = ndimage.label(seg_no_consensus > 0)
    print(f"  No consensus (old): {n_regions_old} regions detected")
    
    results['no_consensus'] = {
        'consensus_map': stacked_no_consensus,
        'segmentation': seg_no_consensus,
        'n_regions': n_regions_old,
        'combined_mask': combined_mask
    }
    
    return results, n_used


def create_comparison_figure(all_results, patient_names, batch_idx=0):
    """Create comparison with maps and binary for each agreement level (3 patients per figure)."""
    n_patients = len(all_results)
    
    # Output folder for individual patient figures
    output_folder = r"C:\Users\liene\Documents\DATA OUTPUT\3D-BIOHEAT-DIRT\consensus per patients"
    os.makedirs(output_folder, exist_ok=True)
    
    # Columns: Stacked | ≥2 Map | ≥2 Seg | ≥2 Bin | ≥3 Map | ≥3 Seg | ≥3 Bin | ≥4 Map | ≥4 Seg | ≥4 Bin
    n_cols = 10
    fig, axes = plt.subplots(n_patients, n_cols, figsize=(2.0*n_cols, 3*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    # Reduce spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
    
    for i, (results, name) in enumerate(zip(all_results, patient_names)):
        if results is None:
            continue
        
        def plot_map_only(ax, data, title):
            """Plot just the bioheat-gradient map."""
            cmap_data = data['consensus_map']
            mask = data['combined_mask']
            valid = cmap_data[mask & (cmap_data > 0)]
            vmax = np.percentile(valid, 99) if len(valid) > 0 else 1
            ax.imshow(cmap_data, cmap='magma', vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.axis('off')
        
        def plot_map_with_contours(ax, data, title, contour_color='lime'):
            """Plot the bioheat-gradient map with segmentation contours."""
            cmap_data = data['consensus_map']
            mask = data['combined_mask']
            seg = data['segmentation'] > 0
            valid = cmap_data[mask & (cmap_data > 0)]
            vmax = np.percentile(valid, 99) if len(valid) > 0 else 1
            
            ax.imshow(cmap_data, cmap='magma', vmin=0, vmax=vmax)
            contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                c = c.squeeze()
                if len(c.shape) == 2:
                    ax.plot(c[:, 0], c[:, 1], contour_color, linewidth=1.2)
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.axis('off')
        
        def plot_binary(ax, data, title):
            """Plot binary segmentation mask."""
            seg = data['segmentation'] > 0
            ax.imshow(seg, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.axis('off')
        
        # Column 0: Stacked Average (original)
        if 'no_consensus' in results:
            d = results['no_consensus']
            plot_map_only(axes[i, 0], d, f"{name}\nStacked")
        
        # Column 1: ≥2 Map
        if 'consensus_simple_2' in results:
            d = results['consensus_simple_2']
            plot_map_only(axes[i, 1], d, "≥2 Map")
        
        # Column 2: ≥2 Segmentation
        if 'consensus_simple_2' in results:
            d = results['consensus_simple_2']
            plot_map_with_contours(axes[i, 2], d, f"≥2 ({d['n_regions']})", 'cyan')
        
        # Column 3: ≥2 Binary
        if 'consensus_simple_2' in results:
            d = results['consensus_simple_2']
            plot_binary(axes[i, 3], d, "≥2 Bin")
        
        # Column 4: ≥3 Map
        if 'consensus_simple_3' in results:
            d = results['consensus_simple_3']
            plot_map_only(axes[i, 4], d, "≥3 Map")
        else:
            axes[i, 4].set_title("N/A", fontsize=8)
            axes[i, 4].axis('off')
        
        # Column 5: ≥3 Segmentation
        if 'consensus_simple_3' in results:
            d = results['consensus_simple_3']
            plot_map_with_contours(axes[i, 5], d, f"≥3 ({d['n_regions']})", 'lime')
        else:
            axes[i, 5].axis('off')
        
        # Column 6: ≥3 Binary
        if 'consensus_simple_3' in results:
            d = results['consensus_simple_3']
            plot_binary(axes[i, 6], d, "≥3 Bin")
        else:
            axes[i, 6].axis('off')
        
        # Column 7: ≥4 Map
        if 'consensus_simple_4' in results:
            d = results['consensus_simple_4']
            plot_map_only(axes[i, 7], d, "≥4 Map")
        else:
            axes[i, 7].set_title("N/A", fontsize=8)
            axes[i, 7].axis('off')
        
        # Column 8: ≥4 Segmentation
        if 'consensus_simple_4' in results:
            d = results['consensus_simple_4']
            plot_map_with_contours(axes[i, 8], d, f"≥4 ({d['n_regions']})", 'yellow')
        else:
            axes[i, 8].axis('off')
        
        # Column 9: ≥4 Binary
        if 'consensus_simple_4' in results:
            d = results['consensus_simple_4']
            plot_binary(axes[i, 9], d, "≥4 Bin")
        else:
            axes[i, 9].axis('off')
    
    plt.suptitle("Stacked → ≥2 (Map|Seg|Bin) → ≥3 (Map|Seg|Bin) → ≥4 (Map|Seg|Bin)", 
                 fontsize=11, fontweight='bold', y=0.98)
    
    # Get patient names for filename
    first_patient = patient_names[0].split()[0] if patient_names else "P00"
    last_patient = patient_names[-1].split()[0] if patient_names else "P00"
    output_file = os.path.join(output_folder, f"Consensus_{first_patient}-{last_patient}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSaved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # All patients from P08 to P25
    patient_list = [f"P{i:02d}" for i in range(8, 26)]  # P08 to P25
    
    all_results = []
    valid_names = []
    
    for p in patient_list:
        result = process_patient_early_consensus(p)
        if result is not None:
            results, n_used = result
            all_results.append(results)
            valid_names.append(f"{p} ({n_used} meas)")
    
    # Process in batches of 3 patients per figure
    batch_size = 3
    for i in range(0, len(all_results), batch_size):
        batch_results = all_results[i:i+batch_size]
        batch_names = valid_names[i:i+batch_size]
        if batch_results:
            create_comparison_figure(batch_results, batch_names, batch_idx=i//batch_size)
    
    print(f"\nProcessed {len(all_results)} patients in {(len(all_results) + batch_size - 1) // batch_size} figures!")

