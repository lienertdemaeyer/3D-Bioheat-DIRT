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
from src.registration import (register_and_stack_measurements, compute_stability_map, 
                               compute_union_tiered_map, compute_smart_consensus_map)


def load_ct_perforators_and_umbilicus(h5_dir, patient_base, measurement_id="M01"):
    """
    Load CT perforator coordinates and umbilicus from H5 file.
    
    Args:
        h5_dir: Directory containing H5 files
        patient_base: e.g., "P15"
        measurement_id: e.g., "M01" - to get the correct umbilicus coordinates
    
    Returns:
        perforators: List of dicts with 'id', 'x_mm', 'y_mm' (relative to umbilicus)
        focus_mm: Camera height in mm
        umb_x_px, umb_y_px: Umbilicus pixel coordinates
    """
    h5_path = os.path.join(h5_dir, f"{patient_base}.h5")
    
    if not os.path.exists(h5_path):
        print(f"  H5 not found: {h5_path}")
        return [], None, None, None
    
    perforators = []
    focus_mm = None
    umb_x_px = None
    umb_y_px = None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get focus_mm from root attributes
            if 'focus_mm' in f.attrs:
                focus_mm = float(f.attrs['focus_mm'])
            
            # Get perforators from /coordinates/perforators
            # Columns: id, x, y, score, notes
            if 'coordinates' in f and 'perforators' in f['coordinates']:
                perf_data = f['coordinates']['perforators'][:]
                for row in perf_data:
                    perf_id = row['id'].decode('utf-8') if isinstance(row['id'], bytes) else str(row['id'])
                    x_mm = float(row['x'])
                    y_mm = float(row['y'])
                    perforators.append({
                        'id': perf_id,
                        'x_mm': x_mm,
                        'y_mm': y_mm
                    })
            
            # Get umbilicus pixel coordinates from /coordinates/umbilicus
            # Columns: x, y, measurement_id
            full_meas_id = f"{patient_base}{measurement_id}"  # e.g., "P15M01"
            if 'coordinates' in f and 'umbilicus' in f['coordinates']:
                umb_data = f['coordinates']['umbilicus'][:]
                for row in umb_data:
                    meas = row['measurement_id'].decode('utf-8') if isinstance(row['measurement_id'], bytes) else str(row['measurement_id'])
                    if meas == full_meas_id:
                        umb_x_px = float(row['x'])
                        umb_y_px = float(row['y'])
                        break
                    
    except Exception as e:
        print(f"  Error loading perforators/umbilicus: {e}")
        import traceback
        traceback.print_exc()
    
    return perforators, focus_mm, umb_x_px, umb_y_px


def convert_mm_to_pixels(x_mm, y_mm, focus_mm, umb_x_px, umb_y_px, img_width=640, img_height=480):
    """
    Convert mm coordinates (relative to umbilicus) to pixel coordinates.
    
    Uses camera model with alpha angles (same as threshold_5x5cm_umbilicus.py).
    """
    alpha_x_deg = 12.5
    alpha_y_deg = 9.5
    
    # Calculate mm per pixel
    x_fov_mm = focus_mm * np.tan(np.radians(alpha_x_deg))
    y_fov_mm = focus_mm * np.tan(np.radians(alpha_y_deg))
    
    mm_per_px_x = 2 * x_fov_mm / img_width
    mm_per_px_y = 2 * y_fov_mm / img_height
    
    # Convert mm to pixel offset
    px_offset_x = x_mm / mm_per_px_x
    px_offset_y = -y_mm / mm_per_px_y  # Negative because image y is inverted
    
    # Add to umbilicus position
    px_x = umb_x_px + px_offset_x
    px_y = umb_y_px + px_offset_y
    
    return px_x, px_y


def cluster_hotspots_to_perforators(segment_centroids, perforator_pixels, max_distance_px=30):
    """
    Cluster detected hotspots to nearest CT perforator.
    
    Args:
        segment_centroids: List of (x, y) centroids of detected regions
        perforator_pixels: List of (x, y, id) CT perforator positions in pixels
        max_distance_px: Maximum distance to associate a hotspot with a perforator
    
    Returns:
        clusters: Dict mapping perforator_id -> list of associated hotspot centroids
        unassigned: List of hotspot centroids not near any perforator
    """
    clusters = {p[2]: [] for p in perforator_pixels}
    unassigned = []
    
    for centroid in segment_centroids:
        cx, cy = centroid
        min_dist = float('inf')
        nearest_perf = None
        
        for px, py, pid in perforator_pixels:
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_perf = pid
        
        if min_dist <= max_distance_px and nearest_perf is not None:
            clusters[nearest_perf].append(centroid)
        else:
            unassigned.append(centroid)
    
    return clusters, unassigned

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
    
    # Create stacked maps at different agreement levels
    # stability values: 0.25 = 1/4, 0.5 = 2/4, 0.75 = 3/4, 1.0 = 4/4
    stacked_2of4 = stacked * (stability >= 0.5)   # At least 2 agree (50%)
    stacked_3of4 = stacked * (stability >= 0.75)  # At least 3 agree (75%)
    stacked_4of4 = stacked * (stability >= 1.0)   # All 4 agree (100%)
    
    # Segmentation at ≥2/4 agreement level (no validation - keep all detections)
    _, segment_2of4 = extract_structure(stacked_2of4, combined_mask)
    
    # Use ≥2/4 directly as final segmentation
    from scipy import ndimage
    segment_validated = (segment_2of4 > 0).astype(np.uint8)
    
    # Load CT perforator coordinates and umbilicus from H5
    perforators, focus_mm, umb_x_px, umb_y_px = load_ct_perforators_and_umbilicus(
        config.H5_DIR, patient_base, measurement_id="M01"
    )
    
    # Convert perforators to pixel coordinates
    perforator_pixels = []
    if perforators and focus_mm and umb_x_px is not None:
        h, w = hybrid_maps[0].shape
        for perf in perforators:
            px, py = convert_mm_to_pixels(
                perf['x_mm'], perf['y_mm'], 
                focus_mm, umb_x_px, umb_y_px,
                img_width=w, img_height=h
            )
            perforator_pixels.append((px, py, perf['id']))
        print(f"  Loaded {len(perforator_pixels)} CT perforators (focus={focus_mm:.1f}mm, umb=({umb_x_px:.1f},{umb_y_px:.1f}))")
    
    # Get centroids of validated segments for clustering
    segment_centroids = []
    labeled_validated, n_regions = ndimage.label(segment_validated)
    for region_id in range(1, n_regions + 1):
        region_mask = labeled_validated == region_id
        coords = np.argwhere(region_mask)
        if len(coords) > 0:
            cy, cx = coords.mean(axis=0)
            segment_centroids.append((cx, cy))
    
    # Cluster hotspots to perforators
    clusters, unassigned = cluster_hotspots_to_perforators(
        segment_centroids, perforator_pixels, max_distance_px=40
    )
    
    return {
        'single': hybrid_maps[0],
        'stacked': stacked,
        'stability': stability,
        'smart_map': smart_map,
        'segment_validated': segment_validated,
        'perforator_pixels': perforator_pixels,
        'segment_centroids': segment_centroids,
        'clusters': clusters,
        'unassigned': unassigned,
        'mask': combined_mask,
        'union_mask': union_mask,
        'n_used': n_used
    }

def create_comparison_figure(patients_data, patient_names):
    """
    Create a refined comparison figure for patients with CT perforators.
    Columns: Input + CT | Segmentation + CT | Clustering
    """
    # Filter to only patients with CT perforators
    filtered_data = []
    filtered_names = []
    for data, name in zip(patients_data, patient_names):
        if data is not None and len(data.get('perforator_pixels', [])) > 0:
            filtered_data.append(data)
            filtered_names.append(name)
    
    if len(filtered_data) == 0:
        print("No patients with CT perforator data found!")
        return
    
    n_patients = len(filtered_data)
    print(f"\nCreating figure for {n_patients} patients with CT data...")
    
    n_cols = 3
    fig, axes = plt.subplots(n_patients, n_cols, figsize=(5*n_cols, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    # Distinct colors for perforators
    colors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653', '#F4A261', '#9B5DE5', '#00BBF9', '#00F5D4']
    
    for i, (data, name) in enumerate(zip(filtered_data, filtered_names)):
        union_mask = data['union_mask']
        perforator_pixels = data.get('perforator_pixels', [])
        segment_centroids = data.get('segment_centroids', [])
        clusters = data.get('clusters', {})
        unassigned = data.get('unassigned', [])
        seg_validated = data['segment_validated'] > 0
        
        smart = data['smart_map']
        vmax_s = np.percentile(smart[union_mask & (smart > 0)], 99.5) if np.any(smart > 0) else 1
        
        # Column 0: Input with CT perforators marked
        axes[i, 0].imshow(smart, cmap='magma', vmin=0, vmax=vmax_s)
        
        for j, (px, py, pid) in enumerate(perforator_pixels):
            color = colors[j % len(colors)]
            # Small filled circle for CT perforator
            axes[i, 0].plot(px, py, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            # Label with offset
            axes[i, 0].annotate(pid, (px, py), xytext=(6, -6), textcoords='offset points',
                               color='white', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.15', fc=color, ec='white', lw=0.5, alpha=0.9))
        
        axes[i, 0].set_title(f"{name}: {len(perforator_pixels)} CT perforators", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 1: Segmentation with CT perforators overlaid
        # Create RGB overlay
        seg_rgb = np.zeros((*seg_validated.shape, 3), dtype=np.float32)
        seg_rgb[seg_validated] = [0.3, 1.0, 0.3]  # Green for detected hotspots
        
        axes[i, 1].imshow(smart, cmap='gray', vmin=0, vmax=vmax_s, alpha=0.4)
        axes[i, 1].imshow(seg_rgb, alpha=0.6)
        
        # Draw CT perforators
        for j, (px, py, pid) in enumerate(perforator_pixels):
            color = colors[j % len(colors)]
            axes[i, 1].plot(px, py, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
        
        axes[i, 1].set_title(f"Segmentation ({len(segment_centroids)} hotspots)", fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 2: Clustering visualization
        axes[i, 2].imshow(smart, cmap='magma', vmin=0, vmax=vmax_s, alpha=0.4)
        
        # Draw clusters - lines from CT perforator to associated hotspots
        cluster_stats = []
        for j, (px, py, pid) in enumerate(perforator_pixels):
            color = colors[j % len(colors)]
            associated = clusters.get(pid, [])
            
            # Draw lines to associated hotspots
            for cx, cy in associated:
                axes[i, 2].plot([px, cx], [py, cy], '-', color=color, linewidth=1.2, alpha=0.8)
                axes[i, 2].plot(cx, cy, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
            
            # Draw CT perforator (same size as hotspots but with thicker edge)
            axes[i, 2].plot(px, py, 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=2)
            axes[i, 2].annotate(pid, (px, py), xytext=(5, -5), textcoords='offset points',
                               color='white', fontsize=7, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.1', fc=color, ec='none', alpha=0.85))
            
            cluster_stats.append(f"{pid}:{len(associated)}")
        
        # Draw unassigned hotspots
        for cx, cy in unassigned:
            axes[i, 2].plot(cx, cy, 'o', color='#888888', markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        
        # Title with cluster counts
        stats_str = " | ".join(cluster_stats)
        unassigned_str = f" | ?:{len(unassigned)}" if unassigned else ""
        axes[i, 2].set_title(f"Clusters: {stats_str}{unassigned_str}", fontsize=9)
        axes[i, 2].axis('off')
    
    plt.suptitle("CT Perforator Validation & Clustering", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "Comparison_CT_Clustering.png")
    plt.savefig(output_file, dpi=250, bbox_inches='tight')
    print(f"Saved comparison to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Process patients P07-P25 (will filter to those with CT perforator data)
    patient_list = [f"P{i:02d}" for i in range(7, 26)]  # P07 to P25
    
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

