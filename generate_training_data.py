"""
Generate training data using Overlap segmentation as ground truth.
- Features: extracted from single measurement detections
- Labels: 1 if region overlaps with consensus (Overlap), 0 otherwise
"""
import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import (register_and_stack_measurements, compute_stability_map,
                               compute_smart_consensus_map, estimate_affine_transform, 
                               align_image_affine, compute_optical_flow_warp)
from scipy.ndimage import gaussian_filter

def extract_region_features(contour, hybrid_map, mask):
    """Extract features for a detected region."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    
    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        w, h = rect[1]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        min_dim = min(w, h)
        max_dim = max(w, h)
    else:
        x, y, bw, bh = cv2.boundingRect(contour)
        center = (x + bw/2, y + bh/2)
        aspect_ratio = 1.0
        min_dim = max_dim = np.sqrt(area)
    
    # Create mask for this contour
    contour_mask = np.zeros(hybrid_map.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 1, -1)
    
    # Intensity features
    if np.sum(contour_mask) > 0:
        region_values = hybrid_map[contour_mask > 0]
        mean_intensity = np.mean(region_values)
        max_intensity = np.max(region_values)
        std_intensity = np.std(region_values)
    else:
        mean_intensity = max_intensity = std_intensity = 0
    
    # Normalized intensity (relative to image)
    if hybrid_map.max() > 0:
        norm_intensity = mean_intensity / hybrid_map.max()
    else:
        norm_intensity = 0
    
    # Distance from mask boundary (center vs edge)
    h_img, w_img = mask.shape
    cx, cy = center
    
    # Find distance to nearest mask edge
    dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if 0 <= int(cy) < h_img and 0 <= int(cx) < w_img:
        dist_to_edge = dist_transform[int(cy), int(cx)]
    else:
        dist_to_edge = 0
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'aspect_ratio': float(aspect_ratio),
        'min_dim': float(min_dim),
        'max_dim': float(max_dim),
        'mean_intensity': float(mean_intensity),
        'max_intensity': float(max_intensity),
        'std_intensity': float(std_intensity),
        'norm_intensity': float(norm_intensity),
        'dist_to_edge': float(dist_to_edge),
        'center_x': float(cx),
        'center_y': float(cy)
    }

def compute_overlap_for_patient(patient_base):
    """
    Compute overlap segmentation for a patient (4 measurements).
    Returns: single_map (M01), overlap_segmentation, all features with labels
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
        
        hybrid_map, _, _ = calculate_hybrid_preserving(frames, mask, config)
        hybrid_maps.append(hybrid_map)
        masks.append(mask)
        print(f"  {pid}: Loaded")
    
    if len(hybrid_maps) < 2:
        print("  Need at least 2 measurements")
        return None
    
    # Register measurements using blurred maps
    blur_sigma = 5.0
    blurred_refs = [gaussian_filter(h, sigma=blur_sigma) for h in hybrid_maps]
    
    reference_blurred = blurred_refs[0]
    reference_mask = masks[0]
    aligned_maps = [hybrid_maps[0].copy()]
    
    for i in range(1, len(blurred_refs)):
        combined_mask = reference_mask & masks[i]
        
        warp_matrix, dx, dy, angle, sx, sy = estimate_affine_transform(
            reference_blurred, blurred_refs[i], mask=combined_mask, use_full_affine=True
        )
        
        shift_magnitude = np.sqrt(dy**2 + dx**2)
        
        if shift_magnitude > 50:
            aligned_maps.append(None)
        else:
            affine_aligned = align_image_affine(hybrid_maps[i], warp_matrix)
            coarse_aligned, _ = compute_optical_flow_warp(
                hybrid_maps[0], affine_aligned, mask=combined_mask, fine_mode=False
            )
            final_aligned, _ = compute_optical_flow_warp(
                hybrid_maps[0], coarse_aligned, mask=combined_mask, fine_mode=True
            )
            aligned_maps.append(final_aligned)
    
    valid_aligned = [m for m in aligned_maps if m is not None]
    n_used = len(valid_aligned)
    print(f"  Using {n_used}/{len(aligned_maps)} measurements")
    
    if n_used < 2:
        return None
    
    # Compute OVERLAP: average of aligned maps, thresholded by stability
    stacked = np.mean(valid_aligned, axis=0)
    
    # Stability: how many measurements have signal at each pixel
    binary_maps = [(m > np.percentile(m[m > 0], 80) if np.any(m > 0) else m > 0) for m in valid_aligned]
    stability = np.mean(binary_maps, axis=0)
    
    # Overlap: only keep pixels with stability >= 0.5 (in at least half the measurements)
    overlap_map = stacked * (stability >= 0.5)
    
    # Combined mask
    combined_mask = masks[0].copy()
    for m in masks[1:]:
        combined_mask = combined_mask & m
    
    # Get overlap segmentation (ground truth)
    _, overlap_seg = extract_structure(overlap_map, combined_mask, apply_shape_filter=False)
    
    # Get single measurement segmentation (what we want to classify)
    _, single_seg = extract_structure(hybrid_maps[0], masks[0], apply_shape_filter=False)
    
    # Extract training data
    training_data = []
    
    # Find contours in single measurement
    single_uint8 = (single_seg > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(single_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter tiny regions
    contours = [c for c in contours if cv2.contourArea(c) >= 15]
    
    print(f"  Found {len(contours)} regions in single measurement")
    
    for i, contour in enumerate(contours):
        # Extract features
        features = extract_region_features(contour, hybrid_maps[0], masks[0])
        
        # Check if this region overlaps with ground truth (overlap segmentation)
        contour_mask = np.zeros(overlap_seg.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, -1)
        
        # Calculate overlap ratio
        overlap_pixels = np.sum((contour_mask > 0) & (overlap_seg > 0))
        total_pixels = np.sum(contour_mask > 0)
        overlap_ratio = overlap_pixels / (total_pixels + 1e-6)
        
        # Label: 1 if significant overlap with ground truth, 0 otherwise
        label = 1 if overlap_ratio > 0.3 else 0
        
        training_data.append({
            'patient': patient_base,
            'region_idx': i,
            'label': label,
            'overlap_ratio': float(overlap_ratio),
            'features': features
        })
    
    n_positive = sum(1 for d in training_data if d['label'] == 1)
    n_negative = len(training_data) - n_positive
    print(f"  Labels: {n_positive} perforators, {n_negative} artifacts")
    
    return training_data

def main():
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all patients
    patients = [f"P{i}" for i in range(15, 26)]
    
    all_training_data = []
    
    for patient_base in patients:
        data = compute_overlap_for_patient(patient_base)
        if data:
            all_training_data.extend(data)
    
    # Save training data
    output_file = os.path.join(output_dir, 'training_data_overlap.json')
    with open(output_file, 'w') as f:
        json.dump(all_training_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"{'='*50}")
    print(f"Total training samples: {len(all_training_data)}")
    print(f"  Perforators: {sum(1 for d in all_training_data if d['label'] == 1)}")
    print(f"  Artifacts: {sum(1 for d in all_training_data if d['label'] == 0)}")
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()

