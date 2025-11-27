import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import (register_and_stack_measurements, compute_stability_map, 
                               compute_union_tiered_map, compute_smart_consensus_map)

# Load U-Net model if available
UNET_MODEL = None
def load_unet_model():
    global UNET_MODEL
    model_path = os.path.join(config.OUTPUT_DIR, 'unet_perforator.pth')
    if os.path.exists(model_path):
        from train_unet import UNetSmall
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        UNET_MODEL = UNetSmall().to(device)
        UNET_MODEL.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        UNET_MODEL.eval()
        print(f"Loaded Attention U-Net model from {model_path} ({device})")
        return True
    return False

def unet_predict(hybrid_map, mask, intensity_filter=True):
    """Apply U-Net to predict segmentation from hybrid map."""
    if UNET_MODEL is None:
        return None
    
    device = next(UNET_MODEL.parameters()).device
    
    # Normalize input
    masked = hybrid_map * mask
    if masked.max() > 0:
        normalized = masked / masked.max()
    else:
        normalized = masked
    
    # Resize to 256x256 for U-Net
    import cv2
    h, w = hybrid_map.shape
    resized = cv2.resize(normalized.astype(np.float32), (256, 256))
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(device)
        output = UNET_MODEL(input_tensor)
        pred = output.squeeze().cpu().numpy()
    
    # Resize back
    pred_full = cv2.resize(pred, (w, h))
    
    # Binary threshold
    binary = (pred_full > 0.35).astype(np.uint8)
    
    return binary


def ensemble_segmentation(unet_seg, rule_seg, hybrid_map):
    """
    Simple ensemble: Rule-based (high detail) minus artifacts.
    - Start with rule-based output (high resolution)
    - Remove elongated structures that U-Net didn't detect (artifacts)
    - Keep everything else
    """
    from scipy import ndimage
    import cv2
    
    if unet_seg is None:
        return rule_seg
    
    # Start with rule-based (high detail)
    rule_binary = (rule_seg > 0).astype(np.uint8)
    unet_binary = (unet_seg > 0).astype(np.uint8)
    
    # Label connected components in rule-based
    labeled, num_features = ndimage.label(rule_binary)
    filtered = np.zeros_like(rule_binary)
    
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        region_area = np.sum(region_mask)
        
        # Check if U-Net also detected this region
        overlap_with_unet = np.sum(unet_binary[region_mask]) / region_area if region_area > 0 else 0
        in_unet = overlap_with_unet > 0.2  # At least 20% overlap
        
        # Calculate aspect ratio for artifact detection
        coords = np.argwhere(region_mask)
        if len(coords) > 5:
            try:
                (_, _), (w, h), _ = cv2.minAreaRect(coords.astype(np.float32))
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            except:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0
        
        is_elongated = aspect_ratio > 4.0
        is_very_elongated = aspect_ratio > 8.0
        
        # Decision logic:
        # 1. Confirmed by U-Net → always keep
        # 2. Very elongated (>8) and NOT in U-Net → artifact, remove
        # 3. Moderately elongated (4-8) and NOT in U-Net → artifact, remove
        # 4. Compact shape → keep (likely real perforator)
        
        if in_unet:
            # U-Net confirms it - keep
            filtered[region_mask] = 1
        elif is_very_elongated:
            # Very long and U-Net didn't see it - definitely artifact
            pass  # Don't add
        elif is_elongated:
            # Moderately elongated and U-Net didn't see it - likely artifact
            pass  # Don't add
        else:
            # Compact shape - keep even if U-Net missed it
            filtered[region_mask] = 1
    
    return filtered

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
    
    # U-Net prediction (if model loaded)
    segment_unet = unet_predict(smart_map, union_mask)
    
    # Smart ensemble: U-Net + Rule-based + Intensity validation
    segment_ensemble = ensemble_segmentation(segment_unet, segment_smart, smart_map)
    
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
        'segment_unet': segment_unet,
        'segment_ensemble': segment_ensemble,
        'mask': combined_mask,
        'union_mask': union_mask,
        'n_used': n_used
    }

def create_comparison_figure(patients_data, patient_names):
    """
    Create a side-by-side comparison figure for multiple patients.
    Columns: Smart Consensus | Overlap Seg | Smart Seg | U-Net Seg | Comparison
    """
    n_patients = len(patients_data)
    has_unet = patients_data[0].get('segment_unet') is not None
    
    n_cols = 5 if has_unet else 4
    fig, axes = plt.subplots(n_patients, n_cols, figsize=(5*n_cols, 5*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    title = "U-Net Segmentation vs Rule-based" if has_unet else "Smart Consensus Segmentation"
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    for i, (data, name) in enumerate(zip(patients_data, patient_names)):
        if data is None:
            continue
            
        mask = data['mask']
        union_mask = data['union_mask']
        
        # Column 0: Smart Consensus input
        smart = data['smart_map']
        vmax_s = np.percentile(smart[union_mask & (smart > 0)], 99.5) if np.any(smart > 0) else 1
        axes[i, 0].imshow(smart, cmap='magma', vmin=0, vmax=vmax_s)
        axes[i, 0].set_title(f"{name}: Smart Consensus", fontsize=11)
        axes[i, 0].axis('off')
        
        # Column 1: Intensity-gated union segmentation (ground truth)
        seg_overlap = data['segment_stacked'] > 0
        axes[i, 1].imshow(seg_overlap, cmap='gray')
        axes[i, 1].set_title("Intensity-Gated GT", fontsize=11)
        axes[i, 1].axis('off')
        
        # Column 2: Rule-based segmentation
        seg_smart = data['segment_smart'] > 0
        axes[i, 2].imshow(seg_smart, cmap='gray')
        axes[i, 2].set_title("Rule-based Seg", fontsize=11)
        axes[i, 2].axis('off')
        
        if has_unet:
            # Column 3: Smart Ensemble (U-Net + Rule-based + Intensity)
            seg_ensemble = data['segment_ensemble'] > 0
            axes[i, 3].imshow(seg_ensemble, cmap='gray')
            axes[i, 3].set_title("Smart Ensemble", fontsize=11)
            axes[i, 3].axis('off')
            
            # Column 4: Comparison (GT vs Ensemble)
            overlay = np.zeros((*seg_overlap.shape, 3), dtype=np.uint8)
            # Blue = only in GT (missed by ensemble)
            overlay[seg_overlap & ~seg_ensemble] = [100, 100, 255]
            # Green = only in ensemble (recovered/extra)
            overlay[seg_ensemble & ~seg_overlap] = [0, 255, 100]
            # White = both agree
            overlay[seg_overlap & seg_ensemble] = [255, 255, 255]
            
            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title("White=match, Blue=missed, Green=recovered", fontsize=9)
            axes[i, 4].axis('off')
        else:
            # Column 3: Comparison (Overlap vs Smart)
            overlay = np.zeros((*seg_overlap.shape, 3), dtype=np.uint8)
            overlay[seg_overlap & ~seg_smart] = [100, 100, 255]
            overlay[seg_smart & ~seg_overlap] = [0, 255, 100]
            overlay[seg_overlap & seg_smart] = [255, 255, 255]
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("White=both, Green=smart recovered", fontsize=9)
            axes[i, 3].axis('off')
    
    plt.suptitle("Rule-based (detailed) + U-Net Artifact Filter", fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(config.OUTPUT_DIR, "Comparison_SmartEnsemble.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparison to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Load U-Net model first
    has_unet = load_unet_model()
    if not has_unet:
        print("No U-Net model found - run train_unet.py first!")
    
    # Test with patients
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

