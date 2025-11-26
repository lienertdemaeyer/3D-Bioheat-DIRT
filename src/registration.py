import numpy as np
import cv2
from scipy.ndimage import shift as ndshift, rotate as ndrotate

def phase_correlation_shift(reference, target, mask=None):
    """
    Compute sub-pixel translation shift between two images using Phase Correlation.
    If mask is provided, uses OpenCV's more robust implementation.
    Returns (dy, dx) - the shift needed to align target to reference.
    """
    # Convert to float32
    ref = reference.astype(np.float32)
    tgt = target.astype(np.float32)
    
    # If mask provided, zero out background and use center-weighted window
    if mask is not None:
        ref = ref * mask.astype(np.float32)
        tgt = tgt * mask.astype(np.float32)
    
    # Normalize (only non-zero pixels)
    ref_mean = np.mean(ref[ref > 0]) if np.any(ref > 0) else 0
    ref_std = np.std(ref[ref > 0]) if np.any(ref > 0) else 1
    tgt_mean = np.mean(tgt[tgt > 0]) if np.any(tgt > 0) else 0
    tgt_std = np.std(tgt[tgt > 0]) if np.any(tgt > 0) else 1
    
    ref = (ref - ref_mean) / (ref_std + 1e-8)
    tgt = (tgt - tgt_mean) / (tgt_std + 1e-8)
    
    # Apply Hanning window to reduce edge effects
    h, w = ref.shape
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)
    
    ref_windowed = ref * window
    tgt_windowed = tgt * window
    
    # FFT
    fft_ref = np.fft.fft2(ref_windowed)
    fft_tgt = np.fft.fft2(tgt_windowed)
    
    # Cross-power spectrum
    cross_power = (fft_ref * np.conj(fft_tgt)) / (np.abs(fft_ref * np.conj(fft_tgt)) + 1e-8)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power)
    correlation = np.abs(correlation)
    
    # Find peak
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Convert to shift (handle wrap-around)
    dy = peak_idx[0] if peak_idx[0] < h // 2 else peak_idx[0] - h
    dx = peak_idx[1] if peak_idx[1] < w // 2 else peak_idx[1] - w
    
    return float(dy), float(dx)

def align_image(image, dy, dx):
    """
    Apply sub-pixel shift to align image.
    """
    return ndshift(image, shift=(-dy, -dx), mode='constant', cval=0)

def estimate_affine_transform(reference, target, mask=None, use_full_affine=True):
    """
    Estimate transformation using ECC algorithm.
    
    If use_full_affine=True: Uses full affine (translation + rotation + scale + shear)
    If use_full_affine=False: Uses Euclidean (translation + rotation only)
    
    Returns the transformation matrix and the estimated parameters.
    """
    # Normalize images to 0-255 uint8 for OpenCV
    ref = reference.astype(np.float32)
    tgt = target.astype(np.float32)
    
    # Apply mask if provided
    if mask is not None:
        ref = ref * mask.astype(np.float32)
        tgt = tgt * mask.astype(np.float32)
    
    # Normalize to 0-1 range
    ref_max = np.max(ref) if np.max(ref) > 0 else 1
    tgt_max = np.max(tgt) if np.max(tgt) > 0 else 1
    ref = ref / ref_max
    tgt = tgt / tgt_max
    
    # Convert to uint8
    ref_uint8 = (ref * 255).astype(np.uint8)
    tgt_uint8 = (tgt * 255).astype(np.uint8)
    
    # Define the motion model
    if use_full_affine:
        # Full affine: translation + rotation + scale + shear (6 DOF)
        warp_mode = cv2.MOTION_AFFINE
    else:
        # Euclidean: translation + rotation only (3 DOF)
        warp_mode = cv2.MOTION_EUCLIDEAN
    
    # Initialize the transformation matrix (identity)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-7)
    
    try:
        # Run ECC algorithm with multiple pyramid levels for robustness
        _, warp_matrix = cv2.findTransformECC(
            ref_uint8, tgt_uint8, warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5
        )
        
        # Extract parameters from warp matrix
        dx = warp_matrix[0, 2]
        dy = warp_matrix[1, 2]
        
        # For affine, extract scale and rotation
        # [[a, b, tx], [c, d, ty]] where a=sx*cos(θ), b=-sy*sin(θ), c=sx*sin(θ), d=sy*cos(θ)
        sx = np.sqrt(warp_matrix[0, 0]**2 + warp_matrix[1, 0]**2)
        sy = np.sqrt(warp_matrix[0, 1]**2 + warp_matrix[1, 1]**2)
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        return warp_matrix, dx, dy, angle_deg, sx, sy
        
    except cv2.error as e:
        # ECC failed, return identity
        return warp_matrix, 0.0, 0.0, 0.0, 1.0, 1.0

def align_image_affine(image, warp_matrix):
    """
    Apply affine transformation to align image.
    """
    h, w = image.shape
    aligned = cv2.warpAffine(
        image.astype(np.float32), 
        warp_matrix, 
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return aligned


def compute_optical_flow_warp(reference, target, mask=None, fine_mode=False):
    """
    Compute dense optical flow from target to reference and return the warped target.
    Uses Farneback optical flow for sub-pixel accuracy deformable registration.
    
    fine_mode: Use smaller windows for better local accuracy on small features.
    """
    # Normalize to 0-255 uint8 for optical flow
    ref_norm = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tgt_norm = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Less blur for fine mode to preserve small features
    blur_size = (7, 7) if fine_mode else (15, 15)
    ref_blur = cv2.GaussianBlur(ref_norm, blur_size, 0)
    tgt_blur = cv2.GaussianBlur(tgt_norm, blur_size, 0)
    
    # Fine mode: smaller windows, more levels, more iterations for small features
    if fine_mode:
        flow = cv2.calcOpticalFlowFarneback(
            ref_blur, tgt_blur, None,
            pyr_scale=0.5,
            levels=7,        # More pyramid levels
            winsize=11,      # Smaller window = more local
            iterations=10,   # More iterations
            poly_n=5,        # Smaller neighborhood
            poly_sigma=1.1,
            flags=0
        )
    else:
        flow = cv2.calcOpticalFlowFarneback(
            ref_blur, tgt_blur, None,
            pyr_scale=0.5,
            levels=5,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )
    
    # Create remap coordinates
    h, w = reference.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Warp target to reference using flow
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    
    warped = cv2.remap(target, map_x, map_y, cv2.INTER_LINEAR, 
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Compute average flow magnitude
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    if mask is not None and np.any(mask):
        avg_flow = np.mean(flow_mag[mask])
    else:
        avg_flow = np.mean(flow_mag)
    
    return warped, avg_flow


def register_and_stack_measurements(hybrid_maps, reference_idx=0, max_shift=50, masks=None, use_affine=True):
    """
    Register multiple measurements to a reference and stack them.
    Excludes outliers with shifts > max_shift.
    
    Args:
        hybrid_maps: List of 2D arrays (one per measurement)
        reference_idx: Which measurement to use as reference (default: 0 = M01)
        max_shift: Maximum allowed shift (pixels). Measurements with larger shifts are excluded.
        masks: Optional list of masks (same length as hybrid_maps). If provided, uses mask for registration.
        use_affine: If True, use ECC affine registration (translation + rotation). If False, use phase correlation (translation only).
    
    Returns:
        stacked_map: Averaged aligned maps
        transforms: List of (dy, dx, angle) transforms applied
        aligned_maps: List of aligned maps (None for excluded)
        included: List of booleans indicating which measurements were included
    """
    n = len(hybrid_maps)
    if n == 0:
        return None, [], [], []
    
    reference = hybrid_maps[reference_idx]
    ref_mask = masks[reference_idx] if masks is not None else None
    
    aligned_maps = []
    transforms = []
    included = []
    
    for i, hmap in enumerate(hybrid_maps):
        if i == reference_idx:
            aligned_maps.append(hmap.copy())
            transforms.append((0.0, 0.0, 0.0))  # (dx, dy, angle)
            included.append(True)
        else:
            # Use intersection of masks for registration
            if masks is not None:
                combined_mask = ref_mask & masks[i]
            else:
                combined_mask = None
            
            if use_affine:
                # Use ECC for full affine registration (translation + rotation + scale + shear)
                warp_matrix, dx, dy, angle, sx, sy = estimate_affine_transform(
                    reference, hmap, mask=combined_mask, use_full_affine=True
                )
                shift_magnitude = np.sqrt(dy**2 + dx**2)
                transforms.append((dx, dy, angle, sx, sy))
                
                if shift_magnitude > max_shift:
                    aligned_maps.append(None)
                    included.append(False)
                else:
                    aligned = align_image_affine(hmap, warp_matrix)
                    aligned_maps.append(aligned)
                    included.append(True)
            else:
                # Use phase correlation (translation only)
                dy, dx = phase_correlation_shift(reference, hmap, mask=combined_mask)
                shift_magnitude = np.sqrt(dy**2 + dx**2)
                transforms.append((dx, dy, 0.0))
                
                if shift_magnitude > max_shift:
                    aligned_maps.append(None)
                    included.append(False)
                else:
                    aligned = align_image(hmap, dy, dx)
                    aligned_maps.append(aligned)
                    included.append(True)
    
    # Stack only included maps
    valid_maps = [m for m in aligned_maps if m is not None]
    if len(valid_maps) == 0:
        return None, transforms, aligned_maps, included
    
    stacked = np.mean(valid_maps, axis=0)
    
    return stacked, transforms, aligned_maps, included

def compute_union_tiered_map(aligned_maps, masks=None, threshold_percentile=80):
    """
    Compute a tiered map that preserves unique information from individual measurements.
    
    Returns:
        union_map: Combined map using union of all measurements
        confidence_map: 0-1 indicating how many measurements contain signal at each pixel
        tiered_map: Final map with intensity weighted by confidence
    """
    n = len(aligned_maps)
    if n == 0:
        return None, None, None
    
    # Align masks if provided
    h, w = aligned_maps[0].shape
    
    # Create union mask (any measurement covers this pixel)
    if masks is not None:
        union_mask = np.zeros((h, w), dtype=bool)
        for m in masks:
            if m is not None:
                union_mask = union_mask | m
    else:
        union_mask = np.ones((h, w), dtype=bool)
    
    # Compute confidence: how many measurements have signal at each pixel
    binary_maps = []
    for amap in aligned_maps:
        if amap is not None and np.max(amap) > 0:
            valid_pixels = amap[amap > 0]
            if len(valid_pixels) > 0:
                thresh = np.percentile(valid_pixels, threshold_percentile)
                binary = (amap > thresh).astype(np.float32)
            else:
                binary = np.zeros_like(amap)
        else:
            binary = np.zeros((h, w), dtype=np.float32)
        binary_maps.append(binary)
    
    # Confidence = fraction of measurements with signal
    confidence_map = np.mean(binary_maps, axis=0)
    
    # Union map: take max across all measurements (preserves unique detections)
    valid_maps = [m for m in aligned_maps if m is not None]
    union_map = np.max(valid_maps, axis=0)
    
    # Tiered map: weight by confidence
    # High confidence (>0.5): full intensity
    # Medium confidence (0.25-0.5): 75% intensity  
    # Low confidence (<0.25): 50% intensity (still visible but muted)
    weight = np.ones_like(confidence_map)
    weight[confidence_map < 0.5] = 0.75
    weight[confidence_map < 0.25] = 0.5
    
    tiered_map = union_map * weight * union_mask
    
    return union_map, confidence_map, tiered_map


def compute_smart_consensus_map(aligned_maps, masks=None, 
                                 consensus_threshold=0.5, 
                                 intensity_percentile=75,
                                 detection_percentile=80):
    """
    Smart Consensus: Combines consensus detections with intensity-validated unique detections.
    
    Logic:
    - If a pixel is detected in 50%+ of measurements → HIGH CONFIDENCE (include)
    - If a pixel is detected in <50% of measurements:
        - If its intensity is > 75th percentile → MEDIUM CONFIDENCE (strong unique, include)
        - If its intensity is < 75th percentile → EXCLUDE (weak unique, likely noise)
    
    Returns:
        smart_map: The final smart consensus map
        confidence_map: 3-level confidence (1.0=consensus, 0.66=strong unique, 0=excluded)
        stats: Dictionary with statistics
    """
    n = len(aligned_maps)
    if n == 0:
        return None, None, {}
    
    valid_maps = [m for m in aligned_maps if m is not None]
    n_valid = len(valid_maps)
    
    if n_valid == 0:
        return None, None, {}
    
    h, w = valid_maps[0].shape
    
    # Create union mask
    if masks is not None:
        union_mask = np.zeros((h, w), dtype=bool)
        for m in masks:
            if m is not None:
                union_mask = union_mask | m
    else:
        union_mask = np.ones((h, w), dtype=bool)
    
    # Step 1: Compute detection count per pixel
    binary_maps = []
    for amap in valid_maps:
        if np.max(amap) > 0:
            valid_pixels = amap[amap > 0]
            if len(valid_pixels) > 0:
                thresh = np.percentile(valid_pixels, detection_percentile)
                binary = (amap > thresh).astype(np.float32)
            else:
                binary = np.zeros_like(amap)
        else:
            binary = np.zeros((h, w), dtype=np.float32)
        binary_maps.append(binary)
    
    # Detection frequency (0 to 1)
    detection_freq = np.mean(binary_maps, axis=0)
    
    # Step 2: Compute max intensity across measurements
    max_intensity = np.max(valid_maps, axis=0)
    
    # Step 3: Compute intensity threshold (75th percentile of all positive values)
    all_positive = max_intensity[union_mask & (max_intensity > 0)]
    if len(all_positive) > 0:
        intensity_thresh = np.percentile(all_positive, intensity_percentile)
    else:
        intensity_thresh = 0
    
    # Step 4: Build smart consensus map
    smart_map = np.zeros((h, w), dtype=np.float32)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    
    # Consensus pixels (detected in 50%+ measurements) - HIGH CONFIDENCE
    consensus_mask = detection_freq >= consensus_threshold
    smart_map[consensus_mask] = max_intensity[consensus_mask]
    confidence_map[consensus_mask] = 1.0
    
    # Unique pixels (detected in <50%) - check intensity
    unique_mask = (detection_freq > 0) & (detection_freq < consensus_threshold)
    
    # Strong unique (high intensity) - MEDIUM CONFIDENCE
    strong_unique_mask = unique_mask & (max_intensity >= intensity_thresh)
    smart_map[strong_unique_mask] = max_intensity[strong_unique_mask]
    confidence_map[strong_unique_mask] = 0.66
    
    # Weak unique (low intensity) - EXCLUDED (left as 0)
    weak_unique_mask = unique_mask & (max_intensity < intensity_thresh)
    # These are excluded (remain 0)
    
    # Apply union mask
    smart_map = smart_map * union_mask
    confidence_map = confidence_map * union_mask
    
    # Statistics
    stats = {
        'n_consensus': np.sum(consensus_mask & union_mask),
        'n_strong_unique': np.sum(strong_unique_mask & union_mask),
        'n_weak_excluded': np.sum(weak_unique_mask & union_mask),
        'intensity_threshold': intensity_thresh
    }
    
    return smart_map, confidence_map, stats

def compute_stability_map(aligned_maps, threshold_percentile=80):
    """
    Compute a stability map showing how consistently each pixel is activated.
    Higher threshold_percentile = stricter definition of "active" pixel.
    
    Returns:
        stability_map: Values 0-1 indicating fraction of measurements with signal
    """
    n = len(aligned_maps)
    
    # Binarize each map based on its own threshold (high percentile = only strong signals)
    binary_maps = []
    for amap in aligned_maps:
        if np.max(amap) > 0:
            # Use high percentile to only capture true perforators, not noise
            valid_pixels = amap[amap > 0]
            if len(valid_pixels) > 0:
                thresh = np.percentile(valid_pixels, threshold_percentile)
                binary = (amap > thresh).astype(np.float32)
            else:
                binary = np.zeros_like(amap)
        else:
            binary = np.zeros_like(amap)
        binary_maps.append(binary)
    
    # Stability = fraction of measurements where pixel is active
    stability = np.mean(binary_maps, axis=0)
    
    return stability

