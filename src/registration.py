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

def estimate_affine_transform(reference, target, mask=None):
    """
    Estimate affine transformation (translation + rotation + scale) using ECC algorithm.
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
    
    # Define the motion model (Euclidean = translation + rotation)
    warp_mode = cv2.MOTION_EUCLIDEAN
    
    # Initialize the transformation matrix (identity)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    
    try:
        # Run ECC algorithm
        _, warp_matrix = cv2.findTransformECC(
            ref_uint8, tgt_uint8, warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5
        )
        
        # Extract parameters from warp matrix
        # For Euclidean: [[cos(θ), -sin(θ), tx], [sin(θ), cos(θ), ty]]
        dx = warp_matrix[0, 2]
        dy = warp_matrix[1, 2]
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        return warp_matrix, dx, dy, angle_deg
        
    except cv2.error as e:
        # ECC failed, return identity
        return warp_matrix, 0.0, 0.0, 0.0

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
                # Use ECC for affine registration (translation + rotation)
                warp_matrix, dx, dy, angle = estimate_affine_transform(reference, hmap, mask=combined_mask)
                shift_magnitude = np.sqrt(dy**2 + dx**2)
                transforms.append((dx, dy, angle))
                
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

