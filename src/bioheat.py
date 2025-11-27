import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, laplace


def erode_mask(mask, erosion_px=20):
    """
    Erode the mask to exclude boundary artifacts (water bag edge, creases).
    
    Args:
        mask: Boolean mask
        erosion_px: Number of pixels to erode from boundary
    
    Returns:
        Eroded mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px*2+1, erosion_px*2+1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def suppress_lines(image, line_length=30):
    """
    Suppress linear structures (blood vessels, creases) while preserving blob-like features.
    Uses morphological opening with a circular kernel - removes thin elongated structures.
    
    Args:
        image: Input intensity map
        line_length: Minimum length of lines to suppress
    
    Returns:
        Filtered image with lines suppressed
    """
    # Opening with circular kernel removes thin structures
    kernel_size = max(5, line_length // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Convert to uint8 for morphology
    if image.max() > 0:
        img_norm = (image / image.max() * 255).astype(np.uint8)
    else:
        return image
    
    # Opening removes thin structures smaller than kernel
    opened = cv2.morphologyEx(img_norm, cv2.MORPH_OPEN, kernel)
    
    # Convert back to original scale
    result = opened.astype(np.float64) / 255.0 * image.max()
    
    return result


def calculate_standard_bioheat(frames, mask, config):
    """
    Calculates Standard Bioheat Source (Physics-Based).
    S = Slope - alpha * Laplacian
    """
    start = config.START_FRAME
    end = config.END_FRAME
    
    limit = min(end, frames.shape[0])
    window = frames[start:limit]
    n = len(window)
    if n < 5: return np.zeros((frames.shape[1], frames.shape[2]))
    
    # Temporal Slope (Linear Regression)
    x = np.arange(n)
    mx = np.mean(x)
    dx = x - mx
    ssx = np.sum(dx**2)
    mw = np.mean(window, axis=0)
    slope = np.sum(dx[:, None, None] * (window - mw[None, :, :]), axis=0) / ssx
    
    # Spatial Laplacian (Diffusion)
    mean_frame = np.mean(window, axis=0)
    smooth_frame = gaussian_filter(mean_frame, sigma=config.SMOOTHING_SIGMA)
    
    # Physical Laplacian (d2T/dx2 + d2T/dy2)
    # Note: pixel_size is in mm, convert to m for SI units if alpha is in SI
    # Here we assume consistent units. If alpha is calibrated for mm, this is fine.
    # Typically laplace kernel is just sums, we divide by dx^2
    lap = laplace(smooth_frame) / ((config.PIXEL_SIZE_MM/1000.0)**2)
    
    source = slope - config.ALPHA * lap
    return source * mask

def calculate_3d_gradient_score(frames, mask, config):
    """
    Calculates 3D Gradient Perforator Score (Pattern-Based).
    Score = Temporal_Gradient * (-Spatial_Laplacian)
    """
    start = config.START_FRAME
    end = config.END_FRAME
    
    # Pre-smooth frames for robust gradients
    frames_smooth = np.zeros_like(frames)
    for t in range(frames.shape[0]):
        frames_smooth[t] = gaussian_filter(frames[t], sigma=2.0) # Fixed sigma=2.0 for features
    
    grad_t = np.gradient(frames_smooth, axis=0)
    
    spatial_curvature = np.zeros_like(frames_smooth)
    for t in range(frames.shape[0]):
        spatial_curvature[t] = -laplace(frames_smooth[t]) # Negative Laplacian = Peak
        
    # Integrate over window
    score_map = np.mean(grad_t[start:end] * spatial_curvature[start:end], axis=0)
    score_map = np.clip(score_map, 0, None)
    
    # Robust Normalization (99th percentile = 1.0)
    masked_positive = score_map[mask & (score_map > 0)]
    if len(masked_positive) > 0:
        robust_max = np.percentile(masked_positive, 99.0)
        score_map = score_map / (robust_max + 1e-8)
        score_map = np.clip(score_map, 0, 1.0)
    elif np.max(score_map) > 0:
        score_map = score_map / (np.max(score_map) + 1e-8)
        score_map = np.clip(score_map, 0, 1.0)
        
    return score_map * mask

def calculate_hybrid_preserving(frames, mask, config, boundary_erosion_px=10):
    """
    Hybrid v4: Intensity Preserving Method.
    Combines Physics (Bioheat) with Pattern Rec (Gradient Score).
    
    Args:
        boundary_erosion_px: Pixels to erode from mask boundary to remove edge artifacts.
                            Set to 0 to disable.
    """
    # Erode mask to exclude boundary artifacts (water bag edge, creases)
    if boundary_erosion_px > 0:
        eroded_mask = erode_mask(mask, erosion_px=boundary_erosion_px)
    else:
        eroded_mask = mask
    
    # 1. Get Physics-Based Source (using eroded mask)
    bioheat = calculate_standard_bioheat(frames, eroded_mask, config)
    
    # 2. Get Pattern-Based Confidence Score (0-1)
    score = calculate_3d_gradient_score(frames, eroded_mask, config)
    
    # 3. Apply Saturation Gain (Weighting)
    # Gain=2.0 means confidence > 0.5 yields full intensity
    weight = np.clip(score * 2.0, 0, 1.0)
    
    hybrid_map = bioheat * weight
    
    return hybrid_map, weight, bioheat

