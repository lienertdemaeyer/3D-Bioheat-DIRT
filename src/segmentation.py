import cv2
import numpy as np
import os
import pickle

# Global classifier (loaded once)
_classifier = None
_scaler = None
_feature_names = None

def load_classifier(model_path=None):
    """Load the trained perforator classifier."""
    global _classifier, _scaler, _feature_names
    
    if model_path is None:
        # Default path
        model_path = r"C:\Users\liene\Documents\DATA OUTPUT\3D-BIOHEAT-DIRT\perforator_classifier.pkl"
    
    if not os.path.exists(model_path):
        print(f"Warning: Classifier not found at {model_path}")
        return False
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    _classifier = data['classifier']
    _scaler = data['scaler']
    _feature_names = data['feature_names']
    return True

def extract_region_features(contour, hybrid_map, mask):
    """Extract features for ML classification."""
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
    
    # Normalized intensity
    norm_intensity = mean_intensity / (hybrid_map.max() + 1e-6) if hybrid_map.max() > 0 else 0
    
    # Distance to edge
    cx, cy = center
    h_img, w_img = mask.shape
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
        'dist_to_edge': float(dist_to_edge)
    }

def filter_by_ml(binary_map, hybrid_map, mask, threshold=0.5):
    """
    Filter segmentation using trained ML classifier.
    
    Args:
        binary_map: Binary segmentation mask
        hybrid_map: Intensity map for feature extraction
        mask: ROI mask
        threshold: Probability threshold (default 0.5, lower = keep more)
    
    Returns:
        Filtered binary map
    """
    global _classifier, _scaler, _feature_names
    
    # Load classifier if not loaded
    if _classifier is None:
        if not load_classifier():
            print("ML classifier not available, returning unfiltered")
            return binary_map
    
    binary_uint8 = (binary_map > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered = np.zeros_like(binary_map, dtype=np.uint8)
    
    for contour in contours:
        if cv2.contourArea(contour) < 15:
            continue
        
        # Extract features
        features = extract_region_features(contour, hybrid_map, mask)
        
        # Create feature vector in correct order
        X = [[features.get(fn, 0) for fn in _feature_names]]
        X_scaled = _scaler.transform(X)
        
        # Predict probability
        prob = _classifier.predict_proba(X_scaled)[0][1]  # Probability of being perforator
        
        if prob >= threshold:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
    
    return filtered


def filter_by_shape(binary_map, min_circularity=0.15, max_aspect_ratio=5.0, min_area=15,
                    consensus_map=None, min_consensus_ratio=0.5, hard_max_aspect=15.0,
                    intensity_map=None, high_intensity_percentile=70,
                    min_width_px=4):
    """
    Filter segmentation to remove elongated stripe-like artifacts.
    Keeps blob-like structures AND elongated structures if they have high consensus OR high intensity.
    
    Args:
        binary_map: Binary segmentation mask
        min_circularity: Minimum circularity for shapes without validation
        max_aspect_ratio: Maximum aspect ratio for shapes without validation
        min_area: Minimum area in pixels
        consensus_map: Optional map showing consensus values (0-1)
        min_consensus_ratio: Minimum consensus to keep elongated shapes
        hard_max_aspect: HARD LIMIT for very long thin shapes (edge artifacts)
        intensity_map: Optional intensity map - high intensity shapes are kept
        high_intensity_percentile: Percentile threshold for "high intensity"
    
    Returns:
        Filtered binary map
    """
    binary_uint8 = (binary_map > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered = np.zeros_like(binary_map, dtype=np.uint8)
    
    # Calculate intensity threshold if intensity map provided
    intensity_threshold = 0
    if intensity_map is not None:
        valid_intensities = intensity_map[binary_map > 0]
        if len(valid_intensities) > 0:
            intensity_threshold = np.percentile(valid_intensities, high_intensity_percentile)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Circularity: 1 for circle, approaches 0 for lines
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio and width from rotated bounding box
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            min_dim = min(w, h)  # Width of the structure
        else:
            aspect_ratio = 1.0
            min_dim = 10  # Default
        
        # THIN STRUCTURES are creases/folds, not perforators
        # Skip very thin elongated structures regardless of intensity
        if min_dim < min_width_px and aspect_ratio > 3.0:
            continue
        
        # Create mask for this contour (needed for consensus/intensity checks)
        contour_mask = np.zeros_like(binary_map, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, -1)
        
        # Check if HIGH INTENSITY (bright = real perforator, keep regardless of shape)
        is_high_intensity = False
        if intensity_map is not None and np.sum(contour_mask) > 0:
            mean_intensity = np.mean(intensity_map[contour_mask > 0])
            is_high_intensity = mean_intensity >= intensity_threshold
        
        # High intensity structures bypass shape filters (except extreme cases)
        if is_high_intensity and aspect_ratio <= hard_max_aspect:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
            continue
        
        # HARD LIMIT: Very long thin shapes are artifacts
        if aspect_ratio > hard_max_aspect:
            continue
        
        # Check consensus for moderately elongated shapes
        has_consensus = False
        if consensus_map is not None and aspect_ratio > max_aspect_ratio:
            if np.sum(contour_mask) > 0:
                mean_consensus = np.mean(consensus_map[contour_mask > 0])
                required_consensus = min_consensus_ratio + 0.05 * (aspect_ratio - max_aspect_ratio)
                has_consensus = mean_consensus >= min(required_consensus, 0.7)
        
        # Keep if: blob-like OR has consensus
        is_blob_like = circularity >= min_circularity and aspect_ratio <= max_aspect_ratio
        
        if is_blob_like or has_consensus:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
    
    return filtered


def extract_structure(image, mask, apply_shape_filter=True, consensus_map=None, use_ml_filter=False, ml_threshold=0.65):
    """
    Unsupervised Structural Segmentation.
    Method: Top-Hat Transform (Background Removal) + Otsu Thresholding.
    
    Args:
        consensus_map: Optional map showing consensus values (0-1). If provided,
                      elongated shapes with high consensus are kept (not artifacts).
        use_ml_filter: If True, use trained ML classifier to filter regions.
        ml_threshold: Probability threshold for ML filter (default 0.5, lower = keep more).
    """
    if not np.any(image > 0):
        return np.zeros_like(image), np.zeros_like(image)
        
    # Normalize to 0-255 for morphology
    img_norm = image.copy()
    vmax = np.percentile(image[mask & (image > 0)], 99.5) if np.any(image > 0) else 1.0
    img_uint8 = np.clip((img_norm / vmax) * 255, 0, 255).astype(np.uint8)
    
    # 1. Top-Hat Transform
    # Smaller kernel (15x15) preserves more detail
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    
    # 2. Adaptive Thresholding (softer than Otsu for more detail)
    # Use lower threshold to preserve weaker signals
    if np.sum(mask) > 0:
        otsu_val, _ = cv2.threshold(tophat[mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use 70% of Otsu threshold to keep more detail
        thresh_val = otsu_val * 0.7
    else:
        thresh_val = 0
    
    # Generate Binary Map
    segmentation = np.zeros_like(img_uint8)
    segmentation[mask] = (tophat[mask] > thresh_val) * 255
    
    # Apply filtering to remove artifacts
    if use_ml_filter:
        # Use trained ML classifier
        segmentation = filter_by_ml(segmentation, image, mask, threshold=ml_threshold)
        # ALSO apply hard rule: remove very thin or very long structures
        segmentation = filter_by_shape(segmentation, consensus_map=consensus_map, intensity_map=image,
                                        min_circularity=0.05, max_aspect_ratio=12.0, hard_max_aspect=15.0,
                                        min_width_px=3)
    elif apply_shape_filter:
        # Fallback to rule-based filter only
        segmentation = filter_by_shape(segmentation, consensus_map=consensus_map, intensity_map=image)
    
    return tophat, segmentation

