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
                    consensus_map=None, min_consensus_ratio=0.4, hard_max_aspect=15.0,
                    intensity_map=None, high_intensity_percentile=70,
                    min_width_px=4):
    """
    Filter segmentation to remove elongated stripe-like artifacts.
    
    Key insight: Real perforators appear CONSISTENTLY across measurements.
    Artifacts (creases/folds) move between measurements → low consensus.
    
    Decision logic:
    1. Blob-like shape (aspect < 4) → KEEP (real perforator shape)
    2. Elongated (aspect >= 4) but has CONSENSUS → KEEP (consistent across measurements)
    3. Elongated without consensus → REMOVE (artifact that moved)
    4. Very thin (< 4px) elongated → REMOVE (definitely crease)
    5. Extremely elongated (aspect > 15) thin → REMOVE (edge artifact)
    """
    binary_uint8 = (binary_map > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered = np.zeros_like(binary_map, dtype=np.uint8)
    
    # Calculate intensity thresholds
    peak_intensity_threshold = 0
    if intensity_map is not None:
        valid_intensities = intensity_map[intensity_map > 0]
        if len(valid_intensities) > 0:
            # Top 20% = very bright peak
            peak_intensity_threshold = np.percentile(valid_intensities, 80)
    
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
        
        # Create mask for this contour
        contour_mask = np.zeros_like(binary_map, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, -1)
        
        # Get consensus (how consistently this appears across measurements)
        has_good_consensus = False
        mean_consensus = 0
        if consensus_map is not None and np.sum(contour_mask) > 0:
            mean_consensus = np.mean(consensus_map[contour_mask > 0])
            # Require higher consensus for more elongated shapes
            required_consensus = min_consensus_ratio + 0.03 * max(0, aspect_ratio - 4)
            required_consensus = min(required_consensus, 0.7)  # Cap at 70%
            has_good_consensus = mean_consensus >= required_consensus
        
        # Get intensity features
        has_very_bright_peak = False
        if intensity_map is not None and np.sum(contour_mask) > 0:
            max_intensity = np.max(intensity_map[contour_mask > 0])
            has_very_bright_peak = max_intensity >= peak_intensity_threshold
        
        # Decision logic:
        
        # 1. Very thin elongated = definitely crease - REMOVE
        is_thin_crease = min_dim < min_width_px and aspect_ratio > 3.0
        if is_thin_crease:
            continue
        
        # 2. Extremely elongated thin = edge artifact - REMOVE
        is_extreme_artifact = aspect_ratio > hard_max_aspect and min_dim < 10
        if is_extreme_artifact:
            continue
        
        # 3. Blob-like shape (low aspect ratio) - KEEP
        is_blob = aspect_ratio < 4.0
        if is_blob:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
            continue
        
        # 4. Elongated (aspect >= 4) - ONLY keep if has consensus
        # This is the key: artifacts move between measurements, real structures don't
        is_elongated = aspect_ratio >= 4.0
        
        if is_elongated:
            # Must have consensus OR be very bright AND wide
            if has_good_consensus:
                cv2.drawContours(filtered, [contour], -1, 255, -1)
            elif has_very_bright_peak and min_dim >= 8:
                # Wide bright structures are likely real even without consensus
                cv2.drawContours(filtered, [contour], -1, 255, -1)
            # else: skip (artifact without consensus)
    
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
    # Check if there's any valid data
    valid_pixels = image[mask & (image > 0)]
    if len(valid_pixels) == 0:
        return np.zeros_like(image), np.zeros_like(image)
        
    # Normalize to 0-255 for morphology
    img_norm = image.copy()
    vmax = np.percentile(valid_pixels, 99.5)
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

