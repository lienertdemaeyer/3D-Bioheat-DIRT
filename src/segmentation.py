import cv2
import numpy as np

def extract_structure(image, mask):
    """
    Unsupervised Structural Segmentation.
    Method: Top-Hat Transform (Background Removal) + Otsu Thresholding.
    """
    if not np.any(image > 0):
        return np.zeros_like(image), np.zeros_like(image)
        
    # Normalize to 0-255 for morphology
    img_norm = image.copy()
    vmax = np.percentile(image[mask & (image > 0)], 99.5) if np.any(image > 0) else 1.0
    img_uint8 = np.clip((img_norm / vmax) * 255, 0, 255).astype(np.uint8)
    
    # 1. Top-Hat Transform
    # Kernel size 25x25 fits typical perforator size (approx 1cm at standard zoom)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(img_uint8, cv2.MORPH_TOPHAT, kernel)
    
    # 2. Otsu Thresholding
    # Only consider pixels inside the mask for threshold calculation
    if np.sum(mask) > 0:
        thresh_val, _ = cv2.threshold(tophat[mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh_val = 0
    
    # Generate Binary Map
    segmentation = np.zeros_like(img_uint8)
    segmentation[mask] = (tophat[mask] > thresh_val) * 255
    
    return tophat, segmentation

