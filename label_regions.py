"""
Quick labeling tool for perforator vs artifact classification.
Shows each detected region, you press Y (keep) or N (remove).
Saves labels to JSON for training.
"""
import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure

def extract_region_features(contour, hybrid_map, consensus_map=None):
    """Extract features for a detected region."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    
    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        min_dim = min(w, h)
        max_dim = max(w, h)
    else:
        aspect_ratio = 1.0
        min_dim = max_dim = np.sqrt(area)
    
    # Intensity features
    mask = np.zeros(hybrid_map.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    
    if np.sum(mask) > 0:
        mean_intensity = np.mean(hybrid_map[mask > 0])
        max_intensity = np.max(hybrid_map[mask > 0])
    else:
        mean_intensity = max_intensity = 0
    
    # Consensus if available
    if consensus_map is not None and np.sum(mask) > 0:
        mean_consensus = np.mean(consensus_map[mask > 0])
    else:
        mean_consensus = 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'min_dim': min_dim,
        'max_dim': max_dim,
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity,
        'mean_consensus': mean_consensus
    }

def label_patient(patient_id, output_dir):
    """Interactive labeling for one patient."""
    print(f"\nLoading {patient_id}...")
    
    frames, dims = load_frames_h5(config.H5_DIR, patient_id, config.MAX_FRAMES)
    if frames is None:
        print(f"  No data for {patient_id}")
        return []
    
    h, w = frames.shape[1], frames.shape[2]
    mask = load_coco_mask(config.COCO_PATH, patient_id, h, w)
    
    hybrid_map, weight_map, bioheat_raw = calculate_hybrid_preserving(frames, mask, config)
    tophat, segmentation = extract_structure(hybrid_map, mask, apply_shape_filter=False)
    
    # Find contours
    seg_uint8 = (segmentation > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter tiny regions
    contours = [c for c in contours if cv2.contourArea(c) >= 15]
    
    print(f"  Found {len(contours)} regions to label")
    
    labels = []
    
    # Create figure for display
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.ion()  # Interactive mode
    
    for i, contour in enumerate(contours):
        # Get bounding box with padding
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        
        # Extract patches
        hybrid_patch = hybrid_map[y1:y2, x1:x2]
        seg_patch = segmentation[y1:y2, x1:x2]
        
        # Create overlay
        overlay = np.zeros((y2-y1, x2-x1, 3))
        overlay[:,:,0] = hybrid_patch / (hybrid_patch.max() + 1e-6)
        overlay[:,:,1] = hybrid_patch / (hybrid_patch.max() + 1e-6)
        overlay[:,:,2] = hybrid_patch / (hybrid_patch.max() + 1e-6)
        overlay[seg_patch > 0, 1] = 1  # Green overlay for segmentation
        
        # Extract features
        features = extract_region_features(contour, hybrid_map)
        
        # Display
        axes[0].clear()
        axes[0].imshow(hybrid_patch, cmap='magma')
        axes[0].set_title('Hybrid Map')
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(seg_patch, cmap='gray')
        axes[1].set_title('Segmentation')
        axes[1].axis('off')
        
        axes[2].clear()
        axes[2].imshow(overlay)
        axes[2].set_title(f'Region {i+1}/{len(contours)}\nAspect: {features["aspect_ratio"]:.1f}, Width: {features["min_dim"]:.1f}px')
        axes[2].axis('off')
        
        fig.suptitle(f'{patient_id} - Press Y (perforator) or N (artifact) or Q (quit)')
        plt.draw()
        plt.pause(0.01)
        
        # Wait for keypress
        while True:
            if plt.waitforbuttonpress(timeout=0.1):
                break
        
        # Get the key (this is a bit hacky but works)
        key = input(f"  [{i+1}/{len(contours)}] Y=perforator, N=artifact, S=skip: ").strip().lower()
        
        if key == 'q':
            break
        elif key == 'y':
            label = 1  # Perforator
        elif key == 'n':
            label = 0  # Artifact
        else:
            continue  # Skip
        
        # Save label with features
        labels.append({
            'patient': patient_id,
            'region_idx': i,
            'label': label,
            'features': features,
            'bbox': [int(x), int(y), int(bw), int(bh)]
        })
        
        print(f"    Labeled as {'PERFORATOR' if label == 1 else 'ARTIFACT'}")
    
    plt.close()
    return labels

def main():
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    labels_file = os.path.join(output_dir, 'region_labels.json')
    
    # Load existing labels if any
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            all_labels = json.load(f)
        print(f"Loaded {len(all_labels)} existing labels")
    else:
        all_labels = []
    
    # Label a few patients
    patients_to_label = ["P15M01", "P16M01", "P17M01", "P18M01"]
    
    for pid in patients_to_label:
        labels = label_patient(pid, output_dir)
        all_labels.extend(labels)
        
        # Save after each patient
        with open(labels_file, 'w') as f:
            json.dump(all_labels, f, indent=2)
        print(f"  Saved {len(all_labels)} total labels")
    
    print(f"\nDone! Labels saved to {labels_file}")
    print(f"Total: {len(all_labels)} labeled regions")
    print(f"  Perforators: {sum(1 for l in all_labels if l['label'] == 1)}")
    print(f"  Artifacts: {sum(1 for l in all_labels if l['label'] == 0)}")

if __name__ == "__main__":
    main()

