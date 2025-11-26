import h5py
import numpy as np
import json
import cv2
import os
import matplotlib.pyplot as plt

def load_frames_h5(h5_dir, patient_id, max_frames=100):
    """Loads thermal frames from H5 file."""
    base_id = patient_id[:3]
    h5_path = os.path.join(h5_dir, f"{base_id}.h5")
    
    if not os.path.exists(h5_path):
        print(f"H5 not found: {h5_path}")
        return None, None

    meas_id = patient_id[3:]
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'Measurements' in f and 'Cooling' in f['Measurements']:
                cooling = f['Measurements']['Cooling']
                if meas_id in cooling:
                    data = cooling[meas_id]['frames'][:]
                    limit = min(max_frames, data.shape[0])
                    frames = data[0:limit]
                    
                    # Convert to Celsius if needed (heuristic)
                    median_val = np.nanmedian(frames)
                    if median_val > 200: 
                        frames = frames - 273.15
                    elif median_val > 50: 
                        # Likely raw counts, normalize crudely (not ideal but consistent)
                        frames = (frames / median_val) * 30.0
                        
                    return frames, (frames.shape[1], frames.shape[2])
    except Exception as e:
        print(f"Error loading H5: {e}")
        
    return None, None

def load_coco_mask(coco_path, patient_id, h, w):
    """Loads binary mask from COCO annotations."""
    try:
        with open(coco_path, 'r') as f: 
            coco = json.load(f)
            
        img_id = None
        for img in coco['images']:
            if patient_id in img['file_name']:
                img_id = img['id']
                break
        
        if img_id is None:
            # Fallback: return full mask
            return np.ones((h, w), dtype=bool)
            
        mask = np.zeros((h, w), dtype=np.uint8)
        anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
        
        for a in anns:
            for seg in a['segmentation']:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
                
        return mask.astype(bool)
        
    except Exception as e:
        print(f"Error loading COCO mask: {e}")
        return np.ones((h, w), dtype=bool)

def save_plot(image, title, output_path, cmap='magma', vmin=None, vmax=None):
    """Helper to save a clean plot."""
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

