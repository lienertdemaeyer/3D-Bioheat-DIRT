import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure

def get_all_stages(pid):
    """
    Returns (original_gray, bioheat_raw, hybrid_v4, segmentation, mask) for a patient.
    """
    print(f"Processing {pid}...")
    
    frames, dims = load_frames_h5(config.H5_DIR, pid, config.MAX_FRAMES)
    if frames is None: 
        return None, None, None, None, None
        
    h, w = frames.shape[1], frames.shape[2]
    mask = load_coco_mask(config.COCO_PATH, pid, h, w)
    
    # 1. Original Grayscale (mean thermal frame)
    mean_thermal = np.mean(frames[15:45], axis=0)
    p2, p98 = np.percentile(mean_thermal, (2, 98))
    original_gray = np.clip((mean_thermal - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # 2. Calculate Hybrid Maps
    hybrid_v4, weight_map, bioheat_raw = calculate_hybrid_preserving(frames, mask, config)
    
    # 3. Segmentation
    tophat_map, segment_map = extract_structure(hybrid_v4, mask)
    
    return original_gray, bioheat_raw, hybrid_v4, segment_map, mask

def main():
    patients = ["P15M04", "P17M03"]
    column_labels = ["(a) Input", "(b) Bioheat Source", "(c) Hybrid Fusion", "(d) Segmentation"]
    
    # Create figure with GridSpec for precise control
    fig = plt.figure(figsize=(14, 7))
    
    # GridSpec: 2 rows for images
    # wspace=0 (columns touch), hspace=0.05 (gap between patients)
    gs = gridspec.GridSpec(2, 4, figure=fig, 
                           wspace=0.0, hspace=0.05,
                           left=0.02, right=1.0, bottom=0.0, top=0.95)
    
    for i, pid in enumerate(patients):
        original_gray, bioheat_raw, hybrid_v4, segment_map, mask = get_all_stages(pid)
        
        if original_gray is not None:
            # Scaling
            bio_vmax = np.percentile(bioheat_raw[mask & (bioheat_raw > 0)], 99.5) if np.any(bioheat_raw > 0) else 1.0
            hyb_vmax = np.percentile(hybrid_v4[mask & (hybrid_v4 > 0)], 99.5) if np.any(hybrid_v4 > 0) else 1.0
            
            images = [
                (original_gray, 'gray', 0, 1),
                (bioheat_raw, 'magma', 0, bio_vmax),
                (hybrid_v4, 'magma', 0, hyb_vmax),
                (segment_map, 'gray', 0, 255)
            ]
            
            for j, (img, cmap, vmin, vmax) in enumerate(images):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                
                # Add column labels only on first row
                if i == 0:
                    ax.set_title(column_labels[j], fontsize=11, fontweight='normal', pad=4)
                
    # Add row labels on the left side, moved closer to images
    fig.text(0.008, 0.72, 'Patient A', fontsize=10, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.008, 0.25, 'Patient B', fontsize=10, fontweight='bold', 
             rotation=90, va='center', ha='center')
    
    output_file = os.path.join(config.OUTPUT_DIR, "Paper_Figure_Pipeline.png")
    plt.savefig(output_file, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
