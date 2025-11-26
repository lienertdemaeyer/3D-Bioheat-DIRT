import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add current dir to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure

def process_patient(pid):
    print(f"Processing {pid}...")
    
    # Load Data
    frames, dims = load_frames_h5(config.H5_DIR, pid, config.MAX_FRAMES)
    if frames is None:
        return
        
    h, w = frames.shape[1], frames.shape[2]
    mask = load_coco_mask(config.COCO_PATH, pid, h, w)
    
    # 1. Calculate Hybrid Bioheat Map (v4)
    hybrid_map, weight_map, bioheat_raw = calculate_hybrid_preserving(frames, mask, config)
    
    # 2. Perform Unsupervised Segmentation
    tophat_map, segment_map = extract_structure(hybrid_map, mask)
    
    # 3. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.suptitle(f"Automated Perforator Mapping: {pid}", fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    # Scale
    vmax = np.percentile(hybrid_map[mask & (hybrid_map > 0)], 99.5) if np.any(hybrid_map > 0) else 1.0
    
    # Plot A: Hybrid Map
    im0 = axes[0].imshow(hybrid_map, cmap='magma', vmin=0, vmax=vmax)
    axes[0].set_title("Hybrid Bioheat-Gradient Map", fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Plot B: Segmentation
    axes[1].imshow(segment_map, cmap='gray')
    axes[1].set_title("Unsupervised Segmentation\n(Top-Hat + Otsu)", fontsize=14)
    axes[1].axis('off')
    
    # Plot C: Overlay
    # Create RGB overlay: Green spots on original thermal background (mean frame)
    mean_thermal = np.mean(frames[15:45], axis=0)
    norm_thermal = (mean_thermal - np.min(mean_thermal)) / (np.max(mean_thermal) - np.min(mean_thermal) + 1e-8)
    overlay = cv2.cvtColor((norm_thermal * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Find contours of segmentation
    contours, _ = cv2.findContours(segment_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2) # Green contours
    
    axes[2].imshow(overlay)
    axes[2].set_title("Clinical Overlay", fontsize=14)
    axes[2].axis('off')
    
    # Plot D: Weight Map (Confidence)
    im3 = axes[3].imshow(weight_map, cmap='cividis', vmin=0, vmax=1)
    axes[3].set_title("Confidence Weight Map", fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(config.OUTPUT_DIR, f"{pid}_automated_mapping.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")
    plt.close()

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("Starting Automated Perforator Mapping Pipeline...")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    
    for pid in config.PATIENTS:
        try:
            process_patient(pid)
        except Exception as e:
            print(f"  Error processing {pid}: {e}")
            
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()

