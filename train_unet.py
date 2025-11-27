"""
Train a lightweight U-Net for perforator segmentation.
Input: Smart Consensus map
Target: Overlap segmentation (consensus-validated)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import load_frames_h5, load_coco_mask
from src.bioheat import calculate_hybrid_preserving
from src.segmentation import extract_structure
from src.registration import estimate_affine_transform, align_image_affine, compute_optical_flow_warp
from scipy.ndimage import gaussian_filter
from scipy import ndimage


def filter_elongated_artifacts(rule_seg, intensity_map):
    """
    Filter elongated artifacts from rule-based segmentation.
    Keeps compact structures, removes long thin artifacts.
    """
    rule_binary = (rule_seg > 0).astype(np.uint8)
    
    # Label connected components
    labeled, num_features = ndimage.label(rule_binary)
    filtered = np.zeros_like(rule_binary)
    
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        region_area = np.sum(region_mask)
        
        if region_area < 5:
            continue  # Skip tiny regions
        
        # Calculate aspect ratio
        coords = np.argwhere(region_mask)
        if len(coords) > 5:
            try:
                (_, _), (w, h), _ = cv2.minAreaRect(coords.astype(np.float32))
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            except:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0
        
        # Check intensity
        region_intensity = intensity_map[region_mask].max() if np.any(region_mask) else 0
        intensity_threshold = np.percentile(intensity_map[intensity_map > 0], 60) if np.any(intensity_map > 0) else 0
        is_bright = region_intensity > intensity_threshold
        
        # Decision: remove elongated artifacts
        is_elongated = aspect_ratio > 4.0
        
        if is_elongated and not is_bright:
            # Elongated and not bright - likely artifact
            pass
        elif aspect_ratio > 8.0:
            # Very elongated - likely artifact even if bright
            pass
        else:
            # Keep compact shapes and bright elongated ones
            filtered[region_mask] = 1
    
    return filtered


# Attention Gate module
class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gate signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Attended skip connection


# Attention U-Net architecture
class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Attention gates (gate_ch = upsampled channels, skip_ch = encoder channels)
        self.att3 = AttentionGate(128, 128, 64)   # up3 has 128ch, e3 has 128ch
        self.att2 = AttentionGate(64, 64, 32)     # up2 has 64ch, e2 has 64ch
        self.att1 = AttentionGate(32, 32, 16)     # up1 has 32ch, e1 has 32ch
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # Output
        self.out = nn.Conv2d(32, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with attention-gated skip connections
        up3 = self.up3(b)
        att3 = self.att3(up3, e3)  # Attention on skip connection
        d3 = self.dec3(torch.cat([up3, att3], dim=1))
        
        up2 = self.up2(d3)
        att2 = self.att2(up2, e2)
        d2 = self.dec2(torch.cat([up2, att2], dim=1))
        
        up1 = self.up1(d2)
        att1 = self.att1(up1, e1)
        d1 = self.dec1(torch.cat([up1, att1], dim=1))
        
        return torch.sigmoid(self.out(d1))


class PerforatorDataset(Dataset):
    def __init__(self, data_list, img_size=256, augment=True):
        self.data = data_list
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        # With augmentation, we effectively have 4x more data
        return len(self.data) * (4 if self.augment else 1)
    
    def __getitem__(self, idx):
        # Get base index and augmentation type
        base_idx = idx % len(self.data)
        aug_type = idx // len(self.data) if self.augment else 0
        
        item = self.data[base_idx]
        
        # Resize to fixed size
        img = cv2.resize(item['input'].astype(np.float32), (self.img_size, self.img_size))
        target = cv2.resize(item['target'].astype(np.float32), (self.img_size, self.img_size))
        
        # Apply augmentation
        if aug_type == 1:  # Horizontal flip
            img = np.fliplr(img).copy()
            target = np.fliplr(target).copy()
        elif aug_type == 2:  # Vertical flip
            img = np.flipud(img).copy()
            target = np.flipud(target).copy()
        elif aug_type == 3:  # Both flips (180° rotation)
            img = np.flipud(np.fliplr(img)).copy()
            target = np.flipud(np.fliplr(target)).copy()
        
        # Normalize input
        if img.max() > 0:
            img = img / img.max()
        
        # Binarize target
        target = (target > 0).astype(np.float32)
        
        # Add channel dimension
        img = img[np.newaxis, :, :].astype(np.float32)
        target = target[np.newaxis, :, :].astype(np.float32)
        
        return torch.from_numpy(img), torch.from_numpy(target)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for better segmentation."""
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        # BCE loss
        bce_loss = self.bce(pred, target)
        
        # Dice loss
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss


def generate_training_pairs():
    """Generate input/target pairs from all patients."""
    patients = [f"P{i:02d}" for i in range(9, 26)]  # P09 to P25 (17 patients)
    data_pairs = []
    
    for patient_base in patients:
        print(f"Processing {patient_base}...")
        measurements = [f"{patient_base}M0{i}" for i in range(1, 5)]
        
        hybrid_maps = []
        masks = []
        
        for pid in measurements:
            frames, dims = load_frames_h5(config.H5_DIR, pid, config.MAX_FRAMES)
            if frames is None:
                continue
            
            h, w = frames.shape[1], frames.shape[2]
            mask = load_coco_mask(config.COCO_PATH, pid, h, w)
            
            hybrid_map, _, _ = calculate_hybrid_preserving(frames, mask, config)
            hybrid_maps.append(hybrid_map)
            masks.append(mask)
        
        if len(hybrid_maps) < 2:
            continue
        
        # Register measurements
        blur_sigma = 5.0
        blurred_refs = [gaussian_filter(h, sigma=blur_sigma) for h in hybrid_maps]
        
        aligned_maps = [hybrid_maps[0].copy()]
        
        for i in range(1, len(blurred_refs)):
            combined_mask = masks[0] & masks[i]
            warp_matrix, dx, dy, angle, sx, sy = estimate_affine_transform(
                blurred_refs[0], blurred_refs[i], mask=combined_mask, use_full_affine=True
            )
            
            if np.sqrt(dy**2 + dx**2) <= 50:
                affine_aligned = align_image_affine(hybrid_maps[i], warp_matrix)
                final_aligned, _ = compute_optical_flow_warp(
                    hybrid_maps[0], affine_aligned, mask=combined_mask, fine_mode=True
                )
                aligned_maps.append(final_aligned)
        
        if len(aligned_maps) < 2:
            continue
        
        # Compute intensity-gated union GT (improved approach)
        stacked = np.mean(aligned_maps, axis=0)
        max_map = np.max(aligned_maps, axis=0)  # MAX for intensity check
        
        binary_maps = [(m > np.percentile(m[m > 0], 80) if np.any(m > 0) else m > 0) for m in aligned_maps]
        stability = np.mean(binary_maps, axis=0)
        
        # Combined mask
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask & m
        
        # Smart consensus map
        smart_map = stacked * (stability >= 0.25)
        
        # Get rule-based segmentation (high detail)
        _, rule_seg = extract_structure(smart_map, combined_mask, use_ml_filter=False, apply_shape_filter=False)
        
        # Filter elongated artifacts from rule-based
        target_seg = filter_elongated_artifacts(rule_seg, smart_map)
        
        # For each single measurement, create a training pair
        for i, hmap in enumerate(hybrid_maps):
            data_pairs.append({
                'input': hmap * masks[i],  # Single measurement
                'target': target_seg,       # Intensity-gated union segmentation
                'patient': patient_base,
                'measurement': i
            })
        
        # Also add the stacked map as input
        data_pairs.append({
            'input': stacked * combined_mask,
            'target': target_seg,
            'patient': patient_base,
            'measurement': 'stacked'
        })
    
    print(f"Generated {len(data_pairs)} training pairs")
    return data_pairs


def train_unet(data_pairs, epochs=50, batch_size=4, lr=1e-3):
    """Train the U-Net model with augmentation and Dice+BCE loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset with augmentation
    dataset = PerforatorDataset(data_pairs, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Training samples: {len(dataset)} (with 4x augmentation)")
    
    # Initialize model
    model = UNetSmall().to(device)
    
    # Loss and optimizer
    criterion = DiceBCELoss(dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
    
    return model


def save_training_pairs(data_pairs, path):
    """Save training pairs to disk for faster loading."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data_pairs, f)
    print(f"Saved {len(data_pairs)} training pairs to {path}")

def load_training_pairs(path):
    """Load training pairs from disk."""
    import pickle
    with open(path, 'rb') as f:
        data_pairs = pickle.load(f)
    print(f"Loaded {len(data_pairs)} training pairs from {path}")
    return data_pairs

def main():
    pairs_path = os.path.join(config.OUTPUT_DIR, 'training_pairs_unet.pkl')
    
    # Check if pairs already exist
    if os.path.exists(pairs_path):
        print("Loading existing training pairs...")
        data_pairs = load_training_pairs(pairs_path)
    else:
        print("Generating training pairs (this takes a few minutes)...")
        data_pairs = generate_training_pairs()
        save_training_pairs(data_pairs, pairs_path)
    
    if len(data_pairs) < 10:
        print("Not enough training data!")
        return
    
    print(f"\nTraining U-Net on {len(data_pairs)} samples (x4 with augmentation)...")
    model = train_unet(data_pairs, epochs=200, batch_size=8, lr=5e-4)  # More epochs, lower LR
    
    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, 'unet_perforator.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

