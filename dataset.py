# dataset.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from patchify import patchify

class FiveDataset(Dataset):
    def __init__(self, root, mode="train", patch_size=512):
        assert mode in {"train", "test"}, "Mode should be 'train' or 'test'"
        
        self.root = root
        self.mode = mode
        self.patch_size = patch_size
        
        self.image_dir = os.path.join(self.root, mode, "Original")
        self.mask_dir = os.path.join(self.root, mode, "Ground truth")
        self.filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # Pre-calculate all patch indices
        self.patch_indices = []
        for idx in range(len(self.filenames)):
            image_path = os.path.join(self.image_dir, self.filenames[idx])
            image = np.array(Image.open(image_path).convert("RGB"))
            h, w = image.shape[:2]
            num_h = (h//patch_size) 
            num_w = (w//patch_size)
            self.patch_indices.extend([(idx, i, j) for i in range(num_h) for j in range(num_w)])
    
    def __len__(self):
        return len(self.patch_indices)
    
    def _apply_clahe(self, image):
        """Apply CLAHE preprocessing to enhance vessel contrast"""
        green_channel = image[:, :, 1]  # Extract green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(green_channel)
    
    def _process_patch(self, img_array, is_mask=False):
        """Process image into patches"""
        # Calculate padding needed to make dimensions divisible by patch_size
        h, w = img_array.shape[:2]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        # Apply padding
        if len(img_array.shape) > 2 and not is_mask:
            # For RGB images
            padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            # For grayscale images or masks
            padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        if is_mask:
            if len(padded_img.shape) > 2:
                padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2GRAY)
            padded_img = (padded_img > 127).astype(np.float32)  # Binarize mask
            
        patches = patchify(padded_img, (self.patch_size, self.patch_size), step=self.patch_size)
        return patches
    
    def __getitem__(self, index):
        idx, i, j = self.patch_indices[index]
        filename = self.filenames[idx]
        
        # Load original data
        image = np.array(Image.open(os.path.join(self.image_dir, filename)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, filename)).convert("L"))
        
        # Image preprocessing
        processed_image = self._apply_clahe(image)
        image_patches = self._process_patch(processed_image)
        mask_patches = self._process_patch(mask, is_mask=True)
        
        # Extract patches
        img_patch = image_patches[i, j, :, :]
        mask_patch = mask_patches[i, j, :, :]
        
        # Convert to PyTorch Tensors
        img_tensor = torch.from_numpy(img_patch).float().unsqueeze(0) / 255.0  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)        # [1, H, W]
        
        return {
            "image": img_tensor,
            "mask": mask_tensor
        }

def load_dataset(data_path, mode, patch_size=512):
    return FiveDataset(
        root=data_path,
        mode=mode,
        patch_size=patch_size
    )

class CompleteImageDataset(Dataset):
    def __init__(self, root, mode="test"):
        assert mode in {"train", "test"}, "Mode should be 'train' or 'test'"
        
        self.root = root
        self.mode = mode
        
        self.image_dir = os.path.join(self.root, mode, "Original")
        self.mask_dir = os.path.join(self.root, mode, "Ground truth")
        self.filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.filenames)
    
    def _apply_clahe(self, image):
        """Apply CLAHE preprocessing to enhance vessel contrast"""
        green_channel = image[:, :, 1]  # Extract green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(green_channel)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load original data
        image = np.array(Image.open(os.path.join(self.image_dir, filename)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, filename)).convert("L"))
        
        # Image preprocessing
        processed_image = self._apply_clahe(image)
        mask = (mask > 127).astype(np.float32)  # Binarize mask
        
        # Convert to PyTorch Tensors
        img_tensor = torch.from_numpy(processed_image).float().unsqueeze(0) / 255.0  # [1, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)        # [1, H, W]
        
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": filename
        }

def load_complete_dataset(data_path, mode):
    return CompleteImageDataset(
        root=data_path,
        mode=mode
    )