# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import load_dataset  # Ensure dataset.py is in the same directory
from model import UNet, get_model  # Ensure model.py has the correct UNet definition

# Parse configuration parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=512, help='Input image size')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save path')
    return parser.parse_args()

# Custom IoU calculation function
def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# Training function
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Print input data shape and label min/max values
        #print(f"Images shape: {images.shape}")
        #print(f"Masks shape: {masks.shape}")
        #print(f"Masks min: {masks.min()}, max: {masks.max()}")
        
        # Forward pass
        outputs = model(images)
        print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")
        
        # Calculate loss
        loss = criterion(outputs, masks)
        if loss is None:
            print("Loss is None!")
        else:
            print(f"Batch Loss: {loss.item()}")
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        epoch_loss += loss.item()
        epoch_iou += calculate_iou(outputs, masks).item()
    
    return epoch_loss / len(train_loader), epoch_iou / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Print input data shape and label min/max values
            print(f"Images shape: {images.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Masks min: {masks.min()}, max: {masks.max()}")
            
            outputs = model(images)
            # print(f"Outputs min: {outputs.min()}, max: {outputs.max()}")
            
            loss = criterion(outputs, masks)
            if loss is None:
                print("Loss is None!")
            else:
                print(f"Batch Loss: {loss.item()}")
            
            epoch_loss += loss.item()
            epoch_iou += calculate_iou(outputs, masks).item()
    
    return epoch_loss / len(val_loader), epoch_iou / len(val_loader)

# Main function
def main():
    args = parse_args()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    train_dataset = load_dataset(args.data_path, 'train', args.patch_size)
    val_dataset = load_dataset(args.data_path, 'test', args.patch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = get_model('resnet34unet', in_channels=1, out_channels=1).to(device) # resnet34unet or unet
    model_name = type(model).__name__
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Try to load saved best model
    checkpoint_path = os.path.join(args.save_dir, f'{model_name}_best_model.pth')
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['iou']
        print(f"Successfully loaded saved model weights, continuing training from epoch {start_epoch} (best IoU: {best_iou:.4f})")
    except FileNotFoundError:
        print("No saved model found, starting training from scratch")
        start_epoch = 0
        best_iou = 0.0

    # Training records
    best_iou = 0.0
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_iou = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_iou = validate(model, device, val_loader, criterion)
        
        # Record metrics
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_iou_history.append(train_iou)
        val_iou_history.append(val_iou)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'iou': val_iou
            }, f'{args.save_dir}/{model_name}_best_model.pth')
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}')
        print('-'*50)
    
    # Save final model
    torch.save(model.state_dict(), f'{args.save_dir}/{model_name}_final_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train')
    plt.plot(val_loss_history, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_iou_history, label='Train')
    plt.plot(val_iou_history, label='Validation')
    plt.title('IoU Curve')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.savefig(f'{args.save_dir}/training_curves.png')
    plt.show()

if __name__ == '__main__':
    main()