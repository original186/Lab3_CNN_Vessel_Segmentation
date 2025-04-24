import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import UNet, get_model
from dataset import load_complete_dataset

def visualize_predictions(image, ground_truth, unet_pred, resnet_pred, filename, save_path=None):
    """
    Visualize the original image, ground truth, and predictions from both models
    """
    # Convert model outputs to numpy arrays and ensure correct dimensions
    unet_pred = torch.sigmoid(unet_pred).cpu().detach().numpy()[0, 0]
    resnet_pred = torch.sigmoid(resnet_pred).cpu().detach().numpy()[0, 0]
    image = image[0, 0].cpu().numpy()
    ground_truth = ground_truth[0, 0].cpu().numpy()

    # Set up the display grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Display original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display ground truth
    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Display UNet prediction
    axes[2].imshow(unet_pred, cmap='gray')
    axes[2].set_title("UNet Prediction")
    axes[2].axis('off')

    # Display ResNet34UNet prediction
    axes[3].imshow(resnet_pred, cmap='gray')
    axes[3].set_title("ResNet34UNet Prediction")
    axes[3].axis('off')

    plt.suptitle(f"File: {filename}")
    
    # Save or display the visualization
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_and_predict(model_name, device, image, checkpoint_path):
    """
    Load saved model checkpoint and make predictions
    """
    # Load model architecture
    model = get_model(model_name, in_channels=1, out_channels=1).to(device)
    model.eval()

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded checkpoint for {model_name} from {checkpoint_path}")

    # Move data to device
    image = image.to(device)

    # Get model predictions
    with torch.no_grad():
        output = model(image)

    return output

def main():
    # Configuration parameters
    data_path = './dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_dataset = load_complete_dataset(data_path, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Set checkpoint paths for both models
    unet_checkpoint_path = './checkpoints/UNet_best_model.pth'
    resnet_checkpoint_path = './checkpoints/ResNet34UNet_best_model.pth'
    
    # Evaluate 3 different complete images
    for i, batch in enumerate(test_loader):
        if i >= 3:  # Only process 3 images
            break
            
        images = batch['image']
        ground_truth = batch['mask']
        filename = batch['filename']

        # Get predictions from both models
        unet_output = load_and_predict('unet', device, images, unet_checkpoint_path)
        resnet_output = load_and_predict('resnet34unet', device, images, resnet_checkpoint_path)
        
        # Save visualization results
        save_path = f'./visualization_result_{i+1}.png'
        visualize_predictions(images, ground_truth, unet_output, resnet_output, filename[0], save_path=save_path)
        print(f"Saved visualization {i+1} to {save_path}")

if __name__ == '__main__':
    main()
