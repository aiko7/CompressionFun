import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Copy the exact UNetAE architecture from your AzureML utils.py
class UNetAE(nn.Module):
    def __init__(self, use_resblocks=False):
        super().__init__()
        self.use_resblocks = use_resblocks
        print("Building UNetAE architecture")
        
        # Encoder - adjusted for 64x64 input
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()  # Output [-1, 1] to match normalization
        )
        
        print("UNetAE ready")
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # 64 channels
        e2 = self.enc2(e1)   # 128 channels
        
        # Decoder
        d2 = self.dec2(e2)   # 64 channels
        output = self.dec1(d2)  # 3 channels
        
        return output

# Transform to match training (exactly from AzureML setup)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)  # Normalize to [-1, 1]
])

# Inverse transform for visualization
def denormalize(tensor):
    """Convert from [-1, 1] back to [0, 1] for visualization"""
    return (tensor + 1.0) / 2.0

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = UNetAE(use_resblocks=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def inference_single_image(model, image_path, device='cuda'):
    """Run inference on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Apply same transform as training
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        reconstructed = model(input_tensor)
    
    # Convert back to PIL images for visualization
    original_resized = transforms.Resize((64, 64))(image)
    reconstructed_img = denormalize(reconstructed.cpu().squeeze(0))
    
    return original_resized, reconstructed_img

def visualize_reconstruction(original, reconstructed, save_path=None):
    """Visualize original vs reconstructed image"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original (64x64)')
    axes[0].axis('off')
    
    # Reconstructed image
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Difference
    original_tensor = transforms.ToTensor()(original)
    diff = torch.abs(original_tensor - reconstructed).mean(dim=0)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def batch_inference_from_folder(model, folder_path, output_dir, device='cuda', max_images=10):
    """Run inference on multiple images from a folder"""
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in image_extensions][:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        try:
            original, reconstructed = inference_single_image(model, img_path, device)
            
            # Save visualization
            save_path = output_dir / f"reconstruction_{i:03d}_{img_path.stem}.png"
            visualize_reconstruction(original, reconstructed, save_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def compute_reconstruction_loss(model, image_path, device='cuda'):
    """Compute reconstruction loss for an image"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(input_tensor)
        mse_loss = nn.MSELoss()(reconstructed, input_tensor).item()
        l1_loss = nn.L1Loss()(reconstructed, input_tensor).item()
    
    return mse_loss, l1_loss

def get_dataset_images(data_dir, class_id, split='val', num_images=5):
    """Get image paths from Tiny ImageNet dataset for a specific class"""
    images = []
    
    # Read class mapping from wnids.txt
    wnids_file = os.path.join(data_dir, 'wnids.txt')
    if os.path.exists(wnids_file):
        with open(wnids_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        if class_id < len(class_names):
            class_name = class_names[class_id]
            print(f"Class {class_id} corresponds to {class_name}")
        else:
            print(f"Error: Class {class_id} not found in dataset")
            return images
    else:
        print(f"Error: wnids.txt not found in {data_dir}")
        return images
    
    if split == 'val':
        # Your val structure: val/n02124075/images/ (not val/images/)
        val_class_dir = os.path.join(data_dir, 'val', class_name, 'images')
        print(f"Looking for validation images at: {val_class_dir}")
        
        if os.path.exists(val_class_dir):
            val_files = [f for f in os.listdir(val_class_dir) 
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in sorted(val_files)[:num_images]:
                images.append(os.path.join(val_class_dir, img_file))
            print(f"Found {len(images)} validation images for class {class_name}")
        else:
            print(f"Validation class directory not found: {val_class_dir}")
            # Fallback: try the annotations approach in case structure is different
            val_annotations = os.path.join(data_dir, 'val', 'val_annotations.txt')
            val_images_dir = os.path.join(data_dir, 'val', 'images')
            
            if os.path.exists(val_annotations) and os.path.exists(val_images_dir):
                print("Trying flat val/images/ structure with annotations...")
                count = 0
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            img_class = parts[1]
                            
                            if img_class == class_name:
                                img_path = os.path.join(val_images_dir, img_name)
                                if os.path.exists(img_path):
                                    images.append(img_path)
                                    count += 1
                                    if len(images) >= num_images:
                                        break
                print(f"Found {count} validation images via annotations")
    
    elif split == 'train':
        # Get training images for this class
        train_class_dir = os.path.join(data_dir, 'train', class_name, 'images')
        print(f"Looking for training images at: {train_class_dir}")
        
        if os.path.exists(train_class_dir):
            train_files = [f for f in os.listdir(train_class_dir) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in sorted(train_files)[:num_images]:
                images.append(os.path.join(train_class_dir, img_file))
            print(f"Found {len(images)} training images for class {class_name}")
        else:
            print(f"Training class directory not found: {train_class_dir}")
    
    elif split == 'test':
        # Get test images (these don't have class labels, so just random ones)
        test_dir = os.path.join(data_dir, 'test', 'images')
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_file in sorted(test_files)[:num_images]:
                images.append(os.path.join(test_dir, img_file))
            print(f"Found {len(images)} test images")
        else:
            print(f"Test directory not found: {test_dir}")
    
    return images

def main():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (modify these for your setup)
    DATA_DIR = "./tiny-imagenet-200"          # Path to your Tiny ImageNet dataset
    CHECKPOINT_PATH = "./ae_class0_final.pt"  # Path to your downloaded .pt file
    CLASS_ID = 0                              # Class ID that the model was trained on
    OUTPUT_DIR = "./inference_results/"
    
    print(f"Using device: {DEVICE}")
    
    # Verify dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset not found at {DATA_DIR}")
        print("Please download Tiny ImageNet dataset and update DATA_DIR")
        return
    
    # Load model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please download your trained model and update CHECKPOINT_PATH")
        return
    
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Test on validation images from the SAME class (should reconstruct well)
    print(f"\n=== Testing on VALIDATION images from Class {CLASS_ID} (same class) ===")
    val_images = get_dataset_images(DATA_DIR, CLASS_ID, 'val', num_images=5)
    if val_images:
        for i, img_path in enumerate(val_images):
            print(f"\nTesting validation image {i+1}: {os.path.basename(img_path)}")
            original, reconstructed = inference_single_image(model, img_path, DEVICE)
            
            # Save with descriptive name
            save_path = os.path.join(OUTPUT_DIR, f"val_class{CLASS_ID}_{i+1}.png")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            visualize_reconstruction(original, reconstructed, save_path)
            
            # Compute losses
            mse, l1 = compute_reconstruction_loss(model, img_path, DEVICE)
            print(f"Reconstruction losses - MSE: {mse:.6f}, L1: {l1:.6f}")
    else:
        print(f"No validation images found for class {CLASS_ID}")
    
    # Test on training images from the SAME class (should reconstruct very well)
    print(f"\n=== Testing on TRAINING images from Class {CLASS_ID} (same class) ===")
    train_images = get_dataset_images(DATA_DIR, CLASS_ID, 'train', num_images=3)
    if train_images:
        for i, img_path in enumerate(train_images):
            print(f"\nTesting training image {i+1}: {os.path.basename(img_path)}")
            original, reconstructed = inference_single_image(model, img_path, DEVICE)
            
            save_path = os.path.join(OUTPUT_DIR, f"train_class{CLASS_ID}_{i+1}.png")
            visualize_reconstruction(original, reconstructed, save_path)
            
            mse, l1 = compute_reconstruction_loss(model, img_path, DEVICE)
            print(f"Reconstruction losses - MSE: {mse:.6f}, L1: {l1:.6f}")
    
    # Test on images from a DIFFERENT class (expected(?) bad reconstruction due to class specificness)
    different_class = (CLASS_ID + 1) % 200  # Different class
    print(f"\n=== Testing on VALIDATION images from Class {different_class} (different class) ===")
    diff_images = get_dataset_images(DATA_DIR, different_class, 'val', num_images=3)
    if diff_images:
        for i, img_path in enumerate(diff_images):
            print(f"\nTesting different class image {i+1}: {os.path.basename(img_path)}")
            original, reconstructed = inference_single_image(model, img_path, DEVICE)
            
            save_path = os.path.join(OUTPUT_DIR, f"diff_class{different_class}_{i+1}.png")
            visualize_reconstruction(original, reconstructed, save_path)
            
            mse, l1 = compute_reconstruction_loss(model, img_path, DEVICE)
            print(f"Reconstruction losses - MSE: {mse:.6f}, L1: {l1:.6f}")
    
    # Test on test images (unlabeled, general performance)
    print(f"\n=== Testing on TEST images (unlabeled) ===")
    test_images = get_dataset_images(DATA_DIR, CLASS_ID, 'test', num_images=3)
    if test_images:
        for i, img_path in enumerate(test_images):
            print(f"\nTesting test image {i+1}: {os.path.basename(img_path)}")
            original, reconstructed = inference_single_image(model, img_path, DEVICE)
            
            save_path = os.path.join(OUTPUT_DIR, f"test_{i+1}.png")
            visualize_reconstruction(original, reconstructed, save_path)
            
            mse, l1 = compute_reconstruction_loss(model, img_path, DEVICE)
            print(f"Reconstruction losses - MSE: {mse:.6f}, L1: {l1:.6f}")
    
    print(f"\nInference complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()