import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from big_local_architecture import UNetAE, DEVICE

DATA_DIR = "./tiny-imagenet-200"
VAL_DIR = os.path.join(DATA_DIR, "val")
BATCH_SIZE = 8
NUM_WORKERS = 8

tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

print("Loading validation dataset...")
valset = ImageFolder(root=VAL_DIR, transform=tf_val)

def get_class_indices(dataset):
    from collections import defaultdict
    idxs = defaultdict(list)
    for i, (_, lbl) in enumerate(dataset):
        idxs[lbl].append(i)
    return dict(idxs)

class_indices_test = get_class_indices(valset)

def visualize_from_checkpoint(class_id, checkpoint_path=None, num_images=BATCH_SIZE):
    """
    Load a trained autoencoder for a given class and visualize reconstructions.

    Args:
        class_id (int): Class label to visualize.
        checkpoint_path (str or None): Path to checkpoint file or directory. If None, uses 'models/ae_{class_id}.pt'.
        num_images (int): Number of images to sample and display.
    """
    if checkpoint_path is None:
        model_path = f"models/ae_{class_id}.pt"
    elif os.path.isdir(checkpoint_path):
        files = [f for f in os.listdir(checkpoint_path) if f.startswith(f"ae_{class_id}_ep") and f.endswith('.pt')]
        if not files:
            print(f"No checkpoint files found in {checkpoint_path} for class {class_id}")
            return
        epochs = [(int(f.split('_ep')[-1].split('.pt')[0]), f) for f in files]
        latest = max(epochs, key=lambda x: x[0])[1]
        model_path = os.path.join(checkpoint_path, latest)
    else:
        model_path = checkpoint_path

    print(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        print(f"Checkpoint not found at {model_path}")
        return
    model = UNetAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    indices = class_indices_test.get(class_id, [])
    if len(indices) == 0:
        print(f"No images found for class {class_id}")
        return
    subset = Subset(valset, indices)
    loader = DataLoader(subset, batch_size=num_images, shuffle=False, num_workers=NUM_WORKERS)
    images, _ = next(iter(loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        recon = model(images)

    images_disp = images.cpu() * 0.5 + 0.5
    recon_disp = recon.cpu() * 0.5 + 0.5

    plt.figure(figsize=(12, 3))
    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(images_disp[i].permute(1, 2, 0))
        plt.axis('off')
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(recon_disp[i].permute(1, 2, 0))
        plt.axis('off')
    plt.suptitle(f"Class {class_id} Reconstructions from {os.path.basename(model_path)}")
    plt.tight_layout()
    plt.savefig(f"reconstructions_class{class_id}.png")
    print(f"Saved plot as reconstructions_class{class_id}.png")

# Example usage
if __name__ == '__main__':
    # Visualize class 0 final model
    visualize_from_checkpoint(class_id=0)
    # Or visualize latest checkpoint in './checkpoints' for class 0
    # visualize_from_checkpoint(class_id=0, checkpoint_path='./checkpoints')
    # Or specify exact file
    # visualize_from_checkpoint(class_id=0, checkpoint_path='checkpoints/ae_0_ep50.pt')
