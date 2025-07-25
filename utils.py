import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict

# Use environment variable for data path (AzureML will set this)
DATA_DIR = os.environ.get('AZUREML_DATA_DIR', './tiny-imagenet-200')

class UNetAE(nn.Module):
    def __init__(self, use_resblocks=False):
        super(UNetAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.layers = [3, 8, 15, 22]
    
    def forward(self, x, y):
        loss = 0
        for layer in self.layers:
            x = self.vgg[:layer](x)
            y = self.vgg[:layer](y)
            loss += self.criterion(x, y)
        return loss

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []
        self.class_to_idx = {}
        
        if split == 'train':
            # Train structure: train/n01443537/images/n01443537_0.JPEG
            train_dir = os.path.join(root_dir, 'train')
            class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            
            for idx, class_dir in enumerate(class_dirs):
                self.class_to_idx[class_dir] = idx
                class_path = os.path.join(train_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.endswith(('.JPEG', '.jpg', '.png')):
                            self.samples.append((os.path.join(class_path, img_file), idx))
        
        elif split == 'val':
            # Val structure: val/images/val_0.JPEG with val_annotations.txt
            val_dir = os.path.join(root_dir, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            # First, build class_to_idx from train directory
            train_dir = os.path.join(root_dir, 'train')
            if os.path.exists(train_dir):
                class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                for idx, class_dir in enumerate(class_dirs):
                    self.class_to_idx[class_dir] = idx
            
            # Read validation annotations
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            img_path = os.path.join(val_dir, 'images', img_name)
                            if os.path.exists(img_path):
                                class_idx = self.class_to_idx.get(class_name, 0)
                                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# These will be initialized in train.py with the correct data directory
trainset = None
valset = None

def initialize_datasets(data_dir):
    """Initialize datasets with the correct data directory"""
    global trainset, valset
    trainset = TinyImageNetDataset(root_dir=data_dir, split='train', transform=transform)
    valset = TinyImageNetDataset(root_dir=data_dir, split='val', transform=transform)
    return trainset, valset

def get_class_indices(dataset):
    """Get indices for each class in the dataset"""
    if dataset is None:
        return {}
    
    idxs = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        idxs[label].append(i)
    return dict(idxs)

def eval_ae(model, class_id, valset, class_indices, criterion_pixel, criterion_perc):
    """Evaluate autoencoder on validation set for a specific class"""
    model.eval()
    device = next(model.parameters()).device
    
    if class_id not in class_indices or len(class_indices[class_id]) == 0:
        print(f"No validation samples for class {class_id}")
        model.train()
        return
        
    val_indices = class_indices[class_id]
    total_pix, total_perc, count = 0, 0, 0
    
    with torch.no_grad():
        for idx in val_indices:
            img, _ = valset[idx]
            img = img.unsqueeze(0).to(device)
            recon = model(img)
            total_pix += criterion_pixel(recon, img).item()
            total_perc += criterion_perc(recon, img).item()
            count += 1
    
    if count > 0:
        avg_pix = total_pix / count
        avg_perc = total_perc / count
        print(f"Eval Class {class_id} - Avg Pixel Loss: {avg_pix:.4f}, Avg Perceptual Loss: {avg_perc:.4f}")
    
    model.train()