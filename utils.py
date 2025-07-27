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

class PerceptualLoss(nn.Module):
    def __init__(self, layers=('3','8','15','22')):
        super().__init__()
        print("Initializing perceptual loss (VGG with explicit weights)")
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters(): 
            p.requires_grad = False
        
        # Create slices and store them as ModuleList to ensure proper device handling
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            idx = int(l)
            self.slices.append(vgg[prev:idx])
            prev = idx
        print("Perceptual loss ready")
    
    def forward(self, x, y):
        loss = 0
        x_input = x
        y_input = y
        for slice_block in self.slices:
            x_input = slice_block(x_input)
            y_input = slice_block(y_input)
            loss += nn.functional.l1_loss(x_input, y_input)
        return loss

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

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

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, class_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []
        self.class_to_idx = {}
        self.class_filter = class_filter  # Set of class IDs to load, None means load all
        
        # Build class_to_idx mapping from wnids.txt (consistent ordering)
        wnids_file = os.path.join(root_dir, 'wnids.txt')
        if os.path.exists(wnids_file):
            with open(wnids_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
                self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        else:
            # Fallback: use sorted directory listing
            train_dir = os.path.join(root_dir, 'train')
            if os.path.exists(train_dir):
                class_names = sorted([d for d in os.listdir(train_dir) 
                                    if os.path.isdir(os.path.join(train_dir, d))])
                self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Create reverse mapping for efficient filtering
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
    
    def _load_train_data(self):
        """Load training data from train/*/images/ structure"""
        train_dir = os.path.join(self.root_dir, 'train')
        
        for class_name, class_idx in self.class_to_idx.items():
            # Skip if we're filtering and this class isn't needed
            if self.class_filter is not None and class_idx not in self.class_filter:
                continue
                
            class_images_dir = os.path.join(train_dir, class_name, 'images')
            if os.path.exists(class_images_dir):
                for img_file in os.listdir(class_images_dir):
                    if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        img_path = os.path.join(class_images_dir, img_file)
                        self.samples.append((img_path, class_idx))
    
    def _load_val_data(self):
        """Load validation data using val_annotations.txt"""
        val_dir = os.path.join(self.root_dir, 'val')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')
        val_images_dir = os.path.join(val_dir, 'images')
        
        if os.path.exists(val_annotations):
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_name = parts[1]
                        
                        if class_name in self.class_to_idx:
                            class_idx = self.class_to_idx[class_name]
                            
                            # Skip if we're filtering and this class isn't needed
                            if self.class_filter is not None and class_idx not in self.class_filter:
                                continue
                            
                            img_path = os.path.join(val_images_dir, img_name)
                            if os.path.exists(img_path):
                                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, class_idx
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor if image loading fails
            if self.transform:
                dummy_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
                return self.transform(dummy_img), class_idx
            else:
                return torch.zeros(3, 64, 64), class_idx

# Define transforms to match your local setup
tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)  # Normalize to [-1, 1]
])

tf_val = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)  # Normalize to [-1, 1]
])

# Global variables for datasets
trainset = None
valset = None

def initialize_datasets(data_dir, class_filter=None):
    """Initialize datasets with the correct data directory and transforms
    
    Args:
        data_dir: Path to tiny-imagenet-200 dataset
        class_filter: Set of class IDs to load (e.g., {42} for class 42 only), None for all classes
    """
    global trainset, valset
    print(f"Initializing datasets from {data_dir}")
    if class_filter:
        print(f"Loading only classes: {sorted(class_filter)}")
    
    trainset = TinyImageNetDataset(root_dir=data_dir, split='train', 
                                  transform=tf_train, class_filter=class_filter)
    valset = TinyImageNetDataset(root_dir=data_dir, split='val', 
                                transform=tf_val, class_filter=class_filter)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    if class_filter is None:
        print(f"Number of classes: {len(trainset.class_to_idx)}")
    
    return trainset, valset

def get_class_indices(dataset):
    """Get indices for each class in the dataset - optimized version"""
    if dataset is None:
        return {}
    
    print(f"Building class indices for {len(dataset)} samples...")
    idxs = defaultdict(list)
    
    # More efficient: directly access samples instead of calling __getitem__
    for i, (_, label) in enumerate(dataset.samples):
        idxs[label].append(i)
    
    result = dict(idxs)
    print(f"Found samples for {len(result)} classes")
    for class_id in sorted(result.keys()):
        indices = result[class_id]
        if len(indices) > 0:
            print(f"  Class {class_id}: {len(indices)} samples")
    
    return result

def eval_ae(model, class_id, valset, class_indices, criterion_pixel, criterion_perc):
    """Evaluate autoencoder on validation set for a specific class"""
    model.eval()
    device = next(model.parameters()).device
    
    if class_id not in class_indices or len(class_indices[class_id]) == 0:
        print(f"No validation samples for class {class_id}")
        model.train()
        return
    
    val_indices = class_indices[class_id]
    total_pix, total_perc, count = 0.0, 0.0, 0
    
    print(f"Evaluating class {class_id} on {len(val_indices)} validation samples...")
    
    with torch.no_grad():
        for idx in val_indices:
            try:
                img, _ = valset[idx]
                img = img.unsqueeze(0).to(device)
                recon = model(img)
                
                pix_loss = criterion_pixel(recon, img).item()
                perc_loss = criterion_perc(recon, img).item()
                
                total_pix += pix_loss
                total_perc += perc_loss
                count += 1
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                continue
    
    if count > 0:
        avg_pix = total_pix / count
        avg_perc = total_perc / count
        print(f"[EVAL] Class {class_id} - Pixel: {avg_pix:.4f}, Perceptual: {avg_perc:.4f}")
    else:
        print(f"[EVAL] Class {class_id} - No valid samples processed")
    
    model.train()