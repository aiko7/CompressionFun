import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict

DATA_DIR = "./tiny-imagenet-200"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

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

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = 0
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

trainset = CustomImageDataset(root_dir=TRAIN_DIR, transform=transform)
valset = CustomImageDataset(root_dir=VAL_DIR, transform=transform)

def get_class_indices(dataset):
    idxs = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        idxs[label].append(i)
    return dict(idxs)

def eval_ae(model, class_id, valset, class_indices, criterion_pixel, criterion_perc):
    model.eval()
    device = next(model.parameters()).device
    val_subset = [valset[i] for i in class_indices[class_id]]

    total_pix, total_perc, count = 0, 0, 0
    with torch.no_grad():
        for img, _ in val_subset:
            img = img.unsqueeze(0).to(device)
            recon = model(img)
            total_pix += criterion_pixel(recon, img).item()
            total_perc += criterion_perc(recon, img).item()
            count += 1

    print(f"Eval Class {class_id} - Avg Pixel Loss: {total_pix/count:.4f}, Avg Perceptual Loss: {total_perc/count:.4f}")
    model.train()
