import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.models import VGG16_Weights
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import UNetAE, PerceptualLoss, trainset, valset, class_indices_train, eval_ae


DATA_DIR = "./tiny-imagenet-200"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
NUM_CLASSES = 200
EPOCHS = 800
LATENT_DIM = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Start")

tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

print("Loading datasets")
trainset = ImageFolder(root=TRAIN_DIR, transform=tf_train)
valset = ImageFolder(root=VAL_DIR, transform=tf_val)
print(f"Loaded trainset with {len(trainset)} images")
print(f"Loaded valset with {len(valset)} images")

print("Indexing classes in train and val sets")
def get_class_indices(dataset):
    idxs = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        idxs[label].append(i)
    return dict(idxs)

class_indices_train = get_class_indices(trainset)
class_indices_test = get_class_indices(valset)
print("Finished indexing classes")

# Perceptual loss using pretrained VGG
class PerceptualLoss(nn.Module):
    def __init__(self, layers=('3','8','15','22')):
        super().__init__()
        print("Initializing perceptual loss (VGG with explicit weights)")
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.slices = []
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
    def __init__(self, latent_dim=LATENT_DIM, use_resblocks=True):
        super().__init__()
        self.use_resblocks = use_resblocks
        print("Building UNetAE architecture")
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.ReLU())
        self.fc = nn.Linear(512*4*4, latent_dim)

        self.fc_inv = nn.Linear(latent_dim, 512*4*4)      

        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #ResBlock(256) 
        )

        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #ResBlock(128) 

        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #ResBlock(64) 
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  
            nn.Tanh()
        )
        print("UNetAE ready")

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        flat = e4.view(e4.size(0), -1)
        z = self.fc(flat)
        inv = self.fc_inv(z).view(-1,512,4,4)
        d4 = self.dec4(inv)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1



def train_ae(model, class_id, dataset, criterion_pixel, criterion_perc, optimizer):
    print(f"Starting training for class {class_id}")
    model.train()
    loader = DataLoader(Subset(dataset, class_indices_train[class_id]), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)
    total_pix, total_perc, count = 0,0,0
    for batch_idx, (images, _) in enumerate(loader):
        imgs = images.to(DEVICE)
        recon = model(imgs)
        loss_pix = criterion_pixel(recon, imgs)
        loss_perc = criterion_perc(recon, imgs)
        loss = loss_pix + 0.1*loss_perc
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_pix += loss_pix.item(); total_perc += loss_perc.item(); count+=1
        if batch_idx % 10 == 0:
            print(f"[Class {class_id}] Batch {batch_idx}/{len(loader)} - Pix: {loss_pix:.4f}, Perc: {loss_perc:.4f}")
    avg_pix, avg_perc = total_pix/count, total_perc/count
    print(f"Finished epoch for class {class_id} - Avg Pix: {avg_pix:.4f}, Avg Perc: {avg_perc:.4f}")
    return avg_pix, avg_perc


def eval_ae(model, class_id, dataset, criterion_pixel, criterion_perc):
    print(f"Starting evaluation for class {class_id}")
    model.eval()
    loader = DataLoader(Subset(dataset, class_indices_test[class_id]), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    total_pix, total_perc, count = 0,0,0
    with torch.no_grad():
        for images, _ in loader:
            imgs = images.to(DEVICE)
            recon = model(imgs)
            total_pix += criterion_pixel(recon, imgs).item()
            total_perc += criterion_perc(recon, imgs).item()
            count+=1
    avg_pix, avg_perc = total_pix/count, total_perc/count
    print(f"Eval complete for class {class_id} - Pix: {avg_pix:.4f}, Perc: {avg_perc:.4f}")
    return avg_pix, avg_perc

def main():
    from azureml.core import Run
    run = Run.get_context()
    
    print("Beginning full training and evaluation loop")
    criterion_pixel = nn.MSELoss()
    criterion_perc = PerceptualLoss()
    results = {}

    # Use AzureML output directory if available
    output_dir = os.environ.get("AZUREML_OUTPUT_DIR", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    for class_id in range(1):
        model = UNetAE(use_resblocks=False).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        history = {'pix': [], 'perc': []}

        for epoch in range(1, EPOCHS + 1):
            pix, perc = train_ae(model, class_id, trainset, criterion_pixel, criterion_perc, optimizer)

            # Log losses to AzureML
            run.log("pixel_loss", pix)
            run.log("perceptual_loss", perc)

            history['pix'].append(pix)
            history['perc'].append(perc)

            if epoch % 10 == 0:
                print(f"[Class {class_id}] Completed epoch {epoch}/{EPOCHS}")

            if epoch % SAVE_EVERY == 0:
                path = os.path.join(output_dir, f"ae_{class_id}_ep{epoch}.pt")
                torch.save(model.state_dict(), path)
                print(f"Saved checkpoint: {path}")

        results[class_id] = history
        model_path = os.path.join(output_dir, f"ae_{class_id}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved final model for class {class_id}: {model_path}")
        eval_ae(model, class_id, valset, criterion_pixel, criterion_perc)

    print("Benchmark Results (Pixel vs Perceptual) per class:")
    for cid, h in results.items():
        print(f"Class {cid}: Start Pix {h['pix'][0]:.4f} -> End Pix {h['pix'][-1]:.4f}; "
              f"Start Perc {h['perc'][0]:.4f} -> End Perc {h['perc'][-1]:.4f}")
    print("Done")

