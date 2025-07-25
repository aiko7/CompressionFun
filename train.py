import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from utils import (
    UNetAE,
    PerceptualLoss,
    initialize_datasets,
    get_class_indices,
    eval_ae,
)

EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_id", type=int, required=True,
                        help="Class ID to train on (0‑199)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs",
                        help="Directory to save / resume checkpoints")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the dataset")
    return parser.parse_args()

def train_epoch(model, class_id, dataset, indices_dict,
                criterion_pixel, criterion_perc, optimizer, batch_size):
    model.train()
    
    if class_id not in indices_dict or len(indices_dict[class_id]) == 0:
        print(f"No training samples found for class {class_id}")
        return 0.0, 0.0
    
    loader = DataLoader(
        Subset(dataset, indices_dict[class_id]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    total_pix, total_perc, count = 0.0, 0.0, 0
    for batch_idx, (images, _) in enumerate(loader):
        imgs = images.to(DEVICE)
        recon = model(imgs)
        loss_pix = criterion_pixel(recon, imgs)
        loss_perc = criterion_perc(recon, imgs)
        loss = loss_pix + 0.1 * loss_perc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_pix += loss_pix.item()
        total_perc += loss_perc.item()
        count += 1

        if batch_idx % 10 == 0:
            print(f"[Class {class_id}] Batch {batch_idx}/{len(loader)} "
                  f"- Pixel {loss_pix:.4f} | Perc {loss_perc:.4f}")

    avg_pix = total_pix / count if count > 0 else 0.0
    avg_perc = total_perc / count if count > 0 else 0.0
    print(f"[Class {class_id}] Epoch finished "
          f"- Avg Pixel {avg_pix:.4f} | Avg Perc {avg_perc:.4f}")
    return avg_pix, avg_perc

def main():
    args = parse_args()

    # Try to get AzureML run context
    try:
        from azureml.core import Run
        run = Run.get_context()
    except Exception:
        run = None

    print(f"Training autoencoder for class {args.class_id} on {DEVICE}")
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Try to get from environment variable (set by AzureML)
        data_dir = os.environ.get('AZUREML_DATA_DIR', './tiny-imagenet-200')
    
    print(f"Using data directory: {data_dir}")
    
    # Initialize datasets
    trainset, valset = initialize_datasets(data_dir)
    print(f"Initialized datasets - Train: {len(trainset)}, Val: {len(valset)}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get class indices
    class_idx_train = get_class_indices(trainset)
    class_idx_val = get_class_indices(valset)
    
    print(f"Found {len(class_idx_train)} training classes")
    print(f"Found {len(class_idx_val)} validation classes")
    
    if args.class_id not in class_idx_train:
        print(f"ERROR: Class {args.class_id} not found in training data")
        return
    
    print(f"Training samples for class {args.class_id}: {len(class_idx_train[args.class_id])}")

    # Initialize model and training components
    model = UNetAE(use_resblocks=False).to(DEVICE)
    criterion_pixel = nn.MSELoss()
    criterion_perc = PerceptualLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if available
    start_epoch = 1
    ckpt_pattern = f"ae_class{args.class_id}_epoch*.pt"
    checkpoints = sorted(
        checkpoint_dir.glob(ckpt_pattern),
        key=lambda p: int(p.stem.split("_epoch")[-1]),
    )
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
        start_epoch = int(latest_ckpt.stem.split("_epoch")[-1]) + 1
        print(f"Resuming from {latest_ckpt} (starting at epoch {start_epoch})")

    history = {"pix": [], "perc": []}

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        pix_loss, perc_loss = train_epoch(
            model, args.class_id, trainset, class_idx_train,
            criterion_pixel, criterion_perc, optimizer, args.batch_size
        )

        # Log to AzureML if available
        if run:
            run.log("pixel_loss", pix_loss)
            run.log("perceptual_loss", perc_loss)
            run.log("epoch", epoch)

        history["pix"].append(pix_loss)
        history["perc"].append(perc_loss)

        if epoch % 10 == 0:
            print(f"[Class {args.class_id}] Completed epoch {epoch}/{args.epochs}")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = checkpoint_dir / f"ae_class{args.class_id}_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at {ckpt_path}")

    # Save final model
    final_path = checkpoint_dir / f"ae_class{args.class_id}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model at {final_path}")

    # Evaluate on validation set
    eval_ae(model, args.class_id, valset, class_idx_val,
            criterion_pixel, criterion_perc)
    print(f"Training completed for class {args.class_id}")

if __name__ == "__main__":
    main()