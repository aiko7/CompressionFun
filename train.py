import os
import argparse
import time
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
NUM_WORKERS = 4  # Reduced for AzureML stability
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
    epoch_start_time = time.time()
    model.train()
    
    if class_id not in indices_dict or len(indices_dict[class_id]) == 0:
        print(f"No training samples found for class {class_id}")
        return 0.0, 0.0, 0.0
    
    print(f"Training class {class_id} with {len(indices_dict[class_id])} samples")
    
    # Time dataloader creation
    loader_start = time.time()
    loader = DataLoader(
        Subset(dataset, indices_dict[class_id]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        drop_last=True  # Avoid issues with batch norm for small last batches
    )
    loader_time = time.time() - loader_start
    print(f"[TIMER] DataLoader creation: {loader_time:.3f}s")

    total_pix, total_perc, count = 0.0, 0.0, 0
    data_load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    
    batch_start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(loader):
        # Time data loading/transfer
        data_start = time.time()
        imgs = images.to(DEVICE, non_blocking=True)
        data_load_time += time.time() - data_start
        
        # Time forward pass
        forward_start = time.time()
        recon = model(imgs)
        loss_pix = criterion_pixel(recon, imgs)
        loss_perc = criterion_perc(recon, imgs)
        loss = loss_pix + 0.1 * loss_perc
        forward_time += time.time() - forward_start

        # Time backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - backward_start

        total_pix += loss_pix.item()
        total_perc += loss_perc.item()
        count += 1

        if batch_idx % 10 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(f"[Class {class_id}] Batch {batch_idx}/{len(loader)} "
                  f"- Pixel {loss_pix:.4f} | Perc {loss_perc:.4f} "
                  f"| Time: {batch_elapsed:.2f}s")

    avg_pix = total_pix / count if count > 0 else 0.0
    avg_perc = total_perc / count if count > 0 else 0.0
    epoch_time = time.time() - epoch_start_time
    
    print(f"[Class {class_id}] Epoch finished "
          f"- Avg Pixel {avg_pix:.4f} | Avg Perc {avg_perc:.4f}")
    print(f"[TIMER] Epoch total: {epoch_time:.3f}s | "
          f"Data loading: {data_load_time:.3f}s | "
          f"Forward: {forward_time:.3f}s | "
          f"Backward: {backward_time:.3f}s")
    
    return avg_pix, avg_perc, epoch_time

def main():
    script_start_time = time.time()
    args = parse_args()

    init_start = time.time()
    try:
        from azureml.core import Run
        run = Run.get_context()
        print("AzureML context found")
    except Exception:
        run = None
        print("No AzureML context, running locally")
    init_time = time.time() - init_start
    print(f"[TIMER] AzureML initialization: {init_time:.3f}s")

    print(f"Training autoencoder for class {args.class_id} on {DEVICE}")
    print(f"Using data directory: {args.data_dir}")
    
    validation_start = time.time()
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory {args.data_dir} does not exist")
        return
    
    expected_dirs = ['train', 'val', 'wnids.txt']
    for item in expected_dirs:
        path = os.path.join(args.data_dir, item)
        if not os.path.exists(path):
            print(f"WARNING: Expected {item} not found at {path}")
    validation_time = time.time() - validation_start
    print(f"[TIMER] Data directory validation: {validation_time:.3f}s")
    
    print("Initializing datasets...")
    dataset_start = time.time()
    class_filter = {args.class_id}  
    trainset, valset = initialize_datasets(args.data_dir, class_filter=class_filter)
    dataset_time = time.time() - dataset_start
    print(f"[TIMER] Dataset initialization: {dataset_time:.3f}s")
    
    if len(trainset) == 0:
        print("ERROR: No training samples found")
        return
    
    # Create checkpoint directory
    checkpoint_start = time.time()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    checkpoint_setup_time = time.time() - checkpoint_start
    print(f"[TIMER] Checkpoint directory setup: {checkpoint_setup_time:.3f}s")

    # Get class indices
    print("Computing class indices...")
    indices_start = time.time()
    class_idx_train = get_class_indices(trainset)
    class_idx_val = get_class_indices(valset)
    indices_time = time.time() - indices_start
    print(f"[TIMER] Class indices computation: {indices_time:.3f}s")
    
    if args.class_id not in class_idx_train:
        print(f"ERROR: Class {args.class_id} not found in training data")
        available_classes = sorted(class_idx_train.keys())
        print(f"Available classes: {available_classes[:10]}..." if len(available_classes) > 10 else f"Available classes: {available_classes}")
        return
    
    train_samples = len(class_idx_train[args.class_id])
    val_samples = len(class_idx_val.get(args.class_id, []))
    print(f"Class {args.class_id} - Train: {train_samples}, Val: {val_samples}")

    # Initialize model and training components
    print("Initializing model...")
    model_start = time.time()
    model = UNetAE(use_resblocks=False).to(DEVICE)
    criterion_pixel = nn.MSELoss()
    criterion_perc = PerceptualLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)
    model_init_time = time.time() - model_start
    print(f"[TIMER] Model initialization: {model_init_time:.3f}s")

    # Resume from checkpoint if available
    checkpoint_load_start = time.time()
    start_epoch = 1
    ckpt_pattern = f"ae_class{args.class_id}_epoch*.pt"
    checkpoints = sorted(
        checkpoint_dir.glob(ckpt_pattern),
        key=lambda p: int(p.stem.split("_epoch")[-1]),
    )
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            start_epoch = int(latest_ckpt.stem.split("_epoch")[-1]) + 1
            print(f"Resuming from {latest_ckpt} (starting at epoch {start_epoch})")
        except Exception as e:
            print(f"Failed to load checkpoint {latest_ckpt}: {e}")
            print("Starting from scratch")
    checkpoint_load_time = time.time() - checkpoint_load_start
    print(f"[TIMER] Checkpoint loading: {checkpoint_load_time:.3f}s")

    history = {"pix": [], "perc": [], "epoch_times": []}

    # Training loop
    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== EPOCH {epoch}/{args.epochs} ===")
        pix_loss, perc_loss, epoch_time = train_epoch(
            model, args.class_id, trainset, class_idx_train,
            criterion_pixel, criterion_perc, optimizer, args.batch_size
        )

        # Log to AzureML if available
        if run:
            run.log("pixel_loss", pix_loss)
            run.log("perceptual_loss", perc_loss)
            run.log("epoch", epoch)
            run.log("epoch_time", epoch_time)

        history["pix"].append(pix_loss)
        history["perc"].append(perc_loss)
        history["epoch_times"].append(epoch_time)

        if epoch % 10 == 0:
            avg_epoch_time = sum(history["epoch_times"][-10:]) / min(10, len(history["epoch_times"]))
            print(f"[Class {args.class_id}] Completed epoch {epoch}/{args.epochs} "
                  f"| Avg last 10 epochs: {avg_epoch_time:.2f}s")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            save_start = time.time()
            ckpt_path = checkpoint_dir / f"ae_class{args.class_id}_epoch{epoch}.pt"
            try:
                torch.save(model.state_dict(), ckpt_path)
                save_time = time.time() - save_start
                print(f"Saved checkpoint at {ckpt_path} (took {save_time:.3f}s)")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    training_time = time.time() - training_start_time
    print(f"\n[TIMER] Total training time: {training_time:.3f}s ({training_time/60:.1f}m)")

    # Save final model
    final_save_start = time.time()
    final_path = checkpoint_dir / f"ae_class{args.class_id}_final.pt"
    try:
        torch.save(model.state_dict(), final_path)
        final_save_time = time.time() - final_save_start
        print(f"Saved final model at {final_path} (took {final_save_time:.3f}s)")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    # Evaluate on validation set
    if args.class_id in class_idx_val and len(class_idx_val[args.class_id]) > 0:
        eval_start = time.time()
        eval_ae(model, args.class_id, valset, class_idx_val,
                criterion_pixel, criterion_perc)
        eval_time = time.time() - eval_start
        print(f"[TIMER] Evaluation time: {eval_time:.3f}s")
    else:
        print(f"No validation data available for class {args.class_id}")
    
    total_script_time = time.time() - script_start_time
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total script runtime: {total_script_time:.3f}s ({total_script_time/60:.1f}m)")
    print(f"Setup time: {dataset_time + model_init_time + indices_time:.3f}s")
    print(f"Training time: {training_time:.3f}s ({training_time/total_script_time*100:.1f}%)")
    if history["epoch_times"]:
        avg_epoch = sum(history["epoch_times"]) / len(history["epoch_times"])
        print(f"Average epoch time: {avg_epoch:.3f}s")
        print(f"Fastest epoch: {min(history['epoch_times']):.3f}s")
        print(f"Slowest epoch: {max(history['epoch_times']):.3f}s")
    
    print(f"Training completed for class {args.class_id}")

if __name__ == "__main__":
    main()