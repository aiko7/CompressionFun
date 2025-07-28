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

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class_id", type=int, help="Single class ID to train on (0‑199)")
    group.add_argument("--class_list", type=str, help="Comma-separated list of class IDs to train on")
    
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
    
    loader_start = time.time()
    loader = DataLoader(
        Subset(dataset, indices_dict[class_id]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        drop_last=True  
    )
    loader_time = time.time() - loader_start
    print(f"[TIMER] DataLoader creation: {loader_time:.3f}s")

    total_pix, total_perc, count = 0.0, 0.0, 0
    data_load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    
    batch_start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(loader):
        data_start = time.time()
        imgs = images.to(DEVICE, non_blocking=True)
        data_load_time += time.time() - data_start
        
        forward_start = time.time()
        recon = model(imgs)
        loss_pix = criterion_pixel(recon, imgs)
        loss_perc = criterion_perc(recon, imgs)
        loss = loss_pix + 0.1 * loss_perc
        forward_time += time.time() - forward_start

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

def train_single_class(class_id, args, run=None):
    """Train autoencoder for a single class"""
    print(f"\n{'='*60}")
    print(f"TRAINING AUTOENCODER FOR CLASS {class_id}")
    print(f"{'='*60}")
    
    class_start_time = time.time()
    
    print(f"Training autoencoder for class {class_id} on {DEVICE}")
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
    class_filter = {class_id}  
    trainset, valset = initialize_datasets(args.data_dir, class_filter=class_filter)
    dataset_time = time.time() - dataset_start
    print(f"[TIMER] Dataset initialization: {dataset_time:.3f}s")
    
    if len(trainset) == 0:
        print("ERROR: No training samples found")
        return
    
    checkpoint_start = time.time()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    checkpoint_setup_time = time.time() - checkpoint_start
    print(f"[TIMER] Checkpoint directory setup: {checkpoint_setup_time:.3f}s")
    
    print("Computing class indices...")
    indices_start = time.time()
    class_idx_train = get_class_indices(trainset)
    class_idx_val = get_class_indices(valset)
    indices_time = time.time() - indices_start
    print(f"[TIMER] Class indices computation: {indices_time:.3f}s")
    
    if class_id not in class_idx_train:
        print(f"ERROR: Class {class_id} not found in training data")
        available_classes = sorted(class_idx_train.keys())
        print(f"Available classes: {available_classes[:10]}..." if len(available_classes) > 10 else f"Available classes: {available_classes}")
        return
    
    train_samples = len(class_idx_train[class_id])
    val_samples = len(class_idx_val.get(class_id, []))
    print(f"Class {class_id} - Train: {train_samples}, Val: {val_samples}")

    print("Initializing model...")
    model_start = time.time()
    model = UNetAE(use_resblocks=False).to(DEVICE)
    criterion_pixel = nn.MSELoss()
    criterion_perc = PerceptualLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)
    model_init_time = time.time() - model_start
    print(f"[TIMER] Model initialization: {model_init_time:.3f}s")

    checkpoint_load_start = time.time()
    start_epoch = 1
    ckpt_pattern = f"ae_class{class_id}_epoch*.pt"
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

    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== EPOCH {epoch}/{args.epochs} ===")
        pix_loss, perc_loss, epoch_time = train_epoch(
            model, class_id, trainset, class_idx_train,
            criterion_pixel, criterion_perc, optimizer, args.batch_size
        )

        if run:
            run.log(f"class_{class_id}_pixel_loss", pix_loss)
            run.log(f"class_{class_id}_perceptual_loss", perc_loss)
            run.log(f"class_{class_id}_epoch", epoch)
            run.log(f"class_{class_id}_epoch_time", epoch_time)

        history["pix"].append(pix_loss)
        history["perc"].append(perc_loss)
        history["epoch_times"].append(epoch_time)

        if epoch % 10 == 0:
            avg_epoch_time = sum(history["epoch_times"][-10:]) / min(10, len(history["epoch_times"]))
            print(f"[Class {class_id}] Completed epoch {epoch}/{args.epochs} "
                  f"| Avg last 10 epochs: {avg_epoch_time:.2f}s")

        if epoch % SAVE_EVERY == 0:
            save_start = time.time()
            ckpt_path = checkpoint_dir / f"ae_class{class_id}_epoch{epoch}.pt"
            
            prev_epoch = epoch - SAVE_EVERY
            if prev_epoch > 0:
                prev_ckpt_path = checkpoint_dir / f"ae_class{class_id}_epoch{prev_epoch}.pt"
                if prev_ckpt_path.exists():
                    try:
                        prev_ckpt_path.unlink()
                        print(f"Deleted previous checkpoint: {prev_ckpt_path}")
                    except Exception as e:
                        print(f"Failed to delete previous checkpoint {prev_ckpt_path}: {e}")
            
            try:
                torch.save(model.state_dict(), ckpt_path)
                save_time = time.time() - save_start
                print(f"Saved checkpoint at {ckpt_path} (took {save_time:.3f}s)")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    training_time = time.time() - training_start_time
    print(f"\n[TIMER] Total training time for class {class_id}: {training_time:.3f}s ({training_time/60:.1f}m)")

    # Save final model
    final_save_start = time.time()
    final_path = checkpoint_dir / f"ae_class{class_id}_final.pt"
    try:
        torch.save(model.state_dict(), final_path)
        final_save_time = time.time() - final_save_start
        print(f"Saved final model at {final_path} (took {final_save_time:.3f}s)")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    if class_id in class_idx_val and len(class_idx_val[class_id]) > 0:
        eval_start = time.time()
        eval_ae(model, class_id, valset, class_idx_val,
                criterion_pixel, criterion_perc)
        eval_time = time.time() - eval_start
        print(f"[TIMER] Evaluation time: {eval_time:.3f}s")
    else:
        print(f"No validation data available for class {class_id}")
    
    total_class_time = time.time() - class_start_time
    print(f"\n[TIMER] Total time for class {class_id}: {total_class_time:.3f}s ({total_class_time/60:.1f}m)")
    print(f"Training completed for class {class_id}")
    
    # Clean up to free memory for next class
    del model, criterion_pixel, criterion_perc, optimizer, trainset, valset
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

    if args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = [int(x.strip()) for x in args.class_list.split(',')]
    
    print(f"Training autoencoders for classes: {class_ids}")
    print(f"Total classes to train: {len(class_ids)}")
    
    successful_classes = 0
    failed_classes = 0
    
    for class_id in class_ids:
        try:
            train_single_class(class_id, args, run)
            successful_classes += 1
            print(f"\n✓ Successfully completed training for class {class_id}")
        except Exception as e:
            failed_classes += 1
            print(f"\n✗ Failed to train class {class_id}: {e}")
            continue
    
    total_script_time = time.time() - script_start_time
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total script runtime: {total_script_time:.3f}s ({total_script_time/60:.1f}m)")
    print(f"Classes trained successfully: {successful_classes}")
    print(f"Classes failed: {failed_classes}")
    print(f"Success rate: {successful_classes/(successful_classes+failed_classes)*100:.1f}%")
    
    if run:
        run.log("total_classes_trained", successful_classes)
        run.log("total_classes_failed", failed_classes)
        run.log("total_runtime_minutes", total_script_time/60)

if __name__ == "__main__":
    main()