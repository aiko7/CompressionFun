import os
import argparse
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import gc

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
BATCH_SIZE = 32  # Reduced from 64 to accommodate multiple models
LEARNING_RATE = 1e-4
NUM_WORKERS = 2  # Reduced to avoid too many threads
SAVE_EVERY = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global shared resources
SHARED_CRITERION_PERC = None
SHARED_TRAINSET = None
SHARED_VALSET = None
SHARED_CLASS_IDX_TRAIN = None
SHARED_CLASS_IDX_VAL = None
RESOURCE_LOCK = threading.Lock()

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
    parser.add_argument("--parallel_models", type=int, default=6,
                        help="Number of models to train in parallel")
    return parser.parse_args()

def initialize_shared_resources(data_dir, all_class_ids):
    """Initialize shared resources once for all parallel training"""
    global SHARED_CRITERION_PERC, SHARED_TRAINSET, SHARED_VALSET
    global SHARED_CLASS_IDX_TRAIN, SHARED_CLASS_IDX_VAL
    
    print("Initializing shared resources (reusing existing environment data)...")
    start_time = time.time()
    
    # Check if datasets are already initialized globally
    from utils import trainset, valset
    if trainset is None or valset is None:
        print("Datasets not found in environment, initializing...")
        class_filter = set(all_class_ids)
        SHARED_TRAINSET, SHARED_VALSET = initialize_datasets(data_dir, class_filter=class_filter)
    else:
        print("Reusing existing datasets from environment")
        SHARED_TRAINSET, SHARED_VALSET = trainset, valset
    
    # Compute class indices once
    print("Computing class indices...")
    SHARED_CLASS_IDX_TRAIN = get_class_indices(SHARED_TRAINSET)
    SHARED_CLASS_IDX_VAL = get_class_indices(SHARED_VALSET)
    
    # Initialize shared perceptual loss (VGG model) - this will reuse cached weights
    print("Initializing shared perceptual loss...")
    SHARED_CRITERION_PERC = PerceptualLoss().to(DEVICE)
    
    init_time = time.time() - start_time
    print(f"[TIMER] Shared resources initialized in {init_time:.3f}s")
    print(f"Memory usage: Train samples: {len(SHARED_TRAINSET)}, Val samples: {len(SHARED_VALSET)}")
    
    return SHARED_TRAINSET, SHARED_VALSET, SHARED_CLASS_IDX_TRAIN, SHARED_CLASS_IDX_VAL

def train_epoch_parallel(model, class_id, criterion_pixel, optimizer, batch_size, thread_id):
    """Modified train_epoch for parallel execution"""
    model.train()
    
    if class_id not in SHARED_CLASS_IDX_TRAIN or len(SHARED_CLASS_IDX_TRAIN[class_id]) == 0:
        print(f"[Thread {thread_id}] No training samples found for class {class_id}")
        return 0.0, 0.0, 0.0
    
    # Create DataLoader for this specific class
    loader = DataLoader(
        Subset(SHARED_TRAINSET, SHARED_CLASS_IDX_TRAIN[class_id]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        drop_last=True
    )

    total_pix, total_perc, count = 0.0, 0.0, 0
    
    for batch_idx, (images, _) in enumerate(loader):
        imgs = images.to(DEVICE, non_blocking=True)
        
        # Forward pass
        recon = model(imgs)
        loss_pix = criterion_pixel(recon, imgs)
        
        # Use shared perceptual loss (thread-safe)
        with RESOURCE_LOCK:
            loss_perc = SHARED_CRITERION_PERC(recon, imgs)
        
        loss = loss_pix + 0.1 * loss_perc
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_pix += loss_pix.item()
        total_perc += loss_perc.item()
        count += 1

        # Less frequent logging to reduce output spam
        if batch_idx % 20 == 0:
            print(f"[Thread {thread_id} | Class {class_id}] Batch {batch_idx}/{len(loader)} "
                  f"- Pixel {loss_pix:.4f} | Perc {loss_perc:.4f}")

    avg_pix = total_pix / count if count > 0 else 0.0
    avg_perc = total_perc / count if count > 0 else 0.0
    
    return avg_pix, avg_perc, count

def train_single_class_parallel(class_id, args, thread_id, run=None):
    """Train autoencoder for a single class in parallel"""
    print(f"\n[Thread {thread_id}] Starting training for class {class_id}")
    class_start_time = time.time()
    
    # Check if we have training data for this class
    if class_id not in SHARED_CLASS_IDX_TRAIN:
        print(f"[Thread {thread_id}] ERROR: Class {class_id} not found in training data")
        return False
    
    train_samples = len(SHARED_CLASS_IDX_TRAIN[class_id])
    val_samples = len(SHARED_CLASS_IDX_VAL.get(class_id, []))
    print(f"[Thread {thread_id}] Class {class_id} - Train: {train_samples}, Val: {val_samples}")

    # Initialize model and optimizer (each thread gets its own)
    model = UNetAE(use_resblocks=False).to(DEVICE)
    criterion_pixel = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoints
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
            print(f"[Thread {thread_id}] Resuming class {class_id} from epoch {start_epoch}")
        except Exception as e:
            print(f"[Thread {thread_id}] Failed to load checkpoint for class {class_id}: {e}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        pix_loss, perc_loss, batch_count = train_epoch_parallel(
            model, class_id, criterion_pixel, optimizer, args.batch_size, thread_id
        )

        # Log metrics if AzureML context is available
        if run and batch_count > 0:
            run.log(f"class_{class_id}_pixel_loss", pix_loss)
            run.log(f"class_{class_id}_perceptual_loss", perc_loss)
            run.log(f"class_{class_id}_epoch", epoch)

        # Progress update every 20 epochs
        if epoch % 20 == 0:
            print(f"[Thread {thread_id}] Class {class_id} - Epoch {epoch}/{args.epochs} "
                  f"- Pixel {pix_loss:.4f} | Perc {perc_loss:.4f}")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = checkpoint_dir / f"ae_class{class_id}_epoch{epoch}.pt"
            
            # Clean up previous checkpoint
            prev_epoch = epoch - SAVE_EVERY
            if prev_epoch > 0:
                prev_ckpt_path = checkpoint_dir / f"ae_class{class_id}_epoch{prev_epoch}.pt"
                if prev_ckpt_path.exists():
                    try:
                        prev_ckpt_path.unlink()
                    except Exception:
                        pass
            
            # Save current checkpoint
            try:
                torch.save(model.state_dict(), ckpt_path)
            except Exception as e:
                print(f"[Thread {thread_id}] Failed to save checkpoint for class {class_id}: {e}")

    # Save final model
    final_path = checkpoint_dir / f"ae_class{class_id}_final.pt"
    try:
        torch.save(model.state_dict(), final_path)
        print(f"[Thread {thread_id}] Saved final model for class {class_id}")
    except Exception as e:
        print(f"[Thread {thread_id}] Failed to save final model for class {class_id}: {e}")

    # Final evaluation
    if class_id in SHARED_CLASS_IDX_VAL and len(SHARED_CLASS_IDX_VAL[class_id]) > 0:
        try:
            eval_ae(model, class_id, SHARED_VALSET, SHARED_CLASS_IDX_VAL,
                    criterion_pixel, SHARED_CRITERION_PERC)
        except Exception as e:
            print(f"[Thread {thread_id}] Evaluation failed for class {class_id}: {e}")

    # Cleanup
    del model, criterion_pixel, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_time = time.time() - class_start_time
    print(f"[Thread {thread_id}] Completed class {class_id} in {total_time/60:.1f}m")
    return True

def main():
    script_start_time = time.time()
    args = parse_args()

    # AzureML context
    try:
        from azureml.core import Run
        run = Run.get_context()
        print("AzureML context found")
    except Exception:
        run = None
        print("No AzureML context, running locally")

    # Parse class IDs
    if args.class_id is not None:
        class_ids = [args.class_id]
    else:
        class_ids = [int(x.strip()) for x in args.class_list.split(',')]
    
    print(f"Training autoencoders for {len(class_ids)} classes: {class_ids[:10]}{'...' if len(class_ids) > 10 else ''}")
    print(f"Using {args.parallel_models} parallel threads")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "No GPU")

    # Initialize shared resources
    initialize_shared_resources(args.data_dir, class_ids)
    
    # Parallel training using ThreadPoolExecutor
    successful_classes = 0
    failed_classes = 0
    
    print(f"\nStarting parallel training with {args.parallel_models} threads...")
    
    with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
        # Submit all training tasks
        future_to_class = {
            executor.submit(train_single_class_parallel, class_id, args, i % args.parallel_models, run): class_id
            for i, class_id in enumerate(class_ids)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_class):
            class_id = future_to_class[future]
            try:
                success = future.result()
                if success:
                    successful_classes += 1
                    print(f"✓ Successfully completed class {class_id}")
                else:
                    failed_classes += 1
                    print(f"✗ Failed to train class {class_id}")
            except Exception as e:
                failed_classes += 1
                print(f"✗ Exception training class {class_id}: {e}")

    # Final summary
    total_script_time = time.time() - script_start_time
    print(f"\n{'='*60}")
    print(f"PARALLEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total runtime: {total_script_time/60:.1f} minutes")
    print(f"Classes trained successfully: {successful_classes}")
    print(f"Classes failed: {failed_classes}")
    print(f"Success rate: {successful_classes/(successful_classes+failed_classes)*100:.1f}%")
    print(f"Average time per class: {total_script_time/len(class_ids)/60:.1f} minutes")
    
    if run:
        run.log("total_classes_trained", successful_classes)
        run.log("total_classes_failed", failed_classes)
        run.log("total_runtime_minutes", total_script_time/60)
        run.log("parallel_threads_used", args.parallel_models)

    # Cleanup shared resources
    global SHARED_CRITERION_PERC, SHARED_TRAINSET, SHARED_VALSET
    del SHARED_CRITERION_PERC, SHARED_TRAINSET, SHARED_VALSET
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()