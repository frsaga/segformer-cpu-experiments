"""
SegFormer Training Script
==========================
SegFormer semantic segmentation training on ADE20K dataset.
Uses HuggingFace Transformers for the SegFormer model implementation.

Supports:
- nateraw/ade20k-tiny: Small 20-sample subset for quick testing (auto-download)
- Full ADE20K: Manual download from official source for real training
- CPU and GPU training (auto-detected)

Quick start:
    # Test with tiny dataset (10 train + 10 val samples)
    python train_segformer_cpu.py

    # Train with more epochs
    python train_segformer_cpu.py --num_epochs 10 --dataset tiny

Usage:
    python train_segformer_cpu.py --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerConfig,
)
from datasets import load_dataset, concatenate_datasets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SegFormer on ADE20K semantic segmentation"
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="nvidia/mit-b0",
        help="Pretrained backbone: nvidia/mit-b0 through mit-b5"
    )
    parser.add_argument(
        "--num_classes", type=int, default=150,
        help="Number of segmentation classes (ADE20K = 150)"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="tiny",
        choices=["tiny", "full"],
        help="Dataset: 'tiny' = nateraw/ade20k-tiny (20 samples), "
             "'full' = local ADE20K in data/ directory"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/ade/ADEChallengeData2016",
        help="Path to full ADE20K dataset (for --dataset full)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Training image size (resized to square)"
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', or 'cuda'")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./output_segformer")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)

    return parser.parse_args()


class SegmentationDataset(Dataset):
    """PyTorch Dataset wrapper for segmentation data from HuggingFace.

    Handles ADE20K label format:
    - 0 = unlabeled/background → -100 (ignore)
    - 1-150 = semantic classes → 0-149
    - 255 = unlabeled → -100 (ignore)

    Uses image processor for image preprocessing only; labels are handled
    manually to avoid issues with PIL's inability to store negative values.
    """

    def __init__(self, hf_dataset, image_processor, num_classes=150, image_size=512):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def _preprocess_label(self, label_np):
        """Convert ADE20K raw labels to 0-indexed training labels.

        Input:  uint8 array with values {0, 1..150, 255}
        Output: int64 array with values {-100, 0..149}
        """
        label_np = label_np.astype(np.int64)

        # Create mask of valid classes (1 to num_classes)
        valid_mask = (label_np >= 1) & (label_np <= self.num_classes)

        # Default all to ignore index
        processed = np.full_like(label_np, -100, dtype=np.int64)

        # Set valid classes (subtract 1 for 0-indexing: 1..150 → 0..149)
        processed[valid_mask] = label_np[valid_mask] - 1

        return processed

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # --- Load image ---
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # --- Load and preprocess label ---
        label = example["label"]
        if not isinstance(label, Image.Image):
            label = Image.fromarray(np.array(label))

        label_np = np.array(label)
        label_processed = self._preprocess_label(label_np)  # int64, -100..149

        # --- Process image with image processor (image only, no labels) ---
        try:
            encoded = self.image_processor(
                images=image,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning(f"Failed to process image {idx}: {e}")
            return {
                "pixel_values": torch.zeros(3, self.image_size, self.image_size),
                "labels": torch.full((self.image_size, self.image_size), -100, dtype=torch.long),
            }

        pixel_values = encoded["pixel_values"].squeeze(0)

        # --- Manually resize label to match image processor output ---
        # Image processor resizes to (image_size, image_size)
        label_tensor = torch.from_numpy(label_processed).float().unsqueeze(0).unsqueeze(0)
        # Use NEAREST interpolation for labels to preserve class values
        label_resized = nn.functional.interpolate(
            label_tensor,
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        return {"pixel_values": pixel_values, "labels": label_resized}


def load_tiny_dataset():
    """Load nateraw/ade20k-tiny from HuggingFace."""
    logger.info("Loading nateraw/ade20k-tiny dataset...")

    train_ds = load_dataset("nateraw/ade20k-tiny", split="train")
    val_ds = load_dataset("nateraw/ade20k-tiny", split="validation")

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Detect number of classes from the data
    all_label_values = set()
    for i in range(len(train_ds)):
        label_np = np.array(train_ds[i]["label"])
        all_label_values.update(np.unique(label_np).tolist())
    for i in range(len(val_ds)):
        label_np = np.array(val_ds[i]["label"])
        all_label_values.update(np.unique(label_np).tolist())

    num_classes = max(all_label_values)
    logger.info(f"Detected classes from labels: max_value={num_classes}, unique_values={len(all_label_values)}")

    return train_ds, val_ds, num_classes


def load_full_ade20k(data_root):
    """Load full ADE20K from local directory."""
    data_root = Path(data_root)

    if not data_root.exists():
        logger.error(f"ADE20K not found at {data_root}")
        logger.info("Please download from: "
                     "https://groups.csail.mit.edu/vision/datasets/ADE20K/")
        logger.info("Expected structure:")
        logger.info(f"  {data_root}/")
        logger.info("    images/training/")
        logger.info("    images/validation/")
        logger.info("    annotations/training/")
        logger.info("    annotations/validation/")
        raise FileNotFoundError(f"ADE20K not found at {data_root}")

    from pathlib import Path

    # Gather image paths
    train_imgs = sorted((data_root / "images" / "training").glob("*.jpg"))
    train_anns = sorted((data_root / "annotations" / "training").glob("*.png"))
    val_imgs = sorted((data_root / "images" / "validation").glob("*.jpg"))
    val_anns = sorted((data_root / "annotations" / "validation").glob("*.png"))

    # Create HF-style dictionaries
    train_data = [
        {"image": str(img), "label": str(ann)}
        for img, ann in zip(train_imgs, train_anns)
    ]
    val_data = [
        {"image": str(img), "label": str(ann)}
        for img, ann in zip(val_imgs, val_anns)
    ]

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_list(train_data)
    val_ds = HFDataset.from_list(val_data)

    logger.info(f"Full ADE20K - Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds, 150


def create_image_processor(image_size=512):
    """Create SegFormer image processor with ADE20K normalization.

    Note: do_reduce_labels is disabled because we handle label preprocessing
    manually (0/255→-100, 1-150→0-149) before passing to the model.
    """
    return SegformerImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
    )


def compute_metrics(preds, labels, num_classes):
    """Compute mIoU and pixel accuracy."""
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    if len(labels) == 0:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0}

    # Pixel accuracy
    correct = (preds == labels).sum().item()
    total = len(labels)
    pixel_acc = correct / total if total > 0 else 0.0

    # mIoU (mean Intersection over Union)
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)

    miou = float(np.mean(ious)) if ious else 0.0
    return {"mIoU": miou, "pixel_accuracy": pixel_acc}


def train(args):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("  SegFormer Semantic Segmentation Training")
    logger.info("=" * 60)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load dataset
    if args.dataset == "tiny":
        train_hf, val_hf, num_classes = load_tiny_dataset()
    else:
        train_hf, val_hf, num_classes = load_full_ade20k(args.data_root)

    args.num_classes = num_classes
    logger.info(f"Number of classes: {num_classes}")

    # Create image processor
    image_processor = create_image_processor(args.image_size)

    # Create PyTorch datasets
    train_dataset = SegmentationDataset(train_hf, image_processor, num_classes, args.image_size)
    val_dataset = SegmentationDataset(val_hf, image_processor, num_classes, args.image_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Load model
    logger.info(f"Loading SegFormer: {args.model_name}...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            semantic_loss_ignore_index=-100,
        )
        logger.info("Loaded pretrained backbone weights!")
    except Exception as e:
        logger.warning(f"Could not load pretrained: {e}")
        logger.info("Creating model from scratch...")
        config = SegformerConfig(
            num_labels=num_classes,
            semantic_loss_ignore_index=-100,
        )
        model = SegformerForSemanticSegmentation(config)

    model = model.to(device)

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # Optimizer (same settings as original SegFormer paper)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Resume checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)

    # LR scheduler (polynomial decay - same as paper)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=1.0)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nTraining: {args.num_epochs} epochs, {total_steps} total steps")
    logger.info(f"Saving to: {output_dir}")
    logger.info("=" * 60)

    best_miou = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            # Log
            if step % args.log_steps == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Save checkpoint
            if global_step % args.save_steps == 0:
                ckpt_path = output_dir / f"checkpoint-step-{global_step}.pt"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, ckpt_path)
                logger.info(f"Checkpoint -> {ckpt_path}")

        # End of epoch summary
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

        # Validation
        logger.info("Validating...")
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                # Get predictions with upsampling
                logits = outputs.logits
                upsampled = nn.functional.interpolate(
                    logits, size=labels.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                preds = upsampled.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(all_preds, all_labels, num_classes)
        metrics["val_loss"] = val_loss / len(val_loader)

        logger.info(
            f"Validation - Loss: {metrics['val_loss']:.4f}, "
            f"mIoU: {metrics['mIoU']:.4f}, "
            f"Pixel Acc: {metrics['pixel_accuracy']:.4f}"
        )

        # Save best model
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }, best_path)
            logger.info(f"*** New best model! mIoU: {best_miou:.4f} ***")

        # Save epoch checkpoint
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, output_dir / f"checkpoint-epoch-{epoch+1}.pt")

    logger.info("=" * 60)
    logger.info(f"Training complete! Best mIoU: {best_miou:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 60)
    return best_miou


if __name__ == "__main__":
    args = parse_args()
    train(args)
