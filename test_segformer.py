"""
SegFormer Model Testing & Evaluation Script
============================================
Loads the downloaded trained SegFormer model, runs inference on
validation images, and saves results to test_results/ folder.

Outputs:
  test_results/
  ├── predictions/       # Segmentation mask overlays (PNG)
  ├── metrics.json       # mIoU, pixel accuracy, per-class IoU
  └── summary.txt        # Human-readable summary
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ADE20K 150 class names
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting",
    "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat",
    "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path",
    "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower",
    "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag"
]

# Color palette for visualization (ADE20K-style)
def generate_palette(n):
    """Generate a fixed color palette for N classes."""
    np.random.seed(42)
    palette = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # background = black
    return palette

PALETTE = generate_palette(150)


def colorize_segmentation(mask, palette=None):
    """Convert class index mask to RGB visualization."""
    if palette is None:
        palette = PALETTE
    mask = np.clip(mask, 0, len(palette) - 1)
    colored = palette[mask]
    return colored


def create_overlay(image, mask, alpha=0.5):
    """Overlay segmentation mask on original image."""
    image_np = np.array(image)
    if image_np.shape[:2] != mask.shape[:2]:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(image_np.shape[1::-1], Image.NEAREST)
        mask = np.array(mask_img)
    mask_rgb = colorize_segmentation(mask)
    overlay = (image_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay


def compute_per_class_iou(preds, labels, num_classes):
    """Compute IoU for each class."""
    ious = {}
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        iou = intersection / union if union > 0 else 0.0
        if iou > 0.001:  # Only report non-zero IoUs
            ious[ADE20K_CLASSES[cls]] = round(iou, 4)
    return ious


def compute_metrics(preds, labels, num_classes):
    """Compute mIoU and pixel accuracy."""
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    if len(labels) == 0:
        return {"mIoU": 0.0, "pixel_accuracy": 0.0, "per_class_iou": {}}

    correct = (preds == labels).sum().item()
    total = len(labels)
    pixel_acc = correct / total if total > 0 else 0.0

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)

    miou = float(np.mean(ious)) if ious else 0.0
    per_class = compute_per_class_iou(preds.cpu(), labels.cpu(), num_classes)

    return {
        "mIoU": round(miou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "per_class_iou": per_class
    }


def main():
    # Paths
    model_path = "trained_models/segformer-b0-finetuned-ade-512-512"
    output_dir = Path("test_results")
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  SegFormer Model Testing & Evaluation")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Check model path
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please ensure trained_models/segformer-b0-finetuned-ade-512-512/ exists")
        sys.exit(1)

    # Load model
    logger.info(f"Loading trained model from: {model_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
        semantic_loss_ignore_index=-100,
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load image processor
    image_processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
    )

    # Load validation dataset
    logger.info("Loading validation dataset...")
    val_hf = load_dataset("nateraw/ade20k-tiny", split="validation")
    logger.info(f"Validation samples: {len(val_hf)}")

    # Run inference
    all_preds = []
    all_labels = []
    all_losses = []
    results = []

    logger.info("\nRunning inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_hf)), desc="Testing"):
            sample = val_hf[idx]

            # Load image
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            if image.mode != "RGB":
                image = image.convert("RGB")
            orig_size = image.size  # (W, H)

            # Load and preprocess label
            label = sample["label"]
            if not isinstance(label, Image.Image):
                label = Image.fromarray(np.array(label))
            label_np = np.array(label).astype(np.int64)

            # Preprocess label: 0/255 -> -100, 1-150 -> 0-149
            valid_mask = (label_np >= 1) & (label_np <= 150)
            label_processed = np.full_like(label_np, -100, dtype=np.int64)
            label_processed[valid_mask] = label_np[valid_mask] - 1

            # Process image
            encoded = image_processor(images=image, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # (1, 150, H/4, W/4)

            # Compute loss if labels available
            label_tensor = torch.from_numpy(label_processed).long().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            # Resize labels to match logits
            label_tensor_resized = nn.functional.interpolate(
                label_tensor.float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

            loss = nn.functional.cross_entropy(
                logits, label_tensor_resized.unsqueeze(0).to(device),
                ignore_index=-100
            )
            all_losses.append(loss.item())

            # Upsample predictions to original image size
            upsampled = nn.functional.interpolate(
                logits, size=(orig_size[1], orig_size[0]),
                mode="bilinear", align_corners=False,
            )
            pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

            # Resize label to original size for metric computation
            label_resized = nn.functional.interpolate(
                label_tensor.float(),
                size=(orig_size[1], orig_size[0]),
                mode="nearest",
            ).squeeze(0).squeeze(0).long().cpu().numpy()

            all_preds.append(torch.from_numpy(pred))
            all_labels.append(torch.from_numpy(label_resized))

            # --- Save visualizations ---
            # 1. Original image
            image.save(pred_dir / f"sample_{idx:02d}_original.jpg")

            # 2. Ground truth
            gt_colored = colorize_segmentation(np.clip(label_resized, 0, 149))
            Image.fromarray(gt_colored).save(pred_dir / f"sample_{idx:02d}_ground_truth.png")

            # 3. Prediction
            pred_colored = colorize_segmentation(pred)
            Image.fromarray(pred_colored).save(pred_dir / f"sample_{idx:02d}_prediction.png")

            # 4. Overlay (prediction on original)
            overlay = create_overlay(image, pred, alpha=0.5)
            Image.fromarray(overlay).save(pred_dir / f"sample_{idx:02d}_overlay.png")

            # 5. Side-by-side comparison
            gt_resized = Image.fromarray(gt_colored).resize(image.size, Image.NEAREST)
            pred_img = Image.fromarray(pred_colored).resize(image.size, Image.NEAREST)
            overlay_img = Image.fromarray(overlay)

            combined = Image.new("RGB", (image.width * 4, image.height))
            combined.paste(image, (0, 0))
            combined.paste(gt_resized, (image.width, 0))
            combined.paste(pred_img, (image.width * 2, 0))
            combined.paste(overlay_img, (image.width * 3, 0))
            combined.save(pred_dir / f"sample_{idx:02d}_comparison.png")

            results.append({
                "sample": idx,
                "original_size": orig_size,
                "loss": round(loss.item(), 4),
            })

    # Aggregate metrics
    all_preds = torch.cat([p.flatten() for p in all_preds])
    all_labels = torch.cat([l.flatten() for l in all_labels])
    metrics = compute_metrics(all_preds, all_labels, num_classes=150)
    metrics["avg_loss"] = round(float(np.mean(all_losses)), 4)
    metrics["num_samples"] = len(val_hf)

    # --- Save metrics ---
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- Save summary ---
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  SegFormer Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Samples: {metrics['num_samples']}\n\n")
        f.write(f"Average Loss:      {metrics['avg_loss']:.4f}\n")
        f.write(f"mIoU:              {metrics['mIoU']:.4f}\n")
        f.write(f"Pixel Accuracy:    {metrics['pixel_accuracy']:.4f}\n\n")
        f.write("Per-Class IoU (non-zero classes):\n")
        f.write("-" * 40 + "\n")
        for cls_name, iou in sorted(metrics["per_class_iou"].items(), key=lambda x: -x[1]):
            bar = "█" * int(iou * 50) + "░" * (50 - int(iou * 50))
            f.write(f"  {cls_name:<30s} {iou:.4f}  {bar}\n")
        f.write("\n")
        f.write("Per-Sample Results:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            f.write(f"  Sample {r['sample']:02d}: loss={r['loss']:.4f}, "
                     f"size={r['original_size']}\n")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"  mIoU:           {metrics['mIoU']:.4f}")
    logger.info(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    logger.info(f"  Avg Loss:       {metrics['avg_loss']:.4f}")
    logger.info(f"  Samples:        {metrics['num_samples']}")
    logger.info(f"\n  Results saved to: {output_dir.resolve()}")
    logger.info(f"    - {len(list(pred_dir.glob('*.png')))} visualization files")
    logger.info(f"    - metrics.json")
    logger.info(f"    - summary.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
