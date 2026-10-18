"""Independent U-Net inference; no classification, XAI, or training side effects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image
import streamlit as st


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/neurovision_unet_best.keras"
MASK_THRESHOLD = 0.5


@dataclass
class SegmentationResult:
    # Binary mask on the original image grid, with values 0 or 255.
    mask: np.ndarray
    # Area statistics use the native 256×256 thresholded prediction.
    positive_pixels: int
    total_pixels: int
    relative_area: float


def checkpoint_signature(checkpoint_path: str | Path) -> tuple[str, int, int]:
    """Resolve independently of the working directory and detect replaced weights."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {path}")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner=False, max_entries=1)
def load_segmentation_model(signature: tuple[str, int, int]) -> tuple[Any, Any]:
    """Cache only the model/lock, never patient images or per-session predictions."""
    import tensorflow as tf
    from segmentation.segmentation_utils import (
        bce_dice_loss, dice_coefficient, dice_loss, iou_score,
    )

    model = tf.keras.models.load_model(
        signature[0],
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "iou_score": iou_score,
        },
        compile=False,
        safe_mode=True,
    )
    if model.input_shape != (None, 256, 256, 3) or model.output_shape != (None, 256, 256, 1):
        raise ValueError("Expected a 256×256 RGB U-Net with a single-channel sigmoid output.")
    # The cached model is shared between Streamlit sessions. Serialize its calls.
    return model, RLock()


def segment_mri(
    image: Image.Image,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> SegmentationResult:
    """Predict at 256×256, then restore the binary mask to the original image grid."""
    import tensorflow as tf

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    # Match segmentation_utils.load_image_mask: bilinear resize, then divide by 255.
    batch = tf.image.resize(rgb, (256, 256), method="bilinear")[None, ...] / 255.0
    model, lock = load_segmentation_model(checkpoint_signature(checkpoint_path))
    with lock:
        output = np.asarray(model(batch, training=False), dtype=np.float32)
    if output.shape != (1, 256, 256, 1):
        raise ValueError(f"Unexpected segmentation output shape: {output.shape}")
    if not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise ValueError("Segmentation model returned invalid sigmoid probabilities.")

    binary = (output[0, :, :, 0] >= MASK_THRESHOLD).astype(np.uint8) * 255
    mask = np.array(Image.fromarray(binary).resize(image.size, Image.Resampling.NEAREST))
    positive_pixels = int(np.count_nonzero(binary))
    total_pixels = int(binary.size)
    return SegmentationResult(
        mask=mask,
        positive_pixels=positive_pixels,
        total_pixels=total_pixels,
        relative_area=100.0 * positive_pixels / total_pixels,
    )


def segmentation_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Tint only positive pixels; retain the MRI and its original dimensions."""
    rgb = np.array(image.convert("RGB"))
    if mask.shape != rgb.shape[:2]:
        raise ValueError("The segmentation mask must match the original MRI dimensions.")
    positive = mask > 0
    rgb[positive] = np.rint(
        0.58 * rgb[positive].astype(np.float32) + 0.42 * np.array([65, 220, 185])
    ).astype(np.uint8)
    return Image.fromarray(rgb)
