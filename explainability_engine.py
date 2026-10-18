
"""
NeuroVision AI — Advanced Explainability Engine

Research-only explainability utilities for the NeuroVision MRI
classifier. The engine deliberately separates model confidence from
explanation evidence.

Implemented:
- Logit-based Grad-CAM
- Multi-baseline, logit-based Integrated Gradients
- Contrastive Grad-CAM
- Central-anatomy proxy and border-dependency diagnostics
- Anatomy-constrained and matched off-anatomy perturbation
- Multi-baseline deletion / insertion faithfulness
- Baseline sanity checks
- Cross-method agreement
- Mild-input stability
- Transparent heuristic Explanation Trust score

Limitations:
- This is not tumor segmentation or clinical localization.
- The central-anatomy region is a heuristic proxy, not a brain mask.
- Perturbation can create distribution shift.
- Explanation Trust is not confidence, accuracy, or a validated clinical score.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

EPS = 1e-8


# ============================================================
# GENERIC HELPERS
# ============================================================

def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D map, got {arr.shape}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def resize_map(values: np.ndarray, width: int, height: int) -> np.ndarray:
    out = cv2.resize(
        np.asarray(values, dtype=np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    )
    return normalize_map(out)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    m = (normalize_map(heatmap) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    image = np.asarray(original_rgb, dtype=np.uint8)
    h, w = image.shape[:2]
    heat = resize_map(heatmap, w, h)
    colored = colorize_heatmap(heat)
    out = (
        (1.0 - float(alpha)) * image.astype(np.float32)
        + float(alpha) * colored.astype(np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)

    if a.shape != b.shape:
        b = cv2.resize(
            b,
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != a.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        a, b = a[mask], b[mask]
    else:
        a, b = a.reshape(-1), b.reshape(-1)

    if len(a) < 2:
        return 0.0

    if float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return 0.0

    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _trapz_compat(y_values, x_values) -> float:
    y = np.asarray(y_values, dtype=np.float32)
    x = np.asarray(x_values, dtype=np.float32)
    if len(y) < 2:
        return 0.0
    widths = x[1:] - x[:-1]
    heights = (y[1:] + y[:-1]) * 0.5
    return float(np.sum(widths * heights))


# ============================================================
# MODEL EVIDENCE / LOGITS
# ============================================================

def model_probabilities(
    model: tf.keras.Model,
    batch: tf.Tensor | np.ndarray,
) -> np.ndarray:
    output = model(
        tf.convert_to_tensor(batch, dtype=tf.float32),
        training=False,
    )
    return np.asarray(output.numpy(), dtype=np.float32)


def classifier_logits_tensor(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    batch: tf.Tensor | np.ndarray,
) -> tf.Tensor:
    """
    Forward to the pre-softmax logits of the final Dense layer.
    This avoids softmax saturation when measuring model evidence.
    """
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)

    augmented = classifier_layers["augmentation"](
        batch,
        training=False,
    )

    _, base_output = grad_model(
        augmented,
        training=False,
    )

    x = classifier_layers["gap"](base_output)
    x = classifier_layers["dropout1"](x, training=False)
    x = classifier_layers["dense"](x)
    x = classifier_layers["dropout2"](x, training=False)

    prediction_layer = classifier_layers["prediction"]

    logits = tf.linalg.matmul(
        x,
        prediction_layer.kernel,
    )

    if getattr(prediction_layer, "use_bias", False):
        logits = tf.nn.bias_add(
            logits,
            prediction_layer.bias,
        )

    return logits


def classifier_logits(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    batch: tf.Tensor | np.ndarray,
) -> np.ndarray:
    return np.asarray(
        classifier_logits_tensor(
            grad_model,
            classifier_layers,
            batch,
        ).numpy(),
        dtype=np.float32,
    )


def target_margin_from_logits(
    logits: np.ndarray,
    target_index: int,
) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    target = arr[:, int(target_index)]

    mask = np.ones(arr.shape[1], dtype=bool)
    mask[int(target_index)] = False

    runner_up = np.max(arr[:, mask], axis=1)
    return target - runner_up


# ============================================================
# MRI REGION HEURISTICS
# ============================================================

def build_foreground_mask(
    image_batch: tf.Tensor,
    threshold: float = 10.0,
) -> np.ndarray:
    image = image_batch[0].numpy().astype(np.float32)
    gray = cv2.cvtColor(
        np.clip(image, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )

    mask = gray > float(threshold)
    if int(mask.sum()) < 100:
        mask = np.ones_like(gray, dtype=bool)

    return mask


def build_anatomical_focus_proxy(
    image_batch: tf.Tensor,
) -> np.ndarray:
    """
    Conservative central-anatomy proxy.
    It is NOT a brain segmentation mask.
    """
    foreground = build_foreground_mask(image_batch)
    h, w = foreground.shape

    clean = foreground.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )
    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    ellipse = np.zeros((h, w), dtype=np.uint8)
    center = (int(round(w * 0.50)), int(round(h * 0.43)))
    axes = (
        max(1, int(round(w * 0.43))),
        max(1, int(round(h * 0.40))),
    )

    cv2.ellipse(
        ellipse,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1,
    )

    proxy = (clean > 0) & (ellipse > 0)

    minimum_pixels = max(
        100,
        int(0.04 * h * w),
    )

    if int(proxy.sum()) < minimum_pixels:
        proxy = foreground & (ellipse > 0)

    if int(proxy.sum()) < 100:
        proxy = foreground.copy()

    return proxy.astype(bool)


def build_border_mask(
    height: int,
    width: int,
    border_fraction: float = 0.12,
) -> np.ndarray:
    fraction = float(np.clip(border_fraction, 0.02, 0.30))
    by = max(1, int(round(height * fraction)))
    bx = max(1, int(round(width * fraction)))

    mask = np.zeros((height, width), dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    return mask


def attention_mass_ratio(
    saliency_map: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    saliency = normalize_map(saliency_map)
    region = np.asarray(region_mask, dtype=bool)

    if region.shape != saliency.shape:
        region = cv2.resize(
            region.astype(np.uint8),
            (saliency.shape[1], saliency.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    total = float(np.sum(saliency))
    if total < EPS:
        return 0.0

    inside = float(np.sum(saliency[region]))
    return float(np.clip(inside / total, 0.0, 1.0))


def border_dependency_diagnostics(
    gradcam_map_2d: np.ndarray,
    integrated_gradients_map_2d: np.ndarray,
    anatomical_proxy: np.ndarray,
    foreground_mask: np.ndarray,
    border_fraction: float = 0.12,
) -> dict:
    h, w = gradcam_map_2d.shape
    border = build_border_mask(h, w, border_fraction)

    gradcam_anatomy = attention_mass_ratio(
        gradcam_map_2d,
        anatomical_proxy,
    )
    ig_anatomy = attention_mass_ratio(
        integrated_gradients_map_2d,
        anatomical_proxy,
    )

    gradcam_border = attention_mass_ratio(
        gradcam_map_2d,
        border,
    )
    ig_border = attention_mass_ratio(
        integrated_gradients_map_2d,
        border,
    )

    gradcam_foreground = attention_mass_ratio(
        gradcam_map_2d,
        foreground_mask,
    )
    ig_foreground = attention_mass_ratio(
        integrated_gradients_map_2d,
        foreground_mask,
    )

    mean_anatomy = float((gradcam_anatomy + ig_anatomy) / 2.0)
    mean_border = float((gradcam_border + ig_border) / 2.0)
    mean_foreground = float(
        (gradcam_foreground + ig_foreground) / 2.0
    )

    if mean_border >= 0.25 or mean_anatomy < 0.45:
        risk = "High"
    elif mean_border >= 0.12 or mean_anatomy < 0.60:
        risk = "Moderate"
    else:
        risk = "Low"

    # Keep old key for app compatibility, but semantics are now explicit:
    # this is border-dependency risk, not all possible shortcut learning.
    return {
        "gradcam_anatomy_ratio": gradcam_anatomy,
        "ig_anatomy_ratio": ig_anatomy,
        "mean_anatomy_ratio": mean_anatomy,
        "gradcam_border_ratio": gradcam_border,
        "ig_border_ratio": ig_border,
        "mean_border_ratio": mean_border,
        "gradcam_foreground_ratio": gradcam_foreground,
        "ig_foreground_ratio": ig_foreground,
        "mean_foreground_ratio": mean_foreground,
        "shortcut_risk": risk,
        "border_dependency_risk": risk,
        "border_fraction": float(border_fraction),
    }


# ============================================================
# LOGIT-BASED GRAD-CAM
# ============================================================

def gradcam_map(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    batch: tf.Tensor,
    target_index: int,
) -> np.ndarray:
    augmented = classifier_layers["augmentation"](
        batch,
        training=False,
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(
            augmented,
            training=False,
        )

        x = classifier_layers["gap"](base_output)
        x = classifier_layers["dropout1"](x, training=False)
        x = classifier_layers["dense"](x)
        x = classifier_layers["dropout2"](x, training=False)

        prediction_layer = classifier_layers["prediction"]

        logits = tf.linalg.matmul(
            x,
            prediction_layer.kernel,
        )

        if getattr(prediction_layer, "use_bias", False):
            logits = tf.nn.bias_add(
                logits,
                prediction_layer.bias,
            )

        class_score = logits[:, int(target_index)]

    gradients = tape.gradient(
        class_score,
        conv_output,
    )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    activations = conv_output[0]

    heatmap = tf.reduce_sum(
        activations * pooled,
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap).numpy()
    return normalize_map(heatmap)


# ============================================================
# LOGIT-BASED INTEGRATED GRADIENTS
# ============================================================

def _integrated_gradients_single_baseline(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    input_batch: tf.Tensor,
    target_index: int,
    baseline_batch: tf.Tensor,
    steps: int = 32,
) -> np.ndarray:
    input_tensor = tf.cast(input_batch, tf.float32)
    baseline_tensor = tf.cast(baseline_batch, tf.float32)

    alphas = tf.reshape(
        tf.linspace(0.0, 1.0, int(steps) + 1),
        (-1, 1, 1, 1),
    )

    delta = input_tensor - baseline_tensor
    interpolated = baseline_tensor + alphas * delta

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        logits = classifier_logits_tensor(
            grad_model,
            classifier_layers,
            interpolated,
        )
        target_scores = logits[:, int(target_index)]

    gradients = tape.gradient(
        target_scores,
        interpolated,
    )

    if gradients is None:
        raise RuntimeError(
            "Integrated Gradients could not calculate input gradients."
        )

    averaged_grads = (gradients[:-1] + gradients[1:]) / 2.0
    mean_grads = tf.reduce_mean(
        averaged_grads,
        axis=0,
        keepdims=True,
    )

    integrated = (delta * mean_grads)[0]

    attribution = tf.reduce_sum(
        tf.abs(integrated),
        axis=-1,
    ).numpy()

    return normalize_map(attribution)


def integrated_gradients_map(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    input_batch: tf.Tensor,
    target_index: int,
    steps: int = 32,
) -> np.ndarray:
    input_tensor = tf.cast(input_batch, tf.float32)

    zero_baseline = tf.zeros_like(input_tensor)

    input_np = input_tensor[0].numpy().astype(np.float32)

    heavy_blur = cv2.GaussianBlur(
        input_np,
        (81, 81),
        sigmaX=0,
    ).astype(np.float32)

    blur_baseline = tf.expand_dims(
        tf.convert_to_tensor(
            heavy_blur,
            dtype=tf.float32,
        ),
        axis=0,
    )

    zero_map = _integrated_gradients_single_baseline(
        grad_model=grad_model,
        classifier_layers=classifier_layers,
        input_batch=input_tensor,
        target_index=target_index,
        baseline_batch=zero_baseline,
        steps=steps,
    )

    blur_map = _integrated_gradients_single_baseline(
        grad_model=grad_model,
        classifier_layers=classifier_layers,
        input_batch=input_tensor,
        target_index=target_index,
        baseline_batch=blur_baseline,
        steps=steps,
    )

    return normalize_map((zero_map + blur_map) / 2.0)


# ============================================================
# PERTURBATION
# ============================================================

def perturb_top_salient_pixels(
    input_batch: tf.Tensor,
    saliency_map: np.ndarray,
    eligible_mask: np.ndarray,
    pixel_count: int,
    blur_kernel: int = 81,
) -> tuple[tf.Tensor, np.ndarray]:
    image = input_batch[0].numpy().astype(np.float32)
    h, w = image.shape[:2]

    saliency = resize_map(saliency_map, w, h)

    eligible = cv2.resize(
        np.asarray(eligible_mask, dtype=np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    candidates = np.flatnonzero(eligible.reshape(-1))

    selected_mask = np.zeros((h, w), dtype=bool)

    if len(candidates) == 0:
        return (
            tf.expand_dims(
                tf.convert_to_tensor(image, dtype=tf.float32),
                axis=0,
            ),
            selected_mask,
        )

    count = int(np.clip(pixel_count, 1, len(candidates)))

    flat_saliency = saliency.reshape(-1)
    ranked = candidates[
        np.argsort(flat_saliency[candidates])[::-1]
    ]
    selected = ranked[:count]
    selected_mask.reshape(-1)[selected] = True

    k = int(blur_kernel)
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(
        image,
        (k, k),
        sigmaX=0,
    )

    perturbed = image.copy()
    perturbed[selected_mask] = blurred[selected_mask]

    return (
        tf.expand_dims(
            tf.convert_to_tensor(perturbed, dtype=tf.float32),
            axis=0,
        ),
        selected_mask,
    )


def probability_influence_strength(
    original_probability: float,
    perturbed_probability: float,
) -> dict:
    original = max(float(original_probability), EPS)
    perturbed = float(perturbed_probability)

    absolute_drop = original - perturbed
    relative_drop = max(absolute_drop / original, 0.0)

    if relative_drop >= 0.40:
        label = "Strong"
    elif relative_drop >= 0.20:
        label = "Moderate"
    elif relative_drop >= 0.10:
        label = "Limited"
    else:
        label = "Weak"

    return {
        "label": label,
        "absolute_drop": float(absolute_drop),
        "relative_drop": float(relative_drop),
    }


def margin_influence_strength(
    original_margin: float,
    perturbed_margin: float,
) -> dict:
    denominator = max(abs(float(original_margin)), 1.0)
    margin_drop = float(original_margin) - float(perturbed_margin)
    relative_drop = max(margin_drop / denominator, 0.0)

    if relative_drop >= 0.40:
        label = "Strong"
    elif relative_drop >= 0.20:
        label = "Moderate"
    elif relative_drop >= 0.10:
        label = "Limited"
    else:
        label = "Weak"

    return {
        "label": label,
        "margin_drop": float(margin_drop),
        "relative_drop": float(relative_drop),
    }


# ============================================================
# MULTI-BASELINE DELETION / INSERTION
# ============================================================

def _baseline_images(
    image: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, np.ndarray]:
    light = cv2.GaussianBlur(
        image,
        (31, 31),
        sigmaX=0,
    ).astype(np.float32)

    heavy = cv2.GaussianBlur(
        image,
        (81, 81),
        sigmaX=0,
    ).astype(np.float32)

    local_mean = image.copy().astype(np.float32)

    fg = np.asarray(foreground, dtype=bool)

    if int(fg.sum()) > 0:
        mean_rgb = np.mean(
            image[fg],
            axis=0,
        ).astype(np.float32)
        local_mean[fg] = mean_rgb
    else:
        local_mean[:] = np.mean(
            image.reshape(-1, 3),
            axis=0,
        )

    return {
        "light_blur": light,
        "heavy_blur": heavy,
        "local_mean": local_mean,
    }


def _single_baseline_curves(
    model: tf.keras.Model,
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    image: np.ndarray,
    baseline: np.ndarray,
    saliency: np.ndarray,
    foreground: np.ndarray,
    target_index: int,
    fractions: tuple[float, ...],
    original_probability: float,
    original_margin: float,
) -> dict:
    h, w = image.shape[:2]

    flat_saliency = saliency.reshape(-1)
    flat_foreground = foreground.reshape(-1)

    foreground_indices = np.flatnonzero(flat_foreground)

    if len(foreground_indices) == 0:
        foreground_indices = np.arange(flat_saliency.size)

    ordered = foreground_indices[
        np.argsort(flat_saliency[foreground_indices])[::-1]
    ]

    original_flat = image.reshape(-1, 3)
    baseline_flat = baseline.reshape(-1, 3)

    deletion_images = []
    insertion_images = []

    total = len(ordered)

    for fraction in fractions:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        count = int(round(fraction * total))
        selected = ordered[:count]

        deleted = original_flat.copy()
        if count > 0:
            deleted[selected] = baseline_flat[selected]

        inserted = baseline_flat.copy()
        if count > 0:
            inserted[selected] = original_flat[selected]

        deletion_images.append(
            deleted.reshape(h, w, 3)
        )
        insertion_images.append(
            inserted.reshape(h, w, 3)
        )

    combined = np.concatenate(
        [
            np.stack(deletion_images, axis=0),
            np.stack(insertion_images, axis=0),
        ],
        axis=0,
    )

    probabilities = model_probabilities(
        model,
        combined,
    )

    logits = classifier_logits(
        grad_model,
        classifier_layers,
        combined,
    )

    n = len(fractions)

    deletion_probs = probabilities[:n, int(target_index)]
    insertion_probs = probabilities[n:, int(target_index)]

    deletion_logits_all = logits[:n]
    insertion_logits_all = logits[n:]

    deletion_target_logits = deletion_logits_all[:, int(target_index)]
    insertion_target_logits = insertion_logits_all[:, int(target_index)]

    deletion_margins = target_margin_from_logits(
        deletion_logits_all,
        target_index,
    )
    insertion_margins = target_margin_from_logits(
        insertion_logits_all,
        target_index,
    )

    x = np.asarray(fractions, dtype=np.float32)

    deletion_auc = _trapz_compat(deletion_probs, x)
    insertion_auc = _trapz_compat(insertion_probs, x)

    # Normalize target-vs-runner-up margin by the original positive margin.
    # This gives a bounded evidence-retention curve that is less affected
    # by softmax saturation than probability alone.
    original_margin_denominator = max(
        float(original_margin),
        EPS,
    )

    deletion_margin_support = np.clip(
        deletion_margins
        /
        original_margin_denominator,
        0.0,
        1.0,
    )

    insertion_margin_support = np.clip(
        insertion_margins
        /
        original_margin_denominator,
        0.0,
        1.0,
    )

    deletion_margin_auc = _trapz_compat(
        deletion_margin_support,
        x,
    )

    insertion_margin_auc = _trapz_compat(
        insertion_margin_support,
        x,
    )

    final_prob = float(deletion_probs[-1])
    final_logit = float(deletion_target_logits[-1])
    final_margin = float(deletion_margins[-1])

    probability_retention = (
        final_prob
        /
        max(float(original_probability), EPS)
    )

    if abs(float(original_margin)) > 1.0:
        margin_retention = (
            final_margin
            /
            float(original_margin)
        )
    else:
        margin_retention = 0.0

    same_top_class = (
        int(np.argmax(deletion_logits_all[-1]))
        ==
        int(target_index)
    )

    if (
        same_top_class
        and
        final_prob >= 0.80
        and
        margin_retention >= 0.60
    ):
        status = "Weak deletion baseline"
        status_level = "Caution"
    elif (
        same_top_class
        and
        (
            final_prob >= 0.50
            or
            margin_retention >= 0.30
        )
    ):
        status = "Partial evidence removal"
        status_level = "Review"
    else:
        status = "Substantial evidence removal"
        status_level = "OK"

    return {
        "fractions": [float(v) for v in x],
        "deletion_probabilities": [float(v) for v in deletion_probs],
        "insertion_probabilities": [float(v) for v in insertion_probs],
        "deletion_target_logits": [
            float(v) for v in deletion_target_logits
        ],
        "insertion_target_logits": [
            float(v) for v in insertion_target_logits
        ],
        "deletion_margins": [float(v) for v in deletion_margins],
        "insertion_margins": [float(v) for v in insertion_margins],
        "deletion_auc": float(deletion_auc),
        "insertion_auc": float(insertion_auc),
        "deletion_margin_support": [
            float(v)
            for v in deletion_margin_support
        ],
        "insertion_margin_support": [
            float(v)
            for v in insertion_margin_support
        ],
        "deletion_margin_auc": float(
            deletion_margin_auc
        ),
        "insertion_margin_auc": float(
            insertion_margin_auc
        ),
        "final_target_probability": final_prob,
        "final_target_logit": final_logit,
        "final_target_margin": final_margin,
        "probability_retention": float(probability_retention),
        "margin_retention": float(margin_retention),
        "same_top_class": bool(same_top_class),
        "status": status,
        "status_level": status_level,
    }


def deletion_insertion_curves(
    model: tf.keras.Model,
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    input_batch: tf.Tensor,
    saliency_map: np.ndarray,
    target_index: int,
    foreground_mask: np.ndarray,
    original_probability: float,
    original_margin: float,
    fractions: tuple[float, ...] = (
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.65,
        0.80,
        1.00,
    ),
) -> dict:
    """
    Multi-baseline deletion/insertion.

    Primary displayed baseline is heavy Gaussian blur. Light blur and
    local-mean replacement are sensitivity checks.
    """
    image = input_batch[0].numpy().astype(np.float32)
    h, w = image.shape[:2]

    saliency = resize_map(
        saliency_map,
        w,
        h,
    )

    foreground = cv2.resize(
        foreground_mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    baselines = _baseline_images(
        image,
        foreground,
    )

    results = {}

    names = {
        "light_blur": "Light Gaussian blur",
        "heavy_blur": "Heavy Gaussian blur",
        "local_mean": "Local mean replacement",
    }

    for key, baseline in baselines.items():
        result = _single_baseline_curves(
            model=model,
            grad_model=grad_model,
            classifier_layers=classifier_layers,
            image=image,
            baseline=baseline,
            saliency=saliency,
            foreground=foreground,
            target_index=target_index,
            fractions=fractions,
            original_probability=original_probability,
            original_margin=original_margin,
        )
        result["display_name"] = names[key]
        results[key] = result

    primary_key = "heavy_blur"
    primary = results[primary_key]

    caution_count = sum(
        1
        for result in results.values()
        if result["status_level"] == "Caution"
    )

    if caution_count >= 2:
        warning_level = "Caution"
        warning = (
            "Multiple deletion baselines retain substantial target-class "
            "evidence; probability AUC may be insufficiently discriminative "
            "for this sample."
        )
    elif caution_count == 1:
        warning_level = "Review"
        warning = (
            "At least one deletion baseline retains substantial target-class "
            "evidence; interpret AUC together with logit and margin changes."
        )
    else:
        warning_level = "OK"
        warning = (
            "Deletion baselines produced substantial evidence removal under "
            "the current diagnostic setup."
        )

    return {
        # Backward-compatible primary curve fields
        "fractions": primary["fractions"],
        "deletion_probabilities": primary["deletion_probabilities"],
        "insertion_probabilities": primary["insertion_probabilities"],
        "deletion_auc": primary["deletion_auc"],
        "insertion_auc": primary["insertion_auc"],
        "deletion_margin_auc": primary[
            "deletion_margin_auc"
        ],
        "insertion_margin_auc": primary[
            "insertion_margin_auc"
        ],

        # New logit/margin fields
        "deletion_target_logits": primary["deletion_target_logits"],
        "insertion_target_logits": primary["insertion_target_logits"],
        "deletion_margins": primary["deletion_margins"],
        "insertion_margins": primary["insertion_margins"],

        "primary_baseline": primary_key,
        "baselines": results,
        "baseline_warning": warning,
        "baseline_warning_level": warning_level,
    }


# ============================================================
# STABILITY
# ============================================================

def explanation_stability(
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    input_batch: tf.Tensor,
    target_index: int,
    reference_map: np.ndarray,
    foreground_mask: np.ndarray,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    original = input_batch[0].numpy().astype(np.float32)

    variants = [
        np.clip(original * 0.97, 0, 255),
        np.clip(original * 1.03, 0, 255),
    ]

    noise = rng.normal(
        loc=0.0,
        scale=2.0,
        size=original.shape,
    ).astype(np.float32)

    variants.append(
        np.clip(original + noise, 0, 255)
    )

    correlations = []

    for variant in variants:
        batch = tf.expand_dims(
            tf.convert_to_tensor(
                variant,
                dtype=tf.float32,
            ),
            axis=0,
        )

        variant_map = gradcam_map(
            grad_model,
            classifier_layers,
            batch,
            target_index,
        )

        variant_map = resize_map(
            variant_map,
            reference_map.shape[1],
            reference_map.shape[0],
        )

        correlations.append(
            safe_correlation(
                reference_map,
                variant_map,
                valid_mask=foreground_mask,
            )
        )

    return {
        "correlations": [float(v) for v in correlations],
        "mean_correlation": float(np.mean(correlations)),
    }


# ============================================================
# TRUST DIAGNOSTIC
# ============================================================

def explanation_trust_score(
    anatomy_ratio: float,
    border_ratio: float,
    agreement: float,
    stability: float,
    margin_relative_drop: float,
    deletion_margin_auc: float,
    baseline_warning_level: str,
) -> dict:
    """
    Transparent heuristic score, NOT a validated XAI/clinical metric.

    Weights:
      20% central-anatomy focus
      15% border independence
      20% cross-method agreement
      10% mild-input stability
      20% logit-margin perturbation influence
      15% logit-margin deletion faithfulness
    """
    anatomy_component = float(
        np.clip(
            (float(anatomy_ratio) - 0.30) / 0.50,
            0.0,
            1.0,
        )
    )

    border_component = float(
        np.clip(
            1.0 - (float(border_ratio) / 0.30),
            0.0,
            1.0,
        )
    )

    agreement_component = float(
        np.clip(float(agreement), 0.0, 1.0)
    )

    stability_component = float(
        np.clip(float(stability), 0.0, 1.0)
    )

    influence_component = float(
        np.clip(
            float(margin_relative_drop) / 0.40,
            0.0,
            1.0,
        )
    )

    deletion_component = float(
        np.clip(
            (0.95 - float(deletion_margin_auc)) / 0.45,
            0.0,
            1.0,
        )
    )

    components = {
        "anatomy_focus": anatomy_component,
        "border_independence": border_component,
        "method_agreement": agreement_component,
        "stability": stability_component,
        "logit_margin_influence": influence_component,
        "deletion_faithfulness": deletion_component,
    }

    score = 100.0 * (
        0.20 * anatomy_component
        + 0.15 * border_component
        + 0.20 * agreement_component
        + 0.10 * stability_component
        + 0.20 * influence_component
        + 0.15 * deletion_component
    )

    reasons = []

    if anatomy_ratio < 0.60:
        reasons.append("limited central-anatomy focus")

    if border_ratio >= 0.12:
        reasons.append("elevated border attention")

    if agreement < 0.35:
        reasons.append("low cross-method agreement")

    if stability < 0.60:
        reasons.append("limited explanation stability")

    if margin_relative_drop < 0.10:
        reasons.append("weak logit-margin perturbation influence")

    if deletion_margin_auc > 0.90:
        reasons.append("weak logit-margin deletion faithfulness")

    if baseline_warning_level == "Caution":
        reasons.append("deletion baseline retains substantial class evidence")

    # Guardrail: do not allow anatomy/stability alone to make weak
    # faithfulness look trustworthy.
    if (
        agreement <= 0.0
        and
        margin_relative_drop < 0.10
        and
        deletion_margin_auc > 0.90
    ):
        score = min(score, 49.0)

    if baseline_warning_level == "Caution":
        score = min(score, 54.0)

    score = float(np.clip(score, 0.0, 100.0))

    if score >= 75.0:
        label = "Strong"
    elif score >= 55.0:
        label = "Moderate"
    else:
        label = "Caution"

    if not reasons:
        reasons.append("no major heuristic warning detected")

    return {
        "score": score,
        "label": label,
        "components": components,
        "reasons": reasons,
    }


# ============================================================
# VISUAL DIAGNOSTICS
# ============================================================

def render_proxy_outline(
    original_rgb: np.ndarray,
    anatomical_proxy: np.ndarray,
    border_fraction: float = 0.12,
) -> np.ndarray:
    image = np.asarray(
        original_rgb,
        dtype=np.uint8,
    ).copy()

    h, w = image.shape[:2]

    proxy = cv2.resize(
        anatomical_proxy.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    contours, _ = cv2.findContours(
        proxy,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        (200, 255, 71),
        2,
    )

    bx = max(1, int(round(w * border_fraction)))
    by = max(1, int(round(h * border_fraction)))

    cv2.rectangle(
        image,
        (bx, by),
        (max(bx + 1, w - bx - 1), max(by + 1, h - by - 1)),
        (255, 124, 124),
        2,
    )

    return image


def _mask_visual(
    original_rgb: np.ndarray,
    mask: np.ndarray,
    tint_rgb: tuple[int, int, int],
) -> np.ndarray:
    h, w = original_rgb.shape[:2]

    resized = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    tint = np.zeros_like(original_rgb)
    tint[resized] = np.asarray(
        tint_rgb,
        dtype=np.uint8,
    )

    return np.where(
        resized[..., None],
        (
            0.60 * original_rgb.astype(np.float32)
            + 0.40 * tint.astype(np.float32)
        ).astype(np.uint8),
        original_rgb,
    )


# ============================================================
# FULL SUITE
# ============================================================

def run_explainability_suite(
    model: tf.keras.Model,
    grad_model: tf.keras.Model,
    classifier_layers: dict,
    image: Image.Image,
    probabilities: np.ndarray,
    predicted_index: int,
    ig_steps: int = 32,
    perturbation_fraction: float = 0.15,
) -> dict:
    image_array = np.array(
        image.convert("RGB")
    ).astype(np.float32)

    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]

    input_height = int(input_shape[1] or 300)
    input_width = int(input_shape[2] or input_height)

    resized = tf.image.resize(
        image_array,
        (input_height, input_width),
        method="bilinear",
    )

    batch = tf.expand_dims(
        resized,
        axis=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    ).reshape(-1)

    predicted_index = int(predicted_index)

    sorted_indices = np.argsort(probabilities)[::-1]
    alternative_index = int(
        next(
            index
            for index in sorted_indices
            if int(index) != predicted_index
        )
    )

    original_target_probability = float(
        probabilities[predicted_index]
    )
    alternative_probability = float(
        probabilities[alternative_index]
    )

    original_logits = classifier_logits(
        grad_model,
        classifier_layers,
        batch,
    )[0]

    original_target_logit = float(
        original_logits[predicted_index]
    )

    original_margin = float(
        target_margin_from_logits(
            original_logits[None, :],
            predicted_index,
        )[0]
    )

    # 1) Logit-based Grad-CAM
    gradcam_small = gradcam_map(
        grad_model,
        classifier_layers,
        batch,
        predicted_index,
    )
    gradcam_224 = resize_map(
        gradcam_small,
        input_width,
        input_height,
    )

    # 2) Runner-up + contrastive CAM
    alternative_cam_small = gradcam_map(
        grad_model,
        classifier_layers,
        batch,
        alternative_index,
    )
    alternative_cam_224 = resize_map(
        alternative_cam_small,
        input_width,
        input_height,
    )

    contrastive_map = normalize_map(
        np.maximum(
            gradcam_224 - alternative_cam_224,
            0.0,
        )
    )

    # 3) Logit-based multi-baseline IG
    ig_map = integrated_gradients_map(
        grad_model=grad_model,
        classifier_layers=classifier_layers,
        input_batch=batch,
        target_index=predicted_index,
        steps=ig_steps,
    )

    foreground_mask = build_foreground_mask(batch)
    anatomical_proxy = build_anatomical_focus_proxy(batch)

    # 4) Agreement matrix
    agreement_gc_ig = safe_correlation(
        gradcam_224,
        ig_map,
        valid_mask=foreground_mask,
    )

    agreement_gc_contrastive = safe_correlation(
        gradcam_224,
        contrastive_map,
        valid_mask=foreground_mask,
    )

    agreement_ig_contrastive = safe_correlation(
        ig_map,
        contrastive_map,
        valid_mask=foreground_mask,
    )

    agreement_matrix = {
        "gradcam_vs_integrated_gradients": agreement_gc_ig,
        "gradcam_vs_contrastive": agreement_gc_contrastive,
        "integrated_gradients_vs_contrastive": agreement_ig_contrastive,
    }

    # 5) Border dependency / anatomical focus
    shortcut = border_dependency_diagnostics(
        gradcam_map_2d=gradcam_224,
        integrated_gradients_map_2d=ig_map,
        anatomical_proxy=anatomical_proxy,
        foreground_mask=foreground_mask,
        border_fraction=0.12,
    )

    # 6) Anatomy-constrained perturbation
    anatomy_pixel_count = max(
        1,
        int(
            round(
                float(perturbation_fraction)
                * max(int(anatomical_proxy.sum()), 1)
            )
        ),
    )

    perturbed_batch, salient_mask = perturb_top_salient_pixels(
        input_batch=batch,
        saliency_map=gradcam_224,
        eligible_mask=anatomical_proxy,
        pixel_count=anatomy_pixel_count,
        blur_kernel=81,
    )

    perturbed_probabilities = model_probabilities(
        model,
        perturbed_batch,
    )[0]

    perturbed_logits = classifier_logits(
        grad_model,
        classifier_layers,
        perturbed_batch,
    )[0]

    perturbed_target_probability = float(
        perturbed_probabilities[predicted_index]
    )

    perturbed_target_logit = float(
        perturbed_logits[predicted_index]
    )

    perturbed_margin = float(
        target_margin_from_logits(
            perturbed_logits[None, :],
            predicted_index,
        )[0]
    )

    perturbed_predicted_index = int(
        np.argmax(perturbed_probabilities)
    )

    probability_influence = probability_influence_strength(
        original_target_probability,
        perturbed_target_probability,
    )

    margin_influence = margin_influence_strength(
        original_margin,
        perturbed_margin,
    )

    # 7) Matched off-anatomy perturbation
    off_anatomy_eligible = (
        foreground_mask
        &
        (~anatomical_proxy)
    )

    off_anatomy_batch, off_anatomy_mask = perturb_top_salient_pixels(
        input_batch=batch,
        saliency_map=gradcam_224,
        eligible_mask=off_anatomy_eligible,
        pixel_count=max(int(salient_mask.sum()), 1),
        blur_kernel=81,
    )

    off_anatomy_probabilities = model_probabilities(
        model,
        off_anatomy_batch,
    )[0]

    off_anatomy_logits = classifier_logits(
        grad_model,
        classifier_layers,
        off_anatomy_batch,
    )[0]

    off_anatomy_target_probability = float(
        off_anatomy_probabilities[predicted_index]
    )

    off_anatomy_target_logit = float(
        off_anatomy_logits[predicted_index]
    )

    off_anatomy_margin = float(
        target_margin_from_logits(
            off_anatomy_logits[None, :],
            predicted_index,
        )[0]
    )

    # 8) Multi-baseline deletion/insertion
    faithfulness = deletion_insertion_curves(
        model=model,
        grad_model=grad_model,
        classifier_layers=classifier_layers,
        input_batch=batch,
        saliency_map=gradcam_224,
        target_index=predicted_index,
        foreground_mask=foreground_mask,
        original_probability=original_target_probability,
        original_margin=original_margin,
    )

    # 9) Stability
    stability = explanation_stability(
        grad_model=grad_model,
        classifier_layers=classifier_layers,
        input_batch=batch,
        target_index=predicted_index,
        reference_map=gradcam_224,
        foreground_mask=foreground_mask,
    )

    # 10) Trust
    trust = explanation_trust_score(
        anatomy_ratio=shortcut["mean_anatomy_ratio"],
        border_ratio=shortcut["mean_border_ratio"],
        agreement=agreement_gc_ig,
        stability=stability["mean_correlation"],
        margin_relative_drop=margin_influence["relative_drop"],
        deletion_margin_auc=faithfulness[
            "deletion_margin_auc"
        ],
        baseline_warning_level=faithfulness[
            "baseline_warning_level"
        ],
    )

    # 11) Visuals
    original_rgb = np.array(
        image.convert("RGB")
    ).astype(np.uint8)

    original_h, original_w = original_rgb.shape[:2]

    gradcam_original = resize_map(
        gradcam_224,
        original_w,
        original_h,
    )

    ig_original = resize_map(
        ig_map,
        original_w,
        original_h,
    )

    contrastive_original = resize_map(
        contrastive_map,
        original_w,
        original_h,
    )

    alternative_original = resize_map(
        alternative_cam_224,
        original_w,
        original_h,
    )

    anatomy_proxy_visual = render_proxy_outline(
        original_rgb,
        anatomical_proxy,
        shortcut["border_fraction"],
    )

    salient_visual = _mask_visual(
        original_rgb,
        salient_mask,
        (200, 255, 71),
    )

    off_visual = _mask_visual(
        original_rgb,
        off_anatomy_mask,
        (255, 124, 124),
    )

    mask_coverage = float(
        salient_mask.sum()
        /
        max(anatomical_proxy.sum(), 1)
    )

    confidence_drop_pp = (
        original_target_probability
        -
        perturbed_target_probability
    ) * 100.0

    off_anatomy_drop_pp = (
        original_target_probability
        -
        off_anatomy_target_probability
    ) * 100.0

    target_logit_drop = (
        original_target_logit
        -
        perturbed_target_logit
    )

    off_anatomy_logit_drop = (
        original_target_logit
        -
        off_anatomy_target_logit
    )

    return {
        "predicted_index": predicted_index,
        "alternative_index": alternative_index,

        "original_probability": original_target_probability,
        "alternative_probability": alternative_probability,

        # Pre-softmax evidence
        "original_target_logit": original_target_logit,
        "perturbed_target_logit": perturbed_target_logit,
        "target_logit_drop": float(target_logit_drop),
        "original_margin": original_margin,
        "perturbed_margin": perturbed_margin,
        "margin_drop": margin_influence["margin_drop"],
        "margin_relative_drop": margin_influence["relative_drop"],
        "logit_influence_strength": margin_influence["label"],

        "off_anatomy_target_logit": off_anatomy_target_logit,
        "off_anatomy_logit_drop": float(off_anatomy_logit_drop),
        "off_anatomy_margin": off_anatomy_margin,
        "off_anatomy_margin_drop": float(
            original_margin - off_anatomy_margin
        ),

        # Maps
        "gradcam_map": gradcam_original,
        "integrated_gradients_map": ig_original,
        "alternative_gradcam_map": alternative_original,
        "contrastive_map": contrastive_original,

        # Visuals
        "gradcam_rgb": colorize_heatmap(gradcam_original),
        "gradcam_overlay": overlay_heatmap(
            original_rgb,
            gradcam_original,
            alpha=0.35,
        ),
        "integrated_gradients_rgb": colorize_heatmap(ig_original),
        "integrated_gradients_overlay": overlay_heatmap(
            original_rgb,
            ig_original,
            alpha=0.35,
        ),
        "contrastive_rgb": colorize_heatmap(contrastive_original),
        "contrastive_overlay": overlay_heatmap(
            original_rgb,
            contrastive_original,
            alpha=0.38,
        ),
        "salient_region_visual": salient_visual,
        "off_anatomy_region_visual": off_visual,
        "anatomy_proxy_visual": anatomy_proxy_visual,

        # Attention diagnostics
        "shortcut_diagnostics": shortcut,
        "agreement_matrix": agreement_matrix,
        "explanation_trust": trust,

        # Probability perturbation
        "perturbed_probabilities": perturbed_probabilities,
        "perturbed_target_probability": perturbed_target_probability,
        "perturbed_predicted_index": perturbed_predicted_index,
        "off_anatomy_probabilities": off_anatomy_probabilities,
        "off_anatomy_target_probability": off_anatomy_target_probability,
        "confidence_drop_pp": float(confidence_drop_pp),
        "off_anatomy_drop_pp": float(off_anatomy_drop_pp),
        "relative_drop": probability_influence["relative_drop"],
        "influence_strength": probability_influence["label"],
        "mask_coverage": mask_coverage,

        # Faithfulness / consistency
        "faithfulness": faithfulness,
        "cross_method_agreement": agreement_gc_ig,
        "stability": stability,

        # Configuration
        "ig_steps": int(ig_steps),
        "perturbation_fraction": float(perturbation_fraction),
        "attribution_target": "pre-softmax class logit",
    }
