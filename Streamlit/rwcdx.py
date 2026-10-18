from __future__ import annotations

import base64
import hashlib
from html import escape
from contextlib import contextmanager
import io
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIG
# ============================================================

# Locally rendered product favicon; no extra asset files required.
NV_PAGE_ICON = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
_nv_icon_draw = ImageDraw.Draw(NV_PAGE_ICON)
_nv_icon_draw.rounded_rectangle((0, 0, 63, 63), radius=17, fill="#0a304c")
_nv_icon_draw.line([(15, 45), (15, 19), (30, 45), (30, 19)], fill="#d8fffa", width=4)
_nv_icon_draw.line([(36, 19), (43, 45), (51, 19)], fill="#8ef4e8", width=4)
del _nv_icon_draw

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon=NV_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents
                     if (p / "src/quality/image_quality.py").is_file()),
                    Path(__file__).resolve().parents[1])

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.quality.image_quality import assess_mri_quality
from segmentation.streamlit_section import render_segmentation_section, sync_segmentation_state
from src.explainability.explainability_engine import (
    gradcam_map,
    resize_map,
    run_explainability_suite,
)

MAIN_MODEL_PATH = (
    PROJECT_ROOT
    / "Experiments"
    / "Accuracy99"
    / "models"
    / "ResNet50V2_300"
    / "resnet50v2_300_best.keras"
)

GM_SPECIALIST_PATH = (
    PROJECT_ROOT
    / "Experiments"
    / "Accuracy99"
    / "specialists"
    / "Glioma_Meningioma"
    / "gm_specialist_best.keras"
)

MVR_SPECIALIST_PATH = (
    PROJECT_ROOT
    / "Experiments"
    / "Accuracy99"
    / "specialists"
    / "Meningioma_vs_Rest"
    / "meningioma_vs_rest_best.keras"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

CLASS_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

IMG_SIZE = (300, 300)

# Cosmetic-only home-screen MRI visual.
# Embedded so the final Streamlit app remains a single self-contained Python file.
HOME_MRI_DATA_URI = "data:image/jpeg;base64,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"

# Current development-stage performance.
MAIN_VALIDATION_ACCURACY = 95.21
GM_ROUTED_VALIDATION_ACCURACY = 96.44
CURRENT_VALIDATION_ACCURACY = 97.05
CURRENT_VALIDATION_IMAGES = 814
CURRENT_VALIDATION_CORRECT = 790
CURRENT_VALIDATION_WRONG = 24

# Historical baseline for honest comparison only.
HISTORICAL_INDEPENDENT_TEST_ACCURACY = 84.41
HISTORICAL_INDEPENDENT_TEST_IMAGES = 1584

# Locked routing rules selected on the development validation set.
GM_MARGIN_MAX = 0.70
GM_CONFIDENCE_MIN = 0.60
MVR_PROMOTE_MIN = 0.75


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "nv_page": "Home",
    "nv_intro_seen": False,
    "analysis_result": None,
    "analysis_image": None,
    "analysis_hash": None,
    "analysis_module": "Overview",
    "explainability_result": None,
    "progression_results": None,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default




def nv_icon(name: str) -> str:
    paths = {
        "classify": '<rect x="4" y="4" width="6" height="6" rx="2"/><rect x="14" y="4" width="6" height="6" rx="2"/><rect x="4" y="14" width="6" height="6" rx="2"/><rect x="14" y="14" width="6" height="6" rx="2"/>',
        "routing": '<path d="M6 4v12a4 4 0 0 0 4 4h8M6 8h8a4 4 0 0 1 4 4v8M15 17l3 3 3-3"/>',
        "focus": '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3"/><circle cx="12" cy="12" r="4"/>',
        "report": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5"/>',
        "upload": '<path d="M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
    }
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + paths[name] + '</svg>'

def nv_html(html: str) -> None:
    for class_name, icon_name, glyph in [("nv-upload-icon", "upload", "⇧"), ("nv-report-icon", "report", "▤")]:
        html = html.replace(f'<div class="{class_name}">{glyph}</div>', f'<div class="{class_name}">{nv_icon(icon_name)}</div>')
    st.html(html)


# ============================================================
# NAVIGATION / STATE
# ============================================================

def go_to(page: str) -> None:
    st.session_state.nv_page = page
    st.rerun()


def reset_analysis_for_new_image(image_hash: str) -> None:
    if st.session_state.analysis_hash != image_hash:
        st.session_state.analysis_hash = image_hash
        st.session_state.analysis_result = None
        st.session_state.analysis_image = None
        st.session_state.analysis_module = "Overview"
        st.session_state.explainability_result = None


# ============================================================
# MRI SEMANTIC VALIDITY GATE
# ============================================================

def assess_mri_semantic_validity(image: Image.Image) -> dict[str, Any]:
    """
    Conservative heuristic gate for obvious non-MRI inputs.

    This is an OOD-style screening heuristic, not a clinically validated
    MRI authentication system.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    original_height, original_width = rgb.shape[:2]

    resized = cv2.resize(
        rgb,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY,
    )

    channel_spread = (
        resized.max(axis=2).astype(np.int16)
        -
        resized.min(axis=2).astype(np.int16)
    )

    grayscale_ratio = float(
        np.mean(channel_spread <= 12)
    )

    dark_ratio = float(
        np.mean(gray < 25)
    )

    bright_ratio = float(
        np.mean(gray > 220)
    )

    border_width = 32

    border_pixels = np.concatenate(
        [
            gray[:border_width, :].ravel(),
            gray[-border_width:, :].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        ]
    )

    border_dark_ratio = float(
        np.mean(border_pixels < 35)
    )

    border_mean = float(
        np.mean(border_pixels)
    )

    center_region = gray[64:192, 64:192]

    center_mean = float(
        np.mean(center_region)
    )

    aspect_ratio = float(
        max(
            original_width / max(original_height, 1),
            original_height / max(original_width, 1),
        )
    )

    foreground = (
        gray > 20
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    central_component = False
    component_area_ratio = 0.0

    if count > 1:
        candidate = 1 + int(
            np.argmax(stats[1:, cv2.CC_STAT_AREA])
        )

        area = int(
            stats[candidate, cv2.CC_STAT_AREA]
        )

        cx, cy = centroids[candidate]

        component_area_ratio = float(
            area / (256 * 256)
        )

        central_component = bool(
            0.08 <= component_area_ratio <= 0.85
            and 64 <= cx <= 192
            and 48 <= cy <= 208
        )

    score = 0.0

    if grayscale_ratio >= 0.90:
        score += 20
    elif grayscale_ratio >= 0.75:
        score += 10

    if dark_ratio >= 0.12:
        score += 15
    elif dark_ratio >= 0.05:
        score += 8

    if border_dark_ratio >= 0.35:
        score += 20
    elif border_dark_ratio >= 0.15:
        score += 10

    if bright_ratio <= 0.20:
        score += 15
    elif bright_ratio <= 0.40:
        score += 8

    if aspect_ratio <= 1.35:
        score += 10
    elif aspect_ratio <= 1.60:
        score += 5

    if central_component:
        score += 15

    if center_mean >= border_mean + 8:
        score += 5

    critical_mismatch = bool(
        (
            bright_ratio > 0.55
            and dark_ratio < 0.03
        )
        or grayscale_ratio < 0.45
        or (
            border_dark_ratio < 0.02
            and bright_ratio > 0.45
        )
    )

    if score >= 65 and not critical_mismatch:
        status = "MRI-like input detected"
        decision = "PASS"
        is_valid = True
    elif score >= 50 and not critical_mismatch:
        status = "MRI-likeness uncertain"
        decision = "REVIEW"
        is_valid = False
    else:
        status = "Non-MRI / invalid input detected"
        decision = "REJECT"
        is_valid = False

    reasons: list[str] = []

    if bright_ratio > 0.45:
        reasons.append("large bright/white background")

    if border_dark_ratio < 0.15:
        reasons.append("weak dark-border MRI pattern")

    if grayscale_ratio < 0.75:
        reasons.append("strong non-grayscale color content")

    if aspect_ratio > 1.60:
        reasons.append("unusual screenshot-like aspect ratio")

    if not central_component:
        reasons.append("no compact central MRI-like foreground structure")

    if not reasons:
        reasons.append("MRI-like grayscale and spatial structure detected")

    return {
        "is_valid": bool(is_valid),
        "decision": decision,
        "status": status,
        "score": int(round(score)),
        "reasons": reasons,
        "grayscale_ratio": grayscale_ratio,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "border_dark_ratio": border_dark_ratio,
        "aspect_ratio": aspect_ratio,
        "central_component": bool(central_component),
        "component_area_ratio": component_area_ratio,
    }


# ============================================================
# MODEL RESOURCES
# ============================================================

class XAIInputPipeline(tf.keras.layers.Layer):
    """
    Reuses the trained main-model input transforms in the same order:
    augmentation -> ResNet preprocessing.
    """

    def __init__(self, augmentation, preprocessing, **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def call(self, inputs, training=False):
        x = self.augmentation(
            inputs,
            training=False,
        )
        x = self.preprocessing(
            x,
            training=False,
        )
        return x


@st.cache_resource(show_spinner=False)
def load_ai_resources():
    required = [
        MAIN_MODEL_PATH,
        GM_SPECIALIST_PATH,
        MVR_SPECIALIST_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required model file(s) not found:\n"
            +
            "\n".join(missing)
        )

    main_model = tf.keras.models.load_model(
        MAIN_MODEL_PATH,
        compile=False,
    )

    gm_model = tf.keras.models.load_model(
        GM_SPECIALIST_PATH,
        compile=False,
    )

    mvr_model = tf.keras.models.load_model(
        MVR_SPECIALIST_PATH,
        compile=False,
    )

    base_model = main_model.get_layer(
        "resnet50v2"
    )

    xai_input_pipeline = XAIInputPipeline(
        main_model.get_layer("augmentation"),
        main_model.get_layer("resnet_preprocessing"),
        name="xai_input_pipeline",
    )

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.output,
            base_model.output,
        ],
        name="resnet50v2_grad_model",
    )

    classifier_layers = {
        "augmentation": xai_input_pipeline,
        "gap": main_model.get_layer("gap"),
        "dropout1": main_model.get_layer("dropout_1"),
        "dense": main_model.get_layer("classifier_dense"),
        "dropout2": main_model.get_layer("dropout_2"),
        "prediction": main_model.get_layer("prediction"),
    }

    dummy = tf.zeros(
        (1, IMG_SIZE[0], IMG_SIZE[1], 3),
        dtype=tf.float32,
    )

    _ = main_model(dummy, training=False)
    _ = gm_model(dummy, training=False)
    _ = mvr_model(dummy, training=False)
    _ = xai_input_pipeline(dummy, training=False)

    return {
        "main_model": main_model,
        "gm_model": gm_model,
        "mvr_model": mvr_model,
        "grad_model": grad_model,
        "classifier_layers": classifier_layers,
    }


def prepare_model_batch(
    image: Image.Image,
) -> tf.Tensor:
    image_array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    resized = tf.image.resize(
        image_array,
        IMG_SIZE,
        method="bilinear",
    )

    return tf.expand_dims(
        resized,
        axis=0,
    )


def _binary_softmax_probabilities(
    output: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    if len(values) == 1:
        p1 = float(
            np.clip(values[0], 0.0, 1.0)
        )
        return np.asarray(
            [1.0 - p1, p1],
            dtype=np.float32,
        )

    if len(values) >= 2:
        pair = values[:2]
        total = float(pair.sum())

        if (
            np.all(pair >= 0)
            and np.all(pair <= 1)
            and abs(total - 1.0) < 0.05
        ):
            return pair.astype(np.float32)

        shifted = pair - np.max(pair)
        exp = np.exp(shifted)
        return (
            exp / np.sum(exp)
        ).astype(np.float32)

    raise RuntimeError(
        "Specialist model returned an empty output."
    )


def run_prediction_pipeline(
    image: Image.Image,
    resources: dict[str, Any],
) -> dict[str, Any]:
    batch = prepare_model_batch(
        image
    )

    main_probs = np.asarray(
        resources["main_model"](
            batch,
            training=False,
        ).numpy()[0],
        dtype=np.float32,
    )

    gm_raw = resources[
        "gm_model"
    ](
        batch,
        training=False,
    ).numpy()[0]

    mvr_raw = resources[
        "mvr_model"
    ](
        batch,
        training=False,
    ).numpy()[0]

    gm_probs = _binary_softmax_probabilities(
        gm_raw
    )

    mvr_probs = _binary_softmax_probabilities(
        mvr_raw
    )

    # Training order used by the specialist experiment:
    # [Glioma, Meningioma].
    gm_pred_global = (
        CLASS_INDEX["glioma"]
        if int(np.argmax(gm_probs)) == 0
        else CLASS_INDEX["meningioma"]
    )

    gm_confidence = float(
        np.max(gm_probs)
    )

    # Meningioma-vs-Rest specialist:
    # index 1 / sigmoid positive = Meningioma.
    mvr_meningioma_probability = float(
        mvr_probs[1]
    )

    main_pred = int(
        np.argmax(main_probs)
    )

    order = np.argsort(
        main_probs
    )[::-1]

    top1 = int(order[0])
    top2 = int(order[1])

    main_margin = float(
        main_probs[top1]
        -
        main_probs[top2]
    )

    G = CLASS_INDEX["glioma"]
    M = CLASS_INDEX["meningioma"]
    P = CLASS_INDEX["pituitary"]

    final_pred = main_pred
    routing_steps: list[str] = []

    gm_pair = (
        set([top1, top2])
        ==
        set([G, M])
    )

    gm_route = bool(
        gm_pair
        and main_pred == G
        and gm_pred_global == M
        and main_margin <= GM_MARGIN_MAX
        and gm_confidence >= GM_CONFIDENCE_MIN
    )

    if gm_route:
        final_pred = M
        routing_steps.append(
            "Glioma↔Meningioma specialist promoted the main prediction to Meningioma."
        )

    meningioma_in_top2 = bool(
        top1 == M
        or top2 == M
    )

    mvr_route = bool(
        final_pred in [G, P]
        and meningioma_in_top2
        and mvr_meningioma_probability >= MVR_PROMOTE_MIN
    )

    if mvr_route:
        final_pred = M
        routing_steps.append(
            "Meningioma-vs-Rest specialist promoted the prediction to Meningioma."
        )

    return {
        "batch": batch,
        "main_probabilities": main_probs,
        "main_prediction_index": main_pred,
        "main_prediction": CLASS_NAMES[main_pred],
        "main_top_confidence": float(main_probs[main_pred]),
        "main_margin": main_margin,
        "top1_index": top1,
        "top2_index": top2,
        "gm_probabilities": gm_probs,
        "gm_confidence": gm_confidence,
        "gm_prediction_index": gm_pred_global,
        "gm_route": gm_route,
        "mvr_probabilities": mvr_probs,
        "mvr_meningioma_probability": mvr_meningioma_probability,
        "mvr_route": mvr_route,
        "final_prediction_index": final_pred,
        "final_prediction": CLASS_NAMES[final_pred],
        "main_probability_for_final_class": float(main_probs[final_pred]),
        "routing_steps": routing_steps,
        "routing_applied": bool(routing_steps),
    }


# ============================================================
# IMAGE VIEWER + ENHANCEMENT
# ============================================================

def image_to_base64(
    image: Image.Image | np.ndarray,
) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            np.asarray(image, dtype=np.uint8)
        )

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def interactive_image_viewer(
    image: Image.Image | np.ndarray,
    title: str,
    overlay: Image.Image | np.ndarray | None = None,
    overlay_label: str = "Overlay",
    enable_zoom_enhance: bool = True,
) -> None:
    """
    Interactive medical-image viewer.

    When enable_zoom_enhance=True, the viewer can automatically switch to a
    viewing-only enhanced rendering after deep zoom. The enhancement is
    interpolation + local contrast + mild sharpening; it does not recover
    missing diagnostic information.
    """
    if isinstance(image, np.ndarray):
        base_pil = Image.fromarray(
            np.asarray(image, dtype=np.uint8)
        ).convert("RGB")
    else:
        base_pil = image.convert("RGB")

    base64_image = image_to_base64(
        base_pil
    )

    enhanced64 = ""

    if enable_zoom_enhance:
        try:
            enhanced64 = image_to_base64(
                enhance_for_viewing(
                    base_pil
                )
            )
        except Exception:
            enhanced64 = ""

    overlay64 = (
        image_to_base64(overlay)
        if overlay is not None
        else ""
    )

    has_overlay = (
        overlay is not None
    )

    has_enhanced = bool(
        enhanced64
    )

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box}}
html,body{{margin:0;background:#f6f9fa;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#20364a}}
.viewer{{height:700px;position:relative;overflow:hidden;border:1px solid #d6e3e8;border-radius:27px;background:radial-gradient(circle at 50% 45%,#ffffff 0%,#edf4f6 58%,#e6eef1 100%);box-shadow:0 24px 58px rgba(10,38,62,.11),inset 0 0 0 1px rgba(255,255,255,.62)}}
.viewer::before{{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,rgba(10,145,139,.03) 1px,transparent 1px),linear-gradient(rgba(10,145,139,.03) 1px,transparent 1px);background-size:32px 32px;mask-image:radial-gradient(circle at center,#000 0%,transparent 84%)}} .viewer::after{{content:"MRI EVIDENCE VIEWER";position:absolute;right:17px;bottom:58px;color:rgba(10,38,62,.18);font-size:8px;font-weight:850;letter-spacing:1.5px;pointer-events:none}}
.canvas{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:grab;touch-action:none}}
.canvas img{{position:absolute;width:88%;height:82%;object-fit:contain;user-select:none;-webkit-user-drag:none;transform-origin:center center;will-change:transform;filter:drop-shadow(0 18px 38px rgba(10,38,62,.15));border-radius:10px}}
.overlay{{opacity:.38;filter:none!important}}
.toolbar{{position:absolute;z-index:20;left:14px;right:14px;top:13px;display:flex;align-items:flex-start;justify-content:space-between;gap:10px;pointer-events:none}}
.titlebox,.controls{{pointer-events:auto}}
.titlebox{{display:flex;align-items:center;gap:8px;padding:9px 10px;border-radius:13px;border:1px solid #d9e5e9;background:rgba(255,255,255,.91);backdrop-filter:blur(16px);box-shadow:0 10px 25px rgba(10,38,62,.065)}}
.title{{font-size:10.5px;font-weight:800;color:#405a70;white-space:nowrap;max-width:350px;overflow:hidden;text-overflow:ellipsis}}
.mode{{padding:5px 7px;border-radius:8px;background:#eef8f7;color:#08766f;font-size:8.5px;font-weight:850;letter-spacing:.04em;text-transform:uppercase}}
.controls{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px;max-width:450px}}
.controls button{{height:39px;min-width:39px;padding:0 11px;border-radius:11px;border:1px solid #d5e2e7;background:rgba(255,255,255,.94);color:#244258;box-shadow:0 8px 20px rgba(10,38,62,.055);cursor:pointer;font-size:13px;font-weight:800;transition:.14s ease}}
.controls button:hover{{transform:translateY(-1px);border-color:#83c8c3;color:#08766f}}
.controls button.active{{border-color:#78c1bc;background:#eaf8f7;color:#08766f;box-shadow:inset 0 0 0 1px rgba(10,145,139,.06)}}
.controls button.hq{{display:{'inline-flex' if has_enhanced else 'none'};align-items:center;gap:5px}}
.bottom{{position:absolute;z-index:20;left:14px;right:14px;bottom:13px;display:flex;align-items:flex-end;justify-content:space-between;gap:10px;pointer-events:none}}
.tip,.zoomBox,.opacity{{pointer-events:auto;padding:9px 11px;border-radius:11px;border:1px solid #d9e5e9;background:rgba(255,255,255,.92);backdrop-filter:blur(15px);box-shadow:0 8px 20px rgba(10,38,62,.05);font-size:9px;font-weight:720;color:#6c8093}}
.tip b{{color:#0b7f79}}
.zoomBox{{display:flex;align-items:center;gap:8px}}
.zoomBox input{{width:150px;accent-color:#0e9e97}}
.opacity{{display:{'flex' if has_overlay else 'none'};align-items:center;gap:8px}}
.opacity input{{width:105px;accent-color:#0e9e97}}
.notice{{position:absolute;z-index:18;left:50%;bottom:62px;transform:translateX(-50%);padding:7px 10px;border-radius:10px;border:1px solid rgba(14,158,151,.17);background:rgba(248,253,253,.93);color:#668095;font-size:8.5px;font-weight:700;opacity:0;transition:.2s ease;pointer-events:none;white-space:nowrap}}
.notice.show{{opacity:1}}
@media(max-width:760px){{
  .viewer{{height:540px}}
  .tip{{display:none}}
  .zoomBox input{{width:90px}}
  .opacity input{{width:70px}}
  .title{{max-width:170px}}
  .controls button{{height:35px;min-width:35px;padding:0 8px}}
}}

html,body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
button:focus-visible,input:focus-visible{{outline:3px solid #638af2;outline-offset:3px}}
button{{min-height:40px!important;font-size:12px!important;border-radius:10px!important}}
.title{{font-size:14px!important}}.sub,.control-title,.range-row label,.foot,.status,.zoomtip,.tip,.zoomBox,.opacity{{font-size:12px!important;line-height:1.5!important}}
.controlpanel{{height:auto!important;min-height:330px;overflow:auto}}.foot{{padding:14px!important}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr!important}}.steps{{grid-template-columns:1fr 1fr!important}}.toolbar{{flex-wrap:wrap}}.controls{{justify-content:flex-start}}.bottom{{flex-wrap:wrap}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}}}

html,body{{background:transparent;color:#cad4e4}}
.viewer{{height:600px;background:#071526;border:0;border-radius:18px;box-shadow:none}}
.viewer:before,.viewer:after{{display:none}}
.canvas img{{filter:none;border-radius:0;width:90%;height:76%}}
.titlebox,.controls button,.tip,.zoomBox,.opacity{{background:#102239;color:#c5d0e1;border:1px solid #2a3b51;box-shadow:none;backdrop-filter:none}}
.title{{color:#e0e6ef;font-size:13px!important;font-weight:500;max-width:260px}}
.mode{{background:#172c45;color:#c2ccdd;text-transform:none;font-size:10px;font-weight:400}}
.controls button.active{{background:#263b57;color:#fff;border-color:#637392;box-shadow:none}}
.controls button{{font-weight:500}}.controls button:hover{{color:#fff;border-color:#71829a}}
.zoomBox input,.opacity input{{accent-color:#8586ee}}.tip{{display:none}}
.notice{{background:#102239;color:#cad4e4;border-color:#2a3b51;font-size:11px}}
@media(max-width:640px){{.viewer{{height:540px}}.toolbar{{flex-direction:column;align-items:flex-start}}.titlebox{{max-width:100%}}.title{{max-width:180px}}.controls{{max-width:100%;gap:4px}}.controls button{{font-size:11px!important;min-height:36px!important}}.bottom{{flex-wrap:wrap;gap:8px}}}}
</style>
</head>
<body>
<div id="viewer" class="viewer">
  <div class="toolbar">
    <div class="titlebox">
      <div class="title">{escape(title)}</div>
      <div id="modeBadge" class="mode">Original view</div>
    </div>
    <div class="controls">
      <button onclick="zoomBy(1.10)" title="Zoom in">＋</button>
      <button onclick="zoomBy(.9091)" title="Zoom out">−</button>
      <button onclick="fitView()" title="Fit image">Fit</button>
      <button onclick="resetView()" title="Reset view">Reset</button>
      <button id="hqBtn" class="hq" onclick="manualToggleHQ()" title="Viewing-only enhancement">Enhance</button>
      <button id="autoBtn" class="hq active" onclick="toggleAutoHQ()" title="Automatically enhance after deep zoom">Auto HQ</button>
      <button onclick="toggleFullscreen()" title="Fullscreen">⛶</button>
    </div>
  </div>

  <div id="canvas" class="canvas">
    <img id="base" src="data:image/png;base64,{base64_image}">
    {'<img id="overlay" class="overlay" src="data:image/png;base64,' + overlay64 + '">' if has_overlay else ''}
  </div>

  <div id="hqNotice" class="notice">
    Enhanced rendering is for visualization only — no diagnostic detail is recovered.
  </div>

  <div class="bottom">
    <div class="tip"><b>Wheel</b> zoom · <b>Drag</b> pan · <b>Double-click</b> reset · deep zoom up to 1200%</div>
    <div class="opacity"><span>{overlay_label}</span><input type="range" min="0" max="100" value="38" oninput="setOpacity(this.value)"></div>
    <div class="zoomBox">
      <input id="zoomSlider" type="range" min="45" max="1200" value="100" oninput="setZoom(this.value)">
      <span id="zoom">100%</span>
    </div>
  </div>
</div>

<script>
let scale=1,x=0,y=0,drag=false,startX=0,startY=0;
let hq=false,autoHQ={'true' if has_enhanced else 'false'};
const originalSrc="data:image/png;base64,{base64_image}";
const enhancedSrc="data:image/png;base64,{enhanced64}";
const base=document.getElementById('base');
const overlay=document.getElementById('overlay');
const canvas=document.getElementById('canvas');
const viewer=document.getElementById('viewer');
const zoomLabel=document.getElementById('zoom');
const zoomSlider=document.getElementById('zoomSlider');
const hqBtn=document.getElementById('hqBtn');
const autoBtn=document.getElementById('autoBtn');
const modeBadge=document.getElementById('modeBadge');
const hqNotice=document.getElementById('hqNotice');

function apply(){{
  const t=`translate(${{x}}px,${{y}}px) scale(${{scale}})`;
  base.style.transform=t;
  if(overlay)overlay.style.transform=t;
  zoomLabel.textContent=Math.round(scale*100)+'%';
  zoomSlider.value=Math.round(scale*100);
  if(autoHQ && enhancedSrc){{
    if(scale>=1.55 && !hq)setHQ(true,true);
    if(scale<=1.22 && hq)setHQ(false,false);
  }}
}}

function flashNotice(){{
  if(!hqNotice)return;
  hqNotice.classList.add('show');
  setTimeout(()=>hqNotice.classList.remove('show'),2200);
}}

function setHQ(value,notify){{
  if(!enhancedSrc)return;
  hq=!!value;
  base.src=hq?enhancedSrc:originalSrc;
  modeBadge.textContent=hq?'Enhanced view':'Original view';
  if(hqBtn)hqBtn.classList.toggle('active',hq);
  if(notify && hq)flashNotice();
}}

function manualToggleHQ(){{
  autoHQ=false;
  if(autoBtn)autoBtn.classList.remove('active');
  setHQ(!hq,true);
}}

function toggleAutoHQ(){{
  if(!enhancedSrc)return;
  autoHQ=!autoHQ;
  if(autoBtn)autoBtn.classList.toggle('active',autoHQ);
  if(autoHQ){{
    if(scale>=1.55)setHQ(true,true);
    else if(scale<=1.22)setHQ(false,false);
  }}
}}

function clampScale(value){{
  return Math.max(.45,Math.min(12,value));
}}

function zoomBy(mult){{
  scale=clampScale(scale*mult);
  apply();
}}

function zoomAt(clientX,clientY,mult){{
  const oldScale=scale;
  const newScale=clampScale(oldScale*mult);
  if(Math.abs(newScale-oldScale)<0.000001)return;

  const rect=canvas.getBoundingClientRect();
  const cx=clientX-(rect.left+rect.width/2);
  const cy=clientY-(rect.top+rect.height/2);
  const ratio=newScale/oldScale;

  x=cx-(cx-x)*ratio;
  y=cy-(cy-y)*ratio;
  scale=newScale;
  apply();
}}

function setZoom(value){{
  scale=clampScale(Number(value)/100);
  apply();
}}

function fitView(){{
  scale=.92;x=0;y=0;apply();
}}

function resetView(){{
  scale=1;x=0;y=0;
  if(autoHQ)setHQ(false,false);
  apply();
}}

function setOpacity(value){{
  if(overlay)overlay.style.opacity=(value/100);
}}

function toggleFullscreen(){{
  if(!document.fullscreenElement){{
    viewer.requestFullscreen().catch(()=>{{}});
  }}else{{
    document.exitFullscreen().catch(()=>{{}});
  }}
}}

canvas.addEventListener('wheel',e=>{{
  e.preventDefault();

  // Slow, trackpad-friendly zoom. MRI and XAI evidence layers
  // remain registered because they share the same display box and transform.
  const delta=Math.max(-80,Math.min(80,e.deltaY));
  const factor=Math.exp(-delta*0.00030);
  zoomAt(e.clientX,e.clientY,factor);
}},{{passive:false}});

canvas.addEventListener('pointerdown',e=>{{
  drag=true;
  startX=e.clientX-x;
  startY=e.clientY-y;
  canvas.setPointerCapture?.(e.pointerId);
  canvas.style.cursor='grabbing';
}});

canvas.addEventListener('pointerup',e=>{{
  drag=false;
  canvas.releasePointerCapture?.(e.pointerId);
  canvas.style.cursor='grab';
}});

canvas.addEventListener('pointercancel',()=>{{
  drag=false;
  canvas.style.cursor='grab';
}});

canvas.addEventListener('pointermove',e=>{{
  if(!drag)return;
  x=e.clientX-startX;
  y=e.clientY-startY;
  apply();
}});

canvas.addEventListener('dblclick',resetView);
apply();
</script>
</body>
</html>
"""

    st.iframe(html, height="content")


def interactive_roi_inspector(
    reference_image: Image.Image | np.ndarray,
    crop_source_image: Image.Image | np.ndarray,
    reference_label: str = "Original MRI",
) -> None:
    """
    Manual region-of-interest inspection tool.

    The user selects a rectangular ROI using the chosen reference layer.
    The matching crop is taken from the original MRI. The enhanced panel
    automatically opens at 300% zoom and supports wheel zoom and drag pan.
    This remains a viewing tool and does not change model inference.
    """
    if isinstance(reference_image, np.ndarray):
        reference_pil = Image.fromarray(
            np.asarray(reference_image, dtype=np.uint8)
        ).convert("RGB")
    else:
        reference_pil = reference_image.convert("RGB")

    if isinstance(crop_source_image, np.ndarray):
        crop_pil = Image.fromarray(
            np.asarray(crop_source_image, dtype=np.uint8)
        ).convert("RGB")
    else:
        crop_pil = crop_source_image.convert("RGB")

    reference64 = image_to_base64(reference_pil)
    crop64 = image_to_base64(crop_pil)
    safe_label = (
        reference_label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box}}
html,body{{
  margin:0;background:transparent;
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:#20364a
}}
.shell{{
  border:1px solid #dce7ee;border-radius:8px;background:#fff;
  box-shadow:0 5px 16px rgba(22,53,78,.045);overflow:hidden
}}
.head{{
  padding:11px 13px 9px;border-bottom:1px solid #e8eef2;
  display:flex;align-items:center;justify-content:space-between;gap:10px
}}
.title{{font-size:10.5px;font-weight:800;color:#173650}}
.sub{{margin-top:2px;font-size:7.3px;color:#788b9b;line-height:1.4}}
.chip{{
  padding:5px 7px;border-radius:999px;background:#eef7ff;color:#3273a4;
  border:1px solid #d9eafb;font-size:6.8px;font-weight:750;white-space:nowrap
}}
.steps{{
  display:grid;grid-template-columns:repeat(4,1fr);gap:6px;
  padding:8px 10px;border-bottom:1px solid #edf1f4;background:#fbfcfe
}}
.step{{
  padding:7px 8px;border:1px solid #e3eaf0;border-radius:6px;background:#fff;
  font-size:6.9px;color:#657a8c;font-weight:700
}}
.step b{{
  display:inline-grid;place-items:center;width:15px;height:15px;margin-right:5px;
  border-radius:50%;background:#6359e9;color:white;font-size:6px
}}
.grid{{
  display:grid;
  grid-template-columns:minmax(0,1.12fr) minmax(165px,.72fr) minmax(165px,.78fr) minmax(170px,.66fr);
  gap:8px;padding:9px
}}
.card{{
  border:1px solid #dfe8ef;border-radius:7px;background:#fff;overflow:hidden;min-width:0
}}
.cardhead{{
  height:31px;padding:0 9px;display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid #edf1f4;font-size:7.6px;font-weight:750;color:#324d63
}}
.note{{font-size:6.2px;color:#8797a4;font-weight:600}}
.pickstage{{
  height:330px;display:flex;align-items:center;justify-content:center;padding:8px;
  background:#f4f8fa;position:relative
}}
#selectCanvas{{
  display:block;max-width:100%;max-height:310px;width:auto;height:auto;border-radius:5px;
  background:#06111b;box-shadow:0 7px 20px rgba(6,30,49,.13);
  cursor:crosshair;touch-action:none
}}
.cropstage{{
  height:330px;display:flex;align-items:center;justify-content:center;padding:8px;
  background:#f6f9fb;position:relative;overflow:hidden
}}
.cropstage.enh{{
  background:radial-gradient(circle at center,#eef4ff,#f6f7fd 72%);
  cursor:grab
}}
.cropstage.enh:active{{cursor:grabbing}}
.cropstage canvas{{
  max-width:100%;max-height:100%;width:auto;height:auto;border-radius:5px;
  background:#07111b;box-shadow:0 7px 18px rgba(6,30,49,.12)
}}
#enhCanvas{{
  transform-origin:center center;will-change:transform;
  touch-action:none;max-width:none;max-height:none
}}
.empty{{
  padding:12px;text-align:center;color:#8798a6;font-size:6.9px;line-height:1.45;font-weight:600
}}
.controlpanel{{
  height:330px;padding:10px;background:#fbfcfe;display:flex;flex-direction:column;gap:10px
}}
.control-title{{font-size:7.8px;font-weight:750;color:#344f65}}
.range-row{{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center}}
.range-row label{{grid-column:1/-1;color:#657b8e;font-size:6.8px;font-weight:700}}
.range-row input{{width:100%;accent-color:#6658e8}}
.status{{font-size:6.4px;color:#71879a;font-weight:700;white-space:nowrap}}
.buttons{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
button{{
  min-height:29px;border:1px solid #d5e2ea;background:#fff;color:#405a70;
  border-radius:6px;padding:5px 7px;font-size:6.8px;font-weight:750;cursor:pointer
}}
button.primary{{
  grid-column:1/-1;border:0;color:#fff;background:linear-gradient(90deg,#4668df,#6757e8)
}}
button:disabled{{opacity:.45;cursor:not-allowed}}
.zoomtip{{
  margin-top:auto;padding:8px;border:1px solid #e1e8ee;border-radius:6px;background:#fff;
  color:#74889a;font-size:6.6px;line-height:1.45
}}
.zoomtip b{{color:#405e76}}
.foot{{
  padding:8px 11px;border-top:1px solid #edf1f4;background:#fbfcfe;
  color:#7b8e9e;font-size:6.7px;line-height:1.45
}}
.foot b{{color:#405d73}}
@media(max-width:940px){{
  .grid{{grid-template-columns:1fr 1fr}}
  .grid .card:first-child{{grid-column:1/-1}}
  .steps{{grid-template-columns:repeat(2,1fr)}}
}}

html,body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
button:focus-visible,input:focus-visible{{outline:3px solid #638af2;outline-offset:3px}}
button{{min-height:40px!important;font-size:12px!important;border-radius:10px!important}}
.title{{font-size:14px!important}}.sub,.control-title,.range-row label,.foot,.status,.zoomtip,.tip,.zoomBox,.opacity{{font-size:12px!important;line-height:1.5!important}}
.controlpanel{{height:auto!important;min-height:330px;overflow:auto}}.foot{{padding:14px!important}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr!important}}.steps{{grid-template-columns:1fr 1fr!important}}.toolbar{{flex-wrap:wrap}}.controls{{justify-content:flex-start}}.bottom{{flex-wrap:wrap}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}}}

.shell{{border:0;border-radius:18px;box-shadow:none;background:transparent}}
.head,.steps,.zoomtip,.foot{{display:none}}
.grid{{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;padding:0}}
.card{{border:1px solid #e7eaee;border-radius:16px;overflow:hidden}}
.cardhead{{height:auto;min-height:55px;padding:16px 20px;font-size:17px;font-weight:500;color:#131a24}}
.note{{font-size:11px;font-weight:400;color:#687382}}
.pickstage,.cropstage,.cropstage.enh{{height:380px;background:#071526;padding:12px}}
.empty{{font-size:12px;color:#a7b1c2;font-weight:400}}
.card:nth-child(2){{display:none}}
.card:nth-child(4){{grid-column:1/-1}}
.controlpanel{{min-height:0;display:grid;grid-template-columns:1fr 1fr 1.5fr;gap:18px;padding:20px;background:white}}
.control-title{{display:none}}.range-row{{align-content:center}}
.range-row input{{accent-color:#5967e8}}.range-row label{{font-size:12px}}
.buttons{{display:flex;flex-wrap:wrap;align-items:center}}button{{font-weight:500;background:#f4f6f8;border-color:#e7eaee;color:#131a24}}
button.primary{{background:#071526}}.status{{color:#687382}}
@media(max-width:940px){{.grid{{grid-template-columns:1fr 1fr}}.grid .card:first-child{{grid-column:auto}}}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr!important}}.controlpanel{{grid-template-columns:1fr!important}}.pickstage,.cropstage{{height:300px}}}}
</style>
</head>
<body>
<div class="shell">
  <div class="head">
    <div>
      <div class="title">Focused Region Inspector</div>
      <div class="sub">Select the area, review the original crop, enhance it for inspection, then zoom directly inside the enhanced panel.</div>
    </div>
    <div class="chip">Reference: {safe_label}</div>
  </div>

  <div class="steps">
    <div class="step"><b>1</b>Choose View</div>
    <div class="step"><b>2</b>Select Region</div>
    <div class="step"><b>3</b>View & Enhance</div>
    <div class="step"><b>4</b>Zoom & Inspect</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="cardhead"><span>Original MRI</span><span id="coordLabel" class="note">Click and drag</span></div>
      <div class="pickstage"><canvas id="selectCanvas"></canvas></div>
    </div>

    <div class="card">
      <div class="cardhead"><span>Selected Region</span><span class="note">Original MRI crop</span></div>
      <div class="cropstage">
        <canvas id="rawCanvas" style="display:none"></canvas>
        <div id="rawEmpty" class="empty">The selected MRI region will appear here.</div>
      </div>
    </div>

    <div class="card">
      <div class="cardhead"><span>Selected Region</span><span class="note">Wheel zoom · drag pan</span></div>
      <div id="enhStage" class="cropstage enh">
        <canvas id="enhCanvas" style="display:none"></canvas>
        <div id="enhEmpty" class="empty">Drag over the MRI to select a region.</div>
      </div>
    </div>

    <div class="card">
      <div class="cardhead"><span>Region controls</span><span id="sizeLabel" class="note">No ROI</span></div>
      <div class="controlpanel">
        <div class="control-title">Enhancement Strength</div>
        <div class="range-row">
          <label>Contrast + mild sharpening</label>
          <input id="strength" type="range" min="0" max="100" value="58">
          <span id="strengthValue" class="status">58%</span>
        </div>

        <div class="control-title">Inspection Zoom</div>
        <div class="range-row">
          <label>100% – 800%</label>
          <input id="zoomRange" type="range" min="100" max="800" value="300">
          <span id="zoomValue" class="status">300%</span>
        </div>

        <div class="buttons">
          <button onclick="resetSelection()">Reset</button>
          <button onclick="useCenterRegion()">Center ROI</button>
          <button onclick="fitZoom()">Fit Zoom</button>
          <button onclick="resetZoom()">300%</button>
          <button id="saveRawBtn" onclick="saveOriginal()" disabled>Save original crop</button><button id="saveBtn" class="primary" onclick="saveEnhanced()" disabled>Save Enhanced ROI</button>
        </div>

        <div class="zoomtip">
          <b>Inspection:</b> after selection, the enhanced crop opens at 300%. Use the slider or mouse wheel to zoom and drag the image to inspect different parts.
        </div>
      </div>
    </div>
  </div>

  <div class="foot">
    <b>Viewing note:</b> selection and enhancement are inspection tools. They do not create new MRI information, segment a tumor, or modify prediction results.
  </div>
</div>

<script>
function saveOriginal(){{if(!selection)return;const a=document.createElement("a");a.href=rawCanvas.toDataURL("image/png");a.download="NeuroVision_ROI_original.png";a.click();}}
const refImg=new Image(), cropImg=new Image();
refImg.src="data:image/png;base64,{reference64}";
cropImg.src="data:image/png;base64,{crop64}";

const selectCanvas=document.getElementById('selectCanvas');
const rawCanvas=document.getElementById('rawCanvas');
const enhCanvas=document.getElementById('enhCanvas');
const enhStage=document.getElementById('enhStage');
const rawEmpty=document.getElementById('rawEmpty');
const enhEmpty=document.getElementById('enhEmpty');
const coordLabel=document.getElementById('coordLabel');
const sizeLabel=document.getElementById('sizeLabel');
const strength=document.getElementById('strength');
const strengthValue=document.getElementById('strengthValue');
const zoomRange=document.getElementById('zoomRange');
const zoomValue=document.getElementById('zoomValue');
const saveBtn=document.getElementById('saveBtn');
const sctx=selectCanvas.getContext('2d');

let drag=false,start=null,current=null,selection=null;
let zoom=3,panX=0,panY=0,zoomDrag=false,zoomStartX=0,zoomStartY=0;

function clamp(v,lo,hi){{return Math.max(lo,Math.min(hi,v));}}

function canvasPoint(e){{
  const r=selectCanvas.getBoundingClientRect();
  const sx=selectCanvas.width/Math.max(r.width,1);
  const sy=selectCanvas.height/Math.max(r.height,1);
  return {{
    x:clamp((e.clientX-r.left)*sx,0,selectCanvas.width),
    y:clamp((e.clientY-r.top)*sy,0,selectCanvas.height)
  }};
}}

function normalizedRect(a,b){{
  const x=Math.min(a.x,b.x), y=Math.min(a.y,b.y);
  return {{x,y,w:Math.abs(a.x-b.x),h:Math.abs(a.y-b.y)}};
}}

function drawSelection(){{
  if(!refImg.complete||!refImg.naturalWidth)return;
  sctx.clearRect(0,0,selectCanvas.width,selectCanvas.height);
  sctx.drawImage(refImg,0,0,selectCanvas.width,selectCanvas.height);
  const r=drag&&start&&current?normalizedRect(start,current):selection;
  if(r&&r.w>2&&r.h>2){{
    sctx.save();
    sctx.fillStyle='rgba(7,18,28,.30)';
    sctx.fillRect(0,0,selectCanvas.width,selectCanvas.height);
    sctx.drawImage(refImg,r.x,r.y,r.w,r.h,r.x,r.y,r.w,r.h);
    sctx.strokeStyle='#ef4d66';
    sctx.lineWidth=Math.max(2,selectCanvas.width/260);
    sctx.setLineDash([9,6]);
    sctx.strokeRect(r.x,r.y,r.w,r.h);
    sctx.setLineDash([]);
    sctx.restore();
  }}
}}

function selectionToCropRect(r){{
  const nx=r.x/selectCanvas.width, ny=r.y/selectCanvas.height;
  const nw=r.w/selectCanvas.width, nh=r.h/selectCanvas.height;
  return {{
    x:clamp(nx*cropImg.naturalWidth,0,cropImg.naturalWidth-1),
    y:clamp(ny*cropImg.naturalHeight,0,cropImg.naturalHeight-1),
    w:clamp(nw*cropImg.naturalWidth,1,cropImg.naturalWidth),
    h:clamp(nh*cropImg.naturalHeight,1,cropImg.naturalHeight)
  }};
}}

function enhanceCanvas(ctx,w,h,s){{
  if(s<=0)return;
  let img=ctx.getImageData(0,0,w,h), d=img.data;
  const contrast=1+(s/100)*0.34;
  const brightness=1+(s/100)*0.035;
  for(let i=0;i<d.length;i+=4){{
    d[i]=clamp(((d[i]-128)*contrast+128)*brightness,0,255);
    d[i+1]=clamp(((d[i+1]-128)*contrast+128)*brightness,0,255);
    d[i+2]=clamp(((d[i+2]-128)*contrast+128)*brightness,0,255);
  }}
  ctx.putImageData(img,0,0);

  const amount=(s/100)*0.28;
  if(amount<=0.01||w<3||h<3)return;
  const src=ctx.getImageData(0,0,w,h), a=src.data, out=new Uint8ClampedArray(a);
  for(let y=1;y<h-1;y++){{
    for(let x=1;x<w-1;x++){{
      const i=(y*w+x)*4;
      const up=((y-1)*w+x)*4,dn=((y+1)*w+x)*4,lf=(y*w+x-1)*4,rt=(y*w+x+1)*4;
      for(let c=0;c<3;c++){{
        const blur=(a[up+c]+a[dn+c]+a[lf+c]+a[rt+c]+a[i+c]*4)/8;
        out[i+c]=clamp(a[i+c]+amount*(a[i+c]-blur)*2.2,0,255);
      }}
    }}
  }}
  src.data.set(out);ctx.putImageData(src,0,0);
}}

function applyZoom(){{
  enhCanvas.style.transform=`translate(${{panX}}px,${{panY}}px) scale(${{zoom}})`;
  zoomValue.textContent=Math.round(zoom*100)+'%';
  zoomRange.value=Math.round(zoom*100);
}}

function setZoomPct(v){{
  zoom=clamp(Number(v)/100,1,8);
  applyZoom();
}}

function resetZoom(){{
  zoom=3;panX=0;panY=0;applyZoom();
}}

function fitZoom(){{
  zoom=1;panX=0;panY=0;applyZoom();
}}

function drawOutputs(){{
  if(!selection||!cropImg.complete||!cropImg.naturalWidth)return;
  const r=selectionToCropRect(selection);
  if(r.w<3||r.h<3)return;

  const maxDim=620;
  const scale=Math.min(2.8,maxDim/Math.max(r.w,r.h));
  const outW=Math.max(60,Math.round(r.w*Math.max(1,scale)));
  const outH=Math.max(60,Math.round(r.h*Math.max(1,scale)));

  rawCanvas.width=outW;rawCanvas.height=outH;
  enhCanvas.width=outW;enhCanvas.height=outH;
  rawCanvas.style.display='block';enhCanvas.style.display='block';
  rawEmpty.style.display='none';enhEmpty.style.display='none';

  const rctx=rawCanvas.getContext('2d');
  rctx.imageSmoothingEnabled=true;rctx.imageSmoothingQuality='high';
  rctx.clearRect(0,0,outW,outH);
  rctx.drawImage(cropImg,r.x,r.y,r.w,r.h,0,0,outW,outH);

  const ectx=enhCanvas.getContext('2d');
  ectx.imageSmoothingEnabled=true;ectx.imageSmoothingQuality='high';
  ectx.clearRect(0,0,outW,outH);
  ectx.drawImage(rawCanvas,0,0,outW,outH);
  enhanceCanvas(ectx,outW,outH,Number(strength.value));

  coordLabel.textContent=`ROI ${{Math.round(selection.w)}} × ${{Math.round(selection.h)}} px`;
  sizeLabel.textContent=`${{Math.round(r.w)}} × ${{Math.round(r.h)}} px`;
  saveBtn.disabled=false;document.getElementById("saveRawBtn").disabled=false;
  resetZoom();
}}

function finishSelection(r){{
  const minSize=Math.max(8,Math.min(selectCanvas.width,selectCanvas.height)*0.025);
  if(!r||r.w<minSize||r.h<minSize){{
    selection=null;
    coordLabel.textContent='Select a larger region';
    drawSelection();
    return;
  }}
  selection=r;drawSelection();drawOutputs();
}}

function resetSelection(){{
  selection=null;start=null;current=null;drag=false;
  drawSelection();
  rawCanvas.style.display='none';enhCanvas.style.display='none';
  rawEmpty.style.display='block';enhEmpty.style.display='block';
  coordLabel.textContent='Click and drag';
  sizeLabel.textContent='No ROI';
  saveBtn.disabled=true;document.getElementById("saveRawBtn").disabled=true;
  zoom=3;panX=0;panY=0;applyZoom();
}}

function useCenterRegion(){{
  const w=selectCanvas.width*.36,h=selectCanvas.height*.36;
  finishSelection({{x:(selectCanvas.width-w)/2,y:(selectCanvas.height-h)/2,w,h}});
}}

function saveEnhanced(){{
  if(!selection)return;
  const a=document.createElement('a');
  a.download='NeuroVision_Selected_ROI_Enhanced.png';
  a.href=enhCanvas.toDataURL('image/png');
  a.click();
}}

function init(){{
  if(!refImg.complete||!refImg.naturalWidth||!cropImg.complete||!cropImg.naturalWidth)return;
  selectCanvas.width=refImg.naturalWidth;
  selectCanvas.height=refImg.naturalHeight;
  drawSelection();
}}
refImg.onload=init;cropImg.onload=init;
if(refImg.complete&&cropImg.complete)setTimeout(init,30);

selectCanvas.addEventListener('pointerdown',e=>{{
  e.preventDefault();drag=true;start=canvasPoint(e);current=start;
  selectCanvas.setPointerCapture?.(e.pointerId);drawSelection();
}});
selectCanvas.addEventListener('pointermove',e=>{{if(!drag)return;current=canvasPoint(e);drawSelection();}});
selectCanvas.addEventListener('pointerup',e=>{{
  if(!drag)return;current=canvasPoint(e);drag=false;
  selectCanvas.releasePointerCapture?.(e.pointerId);
  finishSelection(normalizedRect(start,current));
}});
selectCanvas.addEventListener('pointercancel',()=>{{drag=false;drawSelection();}});

strength.addEventListener('input',()=>{{
  strengthValue.textContent=strength.value+'%';
  if(selection)drawOutputs();
}});
zoomRange.addEventListener('input',()=>setZoomPct(zoomRange.value));

enhStage.addEventListener('wheel',e=>{{
  if(!selection)return;
  e.preventDefault();
  const delta=Math.max(-80,Math.min(80,e.deltaY));
  const factor=Math.exp(-delta*0.0012);
  zoom=clamp(zoom*factor,1,8);
  applyZoom();
}},{{passive:false}});

enhStage.addEventListener('pointerdown',e=>{{
  if(!selection)return;
  zoomDrag=true;zoomStartX=e.clientX-panX;zoomStartY=e.clientY-panY;
  enhStage.setPointerCapture?.(e.pointerId);
}});
enhStage.addEventListener('pointermove',e=>{{
  if(!zoomDrag)return;
  panX=e.clientX-zoomStartX;panY=e.clientY-zoomStartY;applyZoom();
}});
enhStage.addEventListener('pointerup',e=>{{
  zoomDrag=false;enhStage.releasePointerCapture?.(e.pointerId);
}});
enhStage.addEventListener('pointercancel',()=>{{zoomDrag=false;}});

applyZoom();
</script>
</body>
</html>
"""

    st.iframe(html, height="content")


def enhance_for_viewing(
    image: Image.Image,
) -> Image.Image:
    """
    Viewing-only enhancement:
    3x Lanczos upscaling + CLAHE + mild sharpening.
    This does not reconstruct missing clinical information.
    """
    rgb = image.convert("RGB")

    width, height = rgb.size

    upscaled = rgb.resize(
        (width * 3, height * 3),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        upscaled,
        dtype=np.uint8,
    )

    lab = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.55,
        tileGridSize=(8, 8),
    )

    l_channel = clahe.apply(
        l_channel
    )

    merged = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel,
        )
    )

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB,
    )

    pil = Image.fromarray(
        enhanced
    )

    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=1.15,
            percent=100,
            threshold=3,
        )
    )

    pil = ImageEnhance.Contrast(
        pil
    ).enhance(
        1.025
    )

    return pil


# ============================================================
# XAI HELPERS
# ============================================================

def run_full_xai(
    image: Image.Image,
    result: dict[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    prediction = result[
        "prediction"
    ]

    return run_explainability_suite(
        model=resources["main_model"],
        grad_model=resources["grad_model"],
        classifier_layers=resources["classifier_layers"],
        image=image,
        probabilities=prediction["main_probabilities"],
        predicted_index=prediction["main_prediction_index"],
        ig_steps=32,
        perturbation_fraction=0.15,
    )


def quick_attention_extent_proxy(
    image: Image.Image,
    prediction: dict[str, Any],
    resources: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns a Grad-CAM attention-extent proxy.
    It is NOT tumor area/volume/spread.
    """
    batch = prepare_model_batch(
        image
    )

    small_map = gradcam_map(
        resources["grad_model"],
        resources["classifier_layers"],
        batch,
        prediction["final_prediction_index"],
    )

    map_300 = resize_map(
        small_map,
        IMG_SIZE[1],
        IMG_SIZE[0],
    )

    adaptive = max(
        0.55,
        float(
            np.quantile(
                map_300,
                0.82,
            )
        ),
    )

    extent = float(
        np.mean(
            map_300 >= adaptive
        )
        * 100.0
    )

    original = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    original_map = resize_map(
        map_300,
        original.shape[1],
        original.shape[0],
    )

    colored = cv2.applyColorMap(
        np.uint8(
            255 * original_map
        ),
        cv2.COLORMAP_JET,
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB,
    )

    overlay = np.clip(
        0.65 * original.astype(np.float32)
        +
        0.35 * colored.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return (
        extent,
        colored,
        overlay,
    )


# ============================================================
# DATE PARSING
# ============================================================

def parse_scan_date(
    filename: str,
) -> date | None:
    patterns = [
        (
            r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})",
            "ymd",
        ),
        (
            r"(\d{2})[-_.](\d{2})[-_.](20\d{2})",
            "dmy",
        ),
    ]

    for pattern, order in patterns:
        match = re.search(
            pattern,
            filename,
        )

        if not match:
            continue

        a, b, c = match.groups()

        try:
            if order == "ymd":
                return date(
                    int(a),
                    int(b),
                    int(c),
                )

            return date(
                int(c),
                int(b),
                int(a),
            )

        except ValueError:
            pass

    return None


# ============================================================
# REPORTING
# ============================================================

def _report_image(
    image: Image.Image | np.ndarray,
    max_width=72 * mm,
    max_height=62 * mm,
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            np.asarray(image, dtype=np.uint8)
        )

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=92,
    )

    buffer.seek(0)

    width, height = image.size

    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1),
    )

    return RLImage(
        buffer,
        width=width * scale,
        height=height * scale,
    )


def build_report_text(
    result: dict[str, Any],
) -> str:
    prediction = result["prediction"]
    quality = result["quality_result"]
    validity = result["validity_result"]
    xai = st.session_state.get(
        "explainability_result"
    )

    main_probs = prediction[
        "main_probabilities"
    ]

    lines = [
        "=" * 66,
        "NEUROVISION AI",
        "BRAIN MRI ANALYSIS REPORT",
        "=" * 66,
        "",
        "ANALYSIS INFORMATION",
        "-" * 66,
        f'Date & Time        : {result["analysis_time"]}',
        f'Uploaded MRI       : {result["filename"]}',
        "Main Architecture  : ResNet50V2",
        "Input Size         : 300 x 300 x 3",
        "Decision Pipeline  : ResNet50V2 + targeted specialists",
        "",
        "INPUT VALIDITY",
        "-" * 66,
        f'Semantic Gate      : {validity["decision"]}',
        f'MRI-Likeness Score : {validity["score"]}/100',
        f'Gate Evidence       : {", ".join(validity["reasons"])}',
        "",
        "MRI QUALITY ASSESSMENT",
        "-" * 66,
        f'Overall Quality     : {quality["overall_status"]}',
        f'Quality Score       : {quality["quality_score"]}%',
        f'Sharpness           : {quality["blur_score"]} [{quality["blur_status"]}]',
        f'Brightness          : {quality["brightness_score"]} [{quality["brightness_status"]}]',
        f'Contrast            : {quality["contrast_score"]} [{quality["contrast_status"]}]',
        f'Resolution          : {quality["resolution"]} [{quality["resolution_status"]}]',
        "",
        "PREDICTION",
        "-" * 66,
        f'Final Class         : {DISPLAY_NAMES[prediction["final_prediction"]]}',
        f'Main ResNet Class   : {DISPLAY_NAMES[prediction["main_prediction"]]}',
        f'Main Top Confidence : {prediction["main_top_confidence"] * 100:.2f}%',
        f'Routing Applied     : {prediction["routing_applied"]}',
        "",
        "MAIN MODEL PROBABILITIES",
        "-" * 66,
    ]

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        lines.append(
            f'{DISPLAY_NAMES[class_name]:20s}: {float(main_probs[index]) * 100:.2f}%'
        )

    lines.extend(
        [
            "",
            "SPECIALIST ROUTING",
            "-" * 66,
            f'Main Top-2 Margin   : {prediction["main_margin"]:.4f}',
            f'GM Specialist Conf. : {prediction["gm_confidence"] * 100:.2f}%',
            f'GM Route Applied    : {prediction["gm_route"]}',
            f'M-vs-Rest M Prob.   : {prediction["mvr_meningioma_probability"] * 100:.2f}%',
            f'M-vs-Rest Route     : {prediction["mvr_route"]}',
        ]
    )

    for step in prediction[
        "routing_steps"
    ]:
        lines.append(
            f'- {step}'
        )

    lines.extend(
        [
            "",
            "CURRENT DEVELOPMENT PERFORMANCE",
            "-" * 66,
            f'Main ResNet Validation Accuracy : {MAIN_VALIDATION_ACCURACY:.2f}%',
            f'After GM Routing                : {GM_ROUTED_VALIDATION_ACCURACY:.2f}%',
            f'Current Routed Validation       : {CURRENT_VALIDATION_ACCURACY:.2f}%',
            f'Validation Set                  : {CURRENT_VALIDATION_IMAGES} images',
            "Improved frozen independent test : NOT YET EVALUATED",
            "",
            "HISTORICAL BASELINE",
            "-" * 66,
            f'EfficientNetV2B0 independent test: {HISTORICAL_INDEPENDENT_TEST_ACCURACY:.2f}% on {HISTORICAL_INDEPENDENT_TEST_IMAGES} images',
        ]
    )

    if xai is not None:
        shortcut = xai["shortcut_diagnostics"]
        faith = xai["faithfulness"]
        trust = xai["explanation_trust"]

        xai_target_name = DISPLAY_NAMES[
            CLASS_NAMES[
                int(
                    xai[
                        "predicted_index"
                    ]
                )
            ]
        ]

        xai_alternative_name = DISPLAY_NAMES[
            CLASS_NAMES[
                int(
                    xai[
                        "alternative_index"
                    ]
                )
            ]
        ]

        lines.extend(
            [
                "",
                "MULTI-METHOD EXPLAINABILITY",
                "-" * 66,
                f'XAI Target (Main ResNet) : {xai_target_name}',
                f'Strongest Alternative    : {xai_alternative_name}',
                f'Target Probability        : {xai["original_probability"] * 100:.2f}%',
                f'Alternative Probability   : {xai["alternative_probability"] * 100:.2f}%',
                f'Target Logit Drop         : {xai["target_logit_drop"]:+.4f}',
                f'Class-Margin Drop         : {xai["margin_drop"]:+.4f}',
                f'Off-Anatomy Prob. Change  : {xai["off_anatomy_drop_pp"]:+.2f} pp',
                f'Central-Anatomy Attention : {shortcut["mean_anatomy_ratio"] * 100:.2f}%',
                f'Border Attention          : {shortcut["mean_border_ratio"] * 100:.2f}%',
                f'Border Dependency Risk    : {shortcut["border_dependency_risk"]}',
                f'Grad-CAM / IG Agreement   : {xai["cross_method_agreement"]:.4f}',
                f'Stability Correlation     : {xai["stability"]["mean_correlation"]:.4f}',
                f'Deletion AUC Probability  : {faith["deletion_auc"]:.4f}',
                f'Insertion AUC Probability : {faith["insertion_auc"]:.4f}',
                f'Deletion AUC Margin       : {faith["deletion_margin_auc"]:.4f}',
                f'Insertion AUC Margin      : {faith["insertion_margin_auc"]:.4f}',
                f'Explanation Trust         : {trust["score"]:.1f}/100 ({trust["label"]})',
                f'Trust Notes               : {", ".join(trust["reasons"])}',
                "",
                "Visual attribution methods: Logit Grad-CAM, Integrated Gradients, Contrastive CAM.",
                "Visual attribution explains the main ResNet50V2 classifier. Specialist routing is reported separately through its explicit probabilities and threshold rules.",
            ]
        )

    else:
        lines.extend(
            [
                "",
                "MULTI-METHOD EXPLAINABILITY",
                "-" * 66,
                "Not generated.",
            ]
        )

    lines.extend(
        [
            "",
            "IMPORTANT NOTES",
            "-" * 66,
            "MRI quality is a heuristic technical screening module and does not alter classifier confidence.",
            "The MRI validity/OOD gate is heuristic and is not clinically validated MRI authentication.",
            "Grad-CAM, Integrated Gradients and Contrastive CAM characterize model influence; they are not exact tumor segmentation.",
            "Explanation Trust is a heuristic model-behavior diagnostic, not confidence, accuracy or clinical certainty.",
            "The current 97.05% value is development validation accuracy, not improved independent-test accuracy.",
            "",
            "DISCLAIMER",
            "-" * 66,
            "NeuroVision AI is intended for research and educational use. Its output must not be treated as medical diagnosis or treatment advice.",
            "",
            "DEVELOPED BY",
            "-" * 66,
            "Mayank",
            "Deepanshu",
            "Priyanshu",
            "=" * 66,
        ]
    )

    return "\n".join(lines)


def create_pdf_report(
    result: dict[str, Any],
    image: Image.Image,
) -> bytes:
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NVTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#102338"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "NVSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#718397"),
        spaceAfter=13,
    )

    heading_style = ParagraphStyle(
        "NVHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#102338"),
        spaceBefore=7,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "NVBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    story: list[Any] = []

    story.append(
        Paragraph(
            "NEUROVISION AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Brain MRI Analysis",
            subtitle_style,
        )
    )

    prediction = result["prediction"]
    quality = result["quality_result"]
    validity = result["validity_result"]

    story.append(
        Paragraph(
            "Analysis Summary",
            heading_style,
        )
    )

    summary = [
        ["Uploaded MRI", result["filename"]],
        ["Date & Time", result["analysis_time"]],
        ["Final Class", DISPLAY_NAMES[prediction["final_prediction"]]],
        ["Main ResNet Class", DISPLAY_NAMES[prediction["main_prediction"]]],
        ["Main Top Confidence", f'{prediction["main_top_confidence"] * 100:.2f}%'],
        ["Routing Applied", str(prediction["routing_applied"])],
        ["MRI Gate", f'{validity["decision"]} ({validity["score"]}/100)'],
        ["Image Quality", f'{quality["overall_status"]} ({quality["quality_score"]}%)'],
    ]

    table = Table(
        summary,
        colWidths=[
            50 * mm,
            110 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5F6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5E0E6")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Uploaded Brain MRI",
            heading_style,
        )
    )

    story.append(
        _report_image(
            image
        )
    )

    xai = st.session_state.get(
        "explainability_result"
    )

    if xai is not None:
        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                "Multi-Method Explainability",
                heading_style,
            )
        )

        gradcam_report = Image.fromarray(
            xai[
                "gradcam_overlay"
            ].astype(
                np.uint8
            )
        )

        ig_report = Image.fromarray(
            xai[
                "integrated_gradients_overlay"
            ].astype(
                np.uint8
            )
        )

        contrastive_report = Image.fromarray(
            xai[
                "contrastive_overlay"
            ].astype(
                np.uint8
            )
        )

        visual_table = Table(
            [
                [
                    Paragraph(
                        "<b>Grad-CAM</b>",
                        body_style,
                    ),
                    Paragraph(
                        "<b>Integrated Gradients</b>",
                        body_style,
                    ),
                    Paragraph(
                        "<b>Contrastive CAM</b>",
                        body_style,
                    ),
                ],
                [
                    _report_image(
                        gradcam_report,
                        max_width=49 * mm,
                        max_height=43 * mm,
                    ),
                    _report_image(
                        ig_report,
                        max_width=49 * mm,
                        max_height=43 * mm,
                    ),
                    _report_image(
                        contrastive_report,
                        max_width=49 * mm,
                        max_height=43 * mm,
                    ),
                ],
            ],
            colWidths=[
                53 * mm,
                53 * mm,
                53 * mm,
            ],
        )

        visual_table.setStyle(
            TableStyle(
                [
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#EEF7F7"
                        ),
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor(
                            "#D5E4E8"
                        ),
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.25,
                        colors.HexColor(
                            "#E2EBEE"
                        ),
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(
            visual_table
        )

        trust = xai[
            "explanation_trust"
        ]

        shortcut = xai[
            "shortcut_diagnostics"
        ]

        faith = xai[
            "faithfulness"
        ]

        xai_target_name = DISPLAY_NAMES[
            CLASS_NAMES[
                int(
                    xai[
                        "predicted_index"
                    ]
                )
            ]
        ]

        xai_alternative_name = DISPLAY_NAMES[
            CLASS_NAMES[
                int(
                    xai[
                        "alternative_index"
                    ]
                )
            ]
        ]

        story.append(
            Spacer(
                1,
                8,
            )
        )

        xai_metrics = [
            [
                "XAI target",
                xai_target_name,
            ],
            [
                "Strongest alternative",
                xai_alternative_name,
            ],
            [
                "Explanation Trust",
                f'{trust["score"]:.1f}/100 ({trust["label"]})',
            ],
            [
                "Grad-CAM ↔ IG agreement",
                f'{xai["cross_method_agreement"]:.3f}',
            ],
            [
                "Mild-input stability",
                f'{xai["stability"]["mean_correlation"]:.3f}',
            ],
            [
                "Central-anatomy attention",
                f'{shortcut["mean_anatomy_ratio"] * 100:.1f}%',
            ],
            [
                "Border dependency risk",
                shortcut[
                    "border_dependency_risk"
                ],
            ],
            [
                "Target logit drop",
                f'{xai["target_logit_drop"]:+.3f}',
            ],
            [
                "Class-margin drop",
                f'{xai["margin_drop"]:+.3f}',
            ],
            [
                "Deletion / insertion AUC (margin)",
                f'{faith["deletion_margin_auc"]:.3f} / {faith["insertion_margin_auc"]:.3f}',
            ],
        ]

        xai_table = Table(
            xai_metrics,
            colWidths=[
                66 * mm,
                94 * mm,
            ],
        )

        xai_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor(
                            "#F2F7F8"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.HexColor(
                            "#D8E4E8"
                        ),
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(
            xai_table
        )

        story.append(
            Spacer(
                1,
                7,
            )
        )

        story.append(
            Paragraph(
                (
                    "Visual attribution in this report explains the main ResNet50V2 "
                    "classifier. Specialist routing is documented separately through "
                    "its explicit probabilities and threshold rules. Grad-CAM, Integrated "
                    "Gradients and Contrastive CAM characterize model influence; they do "
                    "not constitute tumor segmentation or medically confirmed localization."
                ),
                body_style,
            )
        )

    story.append(
        Paragraph(
            "Performance Status",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            (
                f'Current optimized development validation accuracy: '
                f'{CURRENT_VALIDATION_ACCURACY:.2f}% '
                f'({CURRENT_VALIDATION_CORRECT}/{CURRENT_VALIDATION_IMAGES} correct). '
                "The improved routed pipeline has not yet been evaluated on the frozen "
                f'{HISTORICAL_INDEPENDENT_TEST_IMAGES}-image independent test set. '
                f'The historical EfficientNetV2B0 baseline achieved '
                f'{HISTORICAL_INDEPENDENT_TEST_ACCURACY:.2f}% on that independent set.'
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Important Disclaimer",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            (
                "NeuroVision AI is intended for research and educational use. "
                "MRI validity screening and image-quality screening are heuristic. "
                "Attribution maps are not exact tumor boundaries or segmentation. "
                "Outputs must not be treated as a medical diagnosis or treatment recommendation."
            ),
            body_style,
        )
    )

    story.append(
        Spacer(1, 14)
    )

    story.append(
        Paragraph(
            "<b>Developed By</b><br/>Mayank · Deepanshu · Priyanshu",
            body_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()




def detailed_xai(result, result_image, xai):
    prediction = result['prediction']
    shortcut = xai['shortcut_diagnostics']
    trust = xai['explanation_trust']
    faith = xai['faithfulness']
    xai_target_index = int(xai['predicted_index'])
    alternative_index = int(xai['alternative_index'])
    xai_target_name = DISPLAY_NAMES[CLASS_NAMES[xai_target_index]]
    alternative_name = DISPLAY_NAMES[CLASS_NAMES[alternative_index]]
    agreement_value = float(xai['cross_method_agreement'])
    stability_value = float(xai['stability']['mean_correlation'])
    agreement_label = 'Strong' if agreement_value >= .65 else 'Moderate' if agreement_value >= .35 else 'Low'
    stability_label = 'High' if stability_value >= .85 else 'Moderate' if stability_value >= .60 else 'Low'
    deep_tab, decision_tab, perturb_tab, faith_tab, trust_tab = st.tabs(
        [
            "Deep Viewer",
            "Decision Contrast",
            "Perturbation Tests",
            "Faithfulness",
            "Trust & Stability",
        ]
    )

    # ================================================
    # DEEP VIEWER
    # ================================================
    with deep_tab:
        nv_html(
            """
            <div class="nv-xai-section">
                <div class="nv-xai-section-title">Interactive explanation viewer</div>
                <div class="nv-xai-section-copy">
                    Zoom up to 1200%, pan, control overlay opacity and enter fullscreen.
                    Auto HQ switches to a viewing-only enhanced rendering during deep zoom.
                    The enhanced rendering does not recover missing diagnostic detail.
                </div>
            </div>
            """
        )

        evidence_layer_labels = {
            "Grad-CAM Overlay": "Grad-CAM · Spatial Influence",
            "Integrated Gradients Overlay": "Integrated Gradients · Pixel Attribution",
            "Contrastive CAM Overlay": "Contrastive CAM · Class Difference",
            "Central-Anatomy Proxy": "Central Anatomy · Heuristic ROI",
            "Salient Perturbation Region": "Salient Region · Perturbation Target",
            "Off-Anatomy Control Region": "Off-Anatomy · Control Region",
        }

        deep_view = st.radio(
            "Evidence layer",
            list(evidence_layer_labels.keys()),
            format_func=lambda value: evidence_layer_labels[value],
            horizontal=True,
            key="xai_deep_view",
        )

        nv_html(
            """
            <div style="
                margin:6px 0 12px;
                color:#6a7f91;
                font-size:10px;
                line-height:1.55;
            ">
                <strong style="color:#17354d">Layer guide:</strong>
                Grad-CAM = class-level spatial influence ·
                Integrated Gradients = pixel attribution ·
                Contrastive CAM = target-vs-alternative evidence ·
                Central Anatomy = heuristic ROI ·
                Salient Region = tested important area ·
                Off-Anatomy = matched control area.
            </div>
            """
        )

        if deep_view == "Grad-CAM Overlay":
            interactive_image_viewer(
                result_image,
                f"{xai_target_name} · Grad-CAM",
                overlay=xai[
                    "gradcam_overlay"
                ],
                overlay_label="Grad-CAM opacity",
                enable_zoom_enhance=True,
            )

            st.write("")
            gc1, gc2, gc3, gc4 = st.columns(4)

            gc1.metric(
                "XAI target",
                xai_target_name,
            )

            gc2.metric(
                "Target probability",
                f'{xai["original_probability"] * 100:.2f}%',
            )

            gc3.metric(
                "Agreement with IG",
                f'{xai["agreement_matrix"]["gradcam_vs_integrated_gradients"]:.3f}',
            )

            gc4.metric(
                "Agreement with Contrastive",
                f'{xai["agreement_matrix"]["gradcam_vs_contrastive"]:.3f}',
            )

            nv_html(
                """
                <div class="nv-notice info" style="margin-top:8px">
                    Grad-CAM shows spatial influence for the selected main-model class.
                    Agreement values measure consistency with other XAI methods, not medical correctness.
                </div>
                """
            )

        elif deep_view == "Integrated Gradients Overlay":
            interactive_image_viewer(
                result_image,
                f"{xai_target_name} · Integrated Gradients",
                overlay=xai[
                    "integrated_gradients_overlay"
                ],
                overlay_label="IG opacity",
                enable_zoom_enhance=True,
            )

            st.write("")
            ig1, ig2, ig3, ig4 = st.columns(4)

            ig1.metric(
                "XAI target",
                xai_target_name,
            )

            ig2.metric(
                "IG integration steps",
                f'{int(xai["ig_steps"])}',
            )

            ig3.metric(
                "Agreement with Grad-CAM",
                f'{xai["agreement_matrix"]["gradcam_vs_integrated_gradients"]:.3f}',
            )

            ig4.metric(
                "Agreement with Contrastive",
                f'{xai["agreement_matrix"]["integrated_gradients_vs_contrastive"]:.3f}',
            )

            nv_html(
                """
                <div class="nv-notice info" style="margin-top:8px">
                    Integrated Gradients reports pixel-level attribution accumulated from the
                    configured baselines to the MRI. Agreement values indicate cross-method consistency.
                </div>
                """
            )

        elif deep_view == "Contrastive CAM Overlay":
            interactive_image_viewer(
                result_image,
                f"{xai_target_name} vs {alternative_name} · Contrastive CAM",
                overlay=xai[
                    "contrastive_overlay"
                ],
                overlay_label="Contrastive opacity",
                enable_zoom_enhance=True,
            )

            st.write("")
            cc1, cc2, cc3, cc4 = st.columns(4)

            cc1.metric(
                "Target class",
                xai_target_name,
            )

            cc2.metric(
                "Strongest alternative",
                alternative_name,
            )

            cc3.metric(
                "Target probability",
                f'{xai["original_probability"] * 100:.2f}%',
            )

            cc4.metric(
                "Alternative probability",
                f'{xai["alternative_probability"] * 100:.2f}%',
            )

            nv_html(
                f"""
                <div class="nv-notice info" style="margin-top:8px">
                    Contrastive CAM highlights evidence that differentiates
                    <b>{xai_target_name}</b> from the strongest competing class
                    <b>{alternative_name}</b>. It is not a tumor-segmentation mask.
                </div>
                """
            )

        elif deep_view == "Central-Anatomy Proxy":
            interactive_image_viewer(
                xai[
                    "anatomy_proxy_visual"
                ],
                "Central-Anatomy Proxy · Heuristic ROI",
                enable_zoom_enhance=True,
            )

            st.write("")
            ca1, ca2, ca3, ca4 = st.columns(4)

            ca1.metric(
                "Central-anatomy attention",
                f'{shortcut["mean_anatomy_ratio"] * 100:.1f}%',
            )

            ca2.metric(
                "Border attention",
                f'{shortcut["mean_border_ratio"] * 100:.1f}%',
            )

            ca3.metric(
                "Foreground attention",
                f'{shortcut["mean_foreground_ratio"] * 100:.1f}%',
            )

            ca4.metric(
                "Border dependency risk",
                shortcut["border_dependency_risk"],
            )

            nv_html(
                """
                <div class="nv-notice info" style="margin-top:8px">
                    Green outline = heuristic central-anatomy proxy ·
                    Red rectangle = border-control boundary.
                    These values describe model-attention distribution and shortcut risk;
                    they are not tumor-localization or segmentation accuracy.
                </div>
                """
            )

        elif deep_view == "Salient Perturbation Region":
            interactive_image_viewer(
                xai[
                    "salient_region_visual"
                ],
                "Central-Anatomy Salient Region Tested",
                enable_zoom_enhance=True,
            )

            st.write("")
            sp1, sp2, sp3, sp4 = st.columns(4)

            sp1.metric(
                "Target logit drop",
                f'{xai["target_logit_drop"]:+.3f}',
            )

            sp2.metric(
                "Class-margin drop",
                f'{xai["margin_drop"]:+.3f}',
            )

            sp3.metric(
                "Target probability Δ",
                f'{xai["confidence_drop_pp"]:+.2f} pp',
            )

            sp4.metric(
                "Influence strength",
                xai["logit_influence_strength"],
            )

            nv_html(
                """
                <div class="nv-notice info" style="margin-top:8px">
                    These values show how strongly the main-model decision changed after the
                    salient central-anatomy region was blurred and inference was repeated.
                    Larger drops support stronger model influence, not clinical causality.
                </div>
                """
            )

        else:
            interactive_image_viewer(
                xai[
                    "off_anatomy_region_visual"
                ],
                "Matched Off-Anatomy Control Region",
                enable_zoom_enhance=True,
            )

            st.write("")
            oa1, oa2, oa3, oa4 = st.columns(4)

            oa1.metric(
                "Off-anatomy probability Δ",
                f'{xai["off_anatomy_drop_pp"]:+.2f} pp',
            )

            oa2.metric(
                "Off-anatomy logit drop",
                f'{xai["off_anatomy_logit_drop"]:+.3f}',
            )

            oa3.metric(
                "Off-anatomy margin drop",
                f'{xai["off_anatomy_margin_drop"]:+.3f}',
            )

            oa4.metric(
                "Salient-region probability Δ",
                f'{xai["confidence_drop_pp"]:+.2f} pp',
            )

            nv_html(
                """
                <div class="nv-notice info" style="margin-top:8px">
                    The off-anatomy region is the matched control. Compare its changes with the
                    salient-region change: a substantially larger salient-region effect supports
                    region-specific influence rather than general sensitivity to any image disturbance.
                </div>
                """
            )

    # ================================================
    # DECISION CONTRAST
    # ================================================
    with decision_tab:
        nv_html(
            """
            <div class="nv-xai-section">
                <div class="nv-xai-section-title">Why this main-model class instead of the strongest alternative?</div>
                <div class="nv-xai-section-copy">
                    Contrastive evidence compares the main ResNet50V2 target class
                    with the strongest competing class. Specialist routing is shown
                    separately so one heatmap is not presented as an explanation of
                    the entire routed decision system.
                </div>
            </div>
            """
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Main XAI target",
            xai_target_name,
        )

        c2.metric(
            "Strongest alternative",
            alternative_name,
        )

        c3.metric(
            "Target probability",
            f'{xai["original_probability"] * 100:.2f}%',
        )

        c4.metric(
            "Probability margin",
            f'{(xai["original_probability"] - xai["alternative_probability"]) * 100:.2f} pp',
        )

        st.write("")

        left_route, right_route = st.columns(
            [1.08, 0.92],
            gap="large",
        )

        with left_route:
            interactive_image_viewer(
                result_image,
                f"Contrastive evidence · {xai_target_name} vs {alternative_name}",
                overlay=xai[
                    "contrastive_overlay"
                ],
                overlay_label="Contrastive opacity",
                enable_zoom_enhance=True,
            )

        with right_route:
            prediction = result[
                "prediction"
            ]

            nv_html(
                f"""
                <div class="nv-xai-summary">
                    <b>Main ResNet decision:</b>
                    {DISPLAY_NAMES[prediction["main_prediction"]]}<br>
                    <b>Final routed decision:</b>
                    {DISPLAY_NAMES[prediction["final_prediction"]]}<br><br>
                    Visual attribution in this workspace explains the main
                    ResNet50V2 classifier. If a specialist changes the final class,
                    the override is explained through its explicit probability,
                    margin and threshold rule rather than pretending the main-model
                    heatmap explains the specialist decision.
                </div>
                """
            )

            st.write("")

            r1, r2 = st.columns(
                2
            )

            r1.metric(
                "Main top-2 margin",
                f'{prediction["main_margin"]:.3f}',
            )

            r2.metric(
                "GM specialist confidence",
                f'{prediction["gm_confidence"] * 100:.1f}%',
            )

            r3, r4 = st.columns(
                2
            )

            r3.metric(
                "M-vs-Rest M probability",
                f'{prediction["mvr_meningioma_probability"] * 100:.1f}%',
            )

            r4.metric(
                "Routing",
                (
                    "Applied"
                    if prediction[
                        "routing_applied"
                    ]
                    else "Not applied"
                ),
            )

            if prediction[
                "routing_steps"
            ]:
                for route_step in prediction[
                    "routing_steps"
                ]:
                    st.info(
                        route_step
                    )
            else:
                st.success(
                    "The main ResNet50V2 decision was retained."
                )

    # ================================================
    # PERTURBATION
    # ================================================
    with perturb_tab:
        nv_html(
            """
            <div class="nv-xai-section">
                <div class="nv-xai-section-title">Counterfactual-style perturbation stress test</div>
                <div class="nv-xai-section-copy">
                    The system blurs the most salient pixels inside the central-anatomy
                    proxy, re-runs inference and compares the result with a matched
                    off-anatomy control. Probability, pre-softmax logit and class margin
                    are reported together to reduce softmax-saturation ambiguity.
                </div>
            </div>
            """
        )

        p1, p2, p3, p4, p5 = st.columns(
            5
        )

        p1.metric(
            "Target logit drop",
            f'{xai["target_logit_drop"]:+.3f}',
        )

        p2.metric(
            "Class-margin drop",
            f'{xai["margin_drop"]:+.3f}',
        )

        p3.metric(
            "Target probability Δ",
            f'{xai["confidence_drop_pp"]:+.2f} pp',
        )

        p4.metric(
            "Off-anatomy probability Δ",
            f'{xai["off_anatomy_drop_pp"]:+.2f} pp',
        )

        p5.metric(
            "Influence strength",
            xai[
                "logit_influence_strength"
            ],
        )

        st.write("")

        pa, pb = st.columns(
            2,
            gap="large",
        )

        with pa:
            st.image(
                xai[
                    "salient_region_visual"
                ],
                caption="Central-anatomy salient region blurred and re-tested",
                width="stretch",
            )

        with pb:
            st.image(
                xai[
                    "off_anatomy_region_visual"
                ],
                caption="Matched off-anatomy control region",
                width="stretch",
            )

        perturb_compare = pd.DataFrame(
            {
                "Test": [
                    "Central-anatomy",
                    "Off-anatomy control",
                ],
                "Probability change (pp)": [
                    float(
                        xai[
                            "confidence_drop_pp"
                        ]
                    ),
                    float(
                        xai[
                            "off_anatomy_drop_pp"
                        ]
                    ),
                ],
                "Target-logit change": [
                    float(
                        xai[
                            "target_logit_drop"
                        ]
                    ),
                    float(
                        xai[
                            "off_anatomy_logit_drop"
                        ]
                    ),
                ],
                "Class-margin change": [
                    float(
                        xai[
                            "margin_drop"
                        ]
                    ),
                    float(
                        xai[
                            "off_anatomy_margin_drop"
                        ]
                    ),
                ],
            }
        )

        st.dataframe(
            perturb_compare,
            hide_index=True,
            width="stretch",
        )

        nv_html(
            """
            <div class="nv-notice warn">
                Perturbation is a model-behavior experiment. Blurring pixels can
                create distribution shift. The central-anatomy proxy is heuristic
                and is not a validated brain mask or lesion segmentation.
            </div>
            """
        )

    # ================================================
    # FAITHFULNESS
    # ================================================
    with faith_tab:
        nv_html(
            """
            <div class="nv-xai-section">
                <div class="nv-xai-section-title">Deletion / insertion faithfulness</div>
                <div class="nv-xai-section-copy">
                    Salient pixels are progressively removed from the MRI or restored
                    from a baseline. Heavy blur is the primary curve; light blur and
                    local-mean replacement are sensitivity checks. Probability and
                    target-vs-runner-up class margin are both inspected.
                </div>
            </div>
            """
        )

        fractions_percent = [
            float(value) * 100.0
            for value
            in faith[
                "fractions"
            ]
        ]

        probability_df = pd.DataFrame(
            {
                "Fraction (%)": fractions_percent,
                "Deletion": [
                    float(value) * 100.0
                    for value
                    in faith[
                        "deletion_probabilities"
                    ]
                ],
                "Insertion": [
                    float(value) * 100.0
                    for value
                    in faith[
                        "insertion_probabilities"
                    ]
                ],
            }
        )

        probability_long = probability_df.melt(
            id_vars=[
                "Fraction (%)"
            ],
            var_name="Series",
            value_name="Target probability (%)",
        )

        probability_chart = (
            alt.Chart(
                probability_long
            )
            .mark_line(
                point=True
            )
            .encode(
                x=alt.X(
                    "Fraction (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                y=alt.Y(
                    "Target probability (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                strokeDash="Series:N",
                tooltip=[
                    "Fraction (%):Q",
                    "Series:N",
                    "Target probability (%):Q",
                ],
            )
            .properties(
                height=310
            )
        )

        margin_denominator = max(
            float(
                xai[
                    "original_margin"
                ]
            ),
            1e-8,
        )

        margin_df = pd.DataFrame(
            {
                "Fraction (%)": fractions_percent,
                "Deletion": [
                    float(
                        np.clip(
                            float(value)
                            /
                            margin_denominator,
                            0.0,
                            1.0,
                        )
                        *
                        100.0
                    )
                    for value
                    in faith[
                        "deletion_margins"
                    ]
                ],
                "Insertion": [
                    float(
                        np.clip(
                            float(value)
                            /
                            margin_denominator,
                            0.0,
                            1.0,
                        )
                        *
                        100.0
                    )
                    for value
                    in faith[
                        "insertion_margins"
                    ]
                ],
            }
        )

        margin_long = margin_df.melt(
            id_vars=[
                "Fraction (%)"
            ],
            var_name="Series",
            value_name="Normalized class-margin support (%)",
        )

        margin_chart = (
            alt.Chart(
                margin_long
            )
            .mark_line(
                point=True
            )
            .encode(
                x=alt.X(
                    "Fraction (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                y=alt.Y(
                    "Normalized class-margin support (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                strokeDash="Series:N",
                tooltip=[
                    "Fraction (%):Q",
                    "Series:N",
                    "Normalized class-margin support (%):Q",
                ],
            )
            .properties(
                height=310
            )
        )

        chart_left, chart_right = st.columns(
            2,
            gap="large",
        )

        with chart_left:
            st.caption(
                "Probability faithfulness"
            )
            st.altair_chart(
                probability_chart,
                width="stretch",
            )

        with chart_right:
            st.caption(
                "Class-margin faithfulness"
            )
            st.altair_chart(
                margin_chart,
                width="stretch",
            )

        f1, f2, f3, f4 = st.columns(
            4
        )

        f1.metric(
            "Deletion AUC · probability",
            f'{faith["deletion_auc"]:.3f}',
        )

        f2.metric(
            "Insertion AUC · probability",
            f'{faith["insertion_auc"]:.3f}',
        )

        f3.metric(
            "Deletion AUC · margin",
            f'{faith["deletion_margin_auc"]:.3f}',
        )

        f4.metric(
            "Insertion AUC · margin",
            f'{faith["insertion_margin_auc"]:.3f}',
        )

        baseline_rows = []

        for baseline in faith[
            "baselines"
        ].values():
            baseline_rows.append(
                {
                    "Baseline": baseline[
                        "display_name"
                    ],
                    "100% deleted probability": f'{baseline["final_target_probability"] * 100:.2f}%',
                    "100% deleted target logit": f'{baseline["final_target_logit"]:.3f}',
                    "100% deleted class margin": f'{baseline["final_target_margin"]:.3f}',
                    "Top class retained": (
                        "Yes"
                        if baseline[
                            "same_top_class"
                        ]
                        else "No"
                    ),
                    "Status": baseline[
                        "status"
                    ],
                }
            )

        st.dataframe(
            pd.DataFrame(
                baseline_rows
            ),
            hide_index=True,
            width="stretch",
        )

        if faith[
            "baseline_warning_level"
        ] == "Caution":
            st.warning(
                faith[
                    "baseline_warning"
                ]
            )

        elif faith[
            "baseline_warning_level"
        ] == "Review":
            st.info(
                faith[
                    "baseline_warning"
                ]
            )

        else:
            st.success(
                faith[
                    "baseline_warning"
                ]
            )

    # ================================================
    # TRUST + STABILITY
    # ================================================
    with trust_tab:
        nv_html(
            """
            <div class="nv-xai-section">
                <div class="nv-xai-section-title">Consistency, shortcut and trust diagnostics</div>
                <div class="nv-xai-section-copy">
                    This panel asks whether attribution methods agree, whether the map
                    remains stable under mild input changes, whether attention is
                    concentrated on central anatomy rather than borders, and which
                    components drive the heuristic Explanation Trust score.
                </div>
            </div>
            """
        )

        a1, a2, a3, a4 = st.columns(
            4
        )

        a1.metric(
            "Central-anatomy attention",
            f'{shortcut["mean_anatomy_ratio"] * 100:.1f}%',
        )

        a2.metric(
            "Border attention",
            f'{shortcut["mean_border_ratio"] * 100:.1f}%',
        )

        a3.metric(
            "Foreground attention",
            f'{shortcut["mean_foreground_ratio"] * 100:.1f}%',
        )

        a4.metric(
            "Border dependency risk",
            shortcut[
                "border_dependency_risk"
            ],
        )

        st.write("")

        agreement = xai[
            "agreement_matrix"
        ]

        agreement_matrix = pd.DataFrame(
            [
                [
                    1.0,
                    float(
                        agreement[
                            "gradcam_vs_integrated_gradients"
                        ]
                    ),
                    float(
                        agreement[
                            "gradcam_vs_contrastive"
                        ]
                    ),
                ],
                [
                    float(
                        agreement[
                            "gradcam_vs_integrated_gradients"
                        ]
                    ),
                    1.0,
                    float(
                        agreement[
                            "integrated_gradients_vs_contrastive"
                        ]
                    ),
                ],
                [
                    float(
                        agreement[
                            "gradcam_vs_contrastive"
                        ]
                    ),
                    float(
                        agreement[
                            "integrated_gradients_vs_contrastive"
                        ]
                    ),
                    1.0,
                ],
            ],
            index=[
                "Grad-CAM",
                "Integrated Gradients",
                "Contrastive CAM",
            ],
            columns=[
                "Grad-CAM",
                "Integrated Gradients",
                "Contrastive CAM",
            ],
        )

        agreement_long = (
            agreement_matrix
            .reset_index(
                names="Method A"
            )
            .melt(
                id_vars="Method A",
                var_name="Method B",
                value_name="Correlation",
            )
        )

        agreement_chart = (
            alt.Chart(
                agreement_long
            )
            .mark_rect(
                cornerRadius=5
            )
            .encode(
                x=alt.X(
                    "Method B:N",
                    title=None,
                ),
                y=alt.Y(
                    "Method A:N",
                    title=None,
                ),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(
                        domain=[
                            -1,
                            1,
                        ]
                    ),
                ),
                tooltip=[
                    "Method A:N",
                    "Method B:N",
                    alt.Tooltip(
                        "Correlation:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=280
            )
        )

        stability_labels = [
            "Brightness −3%",
            "Brightness +3%",
            "Gaussian noise",
        ]

        stability_values = [
            float(value)
            for value
            in xai[
                "stability"
            ][
                "correlations"
            ]
        ]

        stability_df = pd.DataFrame(
            {
                "Perturbation": stability_labels[
                    :len(
                        stability_values
                    )
                ],
                "Correlation": stability_values,
            }
        )

        stability_chart = (
            alt.Chart(
                stability_df
            )
            .mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
            )
            .encode(
                x=alt.X(
                    "Perturbation:N",
                    title=None,
                ),
                y=alt.Y(
                    "Correlation:Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            1,
                        ]
                    ),
                ),
                tooltip=[
                    "Perturbation:N",
                    alt.Tooltip(
                        "Correlation:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(
                height=280
            )
        )

        trust_components = pd.DataFrame(
            {
                "Component": [
                    key.replace(
                        "_",
                        " ",
                    ).title()
                    for key
                    in trust[
                        "components"
                    ].keys()
                ],
                "Score (%)": [
                    float(value) * 100.0
                    for value
                    in trust[
                        "components"
                    ].values()
                ],
            }
        )
        trust_components["Label"] = trust_components[
            "Score (%)"
        ].map(
            lambda value: f"{value:.1f}%"
        )

        trust_bar = (
            alt.Chart(
                trust_components
            )
            .mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
            )
            .encode(
                y=alt.Y(
                    "Component:N",
                    sort="-x",
                    title=None,
                ),
                x=alt.X(
                    "Score (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                tooltip=[
                    "Component:N",
                    alt.Tooltip(
                        "Score (%):Q",
                        format=".1f",
                    ),
                ],
            )
        )

        trust_labels = (
            alt.Chart(
                trust_components
            )
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                fontSize=11,
                fontWeight=600,
            )
            .encode(
                y=alt.Y(
                    "Component:N",
                    sort="-x",
                    title=None,
                ),
                x=alt.X(
                    "Score (%):Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            100,
                        ]
                    ),
                ),
                text="Label:N",
            )
        )

        trust_chart = (
            trust_bar
            +
            trust_labels
        ).properties(
            height=280
        )

        tcol1, tcol2, tcol3 = st.columns(
            3,
            gap="large",
        )

        with tcol1:
            st.caption(
                "Cross-method agreement matrix"
            )
            st.altair_chart(
                agreement_chart,
                width="stretch",
            )

        with tcol2:
            st.caption(
                "Mild-input stability"
            )
            st.altair_chart(
                stability_chart,
                width="stretch",
            )

        with tcol3:
            st.caption(
                "Explanation Trust components"
            )
            st.altair_chart(
                trust_chart,
                width="stretch",
            )

        trust_reasons = ", ".join(
            trust[
                "reasons"
            ]
        )

        nv_html(
            f"""
            <div class="nv-xai-summary">
                <b>Explanation Trust:</b> {trust["score"]:.1f}/100
                ({trust["label"]}).<br>
                <b>Diagnostic notes:</b> {trust_reasons}.<br><br>
                Explanation Trust is a transparent heuristic model-behavior
                diagnostic. It is not classifier confidence, accuracy,
                medical certainty or a clinically validated score.
            </div>
            """
        )



BRAIN_LOADER_DATA_URI = 'data:image/webp;base64,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T/5X/d/az4C/0//5P9B/n/3X+Z//3/4n/I/df4M/Xn/z/2f+5+Q/+df3P/t/4n/Vft3873/s94/7nf+X/j/CZ+vv/P/P9f3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3ZXkIVnwo7cCapEo80d2uQNp2TrpmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZdBSIYJGV2ZRP0VmP2B3hsECs6+cDjvY/lEPN/V+pkRG/A2GfuLAUogB4Vm2F63gKxL5rBJDo/QnMrFFOmCjpDo/cPp1Rnf1pItl41H0gcjijZA2nZOumZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZbSbUB2B9TdQHAv7VCf/3+e8k/xF4c74Q6MgsBIpXYEO7s2NTVvCaJG5t6tRc69RpaBZqi7sIMfEFdlRx0ORk9LUXfwfRf+Ylj1/n0eFqSlznMQ6SW2N5BlmtqnIwdOhwcFtERVc7244RmVL5bw6Af+Jd3mmE/uG/o6zFrIzD/5VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGi5BHF59vH20RQYmva4Sd/lTLouBfXwG72dh/nndCeW0HfDeVmMs3RHJhd1ssQcVSuN9kds6hobNz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'

PRODUCT_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--nv-bg:#f4f6f8;--nv-paper:#fff;--nv-ink:#131a24;--nv-muted:#687382;--nv-line:#e7eaee;--nv-navy:#071526;--nv-accent:#5967e8}
.stApp{background:var(--nv-bg);color:var(--nv-ink);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.stApp header[data-testid=stHeader]{background:transparent}
.block-container{max-width:1400px;padding:2rem 3rem 3rem}
[data-testid=stSidebar], [data-testid=stSidebarCollapsedControl]{display:none}
h1,h2,h3,p,button,input,label{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
h1{font-size:34px!important;font-weight:600!important;letter-spacing:-1.2px!important;line-height:1.2!important}
h2,h3{font-weight:500!important;letter-spacing:-.4px!important}
h3{font-size:19px!important} p,label{font-size:14px} [data-testid=stCaptionContainer]{color:var(--nv-muted)}
[data-testid=stVerticalBlock]{gap:1rem}
[data-testid=stVerticalBlockBorderWrapper]>div{border-color:var(--nv-line);border-radius:18px}
[data-testid=stVerticalBlockBorderWrapper]:has(>.st-key-nv_header){border:0}
.st-key-nv_header{padding:12px 0 27px;border-bottom:1px solid var(--nv-line);margin-bottom:22px}
.nv-brand{display:flex;align-items:center;gap:11px;min-height:46px;font-size:19px;font-weight:500;letter-spacing:-.6px;white-space:nowrap}
.nv-brand-mark{background:var(--nv-navy);border-radius:12px;width:39px;height:39px;color:white;display:grid;place-items:center;font-size:13px;letter-spacing:-1px}.nv-brand em{font-style:normal;color:var(--nv-muted)}
.st-key-nv_navigation [data-testid=stButtonGroup]{background:var(--nv-navy);border-radius:32px;padding:5px;gap:3px}
.st-key-nv_navigation button{background:transparent!important;color:#c2cad7!important;border:0!important;border-radius:25px!important;box-shadow:none!important;min-height:39px;padding:6px 15px}
.st-key-nv_navigation button[aria-pressed=true],.st-key-nv_navigation button[aria-checked=true]{background:#fff!important;color:#071526!important}
.st-key-nv_navigation button p{font-size:12px;white-space:nowrap}
.nv-heading{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:8px 0 10px}.nv-heading h1{margin:0;padding:0}.nv-heading p{color:var(--nv-muted);margin:10px 0 0;font-size:14px}.nv-status{white-space:nowrap;font-size:11px;border:1px solid var(--nv-line);border-radius:24px;background:white;padding:11px 15px}.nv-status:before{content:'';display:inline-block;background:#2d8b70;width:5px;height:5px;border-radius:50%;margin-right:7px}
.st-key-nv_upload_hero{background:var(--nv-navy);border-radius:20px;padding:32px 38px;color:white;overflow:hidden;margin-top:6px}
.st-key-nv_upload_hero h2,.st-key-nv_upload_hero h3{font-size:28px!important;color:white;letter-spacing:-1px}.st-key-nv_upload_hero [data-testid=stCaptionContainer]{color:#b3bdcd}.st-key-nv_upload_hero [data-testid=stFileUploaderDropzone]{background:#101f32;border:1px dashed #455168;border-radius:12px;padding:15px}.st-key-nv_upload_hero [data-testid=stFileUploaderDropzone] *{color:#d4dbe7}.st-key-nv_upload_hero [data-testid=stFileUploaderDropzone] button{background:#fff;color:#071526;border:0}.st-key-nv_upload_hero [data-testid=stFileUploaderDropzone] button *{color:#071526}.st-key-nv_upload_hero [data-testid=stFileUploaderFile] *{color:#d4dbe7}
.nv-mri{background:#050b12;border-radius:12px;overflow:hidden;position:relative;display:flex;align-items:center;justify-content:center;min-height:300px}.nv-mri img{width:100%;height:360px;object-fit:contain}.nv-mri span{position:absolute;bottom:14px;left:16px;color:#aab5c6;font-size:11px;background:#071526cc;padding:7px 9px;border-radius:6px}
.st-key-nv_upload_hero .nv-mri{background:transparent}.st-key-nv_upload_hero .nv-mri img{height:340px;filter:grayscale(1)}
.nv-card{background:#fff;border:1px solid var(--nv-line);border-radius:18px;padding:24px;min-width:0}.nv-card h3{margin:0 0 17px;padding:0;color:var(--nv-ink)}.nv-tags{display:flex;flex-wrap:wrap;gap:12px 22px;color:#596471;font-size:12px}.nv-label{color:var(--nv-muted);font-size:12px;margin-bottom:12px}.nv-prediction{font-size:25px;letter-spacing:-.8px;font-weight:500}.nv-number{font-size:47px;line-height:1.2;letter-spacing:-2px;font-weight:500;margin:15px 0 8px}.nv-prob-row{margin:19px 0 0}.nv-prob-label{display:flex;justify-content:space-between;font-size:12px;margin-bottom:8px;color:#596471}.nv-prob-track{background:#f0f1f6;height:4px;border-radius:5px;overflow:hidden}.nv-prob-fill{background:#7466e8;height:100%;border-radius:5px;transition:width .35s ease}.nv-summary{display:grid;grid-template-columns:1fr 1fr;gap:22px}.nv-summary label{display:block;font-size:11px;color:var(--nv-muted);margin-bottom:7px}.nv-summary b{font-size:13px;font-weight:500}.nv-summary .ready{color:#2d8b70}
.nv-metric{background:white;border:1px solid var(--nv-line);border-radius:16px;padding:23px}.nv-metric label{font-size:12px;color:var(--nv-muted);display:block}.nv-metric strong{font-size:30px;font-weight:500;letter-spacing:-1px;display:block;margin-top:12px}.nv-metric small{display:block;color:var(--nv-muted);font-size:11px;margin-top:8px}
.st-key-nv_scan_card,.st-key-nv_xai_card{background:#fff;border-radius:18px}
.nv-report-head,.nv-report-row{display:grid;grid-template-columns:1.1fr 1.1fr .7fr 1.1fr;gap:18px;align-items:center}.nv-report-head{color:var(--nv-muted);font-size:11px;padding-bottom:15px;border-bottom:1px solid var(--nv-line)}.nv-report-row{padding:18px 0 0;font-size:13px}.nv-report-image{display:flex;align-items:center;gap:12px;min-width:0}.nv-report-image img{width:50px;height:55px;object-fit:contain;background:#071526;border-radius:7px}.nv-report-image span{overflow-wrap:anywhere}
.nv-timeline{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}.nv-date{border:1px solid var(--nv-line);border-radius:22px;background:white;font-size:12px;padding:9px 15px}.nv-date+ .nv-date:before{content:'—';margin-right:14px;color:#a9b2bf}
.nv-notice,.nv-xai-summary{padding:14px 17px;background:#eef1f7;border-radius:10px;font-size:13px;color:#586573;line-height:1.6}.nv-notice.warn{background:#fff7e9;color:#846520}.nv-notice.good{background:#edf7f2;color:#2d765d}.nv-eyebrow{font-size:11px;color:#687382}.nv-card-title,.nv-xai-section-title{font-size:20px;font-weight:500}.nv-card-copy,.nv-xai-section-copy{font-size:13px;color:#687382;line-height:1.6}.nv-xai-section{margin:14px 0}.nv-chip{display:inline-block;padding:5px 10px;border-radius:20px;background:#edf0f8;font-size:12px;margin-right:6px}
[data-testid=stMetric]{padding:18px;background:white;border:1px solid var(--nv-line);border-radius:14px}[data-testid=stMetricValue]{font-size:25px}
.stButton button,.stDownloadButton button{border-radius:10px;min-height:42px;transition:transform .15s,background .15s}.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-1px)}.stButton button[kind=primary]{background:var(--nv-navy);border:1px solid var(--nv-navy);color:white}.st-key-nv_upload_hero .stButton button[kind=primary]{background:white;color:#071526;border:0}
.st-key-nv_xai_method [data-testid=stButtonGroup]{padding:5px;background:#e9ecf1;border-radius:12px}.st-key-nv_xai_method button{border:0;border-radius:9px;background:transparent;color:#687382}.st-key-nv_xai_method button[aria-pressed=true],.st-key-nv_xai_method button[aria-checked=true]{background:white;color:#131a24}
.nv-loader{background:#000;border-radius:18px;padding:42px 20px;display:flex;align-items:center;flex-direction:column;color:white;min-height:520px}.nv-loader-art{position:relative;width:280px;height:300px;background-image:linear-gradient(#191b1c 1px,transparent 1px),linear-gradient(90deg,#191b1c 1px,transparent 1px);background-size:25px 25px}.nv-loader-ring{position:absolute;top:50%;left:50%;width:234px;height:234px;border:1px solid #b1b7bc;border-radius:50%;transform:translate(-50%,-50%)}.nv-loader-ring.small{width:166px;height:166px}.nv-loader-orbit{position:absolute;width:234px;height:234px;top:33px;left:23px;animation:nv-orbit 12s linear infinite}.nv-loader-orbit:before,.nv-loader-orbit:after{content:'';position:absolute;width:15px;height:15px;border:1px solid #d2d7db;background:black;border-radius:50%;left:50%;top:-7px}.nv-loader-orbit:after{top:auto;bottom:-7px;left:32%}.nv-loader-brain{position:absolute;inset:0;display:grid;place-items:center;perspective:700px}.nv-loader-brain img{width:200px;height:173px;object-fit:contain;animation:nv-brain 12s ease-in-out infinite}.nv-loader h2{font-size:25px!important;text-align:center;color:white;margin:30px 0 0}.nv-loader p{font-size:13px;color:#8c9298;text-align:center;margin:15px 0 0}@keyframes nv-orbit{to{transform:rotate(360deg)}}@keyframes nv-brain{0%,100%{transform:translateY(2px) rotateY(-20deg) rotateZ(-4deg)}50%{transform:translateY(-4px) rotateY(20deg) rotateZ(4deg)}}
.nv-loader{position:fixed;inset:0;z-index:999999;min-height:100dvh;border-radius:0;justify-content:center}
.stApp:has(.nv-loader) [data-stale=true]{display:none}
@media(max-width:1000px){.block-container{padding:2rem 1.5rem}.st-key-nv_navigation button{padding:6px 9px}.st-key-nv_navigation button p{font-size:11px}.st-key-nv_header [data-testid=stHorizontalBlock]{flex-wrap:wrap}.st-key-nv_header [data-testid=stColumn]{min-width:100%}.nv-number{font-size:38px}}
@media(max-width:640px){.block-container{padding:1.3rem 1rem}.st-key-nv_upload_hero{padding:23px}.nv-heading{flex-wrap:wrap}h1{font-size:29px!important}.nv-card{padding:20px}.nv-mri img{height:280px}.nv-report-head,.nv-report-row{grid-template-columns:1fr 1fr}.nv-loader-art{width:260px}.nv-loader-orbit{left:13px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>
"""


def product_header():
    pages = ['Home', 'Analysis', 'Segmentation', 'Explainability', 'Deep Viewer', 'ROI', 'Reports', 'Progression']
    with st.container(key='nv_header'):
        brand, navigation = st.columns([1, 3], vertical_alignment='center')
        with brand:
            nv_html('<div class="nv-brand"><span class="nv-brand-mark">NV</span><span>NeuroVision <em>AI</em></span></div>')
        with navigation:
            # The widget is synchronized before creation so programmatic navigation
            # and button navigation share the same selected tab.
            st.session_state.nv_navigation = st.session_state.nv_page
            st.segmented_control('Navigation', pages, key='nv_navigation', required=True,
                                 label_visibility='collapsed', on_change=nav_changed,
                                 disabled=st.session_state.get('nv_run_analysis', False))


def nav_changed():
    st.session_state.nv_page = st.session_state.nv_navigation


def page_heading(title, subtitle='', status=''):
    nv_html(f'<div class="nv-heading"><div><h1>{escape(title)}</h1>'
            f'{"<p>" + escape(subtitle) + "</p>" if subtitle else ""}</div>'
            f'{"<span class=nv-status>" + escape(status) + "</span>" if status else ""}</div>')


def mri_panel(image, caption='Original MRI', height=360):
    uri = image if isinstance(image, str) else 'data:image/png;base64,' + image_to_base64(image)
    nv_html(f'<div class="nv-mri"><img src="{uri}" alt="{escape(caption, quote=True)}" style="height:{height}px">'
            f'<span>{escape(caption)}</span></div>')


def metric_card(label, value, note=''):
    nv_html(f'<div class="nv-metric"><label>{escape(label)}</label><strong>{escape(str(value))}</strong>'
            f'{"<small>" + escape(note) + "</small>" if note else ""}</div>')


@contextmanager
def brain_loading(label='Analyzing your brain MRI'):
    slot = st.empty()
    slot.html(f'<div class="nv-loader" role="status" aria-live="polite"><div class="nv-loader-art">'
              '<div class="nv-loader-ring"></div><div class="nv-loader-ring small"></div>'
              '<div class="nv-loader-orbit"></div><div class="nv-loader-brain">'
              f'<img src="{BRAIN_LOADER_DATA_URI}" alt=""></div></div>'
              f'<h2>{escape(label)}</h2><p>Please keep this window open.</p></div>')
    try:
        yield
    finally:
        slot.empty()


def quality_for_image(image, validity):
    if validity['is_valid']:
        try:
            return assess_mri_quality(image)
        except Exception as error:
            st.warning(f'Image quality assessment unavailable: {error}')
    return {'overall_status': 'Unknown' if validity['is_valid'] else 'Not evaluated',
            'quality_score': 0, 'blur_score': 0, 'blur_status': 'Unknown',
            'brightness_score': 0, 'brightness_status': 'Unknown',
            'contrast_score': 0, 'contrast_status': 'Unknown',
            'resolution': f'{image.width} x {image.height}', 'resolution_status': 'Unknown'}


UPLOAD_CHECK_CSS = """
<style>
.nv-check-stage{--check-color:#93a4ff;position:relative;height:340px;background:#050b12;border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center;isolation:isolate}
.nv-check-stage img{width:100%;height:100%;object-fit:contain}
.nv-check-stage.pass{--check-color:#4ee0a3}.nv-check-stage.fail,.nv-check-stage.error{--check-color:#ff777f}.nv-check-stage.review{--check-color:#f5bd57}
.nv-check-shade{position:absolute;inset:0;background:#02081255}
.nv-check-corners{position:absolute;inset:20px;color:var(--check-color)}
.nv-check-corners i{position:absolute;width:19px;height:19px;border:solid currentColor;border-width:1.5px 0 0 1.5px}
.nv-check-corners i:nth-child(2){right:0;transform:rotate(90deg)}.nv-check-corners i:nth-child(3){right:0;bottom:0;transform:rotate(180deg)}.nv-check-corners i:nth-child(4){bottom:0;transform:rotate(270deg)}
.nv-check-sweep{position:absolute;left:6%;right:6%;height:1px;top:18%;background:#b6c7ff;box-shadow:0 0 18px 3px #859aff77;animation:nv-check-scan 1.4s ease-in-out infinite alternate}
.nv-check-mark{position:absolute;width:74px;height:74px;display:grid;place-items:center;border-radius:50%;color:var(--check-color);background:#071526eb;border:1px solid currentColor;box-shadow:0 8px 30px #0005}
.nv-check-mark svg{width:35px;height:35px;fill:none;stroke:currentColor;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.nv-check-stage.animate .nv-check-mark{animation:nv-check-pop .5s cubic-bezier(.2,.85,.3,1.2) both}
.nv-check-stage.animate .nv-check-mark path{stroke-dasharray:55;stroke-dashoffset:55;animation:nv-check-draw .5s .15s ease forwards}
.nv-check-stage.animate .nv-check-mark:after{content:'';position:absolute;inset:-8px;border:1px solid currentColor;border-radius:50%;animation:nv-check-ring .8s ease-out both}
.nv-check-caption{position:absolute;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:center;gap:7px;padding:30px 12px 17px;background:linear-gradient(transparent,#020812ee);color:#d5deed;font-size:12px}
.nv-check-caption:before{content:'';width:5px;height:5px;border-radius:50%;background:var(--check-color)}
.nv-check-status{--check-color:#93a4ff;border-left:2px solid var(--check-color);padding-left:13px;margin:4px 0 12px}.nv-check-status.pass{--check-color:#4ee0a3}.nv-check-status.fail,.nv-check-status.error{--check-color:#ff777f}.nv-check-status.review{--check-color:#f5bd57}
.nv-check-status strong{display:block;color:var(--check-color);font-size:17px;font-weight:500}.nv-check-status span{display:block;color:#b0bdd0;font-size:12px;line-height:1.6;margin-top:6px}
@keyframes nv-check-scan{from{top:18%}to{top:83%}}
@keyframes nv-check-pop{from{transform:scale(.65);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes nv-check-draw{to{stroke-dashoffset:0}}
@keyframes nv-check-ring{from{transform:scale(.7);opacity:.6}to{transform:scale(1.3);opacity:0}}
@media(prefers-reduced-motion:reduce){.nv-check-stage *{animation:none!important;transition:none!important}.nv-check-mark path{stroke-dashoffset:0!important}.nv-check-mark:after{display:none}}
</style>
"""


def upload_check_state():
    if st.session_state.get('nv_upload_error'):
        return 'error'
    if st.session_state.get('nv_check_pending'):
        return 'checking'
    validity = st.session_state.get('nv_validity')
    if validity is None:
        return 'idle'
    if validity.get('is_valid'):
        return 'pass'
    return 'review' if validity.get('decision') == 'REVIEW' else 'fail'


def upload_check_visual(image, state, animate=False):
    captions = {'checking': 'Checking image', 'pass': 'MRI screening passed',
                'fail': 'Image not accepted', 'review': 'MRI screening uncertain',
                'error': 'Unable to open image', 'idle': 'MRI preview'}
    paths = {'pass': '<path d="M5 12l4 4L19 6"/>',
             'fail': '<path d="M6 6l12 12M18 6L6 18"/>',
             'error': '<path d="M6 6l12 12M18 6L6 18"/>',
             'review': '<path d="M12 5v9M12 18v.1"/>'}
    picture = '' if image is None else (
        f'<img src="data:image/png;base64,{image_to_base64(image)}" alt="Uploaded image">'
    )
    mark = (f'<div class="nv-check-mark" aria-hidden="true"><svg viewBox="0 0 24 24">{paths[state]}</svg></div>'
            if state in paths else '<div class="nv-check-sweep" aria-hidden="true"></div>')
    nv_html(f'<div class="nv-check-stage {state}{" animate" if animate else ""}">'
            f'{picture}<div class="nv-check-shade"></div>'
            '<div class="nv-check-corners" aria-hidden="true"><i></i><i></i><i></i><i></i></div>'
            f'{mark}<div class="nv-check-caption">{captions[state]}</div></div>')


def upload_check_status(state):
    messages = {
        'checking': ('Checking your image', 'Screening for a brain MRI.'),
        'pass': ('MRI-like image detected', 'Your image is ready for analysis.'),
        'fail': ('Image not accepted', 'MRI screening failed. Choose a brain MRI image.'),
        'review': ('MRI check needs review', 'Unable to confirm an MRI-like image. Analysis is blocked.'),
        'error': ('Unable to open image', 'Choose a readable JPG or PNG image.'),
    }
    if state in messages:
        title, copy = messages[state]
        nv_html(f'<div class="nv-check-status {state}" role="status" aria-live="polite">'
                f'<strong>{title}</strong><span>{copy}</span></div>')


def prepare_upload():
    upload = st.session_state.get('single_mri_upload')
    if upload is None:
        for key in ['nv_upload_hash', 'nv_upload_name', 'nv_upload_image', 'nv_upload_error',
                    'nv_validity', 'nv_quality', 'analysis_hash', 'analysis_result',
                    'analysis_image', 'explainability_result', 'nv_xai_error']:
            st.session_state[key] = None
        st.session_state.nv_check_pending = False
        st.session_state.nv_check_animate = False
        st.session_state.nv_run_analysis = False
        return
    file_bytes = upload.getvalue()
    digest = hashlib.md5(file_bytes).hexdigest()
    st.session_state.nv_upload_name = upload.name
    if digest == st.session_state.get('nv_upload_hash'):
        return
    reset_analysis_for_new_image(digest)
    st.session_state.nv_upload_hash = digest
    st.session_state.nv_check_pending = False
    st.session_state.nv_check_animate = False
    st.session_state.nv_run_analysis = False
    for key in ['nv_upload_image', 'nv_upload_error', 'nv_validity', 'nv_quality', 'nv_xai_error']:
        st.session_state[key] = None
    try:
        st.session_state.nv_upload_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        # Render the checking state before running the existing screening gate.
        st.session_state.nv_check_pending = True
    except Exception as error:
        st.session_state.nv_upload_error = f'Unable to open this image: {error}'
        st.session_state.nv_check_animate = True


def finish_upload_check():
    if not st.session_state.get('nv_check_pending'):
        return
    try:
        image = st.session_state.nv_upload_image
        validity = assess_mri_semantic_validity(image)
        quality = quality_for_image(image, validity)
        st.session_state.nv_validity = validity
        st.session_state.nv_quality = quality
    except Exception as error:
        st.session_state.nv_validity = None
        st.session_state.nv_quality = None
        st.session_state.nv_upload_error = f'MRI screening could not finish: {error}'
    finally:
        st.session_state.nv_check_pending = False
        st.session_state.nv_check_animate = True
    st.rerun()


def clear_uploaded_scan():
    st.session_state.pop('single_mri_upload', None)
    prepare_upload()



def run_uploaded_analysis():
    image = st.session_state.get('nv_upload_image')
    validity = st.session_state.get('nv_validity')
    if image is None or not validity or not validity['is_valid']:
        return
    try:
        xai_error = None
        with brain_loading():
            resources = load_ai_resources()
            prediction = run_prediction_pipeline(image, resources)
            payload = {'prediction': prediction, 'quality_result': st.session_state.nv_quality,
                       'validity_result': validity, 'analysis_time': datetime.now().strftime('%d-%m-%Y %I:%M:%S %p'),
                       'filename': st.session_state.nv_upload_name}
            st.session_state.analysis_result = payload
            st.session_state.analysis_image = image
            st.session_state.explainability_result = None
            if st.session_state.get('nv_auto_xai', True):
                try:
                    st.session_state.explainability_result = run_full_xai(image, payload, resources)
                except Exception as error:
                    xai_error = str(error)
            st.session_state.nv_xai_error = xai_error
        go_to('Analysis')
    except Exception as error:
        st.error(f'MRI analysis failed: {error}')



def home_page():
    st.html(UPLOAD_CHECK_CSS)
    page_heading('Brain MRI Analysis', 'Upload an MRI to classify the scan and inspect model evidence.', 'System ready')
    state = upload_check_state()
    image = st.session_state.get('nv_upload_image')
    animate = bool(st.session_state.pop('nv_check_animate', False))
    with st.container(key='nv_upload_hero'):
        left, right = st.columns([1, 1], gap='large', vertical_alignment='center')
        with left:
            st.subheader('Upload Brain MRI')
            st.caption('JPG / PNG')
            st.file_uploader('Choose MRI', type=['jpg', 'jpeg', 'png'], key='single_mri_upload',
                             label_visibility='collapsed', on_change=prepare_upload)
            if image is not None or state == 'error':
                st.caption(st.session_state.get('nv_upload_name') or 'Uploaded image')
                upload_check_status(state)
                with st.container(horizontal=True):
                    if st.button('Checking image…' if state == 'checking' else 'Analyze MRI',
                                 type='primary', key='nv_analyze', disabled=state != 'pass'):
                        st.session_state.nv_run_analysis = True
                        st.rerun()
                    st.button('Remove image', key='nv_remove_image', on_click=clear_uploaded_scan)
        with right:
            if state == 'idle':
                mri_panel(HOME_MRI_DATA_URI, 'MRI preview', 340)
            else:
                upload_check_visual(image, state, animate)
    validity = st.session_state.get('nv_validity')
    if validity is not None:
        with st.expander('Analysis options' if state == 'pass' else 'Screening details'):
            st.toggle('Generate explainability with analysis', key='nv_auto_xai')
            st.caption('MRI screening: ' + validity['decision'] + '. ' + ', '.join(validity['reasons']))
            st.caption('Image quality: ' + st.session_state.nv_quality['overall_status'])
    if st.session_state.get('nv_upload_error'):
        with st.expander('Upload error details'):
            st.write(st.session_state.nv_upload_error)
    a, b = st.columns(2)
    with a:
        nv_html('<div class="nv-card"><h3>Supported Classes</h3><div class="nv-tags"><span>Glioma</span><span>Meningioma</span><span>Pituitary</span><span>No Tumor</span></div></div>')
    with b:
        nv_html('<div class="nv-card"><h3>Analysis Includes</h3><div class="nv-tags"><span>Classification</span><span>Routing</span><span>Explainability</span><span>ROI inspection</span></div></div>')
    finish_upload_check()


def current_analysis():
    result, image = st.session_state.analysis_result, st.session_state.analysis_image
    if result is None or image is None:
        with st.container(border=True):
            st.subheader('Upload a brain MRI to begin')
            st.caption('Your scan and results will appear here after analysis.')
            if st.button('Choose MRI', type='primary', key='nv_empty_upload'):
                go_to('Home')
        return None, None
    return result, image


def probability_html(probabilities):
    rows = []
    for name, value in zip(CLASS_NAMES, probabilities):
        pct = float(value) * 100
        rows.append(f'<div class="nv-prob-row"><div class="nv-prob-label"><span>{DISPLAY_NAMES[name]}</span><span>{pct:.2f}%</span></div>'
                    f'<div class="nv-prob-track"><div class="nv-prob-fill" style="width:{max(0,min(100,pct)):.2f}%"></div></div></div>')
    return ''.join(rows)


def analysis_page():
    page_heading('Analysis')
    result, image = current_analysis()
    if result is None:
        return
    prediction = result['prediction']
    left, right = st.columns([1.4, 1], gap='medium')
    with left:
        with st.container(border=True, key='nv_scan_card'):
            st.subheader('Uploaded MRI')
            mri_panel(image, result['filename'])
            if st.button('Open viewer', key='nv_open_viewer', icon=':material/arrow_outward:'):
                go_to('Deep Viewer')
    with right:
        # Specialist outputs are binary decisions, not calibrated four-class probabilities.
        confidence = float(prediction['main_probabilities'][CLASS_INDEX[prediction['final_prediction']]])
        label = 'Main-model probability for routed class' if prediction['routing_applied'] else 'Main-model probability'
        nv_html(f'<div class="nv-card"><div class="nv-label">Prediction</div><div class="nv-prediction">{DISPLAY_NAMES[prediction["final_prediction"]]}</div>'
                f'<div class="nv-number">{confidence*100:.2f}%</div><div class="nv-label">{label}</div>'
                + probability_html(prediction['main_probabilities']) + '</div>')
    left, right = st.columns([1.4, 1])
    with left:
        route = ' → '.join(prediction['routing_steps']) if prediction['routing_steps'] else 'Not required'
        nv_html('<div class="nv-card"><h3>Decision Summary</h3><div class="nv-summary">'
                f'<div><label>Main model</label><b>ResNet50V2</b></div><div><label>Routing</label><b>{escape(route)}</b></div>'
                f'<div><label>Image quality</label><b>{escape(result["quality_result"]["overall_status"])}</b></div>'
                f'<div><label>Validity</label><b>{escape(result["validity_result"]["decision"])}</b></div></div></div>')
    with right:
        ready = st.session_state.explainability_result is not None
        evidence = ''.join(f'<div><label>{name}</label><b class="{"ready" if ready else ""}">{"Ready" if ready else "Not generated"}</b></div>' for name in ['Grad-CAM', 'Integrated Gradients', 'Contrastive', 'Perturbation'])
        nv_html('<div class="nv-card"><h3>Evidence Summary</h3><div class="nv-summary">' + evidence + '</div></div>')
        if st.button('Open explainability', key='nv_open_xai'):
            go_to('Explainability')
    if st.session_state.get('nv_xai_error'):
        st.warning('Classification completed. Explainability needs to be retried from its workspace.')
    with st.expander('Quality and routing details'):
        st.dataframe(pd.DataFrame([result['quality_result']]), hide_index=True)
        st.caption('MRI screening: ' + ', '.join(result['validity_result']['reasons']))
        rows = [{'Stage': 'Main ResNet50V2', 'Decision': DISPLAY_NAMES[prediction['main_prediction']], 'Rule': 'Always'},
                {'Stage': 'Glioma / Meningioma', 'Decision': 'Applied' if prediction['gm_route'] else 'Not applied', 'Rule': f'Top-2 G/M; main G; margin ≤ {GM_MARGIN_MAX}; specialist ≥ {GM_CONFIDENCE_MIN}'},
                {'Stage': 'Meningioma vs Rest', 'Decision': 'Applied' if prediction['mvr_route'] else 'Not applied', 'Rule': f'M in top-2; current G/P; M probability ≥ {MVR_PROMOTE_MIN}'}]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Main margin {prediction['main_margin']:.3f} · G/M confidence {prediction['gm_confidence']:.3f} · M-vs-Rest probability {prediction['mvr_meningioma_probability']:.3f}")


def segmentation_page():
    page_heading('Segmentation', 'Inspect the predicted tumor mask on your uploaded MRI.')
    render_segmentation_section(
        image=st.session_state.get('nv_upload_image'),
        validity=st.session_state.get('nv_validity'),
        upload_hash=st.session_state.get('nv_upload_hash'),
        image_panel=mri_panel,
        pending=st.session_state.get('nv_check_pending', False),
    )


def explainability_page():
    page_heading('Explainability')
    result, image = current_analysis()
    if result is None:
        return
    xai = st.session_state.explainability_result
    if xai is None:
        st.caption('Generate explanations for the analyzed MRI.')
        if st.button('Generate explainability', type='primary'):
            try:
                with brain_loading('Building MRI explanations'):
                    st.session_state.explainability_result = run_full_xai(image, result, load_ai_resources())
                    st.session_state.nv_xai_error = None
                st.rerun()
            except Exception as error:
                st.error(f'Explainability could not be generated: {error}')
        return
    method = st.segmented_control('Explanation method', ['Grad-CAM', 'Integrated Gradients', 'Contrastive CAM', 'Perturbation', 'Controls'],
                                  default='Grad-CAM', required=True, key='nv_xai_method', label_visibility='collapsed')
    layer_map = {'Grad-CAM': 'gradcam_overlay', 'Integrated Gradients': 'integrated_gradients_overlay',
                 'Contrastive CAM': 'contrastive_overlay', 'Perturbation': 'salient_region_visual', 'Controls': 'off_anatomy_region_visual'}
    left, right = st.columns([1.65, 1])
    with left:
        with st.container(border=True, key='nv_xai_card'):
            st.subheader(method)
            mri_panel(xai[layer_map[method]], method, 405)
    with right:
        metric_card('Target Probability', f"{xai['original_probability']*100:.2f}%", DISPLAY_NAMES[CLASS_NAMES[xai['predicted_index']]])
        if method == 'Perturbation':
            metric_card('Target logit drop', f"{xai['target_logit_drop']:+.3f}")
            metric_card('Class-margin drop', f"{xai['margin_drop']:+.3f}")
        elif method == 'Controls':
            metric_card('Off-anatomy probability change', f"{xai['off_anatomy_drop_pp']:+.2f} pp")
            metric_card('Off-anatomy logit drop', f"{xai['off_anatomy_logit_drop']:+.3f}")
        else:
            metric_card('Agreement with IG', f"{xai['agreement_matrix']['gradcam_vs_integrated_gradients']:.3f}", 'Grad-CAM / Integrated Gradients')
            metric_card('Agreement with Contrastive', f"{xai['agreement_matrix']['gradcam_vs_contrastive']:.3f}", 'Grad-CAM / Contrastive CAM')
    trust = xai['explanation_trust']
    signals = [('Localization Focus', xai['shortcut_diagnostics']['mean_anatomy_ratio']*100),
               ('Faithfulness', trust['components'].get('deletion_faithfulness', 0)*100),
               ('Perturbation Influence', trust['components'].get('logit_margin_influence', 0)*100),
               ('Explanation Trust', trust['score'])]
    for col, (name, value) in zip(st.columns(4), signals):
        with col:
            metric_card(name, f'{value:.1f}' + ('/100' if name == 'Explanation Trust' else '%'))
    st.caption('Explanations target the main classifier. Attribution maps are not tumor boundaries; trust describes model behavior.')
    if st.toggle('Show detailed diagnostics', key='nv_show_diagnostics'):
        detailed_xai(result, image, xai)


def viewer_page():
    page_heading('Deep Viewer')
    result, image = current_analysis()
    if result is None:
        return
    layers = {'Original MRI': None}
    xai = st.session_state.explainability_result
    if xai is not None:
        layers.update({'Grad-CAM': xai['gradcam_overlay'], 'Integrated Gradients': xai['integrated_gradients_overlay'], 'Contrastive CAM': xai['contrastive_overlay']})
    layer = st.selectbox('Evidence layer', list(layers), key='nv_viewer_layer') if len(layers)>1 else 'Original MRI'
    interactive_image_viewer(image, layer, overlay=layers[layer], enable_zoom_enhance=True)
    st.caption('Enhancement affects the viewing render only.')


def roi_page():
    page_heading('ROI Inspector')
    result, image = current_analysis()
    if result is None:
        return
    layers = {'Original MRI': image}
    xai = st.session_state.explainability_result
    if xai is not None:
        layers.update({'Grad-CAM': xai['gradcam_overlay'], 'Integrated Gradients': xai['integrated_gradients_overlay'], 'Contrastive CAM': xai['contrastive_overlay']})
    reference = st.selectbox('Selection reference', list(layers), key='nv_roi_layer') if len(layers)>1 else 'Original MRI'
    interactive_roi_inspector(layers[reference], image, reference)


def reports_page():
    page_heading('Reports')
    result, image = current_analysis()
    if result is None:
        return
    prediction = result['prediction']
    uri = 'data:image/png;base64,' + image_to_base64(image)
    nv_html('<div class="nv-card"><h3>Analysis records</h3><div class="nv-report-head"><span>MRI</span><span>Prediction</span><span>Main-model confidence</span><span>Date</span></div>'
            f'<div class="nv-report-row"><div class="nv-report-image"><img src="{uri}" alt="MRI thumbnail"><span>{escape(result["filename"])}</span></div>'
            f'<span>{DISPLAY_NAMES[prediction["final_prediction"]]}</span><span>{prediction["main_top_confidence"]*100:.2f}%</span><span>{escape(result["analysis_time"])}</span></div></div>')
    with st.container(horizontal=True):
        if st.button('View analysis', icon=':material/visibility:'):
            go_to('Analysis')
        st.download_button('Export TXT', build_report_text(result), 'NeuroVision_AI_Report.txt', 'text/plain', icon=':material/download:')
        try:
            st.download_button('Export PDF', create_pdf_report(result, image), 'NeuroVision_AI_Report.pdf', 'application/pdf', icon=':material/download:')
        except Exception as error:
            st.warning(f'PDF export unavailable: {error}')
    with st.expander('Report details'):
        st.code(build_report_text(result), language='text')


def save_progress_upload(index):
    uploaded = st.session_state.get(f'progress_file_{index}')
    files = dict(st.session_state.get('nv_progress_files', {}))
    if uploaded is None:
        files.pop(index, None)
    else:
        files[index] = {'name': uploaded.name, 'bytes': uploaded.getvalue()}
    st.session_state.nv_progress_files = files


def remove_progress_scan(index):
    files = dict(st.session_state.get('nv_progress_files', {}))
    files.pop(index, None)
    st.session_state.nv_progress_files = files
    st.session_state.pop(f'progress_file_{index}', None)
    st.session_state.progression_results = None


def progression_page():
    page_heading('Progression', 'Compare dated scans and measured changes.')
    scans = []
    with st.expander('Add or edit scans', expanded=not bool(st.session_state.progression_results)):
        for index in range(1, 6):
            with st.container(border=True):
                cols = st.columns([1.5, 1, 1], vertical_alignment='bottom')
                with cols[0]:
                    st.file_uploader(f'Scan {index}', type=['jpg', 'jpeg', 'png'], key=f'progress_file_{index}',
                                     on_change=save_progress_upload, args=(index,))
                    uploaded = st.session_state.get('nv_progress_files', {}).get(index)
                    if uploaded:
                        st.caption(uploaded['name'])
                        st.button('Remove scan', key=f'nv_remove_scan_{index}', on_click=remove_progress_scan, args=(index,))
                date_key = f'progress_date_{index}'
                if uploaded:
                    signature = hashlib.sha256(uploaded['bytes']).hexdigest()
                    if st.session_state.get(f'progress_sig_{index}') != signature:
                        st.session_state[date_key] = parse_scan_date(uploaded['name']) or date.today()
                        st.session_state[f'progress_sig_{index}'] = signature
                with cols[1]:
                    scan_date = st.date_input(f'Scan {index} date', key=date_key)
                with cols[2]:
                    measured = st.text_input(f'Measured lesion value · {index}', placeholder='Optional', key=f'progress_measurement_{index}')
                measurement = None
                if measured.strip():
                    try:
                        measurement = float(measured)
                        if not np.isfinite(measurement) or measurement < 0:
                            raise ValueError
                    except ValueError:
                        st.warning(f'Scan {index}: enter a non-negative finite measurement.')
                        measurement = None
                if uploaded:
                    try:
                        image = Image.open(io.BytesIO(uploaded['bytes'])).convert('RGB')
                        scans.append({'slot': index, 'filename': uploaded['name'], 'date': scan_date,
                                      'image': image, 'measurement': measurement, 'signature': signature})
                    except Exception as error:
                        st.error(f'Cannot open Scan {index}: {error}')
        unit = st.text_input('Measurement unit', key='nv_measurement_unit')
    signature = tuple((s['signature'], s['date'].isoformat(), s['measurement']) for s in scans)
    if st.session_state.progression_results and signature != st.session_state.get('nv_progress_signature'):
        st.session_state.progression_results = None
    if st.button('Analyze timeline', type='primary', disabled=len(scans)<2):
        try:
            items = []
            with brain_loading('Comparing your brain MRI scans'):
                resources = load_ai_resources()
                progress = st.progress(0, text='Preparing scans')
                for index, scan in enumerate(sorted(scans, key=lambda s:s['date'])):
                    validity = assess_mri_semantic_validity(scan['image'])
                    prediction, extent, heatmap, overlay = None, None, None, None
                    quality = quality_for_image(scan['image'], validity)
                    if validity['is_valid']:
                        prediction = run_prediction_pipeline(scan['image'], resources)
                        extent, heatmap, overlay = quick_attention_extent_proxy(scan['image'], prediction, resources)
                    items.append({**scan, 'validity': validity, 'quality': quality, 'prediction': prediction,
                                  'attention_extent': extent, 'heatmap': heatmap, 'overlay': overlay})
                    progress.progress((index+1)/len(scans), text=f'Scan {index+1} of {len(scans)}')
                progress.empty()
            st.session_state.progression_results = items
            st.session_state.nv_progress_signature = signature
            st.rerun()
        except Exception as error:
            st.error(f'Timeline analysis failed: {error}')
    items = st.session_state.progression_results
    if not items:
        st.caption('Add at least two dated scans to view the comparison.')
        return
    nv_html('<div class="nv-timeline">' + ''.join(f'<span class="nv-date">{item["date"]:%d %b %Y}</span>' for item in items) + '</div>')
    scan_labels = [f'Scan {item["slot"]} · {item["date"]:%d %b %Y}' for item in items]
    for col, default_index, key in zip(st.columns(2), [0, len(items)-1], ['nv_compare_a', 'nv_compare_b']):
        with col:
            selected = st.selectbox('Comparison scan', range(len(items)), index=default_index, format_func=lambda i:scan_labels[i], key=key)
            item = items[selected]
            mri_panel(item['image'], scan_labels[selected], 330)
            if not item['validity']['is_valid']:
                st.warning('Classification blocked: ' + item['validity']['decision'])
    chart_mode = st.segmented_control('Trend', ['Attention extent', 'Measured lesion'], default='Attention extent', required=True)
    column = 'attention_extent' if chart_mode == 'Attention extent' else 'measurement'
    ylabel = 'Attention extent (%)' if column == 'attention_extent' else f'Measured lesion ({unit or "user-entered unit"})'
    rows = [{'Scan': f'Scan {item["slot"]}', 'Date': item['date'], 'Value': item[column]} for item in items if item[column] is not None]
    if len(rows)>=2:
        chart = alt.Chart(pd.DataFrame(rows)).mark_line(point=True, color='#5967e8').encode(
            x=alt.X('Date:T', title='Scan date'), y=alt.Y('Value:Q', title=ylabel, scale=alt.Scale(zero=False)),
            tooltip=['Scan:N', 'Date:T', alt.Tooltip('Value:Q', title=ylabel, format='.2f')]).properties(height=280)
        st.altair_chart(chart, width='stretch')
    else:
        st.caption('At least two valid values are needed for this trend.')
    st.caption('Attention extent is a Grad-CAM research proxy, not tumor size. Measured values must use the same unit.')
    with st.expander('Scan details and overlays'):
        rows = [{'Scan': scan_labels[i], 'Prediction': DISPLAY_NAMES[item['prediction']['final_prediction']] if item['prediction'] else 'Blocked',
                 'MRI gate': item['validity']['decision'], 'Attention extent (%)': item['attention_extent'],
                 f'Measured lesion ({unit})': item['measurement']} for i,item in enumerate(items)]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        index = st.selectbox('Inspect scan', range(len(items)), format_func=lambda i:scan_labels[i])
        item = items[index]
        interactive_image_viewer(item['image'], scan_labels[index], overlay=item['overlay'])


st.html(PRODUCT_CSS)
st.session_state.setdefault('nv_auto_xai', True)
st.session_state.setdefault('nv_measurement_unit', 'mm²')
# Keep analysis options and dated measurements when navigating away from widgets.
for _key in list(st.session_state):
    if _key.startswith(('progress_date_', 'progress_measurement_')) or _key in ('nv_auto_xai', 'nv_measurement_unit'):
        st.session_state[_key] = st.session_state[_key]
sync_segmentation_state(st.session_state.get('nv_upload_hash'),
                        st.session_state.get('nv_validity'),
                        st.session_state.get('nv_check_pending', False))
if st.session_state.nv_page not in ['Home', 'Analysis', 'Segmentation', 'Explainability', 'Deep Viewer', 'ROI', 'Reports', 'Progression']:
    st.session_state.nv_page = 'Home'
product_header()
if st.session_state.pop('nv_run_analysis', False):
    run_uploaded_analysis()
else:
    {'Home': home_page, 'Analysis': analysis_page, 'Segmentation': segmentation_page, 'Explainability': explainability_page,
     'Deep Viewer': viewer_page, 'ROI': roi_page, 'Reports': reports_page,
     'Progression': progression_page}[st.session_state.nv_page]()
