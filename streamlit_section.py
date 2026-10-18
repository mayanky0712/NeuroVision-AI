"""Streamlit presentation and isolated per-upload state for research segmentation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from PIL import Image
import streamlit as st

from segmentation.inference import (
    DEFAULT_CHECKPOINT, checkpoint_signature, segment_mri, segmentation_overlay,
)


STATE_KEY = "nv_segmentation_result"


def sync_segmentation_state(upload_hash, validity, pending=False) -> None:
    """Invalidate only segmentation state when the current upload changes/is blocked."""
    cached = st.session_state.get(STATE_KEY)
    if cached is not None and (
        not upload_hash
        or cached["upload_hash"] != upload_hash
        or pending
        or not validity
        or not validity.get("is_valid", False)
    ):
        st.session_state.pop(STATE_KEY, None)


def render_segmentation_section(
    image: Image.Image | None,
    validity: dict | None,
    upload_hash: str | None,
    image_panel: Callable,
    pending: bool = False,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> None:
    st.caption("Automated research segmentation — not a clinical contour.")
    sync_segmentation_state(upload_hash, validity, pending)
    if image is None or not upload_hash:
        st.info("Upload a brain MRI on Home and wait for MRI screening to finish.")
        return
    if pending:
        st.info("MRI screening is still in progress. Return to Home to complete it.")
        return
    if not validity or not validity.get("is_valid", False):
        st.warning("Segmentation requires an image that passes the existing MRI screening gate.")
        return

    try:
        signature = checkpoint_signature(checkpoint_path)
    except (OSError, ValueError) as error:
        st.session_state.pop(STATE_KEY, None)
        st.warning("Segmentation is unavailable. The U-Net checkpoint could not be accessed.")
        with st.expander("Segmentation error details"):
            st.text(str(error))
        return

    # Tie results to both decoded image content/dimensions and the exact checkpoint.
    rgb = image.convert("RGB")
    source_key = (rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest(), signature, "native-mask-area")
    cached = st.session_state.get(STATE_KEY)
    if cached is not None and cached["source_key"] != source_key:
        st.session_state.pop(STATE_KEY, None)
        cached = None

    with st.container(border=True):
        st.subheader("Tumor segmentation")
        st.caption("Binary U-Net · 256 × 256 RGB input · mask threshold 0.5")
        run = st.button(
            "Run segmentation" if cached is None else "Run segmentation again",
            key="nv_run_segmentation", type="primary", icon=":material/layers:",
        )
        if run:
            st.session_state.pop(STATE_KEY, None)
            cached = None
            try:
                with st.spinner("Generating the tumor mask…"):
                    result = segment_mri(rgb, checkpoint_path)
                cached = {"upload_hash": upload_hash, "source_key": source_key, "result": result}
                st.session_state[STATE_KEY] = cached
            except Exception as error:
                st.warning("Segmentation could not be completed. You can retry after checking the checkpoint.")
                with st.expander("Segmentation error details"):
                    st.text(str(error))

    if cached is None:
        image_panel(rgb, "Original MRI", 330)
        return

    result = cached["result"]
    original_col, mask_col, overlay_col = st.columns(3, gap="medium")
    with original_col:
        image_panel(rgb, "Original MRI", 330)
    with mask_col:
        image_panel(Image.fromarray(result.mask), "Predicted Tumor Mask", 330)
    with overlay_col:
        image_panel(segmentation_overlay(rgb, result.mask), "Segmentation Overlay", 330)

    area_label = "<0.01%" if 0 < result.relative_area < 0.01 else f"{result.relative_area:.2f}%"
    with st.container(border=True):
        st.metric("Relative Mask Area", area_label)
        st.caption(
            f"{result.positive_pixels:,} predicted positive pixels / {result.total_pixels:,} total mask pixels × 100. "
            "Calculated from the native 256 × 256 thresholded segmentation mask."
        )
    if result.positive_pixels == 0:
        st.info("No region detected at the 0.5 mask threshold. This does not establish the absence of a tumor.")
