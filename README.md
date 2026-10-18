# NeuroVision AI

Research application for brain MRI classification, explainability and tumor segmentation.

**Research and educational use only. Not for clinical diagnosis or treatment decisions. Do not upload patient/private scans.**

## Application

- Four-class ResNet50V2 classification at 300 × 300: glioma, meningioma, no tumor and pituitary.
- Glioma/meningioma and meningioma/rest specialists.
- Grad-CAM, Integrated Gradients and perturbation diagnostics.
- U-Net segmentation at 256 × 256.
- Image viewer, PDF reports and progression workspace.

Production entrypoint: `Streamlit/rwcdx.py`. Runtime source is preserved byte-for-byte from the existing application. This source-only repository excludes training datasets, generated reports, credentials, model binaries and local environments. It contains the production application rather than the entire experimental archive.

## Run locally

Use Python 3.12 with the listed requirements. The Linux dependency constraints were resolver-checked; the Linux Docker image has not yet been built/tested. macOS should use the already working project environment when available.

```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
# Restore the four original checkpoints at the paths in MODELS.md first.
python preflight.py
python -m streamlit run Streamlit/rwcdx.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

The app cannot perform inference without its checkpoints. `preflight.py` checks source and checkpoint hashes against `manifest.json`. Do not substitute checkpoints or change preprocessing/thresholds to work around a hosting limit.

## Deployment

The Dockerfile expects the four original checkpoint files to be restored locally before building. It intentionally fails when required model directories/files are missing. It does not fetch weights from any public service.

```sh
docker build --platform linux/amd64 -t neurovision-demo:review .
docker run --rm -p 127.0.0.1:7860:7860 --memory=16g --cpus=2 neurovision-demo:review
```

Docker startup verifies the manifest, then starts Streamlit on `PORT` (default 7860). The HTTP health endpoint is `/_stcore/health`; it does not prove model readiness. Do not publish a model-containing container image without permission.

A local CPU smoke test passed classification, full XAI, U-Net, PDF generation and eight empty application pages. Full XAI on a synthetic input took approximately 40 seconds; combined process peak RAM was 6.57 GB. These are local packaging checks, not clinical validation or hosted performance guarantees. Start hosting evaluation at 16 GB RAM.

## Sharing and rights

No open-source license has been assigned. Rights to redistribute trained weights, upstream components and the MRI illustration embedded in the app must be confirmed before public release. Keep the repository private until that review is complete. The embedded image remains in the original source; excluding dataset folders does not remove it.

Uploaded images/results use application session memory and report buffers in the inspected runtime code. This is not a guarantee of immediate deletion or provider-side privacy. Use approved de-identified or synthetic demo inputs only.
