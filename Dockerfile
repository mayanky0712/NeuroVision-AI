FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false \
    TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt constraints-linux.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints-linux.txt
COPY --chown=app:app Streamlit/ Streamlit/
COPY --chown=app:app src/ src/
COPY --chown=app:app segmentation/ segmentation/
COPY --chown=app:app Experiments/ Experiments/
COPY --chown=app:app manifest.json preflight.py start.py healthcheck.py ./
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 CMD ["python", "healthcheck.py"]
CMD ["python", "start.py"]
