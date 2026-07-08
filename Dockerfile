# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EBAY_PRICE_HOME=/app
# libgomp: LightGBM/XGBoost OpenMP runtime, absent from python:slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml LICENSE README.md ./
COPY src ./src

# ---- api: CPU-only inference serving (default target) ----
# requires trained pipelines in artifacts/ — run `python -m ebay_price.train`
# (or the train service below) before building
FROM base AS api
RUN pip install .
COPY artifacts ./artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "ebay_price.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- train: scrape/clean/train stack; mount data/, artifacts/, reports/ ----
# XGBoost & CatBoost use the GPU when the container gets one (`gpus: all`);
# torch (the MLP candidate) is left out to keep the image small
FROM base AS train
RUN pip install .[train,scrape]
CMD ["python", "-m", "ebay_price.train"]
