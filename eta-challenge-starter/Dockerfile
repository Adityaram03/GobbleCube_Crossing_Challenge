# Reference Dockerfile for the ETA Challenge.
# Target total image size: ≤ 2.5 GB. This baseline builds to ~2.02 GB
# (xgboost pulls scipy + nvidia-nccl-cu12; trim those if you need headroom).
#
# Build:
#   docker build -t my-eta .
# Test the grader pathway:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required for xgboost at runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Submission surface: predict.py + grade.py + trained weights. We do NOT
# unpickle model.pkl at build time — that runs untrusted candidate code on
# the grader host before any sandbox applies. The first docker-run invocation
# is the smoke test; it runs inside the sandboxed grader container.
COPY predict.py grade.py ./
COPY model.pkl ./

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
