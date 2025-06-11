FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    tzdata \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    notebook \
    tqdm \
    ipywidgets \
    xgboost \
    lightgbm \
    catboost \
    optuna \
    joblib \
    requests

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]