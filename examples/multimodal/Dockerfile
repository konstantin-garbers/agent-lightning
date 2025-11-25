# 1. Base Image: CUDA 12.4 is a safe, modern default compatible with vLLM 0.10.x
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY

# Fix typos in braces and set non-interactive frontend
ENV HTTP_PROXY=$HTTP_PROXY
ENV http_proxy=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV no_proxy=$NO_PROXY
ENV ALL_PROXY=$ALL_PROXY
ENV HF_HUB_DISABLE_XET=1
ENV WANDB_MODE=disabled
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 2. Install System Dependencies
# Removed custom python3.11 to use the system default (3.10) for stability
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    xvfb \
    docker.io \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Clone Repo
RUN git clone -b multimodal-example https://github.com/konstantin-garbers/agent-lightning
WORKDIR /app/agent-lightning
COPY .env .env

# 4. Install Build Dependencies
RUN pip install --upgrade pip
RUN pip install packaging ninja setuptools wheel psutil "numpy<2"

# 5. Install vLLM FIRST (Critical Change)
# We let vLLM 0.10.2 install the exact PyTorch version it needs.
# We do NOT manually install torch beforehand.
RUN pip install vllm==0.10.2

# 6. Install Flash Attention SECOND
# Now that the correct Torch is installed, we compile Flash Attention against it.
# --no-build-isolation: Uses the build tools + Torch we just installed.
RUN pip install flash-attn --no-build-isolation

# 7. Install Project Requirements
# We remove 'vllm' from requirements.txt to prevent it from messing up our installed version
RUN sed -i '/vllm/d' examples/multimodal/requirements.txt && \
    pip install -r examples/multimodal/requirements.txt .[agents]

# 8. Data Downloads
WORKDIR /app/agent-lightning/examples/multimodal/data
RUN pip install gdown
RUN gdown --id '18T5Meu-bABrKz9zhTT_yRJr2oaaTrskh'
RUN gdown --id '1mZ2NhoIq0fKLZzMrNQhmIfQgXweEZmFm'

# 9. Install Verl
WORKDIR /app
RUN pip install verl==0.5.0