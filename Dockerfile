FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DRAMABOX_REPO_DIR=/opt/dramabox/DramaBox
ENV PYTHONPATH=/opt/dramabox/DramaBox:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN mkdir -p /opt/dramabox \
    && git clone https://github.com/resemble-ai/DramaBox.git /opt/dramabox/DramaBox

WORKDIR /opt/dramabox/DramaBox

RUN python -m pip install -r requirements.txt

WORKDIR /app

COPY handler.py dramabox_runtime.py ./

RUN python -m pip install runpod

CMD ["python", "-u", "/app/handler.py"]
