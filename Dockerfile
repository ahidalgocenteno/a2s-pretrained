FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ffmpeg \
        libsm6 \
        vim \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# MUSTER
RUN wget https://github.com/TimFelixBeyer/amtevaluation.github.io/raw/refs/heads/main/MUSTER_v231108.zip \
        -O /tmp/muster.zip \
    && unzip /tmp/muster.zip -d /tmp/MUSTER_tmp \
    && mv "/tmp/MUSTER_tmp/MUSTER 2" /MUSTER \
    && rm -rf /tmp/muster.zip /tmp/MUSTER_tmp \
    && cd /MUSTER \
    && ./compile.sh

# OTHER DEPENDENCIES
RUN pip install --upgrade pip \
    && pip install pybind11

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt