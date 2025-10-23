FROM python:3.8

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
      git \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# JupyterLab
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --upgrade jupyterlab

# Install dependencies
WORKDIR /src/microns-to-nwb
COPY requirements.txt /opt/constraints/requirements.txt
RUN python -m pip install -r /opt/constraints/requirements.txt
