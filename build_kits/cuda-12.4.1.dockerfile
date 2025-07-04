FROM ghcr.io/make87/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-image

ARG VIRTUAL_ENV=/make87/venv

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        build-essential \
        python3-pip \
        python3 \
        libpython3-dev \
        python3-venv \
    && python3 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
    && ${VIRTUAL_ENV}/bin/pip install uv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN set -eux; \
    ${VIRTUAL_ENV}/bin/uv pip install ".[gpu]";

FROM ghcr.io/make87/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
        python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG VIRTUAL_ENV=/make87/venv
ENV ONNX_PROVIDER=CUDAExecutionProvider
COPY --from=base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]
