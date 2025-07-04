FROM ghcr.io/make87/debian:bookworm AS base-image

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
    if [ -f ./pip.conf ]; then \
      echo "Found pip.conf, pointing PIP_CONFIG_FILE at it"; \
      export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    else \
      echo "No pip.conf found, using default indexes"; \
    fi; \
    ${VIRTUAL_ENV}/bin/uv pip install ".[openvino]";

FROM ghcr.io/make87/python3-debian12:latest

ARG VIRTUAL_ENV=/make87/venv
ENV ONNX_PROVIDER=OpenVINOExecutionProvider
COPY --from=base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ENTRYPOINT ["/make87/venv/bin/python3", "-m", "app.main"]
