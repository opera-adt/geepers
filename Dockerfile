# Build stage: build the wheel with setuptools_scm (needs git metadata)
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN python -m pip install --no-cache-dir build \
    && python -m build --wheel --outdir /wheels

# Runtime stage
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/opera-adt/geepers" \
      org.opencontainers.image.description="Download GPS data and compare to InSAR" \
      org.opencontainers.image.licenses="Apache-2.0"

# rasterio/pyogrio/shapely ship manylinux wheels, so no system GDAL is needed
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl matplotlib \
    && rm -rf /wheels

# Run as an unprivileged user; HOME must be writable for the GPS cache dir
RUN useradd --create-home --shell /bin/bash geepers
USER geepers
ENV HOME=/home/geepers
WORKDIR /work

ENTRYPOINT ["geepers"]
CMD ["--help"]
