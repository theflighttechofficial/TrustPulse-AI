# TrustPulse AI live engine and console, as an s390x container on LinuxONE.
# Uses Ubuntu's native s390x NumPy and Flask, so nothing is compiled.
#
# Build and run on the LinuxONE VM (podman or docker):
#   podman build -t trustpulse -f Containerfile .
#   podman run --rm -p 127.0.0.1:8050:8050 \
#       -v ~/trustpulse/artifacts/engine:/app/artifacts/engine:ro trustpulse
#
# The engine state (artifacts/engine) is produced by Part 3 of the notebook and
# is mounted read-only, so the image itself holds no account data.

FROM docker.io/library/ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy python3-flask \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY trustpulse/ trustpulse/
COPY app/ app/
COPY models/ models/

RUN useradd --system --no-create-home trustpulse
USER trustpulse

EXPOSE 8050
CMD ["python3", "app/server.py", "--host", "0.0.0.0", "--port", "8050"]
