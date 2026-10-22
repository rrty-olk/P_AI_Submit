#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/logs
exec > >(tee output/logs/environment.txt) 2>&1
date -u --iso-8601=seconds
uname -a
cat /etc/os-release
df -h / .
free -h
nvidia-smi
docker version
docker compose version
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.0.0-runtime-ubuntu20.04 nvidia-smi
