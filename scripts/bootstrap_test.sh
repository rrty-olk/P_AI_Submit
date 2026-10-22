#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build > output/logs/build_final.txt 2>&1
docker compose run --rm --no-deps ai python -m pytest tests/test_core.py -q --junitxml=output/logs/unit_tests.xml | tee output/logs/unit_tests.txt
docker compose run --rm --no-deps ai bash -lc 'cat /etc/os-release; python -c "import torch,transformers,cv2; print(torch.__version__,transformers.__version__,cv2.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(4,device=\"cuda\").sum().item())"' | tee output/logs/container_probe.txt
docker compose up -d ai
