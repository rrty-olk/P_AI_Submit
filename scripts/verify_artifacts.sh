#!/usr/bin/env bash
set -euo pipefail
cd /app
python -m pytest tests/test_core.py -q --junitxml=output/logs/unit_tests.xml | tee output/logs/unit_tests.txt
python -m inspection.audit
python -m inspection.render
python -m inspection.report
mkdir -p output/qa
pdfinfo output/submission/report.pdf > output/qa/pdfinfo.txt
pdftoppm -scale-to 1400 -png output/submission/report.pdf output/qa/report > /dev/null 2>&1
for sec in 2 12 28 45 66 82; do
  ffmpeg -y -loglevel error -ss "$sec" -i output/submission/inspection_demo.mp4 -frames:v 1 "output/qa/video_${sec}.png"
done
python scripts/package.py
python scripts/check_deliverables.py
