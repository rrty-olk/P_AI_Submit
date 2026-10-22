#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/logs
test -f .env || python3 scripts/fetch_official.py
cleanup() {
    result=$?
    trap - EXIT
    docker compose logs --no-color ai > output/logs/ai_console.txt 2>&1 || true
    docker compose down > output/logs/shutdown.txt 2>&1 || true
    exit "$result"
}
trap cleanup EXIT
docker compose build > output/logs/build.txt 2>&1
docker compose run --rm --no-deps tools python scripts/generate_cases.py
docker compose run --rm --no-deps ai python -m pytest tests/test_core.py -q --junitxml=output/logs/unit_tests.xml | tee output/logs/unit_tests.txt
docker compose up -d --force-recreate ai
# Compose waits for the genuine model health check, including initial download/load.
docker compose run --rm robot bash scripts/safety.sh | tee output/logs/safety_console.txt
docker compose up --abort-on-container-exit --exit-code-from robot robot | tee output/logs/run_console.txt
docker compose run --rm --no-deps robot python -m inspection.audit
docker compose run --rm --no-deps robot python -m inspection.render
docker compose run --rm --no-deps robot python -m inspection.report
docker compose run --rm --no-deps robot pip freeze --local > requirements.lock.txt
docker image inspect physicalai-inspection:local --format '{{json .}}' > output/logs/image_provenance.json
docker compose run --rm --no-deps tools python scripts/package.py
docker compose run --rm --no-deps robot python scripts/check_deliverables.py
