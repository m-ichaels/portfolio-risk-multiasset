#!/usr/bin/env bash
# Full pipeline: (data ->) checks -> the day pipeline -> tests -> figures -> summary -> report
# Usage: scripts/run_all.sh [--download] [--build-book] [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
FROM=2026-07-01; TO=2026-09-17
[[ " $* " == *" --quick "* ]] && { FROM=2026-08-25; TO=2026-08-28; }
[[ " $* " == *" --download "* ]] && python tools/download.py
[[ " $* " == *" --build-book "* ]] && python tools/build_book.py           # needs the sibling execution-ops checkout (../ProjectE)
python -m xrisk futures-check --from 2026-06-01 | tee results/futures_check.log
python -m xrisk fx-check > results/fx_check.log
python -u -m xrisk run --from $FROM --to $TO --faults alternate --seed 1 | tee results/run.log
python -m pytest -q | tee results/tests.txt
python scripts/plots.py
python scripts/summarize.py
python scripts/report.py
python scripts/fill_readme.py
echo done
