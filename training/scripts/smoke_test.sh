#!/usr/bin/env bash
set -euo pipefail
uv run --extra test pytest -q
uv run weatherdiff profile --width 32 --size 256
