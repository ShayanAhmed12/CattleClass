#!/usr/bin/env bash
# Linux / macOS / WSL2 setup, run from the project folder:
#
#   bash setup.sh                # install and test
#   bash setup.sh --gpu          # use TensorFlow with CUDA (Linux/WSL2 with an NVIDIA GPU)
#   bash setup.sh --data         # ...and download the datasets (~1 GB); flags can be combined
#
# Creates .venv with Python 3.12 or 3.13 (TensorFlow has no wheels for newer versions),
# installs requirements.txt and runs the tests. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

gpu=0; data=0
for arg in "$@"; do
  case "$arg" in
    --gpu) gpu=1 ;;
    --data) data=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

# 1. Find Python 3.12 or 3.13.
PY=""
for c in python3.12 python3.13; do
  if command -v "$c" >/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 or 3.13 not found. On Ubuntu/WSL:  sudo apt install python3.12 python3.12-venv"
  exit 1
fi

# 2. Virtual environment and packages.
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt
if [ "$gpu" = 1 ]; then .venv/bin/python -m pip install "tensorflow[and-cuda]"; fi

# 3. Tests, then (optionally) the datasets.
.venv/bin/python -m pytest -q
if [ "$data" = 1 ]; then .venv/bin/python prepare_data.py; fi
.venv/bin/python -c "import tensorflow as tf; print('TensorFlow', tf.__version__, '| GPUs:', len(tf.config.list_physical_devices('GPU')))"

echo
echo "Done. In a new terminal, activate the environment with:  source .venv/bin/activate"
