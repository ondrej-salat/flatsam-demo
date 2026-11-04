#!/bin/bash
# One-time FlatSAM2 setup, safe to re-run:  ./setup.sh && source .envrc
# Needs: Linux x86_64, NVIDIA GPU + driver, git, curl or wget, internet, and a Hugging Face
# account with access to https://huggingface.co/facebook/sam3 (gated).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example (defaults work; edit it to change paths)"
fi
set -a; source .env; set +a

# --- uv: brings its own Python 3.12, so the system python/venv/pip do not matter ---------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    { curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null || wget -qO- https://astral.sh/uv/install.sh; } \
        | env UV_NO_MODIFY_PATH=1 sh
fi

# --- venv + packages --------------------------------------------------------------------
# requirements-lock.txt = complete `pip freeze` of a known-working venv, installed 1:1 with
# --no-deps: the forks' declared versions contradict each other (LoMa: torchvision>=0.23,
# ptlflow: lightning<2.6, ...) so no resolver can rebuild it.
[ -x venv/bin/python ] || uv venv --seed --python 3.12 venv
if ! venv/bin/python -c "import flatsam2, sam3, ptlflow, loma" 2>/dev/null; then
    uv pip install --python venv/bin/python --no-deps -r requirements-lock.txt
    uv pip install --python venv/bin/python --no-deps -e .
fi
venv/bin/python -c "import flatsam2, sam3, ptlflow, loma" || { echo "ERROR: imports fail (see above)"; exit 1; }
echo "Packages OK"
# warn only: a cluster login node has no GPU, its compute nodes do
venv/bin/python -c "import torch; print('GPU', torch.cuda.get_device_name(0))" 2>/dev/null \
    || echo "WARNING: torch sees no CUDA GPU here; the demo needs one (NVIDIA driver installed?)"

# --- .envrc: what every shell sources (from the repo root) ------------------------------
if [ ! -f .envrc ]; then
    printf '%s\n' 'source venv/bin/activate' 'set -a; [ -f .env ] && source .env; set +a' > .envrc
    echo "Wrote .envrc"
fi

# --- weights: SAM3 (gated, Hugging Face) + LoMa (GitHub release) ------------------------
TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "$TOKEN_FILE" ]; then
    read -r -s -p "Hugging Face token (read access to facebook/sam3): " T \
        || { echo "ERROR: no HF token: set HF_TOKEN in .env or run ./setup.sh interactively"; exit 1; }
    echo
    mkdir -p "$(dirname "$TOKEN_FILE")"
    (umask 077; printf '%s' "$T" > "$TOKEN_FILE")
    unset T
fi
venv/bin/python -c "from sam3.model_builder import download_ckpt_from_hf as d; print('SAM3', d(version='sam3'))"
# CPU on purpose: only fetches the weights, so a busy/absent GPU does not matter
CUDA_VISIBLE_DEVICES= venv/bin/python -c "from loma import LoMa, LoMaR; LoMa(LoMaR()); print('LoMa weights ok')"

# --- smoke test: config loads (CPU, seconds) --------------------------------------------
venv/bin/python demo.py params > /dev/null || { echo "ERROR: config smoke check failed (see above)"; exit 1; }
echo "Config OK: $DEMO_CONFIG"

echo
echo "Setup done. In every new shell load the environment first, then run the demo:"
echo "  source .envrc && python demo.py --help"
