#!/usr/bin/env bash
# =============================================================================
# setup.sh — Bootstrap script for Hyperliquid × Fear & Greed Analysis Repo
# Usage:  bash setup.sh
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================================"
echo " Hyperliquid × Bitcoin Sentiment Analysis — Environment Setup"
echo "============================================================"

# 1. Python check
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install it first: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Found Python $PYTHON_VERSION"

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[INFO] Virtual environment already exists — skipping creation."
fi

# 3. Activate venv
source "$VENV_DIR/bin/activate"
echo "[INFO] Virtual environment activated."

# 4. Upgrade pip silently
pip install --upgrade pip --quiet

# 5. Install dependencies
echo "[INFO] Installing dependencies from requirements.txt ..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "[INFO] All dependencies installed successfully."

# 6. Ensure output directories exist
mkdir -p "$PROJECT_DIR/outputs/figures"
mkdir -p "$PROJECT_DIR/outputs/reports"

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Activate the env:  source .venv/bin/activate"
echo " Run full analysis: python main.py"
echo "============================================================"
