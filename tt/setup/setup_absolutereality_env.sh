#!/usr/bin/env bash
set -euo pipefail

# ===== CONFIG =====
ENV_NAME="absreality_env"
PYTHON_VERSION="python3"   # adjust if you prefer python3.11 explicitly
MODEL_ID="digiplay/AbsoluteReality_v1.0_diffusers"
PROJECT_DIR="$HOME/absreality_pipeline"

echo ">>> Updating apt and installing base dependencies..."
sudo apt-get update
sudo apt-get install -y \
    $PYTHON_VERSION \
    ${PYTHON_VERSION}-venv \
    ${PYTHON_VERSION}-distutils \
    git \
    wget \
    curl \
    build-essential \
    git-lfs

git lfs install

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo ">>> Creating Python virtual environment: $ENV_NAME"
$PYTHON_VERSION -m venv "$ENV_NAME"

# shellcheck disable=SC1090
source "$ENV_NAME/bin/activate"

echo ">>> Upgrading pip and installing core Python packages..."
pip install --upgrade pip

# NOTE:
# - This installs the default PyTorch build from PyPI (CPU-only is fine, GPU builds can be swapped in later)
# - If you're on a GPU VM, you can later replace this line with the official CUDA wheel install.
pip install "torch" --index-url https://download.pytorch.org/whl/cpu

# Core libs for Stable Diffusion via diffusers
pip install \
    diffusers[torch] \
    transformers \
    accelerate \
    safetensors \
    pillow \
    tqdm \
    huggingface_hub

echo ">>> (Optional) Login to Hugging Face if the model requires auth..."
echo "    If needed, run:  huggingface-cli login"
echo "    and paste your HF token. Skipping for now."

echo ">>> Pre-downloading AbsoluteReality model: $MODEL_ID"
python - << 'PYCODE'
from huggingface_hub import snapshot_download

model_id = "digiplay/AbsoluteReality_v1.0_diffusers"
print(f"Downloading model: {model_id}")
local_dir = snapshot_download(
    repo_id=model_id,
    local_dir="models/AbsoluteReality",
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {local_dir}")
PYCODE

echo ">>> Environment setup complete."
echo "To activate later, run:"
echo "    cd \"$PROJECT_DIR\""
echo "    source \"$ENV_NAME/bin/activate\""
echo
echo "Your AbsoluteReality model is in:"
echo "    $PROJECT_DIR/models/AbsoluteReality"
echo
echo "Next step: write your Python script using diffusers' StableDiffusionPipeline"
echo "pointing at that directory as model_id or pretrained_model_name_or_path."
