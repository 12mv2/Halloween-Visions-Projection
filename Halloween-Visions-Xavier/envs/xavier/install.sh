#!/bin/bash
# NVIDIA Jetson Xavier NX - Setup Script
# Halloween Hand Detection Projection System
# JetPack 5.0.2 (L4T R35.1.0)

set -e  # Exit on error

echo "============================================"
echo "Xavier NX Setup - Halloween Hand Detection"
echo "============================================"
echo ""

# Verify JetPack version
echo "[1/8] Verifying JetPack version..."
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
    if grep -q "R35" /etc/nv_tegra_release; then
        echo "✓ JetPack 5.0.2 detected"
    else
        echo "⚠ Warning: Expected JetPack 5.0.2 (R35)"
    fi
else
    echo "⚠ Warning: Cannot verify JetPack version"
fi
echo ""

# Install system dependencies
echo "[2/8] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
echo "✓ System dependencies installed"
echo ""

# Create virtual environment
echo "[3/8] Creating virtual environment..."
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
if [ -d "venv" ]; then
    echo "⚠ venv already exists, skipping creation"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate venv
source venv/bin/activate
echo "[4/8] Virtual environment activated"
echo ""

# Upgrade pip
echo "[5/8] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch wheel if available
echo "[6/8] Installing PyTorch..."
TORCH_WHEEL="envs/xavier/wheels/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl"
if [ -f "$TORCH_WHEEL" ]; then
    echo "Installing from local wheel..."
    pip install "$TORCH_WHEEL"
else
    echo "⚠ PyTorch wheel not found in envs/xavier/wheels/"
    echo "Please download from:"
    echo "https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/"
    echo ""
    read -p "Press Enter to continue without PyTorch or Ctrl+C to abort..."
fi
echo ""

# Install Ultralytics and dependencies
echo "[7/8] Installing Ultralytics..."
pip install ultralytics==8.0.196
echo "✓ Ultralytics installed"
echo ""

# Verify installation
echo "[8/8] Verifying installation..."
echo ""
echo "Python version:"
python3 --version
echo ""
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch not installed"
echo ""
echo "CUDA available:"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠ Cannot verify CUDA"
echo ""
echo "Ultralytics version:"
python3 -c "from ultralytics import YOLO; import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python3 simple_projection.py --model Colin1.pt"
echo ""
echo "For SSH access, see: envs/xavier/SSH_SETUP.md"
echo "For full guide, see: envs/xavier/README.md"
echo ""
