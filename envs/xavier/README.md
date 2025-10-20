# NVIDIA Jetson Xavier NX Deployment Guide

**Halloween Hand Detection Projection System**
**Target Platform:** NVIDIA Jetson Xavier NX Developer Kit
**JetPack Version:** 5.0.2 (L4T R35.1.0)
**Last Updated:** 2025-10-20

---

## Quick Start

```bash
# Clone repository
git clone <repository-url> Halloween-Visions-Projection
cd Halloween-Visions-Projection

# Run Xavier setup script
bash envs/xavier/install.sh

# Run application
python3 simple_projection.py --model Colin1.pt
```

---

## System Requirements

### Hardware
- NVIDIA Jetson Xavier NX Developer Kit
- 8GB RAM minimum
- USB camera (UVC compatible)
- HDMI display/projector
- Network connection (for initial setup)

### Software
- JetPack 5.0.2 (L4T R35.1.0)
- Ubuntu 20.04 (included with JetPack)
- Python 3.8.10

---

## Verification Steps

### 1. Verify JetPack Version

```bash
cat /etc/nv_tegra_release
# Expected output:
# R35 (release), REVISION: 1.0, GCID: 31346300, BOARD: t186ref, EABI: aarch64, DATE: Thu Aug 25 18:41:45 UTC 2022

uname -r
# Expected: 5.10.104-tegra
```

### 2. Verify CUDA Installation

```bash
nvcc --version
# Expected: CUDA 11.4

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True

python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
# Expected: CUDA version: 11.4
```

### 3. Verify GPU

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: Xavier
```

---

## Installation Guide

### Method 1: Automated Setup (Recommended)

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
bash envs/xavier/install.sh
```

This script will:
1. Verify JetPack version
2. Install system dependencies
3. Create Python virtual environment
4. Install PyTorch wheel (if available)
5. Install Ultralytics and dependencies
6. Verify GPU inference
7. Test application

### Method 2: Manual Setup

#### Step 1: System Dependencies

```bash
# Update system
sudo apt-get update

# Install OpenCV with CUDA support (recommended)
sudo apt-get install -y python3-opencv

# Install pip
sudo apt-get install -y python3-pip python3-venv

# Install additional dependencies
sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
```

#### Step 2: Create Virtual Environment

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### Step 3: Install PyTorch

**Option A: Use NVIDIA Pre-built Wheel (Recommended)**

```bash
# Download PyTorch wheel for JetPack 5.0.2
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# Install wheel
pip install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: PyTorch: 1.12.0a0+2c916ef.nv22.3
```

**Option B: Use Existing Wheel (if available)**

```bash
# If torch wheel is in envs/xavier/wheels/
pip install envs/xavier/wheels/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
```

#### Step 4: Install Ultralytics

```bash
pip install ultralytics==8.0.196
```

#### Step 5: Verify Installation

```bash
# Test GPU inference
python3 test_gpu_validation.py

# Expected output:
# CUDA available: True
# GPU enabled: Xavier
# Model loaded successfully
# First inference device verified: cuda:0
```

---

## SSH Access Setup

For remote development and deployment, see:
- **Guide:** `~/docs/setup/B10_Xavier_SSH_Setup_Guide.md`
- **VS Code Remote-SSH:** Recommended for development

**Quick SSH Setup:**

```bash
# On Xavier
hostname -I  # Get Xavier IP address

# On development machine
ssh colin@<xavier-ip>
# Default password: (your Xavier password)

# Setup SSH keys (optional but recommended)
ssh-copy-id colin@<xavier-ip>
```

---

## Systemd Service Setup

To run the application automatically on boot:

```bash
# Copy service file
sudo cp halloween.service /etc/systemd/system/

# Enable service
sudo systemctl enable halloween.service

# Start service
sudo systemctl start halloween.service

# Check status
sudo systemctl status halloween.service

# View logs
journalctl -u halloween.service -f
```

---

## Running the Application

### Basic Usage

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
source venv/bin/activate
python3 simple_projection.py
```

### With Options

```bash
# Custom model and camera
python3 simple_projection.py --model quinn_arms_up.pt --source 0

# Start in fullscreen mode
python3 simple_projection.py --fullscreen

# Adjust confidence threshold
python3 simple_projection.py --conf 0.8

# Custom videos
python3 simple_projection.py --video-sleep videos/sleeping_face.mp4 --video-scare videos/angry_face.mp4
```

### Controls (During Runtime)

- **D** - Toggle Debug/Projection mode
- **P** - Toggle Production mode (grey border fix)
- **F** - Toggle fullscreen
- **Q** or **ESC** - Quit

---

## Performance Benchmarks

### Current Production Performance (vB10-final)

**Full Pipeline (Camera → Inference → Display):**
- FPS: 30 FPS (camera-limited)
- Latency: ~33 ms per frame
- GPU Usage: <30%
- CPU Usage: 1-5% per core
- Temperature: 43-45°C
- NaN Rate: 0%

**Inference-Only (No Camera/Display):**
- FPS: 65 FPS
- Latency: ~15 ms per frame
- GPU Usage: Active
- Device: cuda:0 (Xavier GPU)

### Expected Performance After TensorRT Optimization

**With TensorRT FP16:**
- Inference FPS: 90-110 FPS (35-69% improvement)
- Full Pipeline: Still camera-limited at 30 FPS
- Latency: ~9-11 ms per inference

For TensorRT optimization guide, see:
- `~/docs/PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md`

---

## Troubleshooting

### Issue: CUDA Not Available

```bash
# Verify CUDA installation
nvcc --version

# Check PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch wheel if needed
pip uninstall torch
pip install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
```

### Issue: Camera Not Opening

```bash
# Check camera device
ls -la /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Try different camera index
python3 simple_projection.py --source 1
```

### Issue: NaN Errors During Inference

**Cause:** Usually indicates CPU inference instead of GPU
**Solution:**

```bash
# Verify GPU is being used
python3 test_gpu_validation.py

# Check logs for device verification
cat inference_diagnostics.log | grep "device verified"
# Expected: "First inference device verified: cuda:0"
```

### Issue: Low FPS Performance

**Expected:** 30 FPS (camera-limited), 65 FPS (inference-only)

If seeing <15 FPS:
1. Verify GPU inference (see NaN errors section)
2. Check CPU frequency: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`
3. Check thermal throttling: `tegrastats`
4. Consider TensorRT optimization

### Issue: Service Won't Start

```bash
# Check service status
sudo systemctl status halloween.service

# View detailed logs
journalctl -u halloween.service -n 50

# Common fixes:
# - Verify venv path in service file
# - Check file permissions
# - Ensure camera is accessible
```

---

## File Structure

```
Halloween-Visions-Projection/
├── simple_projection.py          # Main application
├── test_gpu_validation.py        # GPU inference validation
├── test_camera.py                # Camera testing utility
├── Colin1.pt                     # Production model
├── quinn_arms_up.pt              # Alternative model
├── halloween.service             # Systemd service file
├── videos/                       # Video assets
│   ├── sleeping_face.mp4
│   └── angry_face.mp4
└── envs/xavier/                  # Xavier-specific files
    ├── README.md                 # This file
    ├── requirements.txt          # Pinned dependencies
    └── install.sh                # Setup script
```

---

## Known Issues & Limitations

### Hardware Limitations
- **Camera FPS:** USB cameras limited to 30 FPS on Xavier
- **Memory:** 7GB shared between CPU/GPU
- **OpenCV Grey Border:** Known issue with cv2.imshow() (not Xavier-specific)

### Software Limitations
- **PyTorch Version:** Must use NVIDIA pre-built wheel (1.12.0a0+2c916ef.nv22.3)
- **Python Version:** Must use Python 3.8 (included with JetPack 5.0.2)
- **OpenCV CUDA:** System python3-opencv recommended for CUDA support

### Optimization Opportunities
- **TensorRT Conversion:** 35-69% inference speedup available (see PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md)
- **Multi-camera Support:** Possible with TensorRT optimization
- **Model Quantization:** INT8 quantization possible but requires calibration

---

## Additional Resources

### Project Documentation
- **System Specs:** `~/docs/SYSTEM_SPECIFICATIONS_PERFORMANCE_REPORT.md`
- **Performance Guide:** `~/docs/PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md`
- **SSH Setup:** `~/docs/setup/B10_Xavier_SSH_Setup_Guide.md`
- **Project Summary:** `~/docs/B10_PROJECT_COMPLETE_SUMMARY.md`

### Build Logs
- `~/docs/build_logs/B10-INFER-GPU-VALIDATE_status_report.md`
- `~/docs/build_logs/B10-PROJ-INFER-REFIT_completion_report.md`
- `~/docs/build_logs/B10-INFER-PIL-REMOVAL_completion_report.md`

### Run Logs
- `~/docs/run_logs/B10-LIVE-CAMERA-VALIDATION_report.md`
- `~/docs/run_logs/B10-FINAL-DEPLOYMENT-READY_report.md`

### External Resources
- **JetPack 5.0.2 Docs:** https://developer.nvidia.com/embedded/jetpack
- **PyTorch Wheels:** https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/
- **Ultralytics Docs:** https://docs.ultralytics.com/
- **TensorRT Guide:** https://docs.nvidia.com/deeplearning/tensorrt/

---

## Support & Contributing

For issues or improvements, see:
- **Root README:** `../../README.md`
- **Contributing Guide:** `../../CONTRIBUTING.md`
- **Maintenance Guide:** `../../MAINTENANCE.md`

---

**Deployment Status:** Production Ready (vB10-final)
**Last Validated:** 2025-10-20
**Performance:** 30 FPS real-time, 0% NaN rate, stable operation
