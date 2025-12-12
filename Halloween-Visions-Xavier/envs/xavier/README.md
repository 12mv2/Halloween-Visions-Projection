# Xavier NX Deployment

**JetPack:** 5.0.2 (L4T R35.1.0) | **Python:** 3.8

## Setup

```bash
# Clone to required path (venv has hardcoded paths)
mkdir -p ~/Documents/MLVisionsProjects
cd ~/Documents/MLVisionsProjects
git clone <repo-url> Halloween-Visions-Projection
cd Halloween-Visions-Projection

# Restore venv from backup
tar -xzvf envs/xavier/xavier_venv_backup_20251201.tar.gz

# Run
source venv/bin/activate
python3 simple_projection.py
```

> **WARNING:** PyTorch 1.12.0 wheel no longer available from NVIDIA.
> The venv backup is the only way to get this environment.

## Auto-Start

```bash
sudo cp halloween.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable halloween.service
sudo systemctl start halloween.service
```

Check: `sudo systemctl status halloween.service`
Logs: `journalctl -u halloween.service -f`

## Verify Installation

```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # True
python3 -c "import torch; print(torch.__version__)"          # 1.12.0a0+2c916ef.nv22.3
```

## Creating New Venv (if needed)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch - try 1.13 (1.12 URL is dead)
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
pip install torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl

pip install ultralytics==8.0.196
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA not available | Check `nvcc --version`, reinstall PyTorch |
| Camera not opening | `ls /dev/video*`, try `--source 1` |
| NaN errors | GPU not being used - run `test_gpu_validation.py` |
| Low FPS (<15) | Check thermal: `tegrastats` |

## Performance

- Full pipeline: 30 FPS (camera-limited)
- Inference only: 65 FPS
- GPU usage: <30%
- Temp: 43-45C

## Known Issues

- PyTorch 1.12 wheel no longer available (use venv backup)
- OpenCV grey border (use fullscreen or projector positioning)
- Ultralytics bug: use `model.model.to(device)` not `model.to(device)`
