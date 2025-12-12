# Hand Detection Projection System

Real-time hand detection with video projection at denhac.

**Location:** Front entrance, right side
**Hardware:** NVIDIA Jetson Xavier NX + USB camera + HDMI projector

## Quick Start

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
source venv/bin/activate
python3 simple_projection.py
```

**Controls:** D=debug, F=fullscreen, Q=quit

## Options

```
--source N          Camera index (default: 0)
--conf X.X          Confidence 0.0-1.0 (default: 0.7)
--model PATH        Model (default: models/Colin1.pt)
--video-sleep PATH  Idle video
--video-scare PATH  Scare video
--fullscreen        Start fullscreen
```

## Auto-Start

```bash
sudo cp halloween.service /etc/systemd/system/
sudo systemctl enable halloween.service
sudo systemctl start halloween.service
```

## Platform Setup

- **Xavier NX:** See [envs/xavier/README.md](envs/xavier/README.md)
- **Ubuntu dev:** See [envs/linux/README.md](envs/linux/README.md)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No display | Run on Xavier terminal, not SSH |
| Camera not found | Try `--source 1` |
| Low accuracy | Try `--conf 0.5` |
| Service fails | `journalctl -u halloween.service -n 50` |

## TODO: Publish to denhac GitHub

1. Create new repo on GitHub: `denhac/Halloween-Visions-Xavier`
2. Run these commands:
   ```bash
   cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
   git remote rename origin old-origin
   git remote add origin git@github.com:denhac/Halloween-Visions-Xavier.git
   git add -A && git commit -m "Initial commit for denhac"
   git push -u origin main
   ```
