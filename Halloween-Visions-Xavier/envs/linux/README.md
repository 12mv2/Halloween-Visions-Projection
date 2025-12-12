# Ubuntu Development Setup

## Quick Start

```bash
git clone https://github.com/denhac/Halloween-Visions-Xavier.git
cd Halloween-Visions-Xavier
python3 -m venv venv
source venv/bin/activate
pip install -r envs/linux/requirements.txt
python3 simple_projection.py
```

## SSH to Xavier

```bash
ssh colin@10.11.3.65
```

Password: ask a denhac member

## File Transfer

```bash
# To Xavier
scp myfile.mp4 colin@10.11.3.65:~/Documents/MLVisionsProjects/Halloween-Visions-Projection/videos/

# From Xavier (whole project, no venv)
rsync -av --exclude='venv' colin@10.11.3.65:~/Documents/MLVisionsProjects/Halloween-Visions-Projection ~/Documents/
```

## Running on Xavier via SSH

Can't display video over SSH. Options:

1. **Run on Xavier terminal directly**
2. **X forwarding:** On Xavier run `xhost +local:`, then SSH with `export DISPLAY=:0`
3. **Systemd:** `sudo systemctl start halloween.service`

## Dev Workflow

1. Edit locally
2. `git push`
3. SSH to Xavier: `git pull`
4. Test on Xavier

## GPU Support (optional)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Without GPU, runs on CPU (~15 FPS vs 30 FPS on Xavier).

---

**Xavier IP:** 10.11.3.65
