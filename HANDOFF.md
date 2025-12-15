# Handoff: Xavier ML Visions Games

## Current State (2024-12-15)

**Phase 2 COMPLETE** - Repo restructured, both games tested locally on Ubuntu.

---

## MASTER PLAN (7 Phases)

### Phase 1: Prep Work ✓ DONE
- [x] Write ONNX inference script (`inference/onnx_infer.py`)
- [x] Write TensorRT inference script (`inference/tensorrt_infer.py`)
- [x] Export model to ONNX (`models/7class_v1/best.onnx`)
- [x] Test ONNX inference locally
- [x] Test SimpleHunt game locally
- [x] Test QuestDemon game locally

### Phase 2: Xavier Repo Restructure ✓ DONE
- [x] Create clean directory structure (`games/`, `models/`, `inference/`)
- [x] Move QuestDemon (Mitch's demon game)
- [x] Copy SimpleHunt (template game)
- [x] Copy production model
- [x] Update import paths
- [x] Delete old cruft (MitchV2, Halloween-Visions-XavierV1)
- [x] Test both games work locally

### Phase 3: Xavier NX Deployment ← START HERE
1. Free disk space on Xavier (target 6GB+)
2. Copy ONNX model to Xavier
3. Convert to TensorRT engine on Xavier
4. Test TensorRT inference on Xavier

### Phase 4: Test Games on Xavier
1. Deploy restructured repo to Xavier
2. Test QuestDemon with PyTorch model
3. Test SimpleHunt with TensorRT engine
4. Fix any Xavier-specific issues

### Phase 5: Educational Game
1. Design educational game concept
2. Build in `games/Educational/`
3. Should teach ML concepts while playing

### Phase 6: Push to Denhac
1. Final testing on Xavier
2. Push xavier repo to denhac remote
3. Verify denhac deployment works

### Phase 7: Cleanup
1. Remove `games/` folder from finetune-workshop repo
2. Keep `models/` and `inference/` in finetune-workshop for export testing
3. Document the two-repo workflow

---

## Repo Structure
```
MLVisionsProjects-from-xavier/
├── games/
│   ├── QuestDemon/         # Mitch's demon game ✓ WORKING
│   │   ├── quest_projection.py
│   │   ├── mitchplayer/    # Video player + assets
│   │   └── requirements.txt
│   └── SimpleHunt/         # Template game ✓ WORKING
│       ├── game.py
│       └── requirements.txt
├── models/
│   └── 7class_v1/          # Production model
│       ├── best.pt         # PyTorch (for QuestDemon)
│       ├── best.onnx       # ONNX (export for TensorRT)
│       └── classes.txt     # Class names
├── inference/              # Shared backends
│   ├── onnx_infer.py       # ONNX Runtime (dev machine)
│   └── tensorrt_infer.py   # TensorRT (Xavier NX)
├── CHANGELOG.md
└── HANDOFF.md
```

---

## Phase 3 Commands (Xavier Deployment)

```bash
# 1. Check/free disk space (MUST have 6GB+)
ssh xavier "df -h"
ssh xavier "sudo apt-get clean"
ssh xavier "sudo journalctl --vacuum-time=7d"
ssh xavier "rm -rf ~/.cache/pip"

# 2. Copy ONNX model to Xavier
scp models/7class_v1/best.onnx xavier:~/
scp models/7class_v1/classes.txt xavier:~/

# 3. Convert to TensorRT on Xavier (this is the critical step)
ssh xavier "/usr/src/tensorrt/bin/trtexec --onnx=~/best.onnx --fp16 --saveEngine=~/best.engine"

# 4. If successful, copy engine back
scp xavier:~/best.engine models/7class_v1/

# 5. Commit the engine file
git add models/7class_v1/best.engine
git commit -m "feat: Add TensorRT engine for Xavier"
```

**If TensorRT conversion fails:**
- Check disk space (needs ~2GB free during conversion)
- Check ONNX opset compatibility with TensorRT 8.4.1
- Fall back to ONNX Runtime on Xavier (slower but works)

---

## Phase 4 Commands (Test on Xavier)

```bash
# Deploy repo to Xavier
rsync -av --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
  /home/ubuntu24/projects/MLVisionsProjects-from-xavier/ xavier:~/MLVisions/

# Test SimpleHunt with TensorRT
ssh xavier "cd ~/MLVisions && python3 games/SimpleHunt/game.py --model models/7class_v1/best.engine"

# Test QuestDemon (uses PyTorch, not TensorRT)
ssh xavier "cd ~/MLVisions/games/QuestDemon && python3 quest_projection.py --model ../../models/7class_v1/best.pt"
```

---

## How to Run Games Locally (Ubuntu)

```bash
# Use finetune-workshop venv (has all deps)
cd /home/ubuntu24/projects/MLVisionsProjects-from-xavier

# SimpleHunt (ONNX inference)
/home/ubuntu24/projects/finetune-workshop/.venv/bin/python games/SimpleHunt/game.py

# QuestDemon (PyTorch/ultralytics)
cd games/QuestDemon
/home/ubuntu24/projects/finetune-workshop/.venv/bin/python quest_projection.py --model ../../models/7class_v1/best.pt
```

**Game Controls:**
- `SPACE` - Start game (SimpleHunt)
- `D` - Debug overlay
- `T` - Trigger demon (skip hand detection)
- `N` - Next object (skip current target)
- `Q` - Quit

---

## Xavier NX Environment
| Component | Version | Notes |
|-----------|---------|-------|
| JetPack   | 5.0.2   | Don't upgrade |
| CUDA      | 11.4    | Bundled with JetPack |
| TensorRT  | 8.4.1   | For inference |
| Disk      | ~4GB free | Need 6GB+ for TensorRT |
| Python    | 3.8     | System Python |

**Warning:** Xavier has fragile dependencies. Don't pip upgrade anything.

---

## Model Info
- **Classes** (7): 9v_battery, background, black_spool, blue_floppy, green_spool, hammer, hand
- **Architecture**: YOLOv8n-cls
- **Input**: 224x224
- **Accuracy**: 98.7% top-1
- **Source**: finetune-workshop/runs/classify/train7

---

## Git Remotes
```bash
origin  https://github.com/12mv2/Halloween-Visions-Projection.git
denhac  https://github.com/Denhac/XavierMLVisions.git
```

---

## Key Files

| File | Purpose |
|------|---------|
| `games/QuestDemon/quest_projection.py` | Mitch's demon game (hand→charge→demon→hunt) |
| `games/QuestDemon/mitchplayer/` | Video player, demon player, assets |
| `games/SimpleHunt/game.py` | Template scavenger hunt (score + timer) |
| `inference/onnx_infer.py` | ONNX Runtime inference class |
| `inference/tensorrt_infer.py` | TensorRT inference class (Xavier) |
| `models/7class_v1/best.pt` | PyTorch model |
| `models/7class_v1/best.onnx` | ONNX export |

---

## Two-Repo Workflow

1. **finetune-workshop** - Training, datasets, model export
   - Train models with YOLO
   - Export to ONNX
   - Test inference locally

2. **MLVisionsProjects-from-xavier** - Games, deployment
   - Copy production models here
   - Build games using inference/ backends
   - Deploy to Xavier/denhac

---

*Last updated: 2024-12-15*
