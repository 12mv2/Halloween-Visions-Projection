# Handoff: Xavier ML Visions Games

## Current State (2024-12-15)

**Phase 2 COMPLETE** - Repo restructured, both games tested locally.

### Repo Structure
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
│       ├── best.pt         # PyTorch
│       ├── best.onnx       # ONNX (for inference/)
│       └── classes.txt
├── inference/              # Shared backends
│   ├── onnx_infer.py       # ONNX Runtime
│   └── tensorrt_infer.py   # TensorRT (Xavier)
└── CHANGELOG.md
```

### Git Remotes
- `origin` → github.com/12mv2/Halloween-Visions-Projection.git
- `denhac` → github.com/Denhac/XavierMLVisions.git

---

## REMAINING WORK

### Phase 3: Xavier NX Deployment
```bash
# 1. Free disk space (target 6GB+)
ssh xavier "sudo apt-get clean; sudo journalctl --vacuum-time=7d"
ssh xavier "df -h"  # Verify space

# 2. Copy ONNX to Xavier
scp models/7class_v1/best.onnx xavier:~/
scp models/7class_v1/classes.txt xavier:~/

# 3. Convert to TensorRT on Xavier
ssh xavier "/usr/src/tensorrt/bin/trtexec --onnx=~/best.onnx --fp16 --saveEngine=~/best.engine"

# 4. Copy back engine file
scp xavier:~/best.engine models/7class_v1/
```

### Phase 4: Test Games on Xavier
```bash
# Deploy repo to Xavier
rsync -av --exclude='.git' --exclude='*.pyc' . xavier:~/MLVisions/

# Test SimpleHunt with TensorRT
ssh xavier "cd ~/MLVisions && python games/SimpleHunt/game.py --model models/7class_v1/best.engine"

# Test QuestDemon (uses PyTorch directly)
ssh xavier "cd ~/MLVisions/games/QuestDemon && python quest_projection.py --model ../../models/7class_v1/best.pt"
```

### Phase 5: Educational Game
- Design and build in `games/Educational/`
- Should teach ML concepts while playing

### Phase 6: Push to Denhac
```bash
git push denhac main
```

### Phase 7: Cleanup finetune-workshop
- Remove `games/` folder from finetune-workshop (now lives here)
- Keep `models/` and `inference/` there for export testing

---

## How to Run Games Locally

```bash
# Use finetune-workshop venv (has all deps)
cd /home/ubuntu24/projects/MLVisionsProjects-from-xavier

# SimpleHunt (ONNX)
/home/ubuntu24/projects/finetune-workshop/.venv/bin/python games/SimpleHunt/game.py

# QuestDemon (PyTorch)
cd games/QuestDemon
/home/ubuntu24/projects/finetune-workshop/.venv/bin/python quest_projection.py --model ../../models/7class_v1/best.pt
```

**Controls:**
- `D` - Debug overlay
- `T` - Trigger demon (skip hand)
- `N` - Next object
- `Q` - Quit

---

## Xavier NX Environment
| Component | Version |
|-----------|---------|
| JetPack   | 5.0.2   |
| CUDA      | 11.4    |
| TensorRT  | 8.4.1   |
| Disk      | ~4GB free (need 6GB+) |

---

## Model Info
- **Classes**: 9v_battery, background, black_spool, blue_floppy, green_spool, hammer, hand
- **Architecture**: YOLOv8n-cls
- **Input**: 224x224
- **Accuracy**: 98.7% top-1

---

## Key Files

| File | Purpose |
|------|---------|
| `games/QuestDemon/quest_projection.py` | Mitch's demon game |
| `games/SimpleHunt/game.py` | Template scavenger hunt |
| `inference/onnx_infer.py` | ONNX inference backend |
| `inference/tensorrt_infer.py` | TensorRT backend (Xavier) |
| `models/7class_v1/best.pt` | PyTorch model |
| `models/7class_v1/best.onnx` | ONNX export |

---

*Last updated: 2024-12-15*
