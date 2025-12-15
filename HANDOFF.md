# Project Handoff: Xavier ML Visions

## What Was Accomplished

### 1. Train7 Model Integration (2024-12-14)
Integrated a new 7-class YOLO classifier into QuestV3:

**Model Details:**
- Path: `QuestV3/models/train7/weights/best.pt`
- Classes: `hand`, `hammer`, `9v_battery`, `black_spool`, `green_spool`, `blue_floppy`, `background`
- Accuracy: 99.4% validation
- Size: ~3MB

**Key Changes:**
- Switched inference from manual tensor preprocessing to `model.predict()` API
- Updated `QUEST_SEQUENCE` to use new classes (replaced orange_ball with blue_floppy)
- Added display name mapping in demonplayer for "blue floppy disk"

### 2. Git LFS Setup
Configured git LFS for large files:
- `*.mp4` - video assets
- `*.pt` - model weights

### 3. QuestV3 Game Working
The object hunt game is fully functional:
```bash
cd ~/projects/MLVisionsProjects-from-xavier/QuestV3
source venv/bin/activate
python3 quest_projection.py
```

**Game Flow:**
1. Show hand → charges up → triggers demon
2. Demon asks for objects in sequence
3. Find and show each object to camera
4. Complete all 5 objects or timeout

## Current Repository State

```
MLVisionsProjects-from-xavier/
├── QuestV3/                 # Active - train7 integrated, working
│   ├── quest_projection.py
│   ├── mitchplayer/
│   └── models/train7/
├── MitchV2/                 # Legacy - original Halloween
├── Halloween-Visions-XavierV1/  # Legacy - untracked
└── TICKET_MultiGame_Platform.md  # Next task
```

**Git Remotes:**
- `origin` → `12mv2/Halloween-Visions-Projection.git` (personal dev)
- `denhac` → `Denhac/XavierMLVisions.git` (org deployment)

**Latest Commit:** `ec3fd87` - feat(QuestV3): Integrate train7 model

## What's Next

See `TICKET_MultiGame_Platform.md` for the restructuring plan to make this a multi-game platform for denhac developers.

## Key Files to Know

| File | Purpose |
|------|---------|
| `QuestV3/quest_projection.py` | Main game entry point |
| `QuestV3/mitchplayer/mitchplayer.py` | DetectorPlayer - handles charge/activate |
| `QuestV3/mitchplayer/demonplayer.py` | DemonPlayer - handles demon dialog sequence |
| `QuestV3/models/train7/weights/best.pt` | Trained 7-class model |
| `finetune-workshop/` | Separate repo for model training |

## Debug Tips

- Press `D` in game for debug overlay
- Press `T` to manually trigger demon (skip hand detection)
- Press `N` to skip to next object
- Debug logging shows `[DP]` for DetectorPlayer, `[DEBUG]` for charge state

## Dependencies

QuestV3 venv has:
- Python 3.12
- ultralytics (YOLO)
- opencv-python
- torch

Note: Xavier has brittle dependencies - avoid updating packages.

---

*Handoff Date: 2024-12-14*
