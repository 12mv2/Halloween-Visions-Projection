# Ticket: Restructure for Multi-Game Platform

## Goal

Transform the denhac XavierMLVisions repo into a platform where multiple developers can create and deploy their own ML-powered games on the Xavier at denhac.

## Current State

- **denhac repo** (`Denhac/XavierMLVisions`): Flat structure with single Halloween projection
- **Local repo**: Multiple sub-projects (QuestV3, MitchV2, XavierV1)
- QuestV3 with train7 model is working and ready

## Proposed Structure

```
XavierMLVisions/
├── games/
│   ├── quest/                    # Object hunt game (QuestV3)
│   │   ├── quest_game.py
│   │   ├── assets/               # Game-specific videos
│   │   └── README.md
│   ├── halloween/                # Original Halloween projection
│   │   ├── halloween_game.py
│   │   └── README.md
│   └── _template/                # Template for new games
│       ├── my_game.py
│       └── README.md
├── shared/
│   ├── mitchplayer/              # Video player components
│   │   ├── mitchplayer.py
│   │   ├── demonplayer.py
│   │   └── styledsubtitler.py
│   ├── models/                   # Trained ML models
│   │   ├── train7/               # 7-class classifier
│   │   └── exp_balanced/         # Legacy model
│   └── inference.py              # Common inference utilities
├── envs/
│   ├── xavier/                   # Xavier NX setup
│   └── linux/                    # Generic Linux setup
├── launcher.py                   # Game selector menu
├── requirements.txt
├── CONTRIBUTING.md               # How to add your game
└── README.md
```

## Tasks

### Phase 1: Restructure Local Repo
- [ ] Create `games/` directory structure
- [ ] Move QuestV3 to `games/quest/`
- [ ] Move Halloween/Mitch to `games/halloween/`
- [ ] Extract shared components to `shared/`
- [ ] Update imports to use shared modules
- [ ] Test both games still work

### Phase 2: Create Developer Experience
- [ ] Create `games/_template/` with minimal working example
- [ ] Write `CONTRIBUTING.md` guide
- [ ] Create `launcher.py` game selector
- [ ] Document how to train custom models

### Phase 3: PR to Denhac
- [ ] Clean up any debug code
- [ ] Ensure all assets use git LFS
- [ ] Create PR with new structure
- [ ] Update denhac README with platform vision

## Technical Notes

### Shared Imports
Games should import shared modules like:
```python
from shared.mitchplayer import DetectorPlayer, DemonPlayer
from shared.inference import load_model, predict
```

### Model Location
Models live in `shared/models/`. Games reference by name:
```python
model = load_model("train7")  # Loads shared/models/train7/weights/best.pt
```

### Game Entry Point
Each game has a main script that can run standalone:
```bash
cd games/quest && python quest_game.py
```

Or via launcher:
```bash
python launcher.py  # Shows menu to select game
```

## Success Criteria

1. Any denhac member can clone repo and run existing games
2. Clear path to add new games (copy template, modify)
3. Shared models/components reduce duplication
4. Works on Xavier NX hardware

## References

- QuestV3 commit: `ec3fd87` (train7 integration)
- denhac repo: https://github.com/Denhac/XavierMLVisions
- finetune-workshop: Training pipeline for new models

---

*Created: 2024-12-14*
