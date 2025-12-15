# Xavier ML Visions

Interactive ML/CV educational games for [denhac](https://denhac.org) hackerspace.

Runs on NVIDIA Jetson Xavier NX with USB camera.

## Games

| Game | Description | Time |
|------|-------------|------|
| [TrainClassifier](games/TrainClassifier/) | Train a REAL neural network on your photos | ~3-4 min |
| [DemonQuest](games/DemonQuest/) | Multi-stage object hunt with demon character | ~5 min |
| [SimpleHunt](games/SimpleHunt/) | Basic object detection game | ~1 min |

## Quick Start (Xavier)

```bash
cd ~/MLVisions

# Educational training game
python3 games/TrainClassifier/game.py

# Object hunt game
python3 games/DemonQuest/quest_projection.py

# Simple detection game
python3 games/SimpleHunt/game.py
```

## Hardware

- NVIDIA Jetson Xavier NX
- USB webcam
- Display/monitor

## Tech Stack

- YOLOv8 (Ultralytics) - object detection & classification
- OpenCV - camera capture & display
- PyTorch - neural network backend

## License

MIT - see [LICENSE](LICENSE)
