# Xavier ML Visions

Interactive ML/CV educational games for [denhac](https://denhac.org) hackerspace.

Runs on NVIDIA Jetson Xavier NX with USB camera + HDMI projector.

## Games

| Game | Description | Time |
|------|-------------|------|
| [TrainClassifier](games/TrainClassifier/) | Train a REAL neural network on your photos | ~3-4 min |
| [DemonQuest](games/DemonQuest/) | Gesture-controlled demon summoning projection | ~2 min |
| [SimpleHunt](games/SimpleHunt/) | Basic object detection game | ~1 min |

## Quick Start (Xavier)

```bash
cd ~/MLVisions

# Educational training game
python3 games/TrainClassifier/game.py

# Demon projection game
python3 games/DemonQuest/quest_projection.py

# Simple detection game
python3 games/SimpleHunt/game.py
```

## Hardware

- NVIDIA Jetson Xavier NX
- USB webcam
- HDMI projector (for DemonQuest)

## Tech Stack

- YOLOv8 (Ultralytics) - object detection & classification
- OpenCV - camera capture & display
- PyTorch - neural network backend

## License

MIT - see [LICENSE](LICENSE)
