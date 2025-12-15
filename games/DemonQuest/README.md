# QuestV3 - Object Hunt Game

A multi-stage object detection game using YOLO classification. Players find physical objects and present them to the camera in sequence.

## Quick Start

```bash
cd ~/projects/MLVisionsProjects-from-xavier/QuestV3
source venv/bin/activate
python3 quest_projection.py
```

## Game Flow

1. Show your **hand** to activate the demon
2. Demon asks for objects in sequence:
   - green spool → black spool → 9v battery → hammer → blue floppy
3. Find each object and show it to the camera
4. Complete all 5 objects or timeout after 5 minutes per object

## Controls

| Key | Action |
|-----|--------|
| D | Toggle debug mode (camera preview) |
| F | Toggle fullscreen |
| Q / ESC | Quit |

## Options

```bash
# Fullscreen mode
python3 quest_projection.py --fullscreen

# Different camera
python3 quest_projection.py --source 1

# Lower confidence threshold
python3 quest_projection.py --conf 0.10

# Different model
python3 quest_projection.py --model models/nano_v2/weights/best.pt
```

## Requirements

- Python 3.8+
- OpenCV, PyTorch, Ultralytics
- USB camera

```bash
pip install ultralytics opencv-python
```

## Project Structure

```
QuestV3/
├── quest_projection.py   # Main entry point
├── mitchplayer/          # Video player components
├── models/               # YOLO classification models
└── envs/                 # Environment setup guides
```
