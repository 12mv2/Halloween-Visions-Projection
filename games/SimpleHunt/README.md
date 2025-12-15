# QuestMitch

**Object Scavenger Hunt Game**

A timed game where players must find and show objects to the camera. Race against the clock to score points!

## How to Play

1. Run the game
2. Press **SPACE** to start
3. Find the object shown on screen
4. Hold it in front of the camera until the progress bar fills
5. Score points and get a new target
6. Beat your high score before time runs out!

## Quick Start

```bash
# From project root
cd /path/to/finetune-workshop

# Run with ONNX model (dev machine)
python games/QuestMitch/game.py --model models/production/7class_v1/best.onnx

# Run with TensorRT engine (Xavier NX)
python games/QuestMitch/game.py --model models/production/7class_v1/best.engine
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `models/production/7class_v1/best.onnx` | Path to model file |
| `--source` | `0` | Camera index |
| `--conf` | `0.7` | Confidence threshold (0-1) |
| `--hold` | `1.5` | Seconds to hold object |
| `--duration` | `60.0` | Game duration in seconds |

## Examples

```bash
# Easier game (lower confidence, shorter hold)
python games/QuestMitch/game.py --conf 0.5 --hold 1.0

# Harder game (higher confidence, longer hold)
python games/QuestMitch/game.py --conf 0.85 --hold 2.5

# Longer game
python games/QuestMitch/game.py --duration 120

# Different camera
python games/QuestMitch/game.py --source 1
```

## Controls

- **SPACE** - Start game / Restart after game over
- **Q** - Quit

## Objects

The default 7-class model recognizes:
- 9V Battery
- Black Filament Spool (3D printer)
- Green Sewing Spool
- Hammer
- Blue Floppy Disk
- Hand
- Background (not used as target)

## Customizing

This game is designed as a template. Fork it to create your own games!

Ideas:
- **Speed Round**: Shorter hold time, more objects
- **Survival Mode**: Lives instead of timer
- **Multiplayer**: Two players, two cameras
- **Sound Effects**: Add audio feedback
- **Leaderboard**: Track high scores

### Code Structure

```python
from inference.onnx_infer import YOLOClassifier  # or tensorrt_infer

class QuestMitch:
    def __init__(self, model_path, ...):
        self.classifier = YOLOClassifier(model_path)
        self.score = 0
        self.target = None

    def new_target(self):
        # Pick random object
        self.target = random.choice(self.target_classes)

    def check_detection(self, pred_class, confidence):
        # Check if player found the target
        if pred_class == self.target and confidence >= threshold:
            # Handle scoring logic
            ...
```
