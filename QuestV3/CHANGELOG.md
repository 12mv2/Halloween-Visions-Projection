# Changelog

## 2025-12-12: Quest Game Flow Implementation

**Status:** Complete - Tested working

### Game Flow Implementation
Implemented full Quest game loop with multi-object detection sequence:

1. **DETECTOR phase:** DetectorPlayer shows idle animation, waiting for hand
2. **CHARGING:** Hand detected, DetectorPlayer shows charging animation
3. **DEMON phase:** DemonPlayer runs intro with snarky dialog, asks for object
4. **Object found:** DemonPlayer plays outro, proceeds to next object
5. **Sequence complete or timeout:** Returns to DETECTOR phase

### Fixed Object Sequence
Objects are detected in fixed order (not random):
- green_spool -> black_spool -> 9v_battery -> hammer -> orange_ball -> orange2

### Timeout Logic
- 5-minute timeout per object (300 seconds)
- If object not found within timeout, returns to hand detection
- Sequence resets on timeout

### Bug Fixes

**Demon text not showing target object:**
- Added `object_display_names` mapping in DemonPlayer for readable names
- Wired body text formatting with `writer.set_text(body_text)`

**Object detection not triggering phase transition:**
- Fixed `fading_out` check - now checked every frame, not just at video end
- Added `and not demon.fading_out` guard to prevent detection spam

**Race condition with sequence_complete flag:**
- Fixed by not resetting `sequence_complete` at top of demon thread loop
- Only reset after main loop has activated demon again

**Blank screen after outro / detector stuck:**
- Added `detector.skip_playback = False` to release detector thread from wait loop
- Properly reset detector state after demon sequence

**Intro text not displaying:**
- Added intro text setting at start of intro phase
- Enabled snarky dialog (was commented out)

**Dead code removal:**
- Removed 15 unreachable lines after return in `render_text_on_frame()`

### Enabled Dialog Text

**Intro phrases (random):**
- "Oh. It's you again."
- "-sniff- You reek of failure."
- "Do you enjoy wasting my time?"

**Body phrase (with target object):**
- "Where is my {object}?"
- "Bring me the {object}."
- "I require the {object}."

**Outro phrases (random):**
- "Your incompetence is disappointing, but not surprising. Be gone."
- "Be gone."

### Files Modified

- `quest_projection.py` - Game state machine, QuestGame class with fixed sequence
- `mitchplayer/demonplayer.py` - Object display names, fading_out fix, race condition fix
- `mitchplayer/styledsubtitler.py` - No changes (working correctly)
- `mitchplayer/mitchplayer.py` - No changes (skip_playback flag already existed)

### Model Files Added

Complete training artifacts copied from finetune-workshop:

**exp_balanced/** (primary Quest model - 8 classes):
- `weights/best.pt` - Trained model weights
- `args.yaml` - Training configuration
- `confusion_matrix.png`, `confusion_matrix_normalized.png` - Validation results
- `results.csv`, `results.png` - Training metrics
- `train_batch*.jpg` - Training batch samples
- `val_batch*_labels.jpg`, `val_batch*_pred.jpg` - Validation predictions

**nano_v2/** (alternative model):
- Same artifact structure as exp_balanced

---

## 2025-12-12: Initial QuestV3 Setup

**Status:** Complete

### Initial Structure
- Created from MitchV2 base
- Added Quest-specific game logic for multi-object detection
- Set up mitchplayer components (DetectorPlayer, DemonPlayer, StyledTypewriter)

### Supported Classes (exp_balanced model)
- hand
- hammer
- 9v_battery
- black_spool
- green_spool
- orange_ball
- orange2
- background

### Controls
- D - Toggle Debug/Projection mode
- F - Toggle fullscreen
- Q/ESC - Quit

---

## Architecture Notes

### Threading Model
- **Main thread:** Camera capture, YOLO inference, state machine, display
- **DetectorPlayer thread:** Video playback for idle/charging/activate phases
- **DemonPlayer thread:** Video playback for intro/body/outro phases

### State Machine Coordination
Flags used for thread synchronization:
- `detector.charging` - Main loop sets True when hand detected
- `detector.activate_complete` - Detector signals charge complete
- `detector.skip_playback` - Skips playback phase for Quest mode handoff
- `demon.active` - Main loop activates demon sequence
- `demon.fading_out` - Main loop signals object found
- `demon.sequence_complete` - Demon signals outro finished
- `demon.phase` - Current demon phase (idle/intro/body/outro)

### Model Inference
Using Ultralytics YOLO classification:
```python
with torch.no_grad():
    output = model.model(frame_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=1)
```

Note: Uses `model.model.to(device)` not `model.to(device)` to avoid Ultralytics training mode bug.
