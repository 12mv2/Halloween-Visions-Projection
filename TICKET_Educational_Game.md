# Ticket: Educational Game for Denhac Visitors

## Priority: HIGH - Needed for tomorrow's visitors

## Goal

Create a quick, simple educational game that teaches ML/computer vision concepts to denhac visitors while being fun and interactive.

## Constraints

- **Time:** Must be designable/buildable quickly
- **Audience:** General public, no ML background assumed
- **Hardware:** Xavier NX with camera + projector/monitor
- **Existing assets:** 7-class model (hand, hammer, 9v_battery, black_spool, green_spool, blue_floppy, background)

## Research Questions

### 1. What ML concepts can we teach visually?
- [ ] Confidence scores (show the percentage bar)
- [ ] Classification vs detection
- [ ] Training data → model → inference pipeline
- [ ] What happens when the model is wrong/uncertain

### 2. Game Format Ideas

**Option A: "Train the AI" Simulation**
- Show visitors what training data looks like
- Let them "feed" objects to camera
- Display confidence building up
- Teach: models learn from examples

**Option B: "Fool the AI" Challenge**
- Try to show objects in ways that confuse the model
- Partial occlusion, weird angles, similar objects
- Scoreboard for "most confusing" presentations
- Teach: AI limitations, adversarial examples

**Option C: "AI Detective" Game**
- AI describes what it sees in real-time
- Visitors try to match what AI is looking for
- Shows classification process live
- Teach: how AI "sees" vs how humans see

**Option D: "Confidence Race"**
- Two players, split screen
- Race to get highest confidence on target object
- Shows how positioning/lighting affects recognition
- Teach: importance of good training data

### 3. UI/UX Considerations
- [ ] Should show confidence bars prominently
- [ ] Should explain what's happening in simple terms
- [ ] Consider adding "what the AI sees" debug view
- [ ] Sound effects for recognition events?

### 4. What can we reuse?
- SimpleHunt already has score/timer mechanics
- QuestDemon has video overlay system
- Both have confidence threshold logic
- Debug overlay shows internal state

## Deliverables

1. **Game concept document** - 1 page max describing the game
2. **Educational talking points** - What docents should explain
3. **Implementation estimate** - How long to build

## Existing Code Reference

```
games/SimpleHunt/game.py     - Score/timer template
games/QuestDemon/            - Video overlay, state machine
inference/onnx_infer.py      - Inference backend
models/7class_v1/            - Trained model
```

## Success Criteria

- Visitors understand that AI learns from examples
- Visitors see confidence scores and understand what they mean
- Game is fun enough to hold attention for 2-3 minutes
- Can be explained by any docent in 30 seconds

## Notes

- Keep it SIMPLE - visitors won't read instructions
- Visual feedback is key - show don't tell
- Consider attention span - short game loops
- Make failures fun, not frustrating

---

*Created: 2024-12-15*
*Target: Tomorrow's denhac visitors*
