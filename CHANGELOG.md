# Changelog

## 2024-12-15 - Repository Restructure

### Added
- `games/` directory with two working games:
  - `QuestDemon/` - Mitch's demon game with video animations
  - `SimpleHunt/` - Simple scavenger hunt template game
- `models/7class_v1/` - Production model (7 classes: hand, hammer, 9v_battery, black_spool, green_spool, blue_floppy, background)
- `inference/` - Shared ONNX and TensorRT inference backends

### Removed
- `MitchV2/` - Old version (consolidated into QuestDemon)
- `Halloween-Visions-XavierV1/` - Old version
- `QuestV3/` - Moved to `games/QuestDemon/`

### Changed
- Flat structure replaced with organized `games/`, `models/`, `inference/` directories
- Model paths updated to use shared `models/7class_v1/`
