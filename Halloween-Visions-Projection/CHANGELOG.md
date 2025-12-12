# Changelog

## 2025-12-01: Repository Cleanup and Multi-Platform Support

**Status:** Complete

### Repository Cleanup
Cleaned repository for sharing on denhac GitHub:

**Deleted:**
- Stray files: `bus.jpg`, `yolov8n.pt`, `inference_diagnostics.log`
- Extra models: `best_final.pt`, `best_runpod.pt`, `colin_arms_up*.pt`, `colin_fetch.pt`, `colin_finger_fetch*.pt`
- Redundant docs: `TODO.md`, `CONTRIBUTING.md`, `MAINTENANCE.md`
- `runs/` directory (training artifacts)

**Reorganized:**
- Models moved to `models/` directory (Colin1.pt, quinn_arms_up.pt)
- Venv backup moved to `envs/xavier/`

### Multi-Platform Environment Structure
Created platform-specific setup guides:

- `envs/xavier/` - Xavier NX deployment (production)
  - Full deployment guide with venv backup restoration
  - Systemd service configuration
  - Troubleshooting guide

- `envs/linux/` - Ubuntu development
  - SSH access instructions (Xavier IP: 10.11.3.65)
  - File transfer commands (SCP/rsync)
  - Development workflow

### Documentation Updates
- README.md rewritten for Xavier/Linux focus
- Added denhac location info (front entrance, right side)
- Removed Mac-specific references

---

## 2025-12-01: Memory Leak Fixes and Ultralytics Workaround

**Status:** Verified working

### Problem
System was degrading from 30 FPS to ~0.05 FPS over 7 days of continuous operation (observed in inference_diagnostics.log from Nov 23-30).

### Root Cause Analysis
- GPU tensors (frame_tensor, output, probs) were not explicitly freed each frame
- Python garbage collection was not keeping up with 30 allocations/second
- FPS benchmark buffer retained 300 entries indefinitely after benchmark

### Fixes Applied (simple_projection.py)
1. **Line 624:** Added `del frame_tensor, output, probs` - explicit GPU memory cleanup each frame
2. **Line 439:** Added `self.fps_buffer.clear()` - frees benchmark memory after completion
3. **Line 510:** Added `model.model.to(device)` - moves inner PyTorch model to GPU

### Ultralytics 8.0.196 Workaround
- **Problem:** `model.to(device)` triggers training mode instead of inference
- **Symptom:** Application starts training on imagenet10 dataset instead of running inference
- **Evidence:** Output showed `engine/trainer: task=classify, mode=train`
- **Solution:** Use `model.model.to(device)` instead of `model.to(device)`
- **Explanation:** `model.model` is the inner PyTorch nn.Module; moving it directly bypasses Ultralytics' buggy `.to()` override

### Documentation Added
- Comprehensive docstrings and comments added to simple_projection.py
- Module-level docstring with architecture overview, state machine, usage, and controls
- Class and method docstrings with Args/Returns documentation
- Inline comments explaining the main loop flow
- Critical workaround documented in code comments

### Technical Notes
- Conservative fixes only - no new dependencies added (Xavier venv is fragile)
- Avoided frequent `torch.cuda.empty_cache()` calls (causes 3-5s stalls per call)
- Based on PyTorch best practices: explicit `del` + infrequent GC is preferred

### Expected Improvement
Stable FPS over extended operation periods (days/weeks).

---

## 2025-09-22: Production Release - OpenCV Solution

### Final System (simple_projection.py)

**Status:** Stable

**Features:**
- Hand detection: 50-99% confidence (trained YOLO classification model)
- Real-time video switching: sleep_face.mp4 and angry_face.mp4
- OpenCV display: Direct window rendering (no VLC dependencies)
- Multi-mode support: Debug mode with camera overlay, clean projection mode
- Camera support: USB external and built-in laptop cameras
- Production controls: D=debug toggle, P=production mode, F=fullscreen, Q=quit
- State machine: 2-second scare duration with debounce logic
- Model: best.pt YOLO classification

**Architecture:** OpenCV window → YOLO classification → Direct video display

**Known Issue:** Grey border at top of OpenCV display
- Workaround: Use mirrored display or physical projector positioning

---

## 2025-09-22: VLC Projection System - Testing Complete

### Production Testing

**Test Results:**
- VLC integration: Working with python-vlc
- Hand detection: 99-100% confidence detection
- Video switching: Seamless idle and scare transitions
- Camera support: USB external and built-in functional
- USB camera fix: Initialization delays and retry logic added
- Video files: sleeping_face.mp4 and angry_face.mp4 configured
- State machine: 2s scare duration with automatic return

**Status:** Ready for projector deployment

---

## 2025-09-22: VLC Direct Projection Implementation

### Architecture Change

**Approach:** Direct video projection using python-vlc
- Eliminated external mapping software dependencies
- Simpler setup and cross-platform compatibility

**VLC Integration:**
- Direct video control via python-vlc
- Fullscreen projection with multi-display support
- Instant switching: sleeping_face.mp4 and angry_face.mp4
- Video looping until state change
- Debounce logic: 0.5s minimum between switches

**Technical Implementation:**
- VLCProjectionController class for video management
- State machine: idle → hand detected → scare → timeout → idle
- YOLO classification with 99% confidence threshold
- Cross-platform display detection (macOS, Windows, Linux)
- Universal video format support

**Code Structure:**
```python
def set_state(self, new_state):
    if new_state == "scare":
        self.play_video("videos/angry_face.mp4")
    else:
        self.play_video("videos/sleeping_face.mp4")

if class_name == 'hand' and confidence >= 0.99:
    controller.set_state("scare")
```

**Project Structure:**
- scripts/yolo_vlc_projection.py - Main VLC projection script
- test_vlc_playback.py - VLC testing and validation
- VLC_PROJECTION_SETUP.md - Setup documentation
- videos/ - Video content directory
- requirements.txt - Simplified dependencies

**Command Line:**
```bash
# Camera preview
python scripts/yolo_vlc_projection.py --show

# Fullscreen projection on display 1
python scripts/yolo_vlc_projection.py --fullscreen-display 1

# Custom videos
python scripts/yolo_vlc_projection.py --video-sleep idle.mp4 --video-scare scare.mp4

# System detection
python scripts/yolo_vlc_projection.py --list-cameras
python scripts/yolo_vlc_projection.py --list-displays
```

**Problems Solved:**
1. Complex mapping software: Eliminated HeavyM/VPT8 dependencies
2. Licensing costs: Free and open source
3. Setup complexity: Simple video file placement
4. Cross-platform issues: Universal VLC support

**Repository Changes:**
- Removed HeavyM dependencies: MIDI, OSC, mapping software
- Simplified requirements: ultralytics, opencv-python, python-vlc
- Streamlined codebase: Single projection approach
- VLC-focused documentation

---

## Legacy Development History

### 2025-09-21: HeavyM MIDI Integration

- MIDI bridge for HeavyM Demo compatibility
- Note 60 (C4) → sleepseq, Note 61 (C#4) → scareseq mapping
- macOS IAC Driver integration for virtual MIDI ports
- Solved HeavyM Demo OSC API limitations

### 2025-09-17: Camera Selection & Documentation

- Multi-camera detection and selection system
- Automatic fallback for failed cameras
- Enhanced error handling and troubleshooting guides

### 2025-09-12: YOLO Hand Detection Implementation

- Fine-tuned YOLO model integration (best.pt)
- 99% confidence threshold
- Real-time classification at 30+ FPS
- State machine: idle and scare with 2-second duration

---

## Migration Summary: Mapping Software to VLC Direct

**Migration Date:** 2025-09-22

**Reasons:**
- Eliminate complex mapping software dependencies
- Reduce setup time and configuration complexity
- Improve cross-platform compatibility
- Remove licensing and cost barriers

**Benefits:**
- Zero external dependencies (just VLC)
- Universal video format support
- Simplified deployment (drag-and-drop video files)
- Cost effective (completely free)
- Reliable operation (VLC stability)
- Easy maintenance (no complex configurations)

**Comparison:**

| Aspect | VLC Approach | Mapping Software |
|--------|-------------|------------------|
| Setup Time | 5 minutes | 30+ minutes |
| Dependencies | 3 Python packages | 10+ with MIDI/OSC |
| Cost | $0 | Potential licensing fees |
| Platforms | Mac/Windows/Linux | Platform-specific |
| Maintenance | Replace video files | Reconfigure mappings |

---

**Note:** The VLC direct projection approach represents a fundamental simplification of the system, prioritizing reliability, ease of use, and universal compatibility.
