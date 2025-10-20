# Halloween Hand Detection Projection

Real-time hand detection system with video projection for Halloween displays.

---

## Quick Start

### Platform-Specific Setup

**NVIDIA Jetson Xavier NX:**
See [envs/xavier/README.md](envs/xavier/README.md) for complete Xavier setup guide.

**macOS/Linux/Windows Development:**

**Install Git LFS** (required for model and video files):
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows: Download from git-lfs.github.io
```

### Setup

```bash
git clone https://github.com/12mv2/Halloween-Visions-Projection.git
cd Halloween-Visions-Projection
git lfs pull

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: Always activate venv before running: `source venv/bin/activate`

### Test Run

```bash
python simple_projection.py --source 0 --conf 0.5
```

Expected output:
- Window with camera feed
- Hand detection confidence values
- Video switching on hand wave

**Controls:**
- D = Toggle debug/clean mode
- F = Fullscreen
- Q or ESC = Quit

---

## Usage

### Basic Operation

```bash
# Standard operation
python simple_projection.py --source 0 --conf 0.7

# Adjust sensitivity
python simple_projection.py --source 0 --conf 0.3  # More sensitive
python simple_projection.py --source 0 --conf 0.9  # Less sensitive

# Custom videos
python simple_projection.py --video-sleep IDLE.mp4 --video-scare SCARE.mp4

# Fullscreen mode
python simple_projection.py --source 0 --conf 0.5 --fullscreen
```

### Command Line Options

```
--source N          Camera index (0=default, 1=alternate)
--conf X.X          Detection confidence threshold (0.0-1.0)
--video-sleep PATH  Custom idle video
--video-scare PATH  Custom scare video
--fullscreen        Start in fullscreen mode
```

---

## System Architecture

**Components:**
1. Camera captures live video
2. YOLO AI model classifies frames (hand/no hand)
3. Video player switches between idle and scare content
4. Display outputs to projector

**Performance:**
- Model: YOLOv8 nano classification (Colin1.pt)
- Classes: {0: 'hand', 1: 'not_hand'}
- Processing: 30+ FPS real-time
- Scare duration: 2 seconds with debounce

**Video System:**
- OpenCV-based playback
- Instant video switching
- Supports standard video formats
- Fullscreen projection ready

---

## System Requirements

**Platform:**
- macOS/Linux/Windows
- Python 3.8+
- Git LFS

**Hardware:**
- USB or built-in camera
- Projector or external display

**Dependencies:**
- ultralytics >= 8.0.0 (YOLO inference)
- opencv-python >= 4.0.0 (video processing)
- python-vlc >= 3.0.0 (legacy support)

**Production Setup:**
- USB camera recommended (works with laptop lid closed)
- Detection range: 3-6 feet optimal
- Adequate lighting required for accuracy
- Mirrored display mode supported

---

## Troubleshooting

**Camera not detected:**
```bash
# Try alternate camera
python simple_projection.py --source 1

# Check permissions (macOS: System Preferences → Privacy → Camera)
```

**Insufficient sensitivity:**
```bash
# Lower confidence threshold
python simple_projection.py --source 0 --conf 0.3
```

**Videos not loading:**
```bash
# Verify Git LFS pulled files
ls -lh *.mp4
git lfs pull
```

**Known Issue:** Grey border at display top (macOS OpenCV limitation)
- Workaround: Position projector to crop border area

---

## Production Status

**Current:** Production Ready
- Hand detection: Accurate and responsive
- Video switching: Smooth and instant
- Cross-platform: macOS/Linux/Windows tested
- Multi-camera: USB + built-in supported

**Platform Notes:**
- macOS: Tested on Darwin 24.6.0
- Linux: Tested on Jetson Xavier NX (JetPack 5.0.2)
- Windows: Standard Python 3.8+ environment

**Platform-Specific Guides:**
- Xavier NX Deployment: [envs/xavier/README.md](envs/xavier/README.md)
- Xavier SSH Setup: [envs/xavier/SSH_SETUP.md](envs/xavier/SSH_SETUP.md)

---

## Project Structure

```
Halloween-Visions-Projection/
├── simple_projection.py          # Main application
├── test_gpu_validation.py        # GPU validation utility
├── test_camera.py                # Camera testing
├── Colin1.pt                     # Production model
├── quinn_arms_up.pt              # Alternative model
├── videos/                       # Video assets
├── envs/                         # Platform-specific environments
│   └── xavier/                   # Xavier NX deployment files
│       ├── README.md             # Xavier setup guide
│       ├── SSH_SETUP.md          # SSH configuration
│       ├── requirements.txt      # Pinned dependencies
│       └── install.sh            # Setup script
└── [documentation files]
```

---

**Documentation:**
- See CHANGELOG.md for version history
- See MAINTENANCE.md for deployment details
- See CONTRIBUTING.md for development guidelines
