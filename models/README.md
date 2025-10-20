# YOLO Model Files

This directory contains trained YOLOv8 classification models for hand detection.

## Available Models

### Colin1.pt (Production Model - Default)
- **Type:** YOLOv8 nano classification
- **Classes:** {0: 'hand', 1: 'not_hand'}
- **Size:** 2.9 MB
- **Performance:** 65 FPS inference-only, 30 FPS full pipeline
- **Training:** Custom trained on hand detection dataset
- **Usage:** Default model used by simple_projection.py

### quinn_arms_up.pt (Alternative Model)
- **Type:** YOLOv8 nano classification
- **Classes:** {0: 'hand', 1: 'not_hand'}
- **Size:** 2.9 MB
- **Training:** Alternative training with different dataset
- **Usage:** `python3 simple_projection.py --model models/quinn_arms_up.pt`

---

## Usage

**Default model (Colin1.pt):**
```bash
python3 simple_projection.py
```

**Alternative model:**
```bash
python3 simple_projection.py --model models/quinn_arms_up.pt
```

**Custom model:**
```bash
python3 simple_projection.py --model models/your_model.pt
```

---

## Model Format

- **Format:** PyTorch (.pt)
- **Framework:** Ultralytics YOLOv8
- **Architecture:** YOLOv8 nano classification
- **Input Size:** 224x224 RGB
- **Output:** 2-class probabilities (hand, not_hand)

---

## Training New Models

To train a new hand detection model:

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n-cls.pt')

# Train on your dataset
model.train(
    data='/path/to/dataset',
    epochs=100,
    imgsz=224,
    batch=32
)

# Save trained model
model.save('models/new_model.pt')
```

---

## Performance Benchmarks

**Tested on NVIDIA Jetson Xavier NX (JetPack 5.0.2):**

| Model | Inference FPS | Full Pipeline FPS | GPU Usage |
|-------|---------------|-------------------|-----------|
| Colin1.pt | 65 FPS | 30 FPS | <30% |
| quinn_arms_up.pt | 65 FPS | 30 FPS | <30% |

**Full pipeline includes:**
- Camera capture (30 FPS camera limit)
- Inference (65 FPS capable)
- Video switching
- Display output

---

## Git LFS

Model files are tracked with Git LFS due to their size:

```bash
# Ensure Git LFS is installed
git lfs install

# Pull model files
git lfs pull
```

---

**Note:** Model files are stored in this directory to keep the project root clean and organized.
