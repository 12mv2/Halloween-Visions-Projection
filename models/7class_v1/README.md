# 7-Class Object Classifier v1

**Source**: `runs/classify/train7`
**Date**: 2024-12-14
**Architecture**: YOLOv8n-cls
**Input Size**: 224x224

## Classes

| Index | Class Name    | Description              |
|-------|---------------|--------------------------|
| 0     | 9v_battery    | 9V battery               |
| 1     | background    | No object detected       |
| 2     | black_spool   | 3D printer filament spool|
| 3     | blue_floppy   | Blue floppy disk         |
| 4     | green_spool   | Green sewing thread spool|
| 5     | hammer        | Hammer                   |
| 6     | hand          | Human hand               |

## Files

- `best.pt` - PyTorch weights (dev/training)
- `best.onnx` - ONNX format (cross-platform inference)
- `classes.txt` - Class names in index order

## Usage

```bash
# PyTorch inference
yolo predict model=models/production/7class_v1/best.pt source=image.jpg

# ONNX inference
python inference/onnx_infer.py --model models/production/7class_v1/best.onnx --source 0
```

## Export to TensorRT (Xavier NX)

```bash
# On Xavier NX
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine
```
