# Research Ticket: Xavier NX Dependency Upgrade Options

## Context

The Xavier NX (JetPack 5.0.2, R35.1.0) has fragile dependencies. We need to understand what upgrades are safe and what the compatibility matrix looks like.

## Current Xavier State

| Component | Current Version | Notes |
|-----------|-----------------|-------|
| JetPack | 5.0.2 (R35.1.0) | Don't upgrade OS |
| CUDA | 11.4 | Bundled with JetPack |
| TensorRT | 8.4.1 | System package |
| Python | 3.8.10 | System Python |
| numpy | 1.17.4 | System package |
| OpenCV | 4.5.4 | System package |
| pip | 20.0.2 | System package |

## Research Questions

### 1. PyTorch Compatibility
- [ ] What PyTorch versions work with JetPack 5.0.2?
- [ ] Can we use PyTorch 2.x or only 1.x?
- [ ] What's the max ultralytics version that works with PyTorch 1.13?

### 2. ONNX Export Compatibility
- [ ] What ONNX opset versions does TensorRT 8.4.1 support?
- [ ] Can we export YOLOv8 models with opset 11-13 from newer ultralytics?
- [ ] Are there onnx-simplifier tools that can downgrade opset?

### 3. TensorRT Conversion
- [ ] Why does `trtexec` fail with "reshape changes volume" error?
- [ ] Is there a workaround for the GlobalAveragePool → Reshape issue?
- [ ] Can we use TensorRT Python API instead of trtexec?

### 4. Alternative Inference Backends
- [ ] Is onnxruntime-gpu available for aarch64 + CUDA 11.4?
- [ ] Can OpenCV DNN use CUDA backend on Xavier?
- [ ] What about TensorFlow Lite as an alternative?

### 5. Safe Upgrade Path
- [ ] Can numpy be safely upgraded to 1.24.x?
- [ ] What other packages can be upgraded without breaking CUDA/TensorRT?
- [ ] Should we use a venv to isolate ML packages from system?

## Error Logs

### TensorRT ONNX Parsing Error
```
[E] Error[4]: [graphShapeAnalyzer.cpp::analyzeShapes::1294] Error Code 4: Miscellaneous
(IShuffleLayer node_view: reshape changes volume. Reshaping [1,1,1,1] to [1,1280].)

Model was exported with:
- PyTorch 2.9.1+cu128
- ONNX opset 18
- ultralytics 8.3.x
```

### OpenCV DNN ONNX Error
```
[ERROR:0] global .../onnx_importer.cpp (718) handleNode DNN/ONNX: ERROR during
processing node with 3 inputs and 1 outputs: [Conv]:(conv2d)
error: (-5:Bad argument) kernel_size (or kernel_h and kernel_w) not specified
```

## Resources

- [PyTorch for Jetson - NVIDIA Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [NVIDIA Install PyTorch Docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [TensorRT ONNX Support Matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- [JetPack 5.0.2 Release Notes](https://developer.nvidia.com/embedded/jetpack-sdk-502)

## Deliverables

1. Compatibility matrix for PyTorch/ultralytics/ONNX versions
2. Recommended export settings for Xavier-compatible ONNX
3. List of safe package upgrades (if any)
4. Alternative inference approach if TensorRT won't work

## Priority

Medium - Current workaround is using PyTorch directly with .pt files

---

*Created: 2024-12-15*
