# Xavier NX Dependency Research Findings

*Research completed: 2024-12-15*

## Current Xavier State (Reference)

| Component | Current Version |
|-----------|-----------------|
| JetPack | 5.0.2 (R35.1.0) |
| CUDA | 11.4 |
| TensorRT | 8.4.1 |
| Python | 3.8.10 |
| numpy | 1.17.4 |
| OpenCV | 4.5.4 |

---

## 1. PyTorch Compatibility

### Available Wheels for JetPack 5.0.2 (L4T R35.1.0)

| PyTorch | Torchvision | Download |
|---------|-------------|----------|
| **1.13.0** (recommended) | 0.14.0 | [torch-1.13.0a0+d0d6b1f2.nv22.10-cp38](https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl) |
| 1.12.0 | 0.13.0 | [torch-1.12.0a0+2c916ef.nv22.3-cp38](https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl) |

### Can We Use PyTorch 2.x?

**No.** PyTorch 2.x requires JetPack 6.x (CUDA 12.x). JetPack 5.0.2 is limited to PyTorch 1.x versions.

### Max Ultralytics Version for PyTorch 1.13

**Good news:** Ultralytics requires PyTorch >= 1.8, so **latest ultralytics (8.3.x) should work** with PyTorch 1.13.

However, models trained with PyTorch 2.x and exported with newer ONNX opsets may have compatibility issues during TensorRT conversion.

### Installation Commands

```bash
# Install PyTorch 1.13.0
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl

# Build torchvision 0.14.0 from source (required)
git clone --branch v0.14.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
```

**Sources:**
- [PyTorch for Jetson - NVIDIA Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [NVIDIA PyTorch Install Docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

---

## 2. ONNX Export Compatibility

### TensorRT 8.4.1 ONNX Support

| Feature | Support |
|---------|---------|
| **Max Opset** | 17 |
| **Backward Compatible** | Down to opset 9 |
| **GlobalAveragePool** | Supported (FP32, FP16, INT8) |
| **Reshape** | Supported (FP32, FP16, INT32, INT8, BOOL) |

### The Problem

Your error shows:
```
Model was exported with:
- PyTorch 2.9.1+cu128  # Training machine
- ONNX opset 18        # Too new for TensorRT 8.4.1
```

TensorRT 8.4.1 supports up to opset 17, so **opset 18 won't work directly**.

### Recommended Export Settings for Xavier

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="onnx",
    opset=12,          # Safe for TensorRT 8.4.1
    simplify=True,     # Run onnx-simplifier
    dynamic=False,     # Fixed input shape
    imgsz=224,         # Match training size
)
```

### ONNX Opset Downgrade Tools

**Option 1: ONNX Version Converter (Official)**
```python
import onnx
from onnx import version_converter

model = onnx.load("best.onnx")
converted = version_converter.convert_version(model, 12)  # Target opset 12
onnx.save(converted, "best_opset12.onnx")
```

**Option 2: Re-export with ultralytics on Xavier**
```bash
# On Xavier with PyTorch 1.13
pip install ultralytics
yolo export model=best.pt format=onnx opset=12 simplify=True
```

**Option 3: simple-onnx-processing-tools**
```bash
pip install simple-onnx-processing-tools[full]
# Provides various ONNX manipulation utilities
```

**Sources:**
- [TensorRT 8.4 Operator Support](https://github.com/onnx/onnx-tensorrt/blob/release/8.4-GA/docs/operators.md)
- [ONNX Version Converter](https://onnx.ai/onnx/repo-docs/VersionConverter.html)

---

## 3. TensorRT Conversion Issues

### "reshape changes volume" Error Analysis

```
Error: IShuffleLayer node_view: reshape changes volume.
Reshaping [1,1,1,1] to [1,1280].
```

**Root Causes:**
1. **Dynamic shape mismatch** - Model expects dynamic input but TensorRT needs fixed shapes
2. **GlobalAveragePool output shape** - After GAP, tensor is [1,C,1,1] but flatten expects [1,C]
3. **Opset incompatibility** - Higher opset operators not fully supported

### Workarounds

**1. Use Polygraphy for debugging:**
```bash
pip install polygraphy
polygraphy run best.onnx --onnxrt  # Verify ONNX is valid first
polygraphy surgeon sanitize best.onnx -o best_fixed.onnx --fold-constants
```

**2. Fix input shapes with Polygraphy:**
```bash
polygraphy surgeon sanitize best.onnx \
    --override-input-shapes images:[1,3,224,224] \
    -o best_fixed.onnx
```

**3. Re-export with explicit batch and fixed shapes:**
```python
model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,  # Critical: fixed shapes
    batch=1,
)
```

**4. Use TensorRT Python API** (more control than trtexec):
```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open("best.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_serialized_network(network, config)
with open("best.engine", "wb") as f:
    f.write(engine)
```

**Sources:**
- [TensorRT Reshape Issues](https://github.com/NVIDIA/TensorRT/issues/2245)
- [GlobalAveragePool Error](https://github.com/NVIDIA/TensorRT/issues/402)

---

## 4. Alternative Inference Backends

### Option A: ONNX Runtime GPU (Recommended Alternative)

**Availability:** Yes, but requires manual installation from Jetson Zoo.

```bash
# Download pre-built wheel for JetPack 5.x
wget https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl \
    -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl

pip3 install onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

**Usage:**
```python
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("best.onnx", providers=providers)

# Run inference
outputs = session.run(None, {"images": input_tensor})
```

**Pros:**
- More forgiving of ONNX variations
- Easier debugging
- Falls back to CPU for unsupported ops

**Cons:**
- Slower than TensorRT (but still GPU accelerated)
- May not support all TensorRT optimizations

### Option B: OpenCV DNN with CUDA

**Availability:** Requires rebuilding OpenCV from source.

The default JetPack OpenCV does **not** include CUDA DNN support. You must compile OpenCV with:
```bash
cmake -D WITH_CUDA=ON \
      -D OPENCV_DNN_CUDA=YES \
      -D CUDA_ARCH_BIN=7.2 \  # Xavier NX compute capability
      ...
```

**Usage (once rebuilt):**
```python
net = cv2.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

**Cons:**
- Significant rebuild effort
- Limited ONNX operator support
- Your error about `kernel_size not specified` suggests ONNX compatibility issues

### Option C: PyTorch Direct (.pt files)

**This is your current workaround and it's valid.**

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(frame)
```

**Pros:**
- Always works
- No export issues
- Full feature support

**Cons:**
- Slower than TensorRT (2-5x)
- Higher memory usage
- Requires full PyTorch installation

**Sources:**
- [Jetson Zoo - ONNX Runtime](https://elinux.org/Jetson_Zoo)
- [OpenCV DNN CUDA on Jetson](https://forums.developer.nvidia.com/t/how-can-i-use-dnn-module-of-opencv-at-gpu-on-jetson-xavier-nx/146321)

---

## 5. Safe Package Upgrades

### Upgrade Risk Assessment

| Package | Current | Target | Risk | Recommendation |
|---------|---------|--------|------|----------------|
| **numpy** | 1.17.4 | 1.24.x | MEDIUM | Use venv, test thoroughly |
| **pip** | 20.0.2 | 24.x | LOW | Safe to upgrade |
| **ultralytics** | - | 8.3.x | LOW | Works with PyTorch 1.13 |
| **onnx** | - | 1.15.x | LOW | Safe in venv |
| **onnxruntime-gpu** | - | 1.16.0 | LOW | Use Jetson Zoo wheel |

### NumPy Upgrade Caution

NumPy 1.24+ can cause issues with:
- TensorRT Python bindings
- Some PyTorch operations
- System packages expecting older numpy

**Recommendation:** Use a virtual environment:
```bash
python3 -m venv ~/ml_env
source ~/ml_env/bin/activate
pip install --upgrade pip
pip install numpy==1.24.3  # Test this version first
```

### DO NOT Upgrade

| Package | Reason |
|---------|--------|
| CUDA | Bundled with JetPack |
| TensorRT | System package, breaks everything |
| System Python | OS dependencies |
| cuDNN | Bundled with JetPack |

---

## 6. Recommended Approach

### Best Path Forward

1. **Export on Xavier** (not training machine):
   ```bash
   # On Xavier with PyTorch 1.13
   pip install ultralytics
   yolo export model=best.pt format=onnx opset=12 simplify=True
   ```

2. **Convert to TensorRT**:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
       --onnx=best.onnx \
       --fp16 \
       --saveEngine=best.engine
   ```

3. **If TensorRT fails, use ONNX Runtime**:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("best.onnx",
       providers=['CUDAExecutionProvider'])
   ```

### Fallback Options (in order of preference)

1. **TensorRT** - Fastest, but strict ONNX requirements
2. **ONNX Runtime GPU** - Good balance of speed and compatibility
3. **PyTorch .pt files** - Always works, slower but reliable

---

## Summary Matrix

| Question | Answer |
|----------|--------|
| PyTorch 2.x on JetPack 5.0.2? | **No** - max is 1.13.0 |
| Max ultralytics for PyTorch 1.13? | **8.3.x works** (requires >= 1.8) |
| TensorRT 8.4.1 max ONNX opset? | **17** (backwards to 9) |
| Can downgrade ONNX opset? | **Yes** - use version_converter |
| onnxruntime-gpu available? | **Yes** - Jetson Zoo wheel |
| OpenCV DNN CUDA? | **Requires rebuild** from source |
| Safe numpy upgrade? | **Use venv**, test 1.24.x carefully |

---

## Action Items

- [ ] Install PyTorch 1.13.0 wheel on Xavier
- [ ] Install ultralytics in venv on Xavier
- [ ] Re-export models with `opset=12, simplify=True`
- [ ] Test TensorRT conversion with fixed ONNX
- [ ] Install onnxruntime-gpu as fallback
- [ ] Update inference code to try TRT -> ORT -> PyTorch

---

*Sources referenced throughout document*
