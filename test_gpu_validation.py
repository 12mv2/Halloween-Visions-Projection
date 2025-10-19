#!/usr/bin/env python3
"""
B10-INFER-GPU-VALIDATE: GPU Validation Test Script
Tests GPU inference without requiring camera/video setup
"""

import torch
import numpy as np
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def test_cuda_availability():
    """Test 1: CUDA Availability"""
    logger.info("=" * 60)
    logger.info("TEST 1: CUDA Availability")
    logger.info("=" * 60)

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        logger.error("❌ CUDA not available - cannot run GPU validation")
        return False


def test_model_loading():
    """Test 2: Model Loading"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Model Loading")
    logger.info("=" * 60)

    model_path = Path("Colin1.pt")
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Looking for YOLO models in current directory...")
        pt_files = list(Path(".").glob("*.pt"))
        if pt_files:
            logger.info(f"Found models: {[str(f) for f in pt_files]}")
            model_path = pt_files[0]
            logger.info(f"Using: {model_path}")
        else:
            return None

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        logger.info(f"✅ Model loaded: {model_path}")
        logger.info(f"Model classes: {model.names}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def test_gpu_inference(model):
    """Test 3: GPU Inference"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: GPU Inference")
    logger.info("=" * 60)

    # Create synthetic test image (OpenCV BGR format)
    import cv2
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Convert BGR to RGB and normalize (HWC -> CHW, 0-1 range)
    frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # Resize to model input size (224x224 for classify models)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    # Convert to tensor: HWC -> CHW, uint8 -> float32, 0-255 -> 0-1
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    # Add batch dimension and move to GPU
    frame_tensor = frame_tensor.unsqueeze(0).to("cuda")
    logger.info(f"✅ Test frame preprocessed: {frame_tensor.shape}, device={frame_tensor.device}")

    # Force model to GPU
    model.to("cuda")
    logger.info("✅ Model moved to CUDA device")

    # Run inference with pre-processed tensor (bypass Ultralytics transforms)
    try:
        # Direct model inference bypassing predictor
        with torch.no_grad():
            output = model.model(frame_tensor)

        # Convert output to probabilities
        probs = torch.nn.functional.softmax(output, dim=1)
        logger.info("✅ Inference completed (direct model call)")

        # Check for NaNs
        has_nan = not torch.isfinite(probs).all()
        if has_nan:
            logger.error("❌ NaN detected in inference output!")
            logger.error(f"   Probs: {probs}")
            return False
        else:
            logger.info("✅ No NaNs detected in inference output")
            logger.info(f"   Probs shape: {probs.shape}")
            logger.info(f"   Probs device: {probs.device}")
            logger.info(f"   Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
            logger.info(f"   Predicted class: {probs.argmax(dim=1).item()}")
            return True

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark(model, num_frames=100):
    """Test 4: Benchmark FPS"""
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 4: Benchmark ({num_frames} frames)")
    logger.info("=" * 60)

    import cv2

    # Create synthetic test image (OpenCV BGR format)
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Convert BGR to RGB and preprocess
    frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).to("cuda")

    model.to("cuda")

    frame_times = []
    nan_count = 0

    # Warmup
    logger.info("Running 10 warmup frames...")
    for _ in range(10):
        with torch.no_grad():
            output = model.model(frame_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

    logger.info(f"Running {num_frames} benchmark frames...")
    for i in range(num_frames):
        start = time.time()
        with torch.no_grad():
            output = model.model(frame_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
        duration = time.time() - start
        frame_times.append(duration)

        # Check for NaNs
        has_nan = not torch.isfinite(probs).all()
        if has_nan:
            nan_count += 1

        if (i + 1) % 25 == 0:
            logger.info(f"  Progress: {i + 1}/{num_frames} frames")

    # Calculate statistics
    frame_times = np.array(frame_times)
    mean_duration = np.mean(frame_times)
    std_duration = np.std(frame_times)
    mean_fps = 1.0 / mean_duration if mean_duration > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("📊 BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total frames: {num_frames}")
    logger.info(f"Mean FPS: {mean_fps:.2f}")
    logger.info(f"Mean frame time: {mean_duration * 1000:.2f} ms")
    logger.info(f"Std frame time: {std_duration * 1000:.2f} ms")
    logger.info(f"Min frame time: {frame_times.min() * 1000:.2f} ms")
    logger.info(f"Max frame time: {frame_times.max() * 1000:.2f} ms")
    logger.info(f"NaN occurrences: {nan_count}")
    logger.info(f"NaN rate: {(nan_count / num_frames) * 100:.2f}%")

    # Pass/Fail
    passed = mean_fps >= 15 and nan_count == 0
    status = "✅ PASS" if passed else "⚠️  FAIL"
    logger.info(f"\nStatus: {status}")

    if mean_fps < 15:
        logger.warning(f"  FPS below target (got {mean_fps:.2f}, expected ≥15)")
    if nan_count > 0:
        logger.warning(f"  NaN outputs detected ({nan_count}/{num_frames})")

    logger.info("=" * 60)

    return passed


def main():
    """Run all GPU validation tests"""
    logger.info("\n" + "🎃" * 30)
    logger.info("B10-INFER-GPU-VALIDATE: GPU Validation Test Suite")
    logger.info("🎃" * 30 + "\n")

    # Test 1: CUDA
    if not test_cuda_availability():
        logger.error("\n❌ VALIDATION FAILED: CUDA not available")
        return 1

    # Test 2: Model Loading
    model = test_model_loading()
    if model is None:
        logger.error("\n❌ VALIDATION FAILED: Could not load model")
        return 1

    # Test 3: GPU Inference
    if not test_gpu_inference(model):
        logger.error("\n❌ VALIDATION FAILED: GPU inference test failed")
        return 1

    # Test 4: Benchmark (B10-PROJ-INFER-REFIT: Extended to 1000 frames)
    if not test_benchmark(model, num_frames=1000):
        logger.warning("\n⚠️  VALIDATION WARNING: Benchmark did not meet all targets")
        # Don't fail - this is informational

    logger.info("\n" + "=" * 60)
    logger.info("✅ GPU VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run simple_projection.py with real camera/video")
    logger.info("2. Check inference_diagnostics.log for detailed metrics")
    logger.info("3. Verify FPS ≥ 15 and NaN count = 0")
    logger.info("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
