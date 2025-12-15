#!/usr/bin/env python3
"""
TensorRT inference for YOLOv8 classification models.

This module provides a TensorRT backend for YOLOv8 classifiers on NVIDIA Jetson
devices. It shares the same interface as onnx_infer.py for drop-in replacement.

Requirements (Jetson):
    - JetPack 5.x with TensorRT
    - tensorrt Python bindings
    - pycuda

Engine creation (on Jetson):
    /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --fp16 --saveEngine=best.engine

Usage:
    from inference.tensorrt_infer import YOLOClassifier

    model = YOLOClassifier("models/production/7class_v1/best.engine")
    pred_class, confidence = model.predict(frame)
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np

# TensorRT imports - only available on Jetson/NVIDIA systems
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None


class YOLOClassifier:
    """TensorRT backend for YOLOv8 classification models."""

    def __init__(
        self,
        model_path: str,
        classes_file: Optional[str] = None,
        input_size: int = 224,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to TensorRT engine file (.engine)
            classes_file: Path to classes.txt (auto-detected if None)
            input_size: Model input size (default 224 for YOLOv8n-cls)
        """
        if not TRT_AVAILABLE:
            raise ImportError(
                "TensorRT not available. This module requires:\n"
                "  - NVIDIA Jetson with JetPack 5.x\n"
                "  - tensorrt and pycuda Python packages\n"
                "Use inference.onnx_infer.YOLOClassifier instead for CPU/ONNX inference."
            )

        self.model_path = Path(model_path)
        self.input_size = input_size

        # Auto-detect classes file
        if classes_file is None:
            classes_file = self.model_path.parent / "classes.txt"
        self.classes = self._load_classes(classes_file)

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

    def _load_classes(self, classes_file: Path) -> List[str]:
        """Load class names from file."""
        if classes_file.exists():
            with open(classes_file) as f:
                return [line.strip() for line in f if line.strip()]
        return [f"class_{i}" for i in range(1000)]

    def _load_engine(self):
        """Load TensorRT engine from file."""
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """Allocate GPU memory for input/output."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # Calculate size
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
                # Update input size from engine
                if len(shape) >= 3:
                    self.input_size = shape[2]
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.

        Args:
            frame: BGR image (HWC format) from OpenCV

        Returns:
            Preprocessed tensor (1, 3, H, W) float32
        """
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, 0)

        return np.ascontiguousarray(img)

    def postprocess(self, output: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Postprocess model output.

        Args:
            output: Raw model output (1, num_classes)

        Returns:
            (top_class, top_confidence, all_predictions)
        """
        probs = output.flatten()

        # Apply softmax if not already applied
        if not np.isclose(probs.sum(), 1.0, atol=0.1):
            probs = self._softmax(probs)

        # Get all predictions sorted by confidence
        sorted_indices = np.argsort(probs)[::-1]
        all_preds = [(self.classes[i], float(probs[i])) for i in sorted_indices]

        # Top prediction
        top_idx = sorted_indices[0]
        top_class = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        return top_class, top_conf, all_preds

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image from OpenCV (any size)

        Returns:
            (predicted_class, confidence) where confidence is 0-1
        """
        # Preprocess
        input_tensor = self.preprocess(frame)

        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output back
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        # Postprocess
        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        pred_class, confidence, _ = self.postprocess(output)

        return pred_class, confidence

    def predict_all(self, frame: np.ndarray) -> List[Tuple[str, float]]:
        """
        Run inference and return all class probabilities.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of (class_name, confidence) sorted by confidence descending
        """
        input_tensor = self.preprocess(frame)

        np.copyto(self.inputs[0]["host"], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        _, _, all_preds = self.postprocess(output)
        return all_preds

    def __del__(self):
        """Clean up CUDA resources."""
        if hasattr(self, "stream"):
            self.stream.synchronize()


def main():
    """Demo: run inference on webcam with TensorRT."""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT classifier demo")
    parser.add_argument("--model", type=str, required=True, help="Path to TensorRT engine")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()

    if not TRT_AVAILABLE:
        print("TensorRT not available on this system.")
        print("This script is designed for NVIDIA Jetson devices.")
        return

    # Load model
    print(f"Loading TensorRT engine: {args.model}")
    classifier = YOLOClassifier(args.model)
    print(f"Classes: {classifier.classes}")
    print(f"Input size: {classifier.input_size}")

    # Open camera
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.source}")

    print("\nPress 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        pred_class, confidence = classifier.predict(frame)

        color = (0, 255, 0) if confidence >= args.conf else (128, 128, 128)
        text = f"{pred_class}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("TensorRT Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
