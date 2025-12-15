#!/usr/bin/env python3
"""
ONNX Runtime inference for YOLOv8 classification models.

This module provides a clean interface for running inference on ONNX-exported
YOLOv8 classifiers. It handles preprocessing (resize, normalize) and
postprocessing (softmax, top-k) matching Ultralytics conventions.

Usage:
    from inference.onnx_infer import YOLOClassifier

    model = YOLOClassifier("models/production/7class_v1/best.onnx")
    pred_class, confidence = model.predict(frame)
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import onnxruntime as ort


class YOLOClassifier:
    """ONNX Runtime backend for YOLOv8 classification models."""

    def __init__(
        self,
        model_path: str,
        classes_file: Optional[str] = None,
        input_size: int = 224,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to ONNX model file
            classes_file: Path to classes.txt (auto-detected if None)
            input_size: Model input size (default 224 for YOLOv8n-cls)
            providers: ONNX Runtime execution providers (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size

        # Auto-detect classes file
        if classes_file is None:
            classes_file = self.model_path.parent / "classes.txt"
        self.classes = self._load_classes(classes_file)

        # Select providers
        if providers is None:
            providers = self._get_available_providers()

        # Load model
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape from model
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] is not None:
            self.input_size = input_shape[2]

    def _load_classes(self, classes_file: Path) -> List[str]:
        """Load class names from file."""
        if classes_file.exists():
            with open(classes_file) as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback: numbered classes
        return [f"class_{i}" for i in range(1000)]

    def _get_available_providers(self) -> List[str]:
        """Get best available execution providers."""
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available]

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

        return img

    def postprocess(self, output: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Postprocess model output.

        Args:
            output: Raw model output (1, num_classes)

        Returns:
            (top_class, top_confidence, all_predictions)
        """
        probs = output[0]

        # Apply softmax if not already applied (check if sums to ~1)
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

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Postprocess
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
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        _, _, all_preds = self.postprocess(output)
        return all_preds


def main():
    """Demo: run inference on webcam."""
    import argparse

    parser = argparse.ArgumentParser(description="ONNX classifier demo")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    classifier = YOLOClassifier(args.model)
    print(f"Classes: {classifier.classes}")
    print(f"Input size: {classifier.input_size}")
    print(f"Providers: {classifier.session.get_providers()}")

    # Open camera
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.source}")

    print("\nPress 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)

        # Predict
        pred_class, confidence = classifier.predict(frame)

        # Draw result
        color = (0, 255, 0) if confidence >= args.conf else (128, 128, 128)
        text = f"{pred_class}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("ONNX Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
