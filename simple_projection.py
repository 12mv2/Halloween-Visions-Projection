#!/usr/bin/env python3
"""
Simple Halloween Hand Detection Projection
- Direct OpenCV window display (no VLC needed)
- Toggle between debug mode and fullscreen projection
- Mirror display setup friendly
"""

import argparse
import time
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_formatter)

# File handler for diagnostics
file_handler = logging.FileHandler("inference_diagnostics.log", mode="w")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Keep backward compatibility for existing logging calls
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[console_handler, file_handler]
)


class SimpleProjectionController:
    def __init__(self, video_sleep_path, video_scare_path):
        self.video_sleep_path = video_sleep_path
        self.video_scare_path = video_scare_path
        self.state = "idle"
        self.confidence_threshold = 0.7
        self.scare_duration = 2.0
        self.last_trigger = 0.0
        self.debug_mode = True
        self.production_mode = False

        # B10-INFER-GPU-VALIDATE: FPS tracking and NaN detection
        self.fps_buffer = deque(maxlen=300)  # Store last 300 frame times
        self.nan_count = 0
        self.total_frames = 0
        self.warmup_frames = 30  # Skip first 30 frames for benchmark
        self.benchmark_complete = False
        self.device_verified = False

        # Load videos
        self.sleep_cap = cv2.VideoCapture(video_sleep_path)
        self.scare_cap = cv2.VideoCapture(video_scare_path)

        if not self.sleep_cap.isOpened():
            raise Exception(f"Could not open sleep video: {video_sleep_path}")
        if not self.scare_cap.isOpened():
            raise Exception(f"Could not open scare video: {video_scare_path}")

        # Get video properties
        self.sleep_fps = self.sleep_cap.get(cv2.CAP_PROP_FPS)
        self.scare_fps = self.scare_cap.get(cv2.CAP_PROP_FPS)

        self.sleep_frame_count = 0
        self.scare_frame_count = 0

        logging.info("Videos loaded successfully")
        logging.info(f"Sleep video: {video_sleep_path} ({self.sleep_fps:.1f} FPS)")
        logging.info(f"Scare video: {video_scare_path} ({self.scare_fps:.1f} FPS)")

    def get_current_video_frame(self):
        """Return current frame based on state."""
        if self.state == "scare":
            ret, frame = self.scare_cap.read()
            if not ret:
                self.scare_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.scare_cap.read()
            return frame
        else:
            ret, frame = self.sleep_cap.read()
            if not ret:
                self.sleep_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.sleep_cap.read()
            return frame

    def process_hand_detection(self, class_name, confidence):
        """Handle hand detection transitions."""
        current_time = time.time()

        if class_name == "hand" and confidence >= self.confidence_threshold:
            if self.state != "scare":
                logging.info(f"HAND DETECTED! Confidence: {confidence:.1%}")
                logging.info("Switching to SCARE state")
                self.state = "scare"
                self.last_trigger = current_time

        # Return to idle after timeout
        if self.state == "scare" and current_time - self.last_trigger > self.scare_duration:
            logging.info("SCARE timeout, returning to IDLE")
            self.state = "idle"

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        mode = "DEBUG" if self.debug_mode else "PROJECTION"
        logging.info(f"Switched to {mode} mode")
        return self.debug_mode

    def toggle_production_mode(self):
        self.production_mode = not self.production_mode
        mode = "PRODUCTION" if self.production_mode else "NORMAL"
        logging.info(f"Switched to {mode} display mode")
        return self.production_mode

    def create_debug_display(self, camera_frame, video_frame, class_name, confidence, model_name="Colin1.pt"):
        cam_h, cam_w = camera_frame.shape[:2]
        scale = 0.3
        small_cam = cv2.resize(camera_frame, (int(cam_w * scale), int(cam_h * scale)))

        display_h, display_w = camera_frame.shape[:2]
        display = cv2.resize(video_frame, (display_w, display_h))

        cam_h_small, cam_w_small = small_cam.shape[:2]
        y_offset = 20
        x_offset = display_w - cam_w_small - 20

        cv2.rectangle(display,
                      (x_offset - 5, y_offset - 5),
                      (x_offset + cam_w_small + 5, y_offset + cam_h_small + 5),
                      (255, 255, 255), 2)
        display[y_offset:y_offset + cam_h_small, x_offset:x_offset + cam_w_small] = small_cam

        info_y = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        state_color = (0, 0, 255) if self.state == "scare" else (0, 255, 0)
        cv2.putText(display, f"State: {self.state.upper()}", (20, info_y),
                    font, font_scale, state_color, thickness)

        detection_color = (0, 255, 255) if class_name == "hand" else (255, 255, 255)
        cv2.putText(display, f"Detection: {class_name} ({confidence:.1%})",
                    (20, info_y + 50), font, font_scale, detection_color, thickness)

        cv2.putText(display, f"Threshold: {self.confidence_threshold:.0%}",
                    (20, info_y + 100), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(display, f"Model: {model_name}",
                    (20, info_y + 150), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(display, "Press 'D' to toggle debug/projection mode",
                    (20, display_h - 120), font, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 'P' for production mode (grey fix)",
                    (20, display_h - 90), font, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 'F' for fullscreen",
                    (20, display_h - 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 'Q' or ESC to quit",
                    (20, display_h - 30), font, 0.7, (255, 255, 255), 2)

        return display

    def create_production_display(self, video_frame):
        h, w = video_frame.shape[:2]
        stretched_h = int(h * 1.1)
        production_frame = cv2.resize(video_frame, (w, stretched_h))
        crop_start = (stretched_h - h) // 2
        return production_frame[crop_start:crop_start + h, :]

    def preprocess_frame_to_tensor(self, camera_frame, device="cuda"):
        """
        B10-PROJ-INFER-REFIT: OpenCV-only preprocessing for direct tensor inference
        Converts BGR camera frame to GPU tensor ready for model input
        Returns: tensor [1, 3, 224, 224] on specified device
        """
        # Convert BGR to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size (224x224 for classify models)
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Convert to tensor: HWC -> CHW, uint8 -> float32, 0-255 -> 0-1
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

        # Add batch dimension and move to GPU
        frame_tensor = frame_tensor.unsqueeze(0).to(device)

        return frame_tensor

    def check_nan_in_probs(self, probs_tensor):
        """
        B10-INFER-GPU-VALIDATE Objective 2: Detect residual NaNs in model outputs
        Returns: (has_nan, probs_array)
        """
        if probs_tensor is None:
            return False, None

        # Convert to tensor if needed
        if isinstance(probs_tensor, np.ndarray):
            probs_tensor = torch.from_numpy(probs_tensor)

        has_nan = not torch.isfinite(probs_tensor).all()

        if has_nan:
            self.nan_count += 1
            logger.warning(f"NaN detected in probs (total NaN count: {self.nan_count})")

            # Abort if too many NaNs
            if self.nan_count > 5 and self.total_frames < 1000:
                logger.error(f"Exceeded NaN threshold: {self.nan_count} NaNs in {self.total_frames} frames")
                logger.error("Aborting due to excessive NaN outputs - GPU inference may be failing")
                raise RuntimeError("Excessive NaN outputs detected - check GPU inference path")

        return has_nan, probs_tensor.cpu().numpy() if isinstance(probs_tensor, torch.Tensor) else probs_tensor

    def record_frame_time(self, frame_start_time):
        """
        B10-INFER-GPU-VALIDATE Objective 3: Benchmark baseline FPS
        """
        frame_duration = time.time() - frame_start_time
        self.total_frames += 1

        # Skip warmup frames for accurate benchmarking
        if self.total_frames > self.warmup_frames:
            self.fps_buffer.append(frame_duration)

        # Report benchmark after 300 frames (post-warmup)
        if len(self.fps_buffer) == 300 and not self.benchmark_complete:
            self.benchmark_complete = True
            durations = np.array(self.fps_buffer)
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            mean_fps = 1.0 / mean_duration if mean_duration > 0 else 0
            std_fps = std_duration / (mean_duration ** 2) if mean_duration > 0 else 0

            logger.info("=" * 60)
            logger.info("B10-INFER-GPU-VALIDATE Benchmark Results (300 frames)")
            logger.info(f"   Warmup frames skipped: {self.warmup_frames}")
            logger.info(f"   Mean FPS: {mean_fps:.2f} +/- {std_fps:.2f}")
            logger.info(f"   Mean frame time: {mean_duration*1000:.2f} ms")
            logger.info(f"   NaN occurrences: {self.nan_count}")
            logger.info(f"   GPU inference: {'PASS' if mean_fps >= 15 else 'BELOW TARGET'}")
            logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Simple Halloween Hand Detection Projection")
    parser.add_argument("--model", default="Colin1.pt", help="YOLO model file")
    parser.add_argument("--source", default=0, help="Camera index")
    parser.add_argument("--video-sleep", default="videos/sleeping_face.mp4", help="Sleep video")
    parser.add_argument("--video-scare", default="videos/angry_face.mp4", help="Scare video")
    parser.add_argument("--conf", type=float, default=0.7, help="Hand detection confidence threshold")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("Simple Halloween Hand Detection Projection")
    logging.info("=" * 60)

    # B10-INFER-GPU-VALIDATE Objective 1: Confirm GPU Execution
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    if cuda_available:
        logger.info(f"GPU enabled: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.info(f"   PyTorch version: {torch.__version__}")
        device = "cuda"
    else:
        logger.warning("CUDA not available - falling back to CPU (NaNs may occur)")
        device = "cpu"

    try:
        logging.info(f"Loading YOLO model: {args.model}")
        model = YOLO(args.model)
        
        # Force model to GPU if available
        if cuda_available:
            model.to(device)
            logger.info(f"Model moved to device: {device}")

        logging.info(f"Model loaded: {model.names}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return 1

    try:
        controller = SimpleProjectionController(args.video_sleep, args.video_scare)
        controller.confidence_threshold = args.conf
    except Exception as e:
        logging.error(f"Failed to initialize controller: {e}")
        return 1

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"Could not open camera: {source}")
        return 1

    logging.info(f"Camera opened: {args.source}")
    logging.info(f"Confidence threshold: {args.conf:.0%}")
    logging.info("Controls:")
    logging.info("  D = Toggle Debug/Projection mode")
    logging.info("  P = Toggle Production mode (grey border fix)")
    logging.info("  F = Toggle fullscreen")
    logging.info("  Q/ESC = Quit")
    logging.info("-" * 60)

    window_name = "Halloween Projection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            frame_start_time = time.time()  # B10-INFER-GPU-VALIDATE: FPS tracking
            
            ret, camera_frame = cap.read()
            if not ret:
                logging.warning("Failed to read camera frame")
                continue

            video_frame = controller.get_current_video_frame()
            if video_frame is None:
                continue

            cam_h, cam_w = camera_frame.shape[:2]
            extended_h = int(cam_h * 1.15)
            video_frame = cv2.resize(video_frame, (cam_w, extended_h))
            video_frame = video_frame[:cam_h, :]

            # B10-PROJ-INFER-REFIT: Direct tensor inference (no file I/O, no PIL)
            # Preprocess frame to GPU tensor
            frame_tensor = controller.preprocess_frame_to_tensor(camera_frame, device=device)

            # Run direct inference bypassing Ultralytics predictor
            with torch.no_grad():
                output = model.model(frame_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)

            # B10-INFER-GPU-VALIDATE: Verify device on first inference
            if not controller.device_verified:
                controller.device_verified = True
                actual_device = probs.device
                logger.info(f"First inference device verified: {actual_device}")
                logger.info(f"   Tensor shape: {probs.shape}")
                if str(actual_device) != device and cuda_available:
                    logger.warning(f"Expected {device} but got {actual_device}")

            # B10-INFER-GPU-VALIDATE Objective 2: NaN detection
            has_nan, probs_np = controller.check_nan_in_probs(probs)

            # Extract classification results
            class_name = "not_hand"
            confidence = 0.0
            if probs_np is not None and len(probs_np.shape) > 0:
                # probs_np is [1, 2] shape, flatten to get class probs
                probs_flat = probs_np.flatten()
                max_idx = probs_flat.argmax()
                confidence = float(probs_flat[max_idx])
                class_name = model.names[int(max_idx)]
            else:
                class_name = "not_hand"
                confidence = 0.5

            if not hasattr(controller, "frame_count"):
                controller.frame_count = 0
            controller.frame_count += 1
            if controller.frame_count % 30 == 0:
                logging.info(f"Classification: {class_name} ({confidence:.1%})")

            controller.process_hand_detection(class_name, confidence)
            
            # B10-INFER-GPU-VALIDATE Objective 3: Record frame time for FPS calculation
            controller.record_frame_time(frame_start_time)

            if controller.debug_mode:
                display_frame = controller.create_debug_display(
                    camera_frame, video_frame, class_name, confidence, args.model)
            elif controller.production_mode:
                display_frame = controller.create_production_display(video_frame)
            else:
                display_frame = video_frame

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("d"), ord("D")]:
                controller.toggle_debug_mode()
            elif key in [ord("p"), ord("P")]:
                controller.toggle_production_mode()
            elif key in [ord("f"), ord("F")]:
                current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_state == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    logging.info("Switched to windowed mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    logging.info("Switched to fullscreen mode")

    except KeyboardInterrupt:
        logging.info("Shutting down...")

    finally:
        # B10-INFER-GPU-VALIDATE: Final summary
        logger.info("=" * 60)
        logger.info("Final Session Summary")
        logger.info(f"   Total frames processed: {controller.total_frames}")
        logger.info(f"   Total NaN occurrences: {controller.nan_count}")
        if controller.total_frames > 0:
            nan_rate = (controller.nan_count / controller.total_frames) * 100
            logger.info(f"   NaN rate: {nan_rate:.2f}%")
        logger.info(f"   Device used: {device}")
        logger.info("=" * 60)

        cap.release()
        controller.sleep_cap.release()
        controller.scare_cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")
        logging.info("Diagnostics saved to: inference_diagnostics.log")


if __name__ == "__main__":
    main()

