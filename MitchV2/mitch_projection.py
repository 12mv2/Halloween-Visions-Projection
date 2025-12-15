#!/usr/bin/env python3
"""
Simple Halloween Hand Detection Projection
===========================================

A real-time hand detection system that switches between "sleeping" and "scare"
video displays based on whether a hand is detected in the camera feed.

ARCHITECTURE OVERVIEW:
----------------------
1. Camera captures frames at ~30 FPS
2. Each frame is preprocessed to a GPU tensor (224x224 RGB)
3. YOLO classification model predicts "hand" vs "not_hand"
4. Based on prediction, displays either sleeping_face.mp4 or angry_face.mp4
5. OpenCV window renders the output (supports fullscreen projection)

STATE MACHINE:
--------------
- IDLE: Shows sleeping_face.mp4 (looped)
- SCARE: Shows angry_face.mp4 for 2 seconds, then returns to IDLE

HARDWARE REQUIREMENTS:
----------------------
- NVIDIA Jetson Xavier NX (or CUDA-capable GPU)
- USB camera (UVC compatible)
- HDMI display/projector

KNOWN ISSUES (Ultralytics 8.0.196):
-----------------------------------
- DO NOT use model.to(device) - triggers training mode bug
- USE model.model.to(device) instead - moves inner PyTorch model safely
- See CHANGELOG.md for details on this workaround

Usage:
    python3 simple_projection.py                    # Default settings
    python3 simple_projection.py --fullscreen       # Start fullscreen
    python3 simple_projection.py --conf 0.8         # Higher confidence threshold
    python3 simple_projection.py --source 1         # Different camera

Controls:
    D - Toggle Debug/Projection mode
    P - Toggle Production mode (grey border fix)
    F - Toggle fullscreen
    Q/ESC - Quit
"""

import argparse
import time
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque


from mitchplayer.mitchplayer import DetectorPlayer
from pathlib import Path

# Use demon video for playback instead of rick roll
_ASSETS_DIR = Path(__file__).parent / "mitchplayer" / "assets"
mitchplayer = DetectorPlayer(
    playback_video_path=_ASSETS_DIR / "D01_DEMON_INTRO.mp4"
)
mitchplayer.start()  # start animation thread


# =============================================================================
# LOGGING SETUP
# =============================================================================
# Dual logging: console for real-time monitoring, file for diagnostics
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler - shows logs in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_formatter)

# File handler - saves to inference_diagnostics.log for post-run analysis
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


# =============================================================================
# PROJECTION CONTROLLER CLASS
# =============================================================================
class SimpleProjectionController:
    """
    Manages the projection state machine and video playback.

    Responsibilities:
    - Load and loop video files (sleeping and scare)
    - Track state transitions (idle <-> scare)
    - Preprocess camera frames for inference
    - Create debug/production display overlays
    - Monitor FPS and detect NaN errors

    Attributes:
        state (str): Current state - "idle" or "scare"
        confidence_threshold (float): Minimum confidence to trigger scare (default 0.7)
        scare_duration (float): How long to show scare video (default 2.0 seconds)
        debug_mode (bool): Show camera overlay and stats
        production_mode (bool): Apply grey border fix
    """

    def __init__(self, video_sleep_path, video_scare_path):
        """
        Initialize controller with video paths.

        Args:
            video_sleep_path: Path to idle/sleeping video (e.g., "videos/sleeping_face.mp4")
            video_scare_path: Path to scare video (e.g., "videos/angry_face.mp4")
        """
        self.video_sleep_path = video_sleep_path
        self.video_scare_path = video_scare_path

        # State machine
        self.state = "idle"  # "idle" or "scare"
        self.confidence_threshold = 0.7  # Minimum confidence to trigger scare
        self.scare_duration = 2.0  # Seconds to show scare video
        self.last_trigger = 0.0  # Timestamp of last scare trigger

        # Display modes
        self.debug_mode = False  # Press D to show camera overlay and stats
        self.production_mode = False  # Apply grey border fix

        # Performance monitoring (B10-INFER-GPU-VALIDATE)
        self.fps_buffer = deque(maxlen=300)  # Store last 300 frame times for FPS calc
        self.nan_count = 0  # Count NaN errors (indicates GPU/CPU mismatch)
        self.total_frames = 0  # Total frames processed
        self.warmup_frames = 30  # Skip first 30 frames for accurate benchmark
        self.benchmark_complete = False  # Only report benchmark once
        self.device_verified = False  # Track if we've verified GPU execution

        # Load video files using OpenCV VideoCapture
        self.sleep_cap = cv2.VideoCapture(video_sleep_path)
        self.scare_cap = cv2.VideoCapture(video_scare_path)

        if not self.sleep_cap.isOpened():
            raise Exception(f"Could not open sleep video: {video_sleep_path}")
        if not self.scare_cap.isOpened():
            raise Exception(f"Could not open scare video: {video_scare_path}")

        # Get video properties for logging
        self.sleep_fps = self.sleep_cap.get(cv2.CAP_PROP_FPS)
        self.scare_fps = self.scare_cap.get(cv2.CAP_PROP_FPS)

        self.sleep_frame_count = 0
        self.scare_frame_count = 0

        logging.info("Videos loaded successfully")
        logging.info(f"Sleep video: {video_sleep_path} ({self.sleep_fps:.1f} FPS)")
        logging.info(f"Scare video: {video_scare_path} ({self.scare_fps:.1f} FPS)")

    def get_current_video_frame(self):
        """
        Return current frame based on state.

        Loops videos automatically when they reach the end.

        Returns:
            numpy.ndarray: BGR video frame, or None if read fails
        """
        return mitchplayer.current_frame


        if self.state == "scare":
            ret, frame = self.scare_cap.read()
            if not ret:
                # Video ended - loop back to start
                self.scare_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.scare_cap.read()
            return frame
        else:
            ret, frame = self.sleep_cap.read()
            if not ret:
                # Video ended - loop back to start
                self.sleep_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.sleep_cap.read()
            return frame

    def process_hand_detection(self, class_name, confidence):
        """
        Handle hand detection state transitions.

        State Machine Logic:
        - If hand detected with high confidence AND not already scaring -> SCARE
        - If in SCARE state AND timeout exceeded -> IDLE

        Args:
            class_name: Predicted class ("hand" or "not_hand")
            confidence: Prediction confidence (0.0 to 1.0)
        """
        current_time = time.time()

        # Trigger scare if hand detected with sufficient confidence
        if class_name == "hand" and confidence >= self.confidence_threshold:
            if self.state != "scare":
                logging.info(f"HAND DETECTED! Confidence: {confidence:.1%}")
                logging.info("Switching to SCARE state")
                self.state = "scare"
                mitchplayer.charging = True
                self.last_trigger = current_time

        # Return to idle after scare timeout
        if self.state == "scare" and current_time - self.last_trigger > self.scare_duration:
            logging.info("SCARE timeout, returning to IDLE")
            mitchplayer.charging = False
            self.state = "idle"

    def toggle_debug_mode(self):
        """Toggle debug mode (shows camera overlay and stats)."""
        self.debug_mode = not self.debug_mode
        mode = "DEBUG" if self.debug_mode else "PROJECTION"
        logging.info(f"Switched to {mode} mode")
        return self.debug_mode

    def toggle_production_mode(self):
        """Toggle production mode (applies grey border fix)."""
        self.production_mode = not self.production_mode
        mode = "PRODUCTION" if self.production_mode else "NORMAL"
        logging.info(f"Switched to {mode} display mode")
        return self.production_mode

    def create_debug_display(self, camera_frame, video_frame, class_name, confidence, model_name="Colin1.pt"):
        """
        Create debug display with camera overlay and stats.

        Shows:
        - Video frame as main display
        - Small camera preview in top-right corner
        - Current state (IDLE/SCARE) with color indicator
        - Detection result and confidence
        - Confidence threshold
        - Model name
        - Control hints

        Args:
            camera_frame: Raw camera frame
            video_frame: Current video frame
            class_name: Predicted class
            confidence: Prediction confidence
            model_name: Model filename for display

        Returns:
            numpy.ndarray: Composed debug display frame
        """
        # Scale camera preview to 30% size
        cam_h, cam_w = camera_frame.shape[:2]
        scale = 0.3
        small_cam = cv2.resize(camera_frame, (int(cam_w * scale), int(cam_h * scale)))

        # Resize video to match camera dimensions
        display_h, display_w = camera_frame.shape[:2]
        display = cv2.resize(video_frame, (display_w, display_h))

        # Position camera preview in top-right corner
        cam_h_small, cam_w_small = small_cam.shape[:2]
        y_offset = 20
        x_offset = display_w - cam_w_small - 20

        # Draw border around camera preview
        cv2.rectangle(display,
                      (x_offset - 5, y_offset - 5),
                      (x_offset + cam_w_small + 5, y_offset + cam_h_small + 5),
                      (255, 255, 255), 2)
        display[y_offset:y_offset + cam_h_small, x_offset:x_offset + cam_w_small] = small_cam

        # Draw status text
        info_y = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        # State indicator (red for scare, green for idle)
        state_color = (0, 0, 255) if self.state == "scare" else (0, 255, 0)
        cv2.putText(display, f"State: {self.state.upper()}", (20, info_y),
                    font, font_scale, state_color, thickness)

        # Detection result (yellow for hand, white for not_hand)
        detection_color = (0, 255, 255) if class_name == "hand" else (255, 255, 255)
        cv2.putText(display, f"Detection: {class_name} ({confidence:.1%})",
                    (20, info_y + 50), font, 1.2, detection_color, thickness)

        cv2.putText(display, f"Threshold: {self.confidence_threshold:.0%}",
                    (20, info_y + 100), font, 1.2, (255, 255, 255), thickness)

        cv2.putText(display, f"Model: {model_name}",
                    (20, info_y + 150), font, 1.2, (255, 255, 255), thickness)

        # Control hints at bottom
        cv2.putText(display, "Press 'D' to toggle debug/projection mode",
                    (20, display_h - 50), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Press 'P' for production mode (grey fix)",
                    (20, display_h - 40), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Press 'F' for fullscreen",
                    (20, display_h - 30), font, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "Press 'Q' or ESC to quit",
                    (20, display_h - 20), font, 0.4, (255, 255, 255), 1)

        return display

    def create_production_display(self, video_frame):
        """
        Create production display with grey border fix.

        OpenCV's cv2.imshow() can show a grey border at the top on some systems.
        This workaround stretches the video vertically by 10% then crops to
        original size, effectively hiding the grey border.

        Args:
            video_frame: Video frame to process

        Returns:
            numpy.ndarray: Processed frame with grey border hidden
        """
        h, w = video_frame.shape[:2]
        stretched_h = int(h * 1.1)  # Stretch 10% taller
        production_frame = cv2.resize(video_frame, (w, stretched_h))
        crop_start = (stretched_h - h) // 2  # Center crop
        return production_frame[crop_start:crop_start + h, :]

    def preprocess_frame_to_tensor(self, camera_frame, device="cuda"):
        """
        Convert camera frame to GPU tensor for inference.

        This is the B10-PROJ-INFER-REFIT optimization: direct tensor conversion
        without saving to disk or using PIL. Much faster than Ultralytics'
        default predictor pipeline.

        Pipeline:
        1. BGR -> RGB (YOLO expects RGB)
        2. Resize to 224x224 (classification model input size)
        3. HWC -> CHW (PyTorch tensor format)
        4. uint8 -> float32, normalize 0-255 -> 0-1
        5. Add batch dimension [1, 3, 224, 224]
        6. Move to GPU

        Args:
            camera_frame: BGR camera frame (numpy array)
            device: Target device ("cuda" or "cpu")

        Returns:
            torch.Tensor: Shape [1, 3, 224, 224] on specified device
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
        Detect NaN values in model output.

        NaN outputs typically indicate a device mismatch: model weights on CPU
        but input tensor on GPU, or vice versa. This is a critical error that
        will cause incorrect predictions.

        B10-INFER-GPU-VALIDATE Objective 2: Monitor for NaN errors.

        Args:
            probs_tensor: Model output probabilities (torch.Tensor)

        Returns:
            tuple: (has_nan: bool, probs_array: numpy.ndarray)

        Raises:
            RuntimeError: If NaN count exceeds threshold (>5 NaNs in <1000 frames)
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

            # Abort if too many NaNs - indicates fundamental GPU inference issue
            if self.nan_count > 5 and self.total_frames < 1000:
                logger.error(f"Exceeded NaN threshold: {self.nan_count} NaNs in {self.total_frames} frames")
                logger.error("Aborting due to excessive NaN outputs - GPU inference may be failing")
                raise RuntimeError("Excessive NaN outputs detected - check GPU inference path")

        return has_nan, probs_tensor.cpu().numpy() if isinstance(probs_tensor, torch.Tensor) else probs_tensor

    def record_frame_time(self, frame_start_time):
        """
        Record frame processing time for FPS calculation.

        B10-INFER-GPU-VALIDATE Objective 3: Benchmark baseline FPS.

        After 300 frames (excluding warmup), reports:
        - Mean FPS and standard deviation
        - Mean frame time in milliseconds
        - NaN count
        - Pass/fail status (target: >= 15 FPS)

        Args:
            frame_start_time: time.time() from start of frame processing
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

            # Free benchmark memory - prevents memory leak over long runs
            self.fps_buffer.clear()


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main application entry point.

    Flow:
    1. Parse command-line arguments
    2. Initialize CUDA/GPU
    3. Load YOLO model (with Ultralytics 8.0.196 workaround)
    4. Initialize projection controller
    5. Open camera
    6. Main loop: capture -> preprocess -> infer -> display
    7. Cleanup on exit
    """
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Simple Halloween Hand Detection Projection")
    parser.add_argument("--model", default="models/Colin1.pt", help="YOLO model file")
    parser.add_argument("--source", default=0, help="Camera index")
    parser.add_argument("--video-sleep", default="videos/Santa_s_Wild_AI_Snowball_Fight.mp4", help="Sleep video")
    parser.add_argument("--video-scare", default="videos/Santa_Chases_Turkey_Technology_Abounds.mp4", help="Scare video")
    parser.add_argument("--conf", type=float, default=0.7, help="Hand detection confidence threshold")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("Simple Halloween Hand Detection Projection")
    logging.info("=" * 60)

    # -------------------------------------------------------------------------
    # GPU INITIALIZATION
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    try:
        logging.info(f"Loading YOLO model: {args.model}")
        model = YOLO(args.model)

        # CRITICAL WORKAROUND for Ultralytics 8.0.196:
        #
        # DO NOT USE: model.to(device)
        #   - This triggers a bug that puts the model in training mode
        #   - Symptoms: Downloads imagenet10 dataset, starts training
        #
        # USE INSTEAD: model.model.to(device)
        #   - model.model is the inner PyTorch nn.Module
        #   - Moving it directly bypasses Ultralytics' buggy .to() override
        #   - This correctly moves weights to GPU for inference
        #
        # See CHANGELOG.md "Ultralytics 8.0.196 Workaround" for details.
        if cuda_available:
            model.model.to(device)
            logger.info(f"Inner model moved to: {device}")

        logging.info(f"Model loaded: {model.names}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return 1

    # -------------------------------------------------------------------------
    # CONTROLLER INITIALIZATION
    # -------------------------------------------------------------------------
    try:
        controller = SimpleProjectionController(args.video_sleep, args.video_scare)
        controller.confidence_threshold = args.conf
    except Exception as e:
        logging.error(f"Failed to initialize controller: {e}")
        return 1

    # -------------------------------------------------------------------------
    # CAMERA INITIALIZATION
    # -------------------------------------------------------------------------
    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # Could be a video file path

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

    # -------------------------------------------------------------------------
    # WINDOW SETUP
    # -------------------------------------------------------------------------
    window_name = "Halloween Projection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    try:
        while True:
            frame_start_time = time.time()  # For FPS tracking

            # --- CAPTURE ---
            ret, camera_frame = cap.read()
            if not ret:
                logging.warning("Failed to read camera frame")
                continue

            # --- GET VIDEO FRAME ---
            video_frame = controller.get_current_video_frame()
            if video_frame is None:
                continue

            # Resize video to match camera dimensions (with slight vertical stretch)
            cam_h, cam_w = camera_frame.shape[:2]
            extended_h = int(cam_h * 1.15)
            video_frame = cv2.resize(video_frame, (cam_w, extended_h))
            video_frame = video_frame[:cam_h, :]  # Crop to camera height

            # --- PREPROCESS ---
            # Convert camera frame to GPU tensor for inference
            # This is the B10-PROJ-INFER-REFIT optimization: direct tensor
            # conversion without file I/O or PIL (much faster)
            frame_tensor = controller.preprocess_frame_to_tensor(camera_frame, device=device)

            # --- INFERENCE ---
            # Run direct inference bypassing Ultralytics predictor
            # Using model.model (inner PyTorch model) for direct forward pass
            with torch.no_grad():
                output = model.model(frame_tensor)
                # Ultralytics 8.3+ returns tuple (logits, features), extract logits
                logits = output[0] if isinstance(output, tuple) else output
                probs = torch.nn.functional.softmax(logits, dim=1)

            # --- VERIFY GPU EXECUTION (first frame only) ---
            if not controller.device_verified:
                controller.device_verified = True
                actual_device = probs.device
                logger.info(f"First inference device verified: {actual_device}")
                logger.info(f"   Tensor shape: {probs.shape}")
                if str(actual_device) != device and cuda_available:
                    logger.warning(f"Expected {device} but got {actual_device}")

            # --- NaN DETECTION ---
            has_nan, probs_np = controller.check_nan_in_probs(probs)

            # --- EXTRACT RESULTS ---
            class_name = "not_hand"
            confidence = 0.0
            if probs_np is not None and len(probs_np.shape) > 0:
                # probs_np is [1, 2] shape: [batch, classes]
                # Classes: 0 = "hand", 1 = "not_hand" (from model.names)
                probs_flat = probs_np.flatten()
                max_idx = probs_flat.argmax()
                confidence = float(probs_flat[max_idx])
                class_name = model.names[int(max_idx)]
            else:
                class_name = "not_hand"
                confidence = 0.5

            # --- MEMORY CLEANUP ---
            # Free GPU memory - prevents memory leak over long runs
            # This is critical for 24/7 operation on Xavier
            del frame_tensor, output, probs

            # --- LOGGING (every 30 frames) ---
            if not hasattr(controller, "frame_count"):
                controller.frame_count = 0
            controller.frame_count += 1
            if controller.frame_count % 30 == 0:
                logging.info(f"Classification: {class_name} ({confidence:.1%})")

            # --- STATE MACHINE UPDATE ---
            controller.process_hand_detection(class_name, confidence)

            # --- FPS TRACKING ---
            controller.record_frame_time(frame_start_time)

            # --- DISPLAY ---
            if controller.debug_mode:
                display_frame = controller.create_debug_display(
                    camera_frame, video_frame, class_name, confidence, args.model)
            elif controller.production_mode:
                display_frame = controller.create_production_display(video_frame)
            else:
                display_frame = video_frame

            cv2.imshow(window_name, display_frame)

            # --- KEYBOARD INPUT ---
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
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

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    finally:
        # Final session summary
        logger.info("=" * 60)
        logger.info("Final Session Summary")
        logger.info(f"   Total frames processed: {controller.total_frames}")
        logger.info(f"   Total NaN occurrences: {controller.nan_count}")
        if controller.total_frames > 0:
            nan_rate = (controller.nan_count / controller.total_frames) * 100
            logger.info(f"   NaN rate: {nan_rate:.2f}%")
        logger.info(f"   Device used: {device}")
        logger.info("=" * 60)

        # Release resources
        cap.release()
        controller.sleep_cap.release()
        controller.scare_cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")
        logging.info("Diagnostics saved to: inference_diagnostics.log")


if __name__ == "__main__":
    main()
