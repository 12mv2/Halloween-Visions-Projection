#!/usr/bin/env python3
"""
Quest Multi-Class Object Detection
===================================

A real-time multi-class object detection system for interactive Quest gameplay.
Detects objects from camera feed and triggers game state transitions.

GAME FLOW:
----------
1. IDLE: DetectorPlayer shows idle animation, waiting for hand
2. CHARGING: Hand detected, DetectorPlayer shows charging animation
3. DEMON: DemonPlayer runs intro, asks for object, waits
4. Object found: DemonPlayer plays outro
5. Back to IDLE

SUPPORTED CLASSES (train7 model):
---------------------------------
- hand, hammer, 9v_battery, black_spool, green_spool, blue_floppy, background

Controls:
    D - Toggle Debug/Projection mode
    F - Toggle fullscreen
    T - Trigger (simulate hand detection, start demon)
    N - Next (simulate finding target object)
    Q/ESC - Quit
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from mitchplayer.mitchplayer import DetectorPlayer
from mitchplayer.demonplayer import DemonPlayer

# Compute absolute model path from script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "7class_v1" / "best.pt")


# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("inference_diagnostics.log", mode="w")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[console_handler, file_handler]
)


# =============================================================================
# QUEST GAME STATE
# =============================================================================
class QuestGame:
    """
    Manages Quest game state and object progression.
    """
    # Fixed sequence of objects (cycles through in order)
    QUEST_SEQUENCE = [
        "green_spool",
        "black_spool",
        "9v_battery",
        "hammer",
        "blue_floppy",
    ]

    TIMEOUT_SECONDS = 300  # 5 minutes per object

    def __init__(self):
        self.state = "detector"  # "detector" or "demon"
        self.target_object = None
        self.sequence_index = 0
        self.object_start_time = None

    def pick_next_target(self):
        """Get next object in sequence."""
        self.target_object = self.QUEST_SEQUENCE[self.sequence_index]
        self.sequence_index = (self.sequence_index + 1) % len(self.QUEST_SEQUENCE)
        self.object_start_time = time.time()
        return self.target_object

    def is_sequence_complete(self):
        """Check if we've gone through all objects."""
        return self.sequence_index == 0 and self.target_object is not None

    def is_timed_out(self):
        """Check if current object hunt has timed out (5 min)."""
        if self.object_start_time is None:
            return False
        return (time.time() - self.object_start_time) > self.TIMEOUT_SECONDS

    def reset_sequence(self):
        """Reset to start of sequence."""
        self.sequence_index = 0
        self.target_object = None
        self.object_start_time = None

    def check_object(self, class_name):
        """Check if detected object matches target."""
        return class_name == self.target_object


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Quest Multi-Class Object Detection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model file")
    parser.add_argument("--source", default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("Quest Multi-Class Object Detection")
    logging.info("=" * 60)

    # -------------------------------------------------------------------------
    # GPU INITIALIZATION
    # -------------------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        logger.info(f"GPU enabled: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    try:
        logging.info(f"Loading YOLO model: {args.model}")
        model = YOLO(args.model)
        if cuda_available:
            model.model.to(device)
        logging.info(f"Model classes: {model.names}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return 1

    # -------------------------------------------------------------------------
    # PLAYER INITIALIZATION
    # -------------------------------------------------------------------------
    # DetectorPlayer: handles idle/charging/activate phases
    detector = DetectorPlayer()
    detector.skip_playback = True  # We'll use DemonPlayer for playback
    detector.start()

    # DemonPlayer: handles demon dialog sequence
    demon = DemonPlayer()
    demon.start()

    # Game state
    game = QuestGame()
    confidence_threshold = args.conf
    debug_mode = False

    # -------------------------------------------------------------------------
    # CAMERA INITIALIZATION
    # -------------------------------------------------------------------------
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
    logging.info("Controls: D=debug  F=fullscreen  T=trigger  N=next  Q=quit")
    logging.info("-" * 60)

    # -------------------------------------------------------------------------
    # WINDOW SETUP
    # -------------------------------------------------------------------------
    window_name = "Quest"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    frame_count = 0
    try:
        while True:
            # --- CAPTURE ---
            ret, camera_frame = cap.read()
            if not ret:
                continue

            # --- GET VIDEO FRAME FROM ACTIVE PLAYER ---
            if game.state == "detector":
                video_frame = detector.current_frame
            else:  # demon
                video_frame = demon.current_frame

            if video_frame is None:
                continue

            # Resize video to match camera
            cam_h, cam_w = camera_frame.shape[:2]
            video_frame = cv2.resize(video_frame, (cam_w, cam_h))

            # --- INFERENCE ---
            results = model.predict(camera_frame, verbose=False)
            probs = results[0].probs
            class_name = model.names[probs.top1]
            confidence = float(probs.top1conf)

            # --- GAME STATE MACHINE ---
            if game.state == "detector":
                # Waiting for hand to trigger charge
                if class_name == "hand" and confidence >= confidence_threshold:
                    detector.charging = True
                else:
                    detector.charging = False

                # Debug: show charge status every 30 frames
                if frame_count % 30 == 0:
                    logging.info(f"[DEBUG] charge={detector.charge}, phase={detector.phase}, activate_complete={detector.activate_complete}")

                # Check if detector completed activate phase
                if detector.activate_complete:
                    logging.info("Charge complete! Starting demon sequence...")
                    game.state = "demon"
                    game.pick_next_target()
                    demon.target_object = game.target_object
                    demon.active = True  # Start demon sequence
                    detector.activate_complete = False
                    detector.skip_playback = True  # Reset for next cycle
                    logging.info(f"Demon asking for: {game.target_object}")

            else:  # demon state
                # Check for timeout (5 min without finding object)
                if game.is_timed_out():
                    logging.info("Timeout! Going back to hand detection...")
                    game.state = "detector"
                    game.reset_sequence()
                    demon.fading_out = True  # End current demon sequence
                    demon.sequence_complete = False
                    detector.charge = 0
                    detector.charging = False
                    detector.skip_playback = False

                # Check if target object is detected
                elif demon.phase == "body" and not demon.fading_out:
                    if class_name == game.target_object and confidence >= confidence_threshold:
                        logging.info(f"Found {class_name}! Playing outro...")
                        demon.fading_out = True

                # Debug: log phase if stuck
                if frame_count % 60 == 0:
                    elapsed = int(time.time() - game.object_start_time) if game.object_start_time else 0
                    logging.info(f"[DEBUG] demon.phase={demon.phase}, target={game.target_object}, elapsed={elapsed}s")

                # Check if demon sequence complete (outro finished)
                if demon.sequence_complete:
                    demon.sequence_complete = False  # Reset flag

                    # Check if all objects found or returning from timeout
                    if game.is_sequence_complete() or game.state == "detector":
                        logging.info("All objects found! Back to hand detection.")
                        game.state = "detector"
                        game.reset_sequence()
                        detector.charge = 0
                        detector.charging = False
                        detector.skip_playback = False
                    else:
                        # Continue to next object immediately
                        game.pick_next_target()
                        demon.target_object = game.target_object
                        demon.active = True  # Start next demon sequence
                        logging.info(f"Next object: {game.target_object}")

            # --- LOGGING ---
            frame_count += 1
            if frame_count % 30 == 0:
                state_info = f"[{game.state.upper()}]"
                if game.state == "demon" and game.target_object:
                    state_info += f" looking for: {game.target_object}"
                logging.info(f"{state_info} Detection: {class_name} ({confidence:.1%})")

            # --- DISPLAY ---
            if debug_mode:
                # Debug overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(video_frame, f"State: {game.state.upper()}", (20, 40),
                            font, 1.0, (0, 255, 0), 2)
                cv2.putText(video_frame, f"Detected: {class_name} ({confidence:.1%})", (20, 80),
                            font, 1.0, (0, 255, 255), 2)
                if game.target_object:
                    cv2.putText(video_frame, f"Target: {game.target_object}", (20, 120),
                                font, 1.0, (255, 0, 255), 2)
                # Camera preview
                small_cam = cv2.resize(camera_frame, (cam_w // 4, cam_h // 4))
                video_frame[10:10+cam_h//4, cam_w-cam_w//4-10:cam_w-10] = small_cam

            cv2.imshow(window_name, video_frame)

            # --- KEYBOARD INPUT ---
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("d"), ord("D")]:
                debug_mode = not debug_mode
                logging.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key in [ord("f"), ord("F")]:
                current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_state == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key in [ord("t"), ord("T")]:
                # Trigger: simulate hand detection complete, start demon sequence
                if game.state == "detector":
                    logging.info("[MANUAL] Triggering demon sequence...")
                    detector.charging = False
                    detector.charge = 0
                    game.state = "demon"
                    game.pick_next_target()
                    demon.target_object = game.target_object
                    demon.active = True
                    logging.info(f"Demon asking for: {game.target_object}")
            elif key in [ord("n"), ord("N")]:
                # Next: simulate finding target object, advance to next
                if game.state == "demon" and demon.phase == "body":
                    logging.info(f"[MANUAL] Simulating {game.target_object} found...")
                    demon.fading_out = True

    except KeyboardInterrupt:
        logging.info("Shutting down...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete")


if __name__ == "__main__":
    main()
