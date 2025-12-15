#!/usr/bin/env python3
"""
SimpleHunt - Object Scavenger Hunt Game

A simple game where players must find and show objects to the camera.
The game randomly selects target objects and players earn points for
correctly identifying them.

This is the foundational game template for denhac members to fork
and customize with their own game mechanics.

Usage:
    python games/SimpleHunt/game.py --model models/7class_v1/best.onnx
    python games/SimpleHunt/game.py --model models/7class_v1/best.engine  # Xavier
"""
import sys
import time
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import inference backend based on model type
def load_classifier(model_path: str):
    """Load appropriate classifier based on model file extension."""
    if model_path.endswith(".engine"):
        from inference.tensorrt_infer import YOLOClassifier
    else:
        from inference.onnx_infer import YOLOClassifier
    return YOLOClassifier(model_path)


# Display names for prettier UI
DISPLAY_NAMES = {
    "9v_battery": "9V Battery",
    "black_spool": "Black Filament Spool",
    "green_spool": "Green Sewing Spool",
    "hammer": "Hammer",
    "blue_floppy": "Blue Floppy Disk",
    "hand": "Hand",
    "background": "Background",
}

# Colors for each class (BGR)
CLASS_COLORS = {
    "hand": (0, 255, 255),
    "9v_battery": (0, 165, 255),
    "black_spool": (64, 64, 64),
    "green_spool": (0, 255, 0),
    "hammer": (0, 0, 255),
    "blue_floppy": (255, 128, 0),
    "background": (200, 200, 200),
}


class QuestMitch:
    """Object scavenger hunt game."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.7,
        hold_time: float = 1.5,
        game_duration: float = 60.0,
    ):
        """
        Initialize the game.

        Args:
            model_path: Path to model file (ONNX or TensorRT engine)
            conf_threshold: Confidence required to register detection (0-1)
            hold_time: Seconds to hold object for it to count
            game_duration: Total game time in seconds
        """
        self.classifier = load_classifier(model_path)
        self.conf_threshold = conf_threshold
        self.hold_time = hold_time
        self.game_duration = game_duration

        # Game state
        self.score = 0
        self.target = None
        self.target_start_time = None
        self.detection_start_time = None
        self.game_start_time = None
        self.game_over = False

        # Exclude background from targets
        self.target_classes = [
            c for c in self.classifier.classes if c != "background"
        ]

    def new_target(self):
        """Select a new random target object."""
        self.target = random.choice(self.target_classes)
        self.target_start_time = time.time()
        self.detection_start_time = None

    def start_game(self):
        """Start a new game."""
        self.score = 0
        self.game_over = False
        self.game_start_time = time.time()
        self.new_target()

    def check_detection(self, pred_class: str, confidence: float) -> dict:
        """
        Check if detection matches target.

        Returns:
            dict with detection status info
        """
        result = {
            "matched": False,
            "hold_progress": 0.0,
            "scored": False,
        }

        # Check if prediction matches target with sufficient confidence
        if pred_class == self.target and confidence >= self.conf_threshold:
            result["matched"] = True

            # Start or continue hold timer
            if self.detection_start_time is None:
                self.detection_start_time = time.time()

            hold_elapsed = time.time() - self.detection_start_time
            result["hold_progress"] = min(1.0, hold_elapsed / self.hold_time)

            # Check if held long enough
            if hold_elapsed >= self.hold_time:
                self.score += 1
                result["scored"] = True
                self.new_target()
        else:
            # Reset hold timer if wrong object or low confidence
            self.detection_start_time = None

        return result

    def get_time_remaining(self) -> float:
        """Get remaining game time."""
        if self.game_start_time is None:
            return self.game_duration
        elapsed = time.time() - self.game_start_time
        return max(0.0, self.game_duration - elapsed)

    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.get_time_remaining() <= 0


def draw_text_with_bg(
    frame, text, position, font_scale=1.0, color=(255, 255, 255), thickness=2
):
    """Draw text with dark background for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1
    )
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, x, y, width, height, progress, color):
    """Draw a progress bar."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    fill_width = int(width * progress)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="SimpleHunt - Object Scavenger Hunt")
    parser.add_argument(
        "--model",
        type=str,
        default="models/7class_v1/best.onnx",
        help="Path to model file",
    )
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--conf", type=float, default=0.7, help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--hold", type=float, default=1.5, help="Seconds to hold object"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Game duration in seconds"
    )
    args = parser.parse_args()

    # Initialize game
    print(f"Loading model: {args.model}")
    game = QuestMitch(
        args.model,
        conf_threshold=args.conf,
        hold_time=args.hold,
        game_duration=args.duration,
    )
    print(f"Classes: {game.classifier.classes}")
    print(f"Target objects: {game.target_classes}")

    # Open camera
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nPress SPACE to start, Q to quit")

    waiting_to_start = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Run inference
        pred_class, confidence = game.classifier.predict(frame)
        display_pred = DISPLAY_NAMES.get(pred_class, pred_class)
        pred_color = CLASS_COLORS.get(pred_class, (255, 255, 255))

        if waiting_to_start:
            # Title screen
            draw_text_with_bg(frame, "QUESTMITCH", (w // 2 - 150, h // 2 - 50), 2.0, (0, 255, 255), 3)
            draw_text_with_bg(frame, "Object Scavenger Hunt", (w // 2 - 140, h // 2 + 10), 0.8, (200, 200, 200), 2)
            draw_text_with_bg(frame, "Press SPACE to start", (w // 2 - 120, h // 2 + 60), 0.7, (100, 255, 100), 2)

            # Show current detection as preview
            draw_text_with_bg(frame, f"Detecting: {display_pred}", (20, h - 40), 0.6, pred_color, 1)

        elif game.is_game_over():
            # Game over screen
            draw_text_with_bg(frame, "GAME OVER", (w // 2 - 130, h // 2 - 50), 2.0, (0, 0, 255), 3)
            draw_text_with_bg(frame, f"Final Score: {game.score}", (w // 2 - 100, h // 2 + 20), 1.2, (255, 255, 255), 2)
            draw_text_with_bg(frame, "Press SPACE to play again", (w // 2 - 150, h // 2 + 80), 0.7, (100, 255, 100), 2)

        else:
            # Active game
            detection = game.check_detection(pred_class, confidence)

            # Draw target prompt
            target_display = DISPLAY_NAMES.get(game.target, game.target)
            target_color = CLASS_COLORS.get(game.target, (255, 255, 255))
            draw_text_with_bg(frame, f"FIND: {target_display}", (20, 50), 1.2, target_color, 2)

            # Draw score and time
            time_remaining = game.get_time_remaining()
            draw_text_with_bg(frame, f"Score: {game.score}", (w - 180, 50), 1.0, (255, 255, 255), 2)
            time_color = (0, 255, 0) if time_remaining > 10 else (0, 165, 255) if time_remaining > 5 else (0, 0, 255)
            draw_text_with_bg(frame, f"Time: {time_remaining:.1f}s", (w - 180, 90), 0.8, time_color, 2)

            # Draw current detection
            if detection["matched"]:
                draw_text_with_bg(frame, f"MATCHED: {display_pred}", (20, h - 80), 1.0, (0, 255, 0), 2)
                # Draw hold progress bar
                draw_progress_bar(frame, 20, h - 50, 200, 25, detection["hold_progress"], (0, 255, 0))
                draw_text_with_bg(frame, "HOLD IT!", (230, h - 40), 0.7, (0, 255, 0), 2)

                if detection["scored"]:
                    # Flash effect when scored
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
            else:
                draw_text_with_bg(frame, f"Seeing: {display_pred} ({confidence*100:.0f}%)", (20, h - 40), 0.7, (150, 150, 150), 1)

        cv2.imshow("SimpleHunt", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if waiting_to_start or game.is_game_over():
                game.start_game()
                waiting_to_start = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
