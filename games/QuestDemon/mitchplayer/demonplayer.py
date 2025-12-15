# MITCH 2025

# System Modules
from pathlib import Path
import threading
import time
import random

# Installed Modules
import cv2

from .styledsubtitler import StyledTypewriter


def open_video(path):
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise Exception(f"Cannot open {path}")
	return cap


class DemonPlayer(threading.Thread):

	# Default paths relative to this file's location
	_ASSETS_DIR = Path(__file__).parent / "assets"

	def __init__(self,
				 intro_video_path=None,
				 body_video_path=None,
				 outro_video_path=None,
				 ):
		# Use assets directory relative to this file if paths not specified
		if intro_video_path is None:
			intro_video_path = self._ASSETS_DIR / "D01_DEMON_INTRO.mp4"
		if body_video_path is None:
			body_video_path = self._ASSETS_DIR / "D10_DEMON_BODY_LOOP.mp4"
		if outro_video_path is None:
			outro_video_path = self._ASSETS_DIR / "D02_DEMON_OUTRO.mp4"

		super().__init__()
		self.daemon = True  # kills thread when program exits

		# Frame captures
		self.intro_cap = open_video(intro_video_path)
		self.body_cap = open_video(body_video_path)
		self.outro_cap = open_video(outro_video_path)

		# Display names for quest objects (model class name -> readable name)
		self.object_display_names = {
			"hammer": "hammer",
			"9v_battery": "9 volt battery",
			"black_spool": "black spool",
			"green_spool": "green spool",
			"blue_floppy": "blue floppy disk",
		}

		self.color_dict = {
			"blue": (255, 0, 0),
			"green": (0, 255, 0),
			"red": (0, 0, 255)
		}

		self.style_dict = {k: {"color": v, "bold": True, "uppercase": True} for k, v in self.color_dict.items()}
		self.style_dict["rather"] = {"italic": True}
		self.style_dict["reek"] = {"bold": True}

		self.writer = StyledTypewriter(
			text = "Testing it out",
			style_dict = self.style_dict,
			position = (640//20, 480-48),
			chars_per_second = 20
			)

		self.intro_texts = [
			("Oh.", "It's you again."),
			("-sniff-", "You reek of failure."),
			("Do you enjoy wasting my time?",)
		]

		self.body_texts = [
			"Where is my {object}?",
			"Bring me the {object}.",
			"I require the {object}.",
		]

		self.outro_texts = [
			("Your incompetence is disappointing, but not surprising.", "Be gone."),
			("Be gone.",)
		]

		self.fading_out = False

		# Phase tracking for main loop coordination
		self.phase = "idle"  # idle, intro, body, outro
		self.active = False  # Main loop sets True to start demon sequence
		self.sequence_complete = False  # Signals main loop when outro finishes

		# Quest object tracking
		self.target_object = None  # Set by main loop (e.g., "orange_ball")

		self.current_frame = None

	def get_random_texts(self):
		random.shuffle(self.intro_texts)
		random.shuffle(self.body_texts)
		random.shuffle(self.outro_texts)
		return self.intro_texts[0] + self.body_texts[0] + self.outro_texts[0]


	def render_text_on_frame(self, frame):
		frame = self.writer.render(frame)
		return frame

	def run(self):

		while True:
			# Wait for main loop to activate demon sequence
			self.phase = "idle"
			# Don't reset sequence_complete here - let main loop see it first
			while not self.active:
				time.sleep(0.05)

			# Reset for new sequence (after main loop has activated us)
			self.fading_out = False
			self.sequence_complete = False

			# INTRO phase
			self.phase = "intro"
			intro_index = 0
			self.intro_cap.set(cv2.CAP_PROP_POS_FRAMES, intro_index)
			# Set intro text (join tuple into single string if multiple parts)
			intro_text = random.choice(self.intro_texts)
			if isinstance(intro_text, tuple):
				intro_text = " ".join(intro_text)
			self.writer.set_text(intro_text)
			while True:
				ret, frame = self.intro_cap.read()
				frame = self.render_text_on_frame(frame)
				intro_index+=1
				if ret:
					self.current_frame = frame
					time.sleep(1001/30000)
				else:
					break

			# BODY phase - loops until fading_out is True
			self.phase = "body"
			self.body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

			# Set the text to show what object we're looking for
			if self.target_object:
				display_name = self.object_display_names.get(self.target_object, self.target_object)
				body_text = random.choice(self.body_texts).format(object=display_name)
				self.writer.set_text(body_text)
			while True:
				# Check fading_out every frame, not just at video end
				if self.fading_out:
					break
				ret, frame = self.body_cap.read()
				if not ret:
					self.body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
					ret, frame = self.body_cap.read()
				frame = self.render_text_on_frame(frame)
				self.current_frame = frame
				time.sleep(1001/30000)

			# OUTRO phase
			self.phase = "outro"
			outro_index = 0
			self.outro_cap.set(cv2.CAP_PROP_POS_FRAMES, outro_index)
			while True:
				ret, frame = self.outro_cap.read()
				outro_index+=1
				if ret:
					self.current_frame = frame
					time.sleep(1001/30000)
				else:
					break

			# Signal completion and reset
			self.sequence_complete = True
			self.active = False



def main():
	player = DemonPlayer()
	player.start()  # start animation thread

	texts = player.get_random_texts()

	print("Press SPACE to charge, ESC to quit.")
	while True:

		if player.current_frame is not None:
			cv2.imshow("Window", player.current_frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(' '):
			player.fading_out = True

		if key == 27:  # ESC
			player.running = False
			break

		# You can update player.space_down from AI/vision here in future

	player.join()


if __name__ == "__main__":
	main()