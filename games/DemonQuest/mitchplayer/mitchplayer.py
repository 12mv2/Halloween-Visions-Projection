# MITCH 2025

# System Modules
from pathlib import Path
import threading
import time

# Installed Modules
import cv2


def open_video(path):
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise Exception(f"Cannot open {path}")
	return cap


class DetectorPlayer(threading.Thread):

	# Default paths relative to this file's location
	_ASSETS_DIR = Path(__file__).parent / "assets"

	def __init__(self,
				 rest_video_path=None,
				 charge_video_path=None,
				 activate_video_path=None,
				 playback_video_path=None,
				 charge_maximum=120,
				 playback_loops=1):
		# Use assets directory relative to this file if paths not specified
		if rest_video_path is None:
			rest_video_path = self._ASSETS_DIR / "0_ANIM_IDLE.mp4"
		if charge_video_path is None:
			charge_video_path = self._ASSETS_DIR / "1_ANIM_HOLD.mp4"
		if activate_video_path is None:
			activate_video_path = self._ASSETS_DIR / "2_ANIM_SUCCESS.mp4"
		if playback_video_path is None:
			playback_video_path = self._ASSETS_DIR / "3_ANIM_PLAYBACK.mp4"

		super().__init__()
		self.daemon = True  # kills thread when program exits

		# Charge logic
		self.charge = 2
		self.charge_max = charge_maximum
		self.charge_min = 0
		self.discharge_rate = 1

		# External input updated by main thread
		self.charging = False

		# Frame captures
		self.idle_capture = open_video(rest_video_path)
		self.charge_cap = open_video(charge_video_path)
		self.activate_cap = open_video(activate_video_path)
		self.playback_cap = open_video(playback_video_path)

		# Control flag
		self.running = True

		# Playback settings
		self.playback_loops = playback_loops
		self.skip_playback = False  # If True, skip playback phase (for Quest mode)

		# Phase tracking for main loop coordination
		self.phase = "idle"  # idle, charging, activate, playback
		self.activate_complete = False  # Signal for main loop

		self.current_frame = None

	def read_frame_loop(self, capture):
		ret, frame = capture.read()
		if not ret:
			capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ret, frame = capture.read()
		return frame

	def run(self):
		import logging
		dp_logger = logging.getLogger("DetectorPlayer")

		while True:
			# Reset signal
			self.activate_complete = False
			self.phase = "idle"
			dp_logger.info("[DP] Waiting for charge...")

			while True:
				if self.charging:
					self.charge+=1
					if self.phase == "idle":
						self.phase = "charging"
						dp_logger.info(f"[DP] Started charging! charge={self.charge}")
				else:
					if self.charge!=0:
						self.charge-=1
						if self.charge==0:
							self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
							self.phase = "idle"

				if self.charge<1:
					self.current_frame = self.read_frame_loop(self.idle_capture)
				else:
					self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, self.charge-1)
					ret, frame = self.charge_cap.read()
					if ret:
						self.current_frame = frame
					else:
						dp_logger.info(f"[DP] Charge video complete at charge={self.charge}, starting activate")
						self.charge = 0
						self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
						break

			# Activate phase
			self.phase = "activate"
			self.activate_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			while True:
				ret, self.current_frame = self.activate_cap.read()
				if ret:
					time.sleep(1001/30000)
				else:
					break

			# Signal that activate is complete (for Quest mode)
			self.activate_complete = True
			dp_logger.info(f"[DP] Activate complete! activate_complete={self.activate_complete}")

			# Playback phase (skip if Quest mode hands off to DemonPlayer)
			if not self.skip_playback:
				self.phase = "playback"
				for _ in range(self.playback_loops):
					self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
					while True:
						ret, self.current_frame = self.playback_cap.read()
						if ret:
							time.sleep(1001/30000)
						else:
							break
			else:
				# Wait for main loop to reset skip_playback
				while self.skip_playback:
					time.sleep(0.1)



def main():
	player = DetectorPlayer()
	player.start()  # start animation thread

	print("Press SPACE to charge, ESC to quit.")
	while True:

		if player.current_frame is not None:
			cv2.imshow("Window", player.current_frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(' '):
			player.charging = True
		elif key != 255:   # any other key
			player.charging = False

		if key == 27:  # ESC
			player.running = False
			break

		# You can update player.space_down from AI/vision here in future

	player.join()


if __name__=="__main__":
	main()