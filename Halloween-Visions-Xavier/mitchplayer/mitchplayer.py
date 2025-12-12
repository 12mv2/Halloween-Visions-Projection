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

	def __init__(self,
				 rest_video_path=Path(R"C:\Drive\Temp\0_ANIM_IDLE.mp4"),
				 charge_video_path=Path(R"C:\Drive\Temp\1_ANIM_HOLD.mp4"),
				 activate_video_path=Path(R"C:\Drive\Temp\2_ANIM_SUCCESS.mp4"),
				 playback_video_path=Path(R"C:\Drive\Temp\3_ANIM_PLAYBACK.mp4"),
				 charge_maximum=120):

		super().__init__()
		self.daemon = True  # kills thread when program exits

		# Charge logic
		self.charge = 0
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

		self.current_frame = None

	def read_frame_loop(self, capture):
		ret, frame = capture.read()
		if not ret:
			capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ret, frame = capture.read()
		return frame

	def run(self):

		while True:

			while True:
				if self.charging:
					self.charge+=1
				else:
					if self.charge!=0:
						self.charge-=1
						if self.charge==0:
							self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

				if self.charge<1:
					self.current_frame = self.read_frame_loop(self.idle_capture)
				else:
					self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, self.charge-1)
					ret, frame = self.charge_cap.read()
					if ret:
						self.current_frame = frame
					else:
						self.charge = 0
						self.charge_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
						break

			self.activate_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			while True:
				ret, self.current_frame = self.activate_cap.read()
				if ret:
					time.sleep(1001/30000)
				else:
					break


			self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			while True:
				ret, self.current_frame = self.playback_cap.read()
				if ret:
					time.sleep(1001/30000)
				else:
					break



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