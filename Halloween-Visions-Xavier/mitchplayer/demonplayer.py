# MITCH 2025

# System Modules
from pathlib import Path
import threading
import time
import random

# Installed Modules
import cv2

from styledsubtitler import StyledTypewriter


def open_video(path):
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise Exception(f"Cannot open {path}")
	return cap


class DemonPlayer(threading.Thread):

	def __init__(self,
				 intro_video_path=Path(R"C:\Drive\Temp\D01_DEMON_INTRO.mp4"),
				 body_video_path=Path(R"C:\Drive\Temp\D10_DEMON_BODY_LOOP.mp4"),
				 outro_video_path=Path(R"C:\Drive\Temp\D02_DEMON_OUTRO.mp4"),
				 ):

		super().__init__()
		self.daemon = True  # kills thread when program exits

		# Frame captures
		self.intro_cap = open_video(intro_video_path)
		self.body_cap = open_video(body_video_path)
		self.outro_cap = open_video(outro_video_path)

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
			#("Oh.", "It's you again"),
			#("-sniff-.", "You reek of failure"),
			("Do you enjoy wasting my time?",)
		]

		self.body_texts = [
			("Where is my {active_color} item?",)
		]

		self.outro_texts = [
			#("Your incompetence is disappointing, but not surprising", "Be gone.")
			("Be gone.",)
		]

		self.fading_out = False

		self.current_frame = None

	def get_random_texts(self):
		random.shuffle(self.intro_texts)
		random.shuffle(self.body_texts)
		random.shuffle(self.outro_texts)
		return self.intro_texts[0] + self.body_texts[0] + self.outro_texts[0]


	def render_text_on_frame(self, frame):
		frame = self.writer.render(frame)
		return frame
		text_index = 0
		writer.set_text(texts[text_index])

		ellipsis_prev = False

		while True:
			
			print(writer.completed_segments)
			# Check if ellipsis just completed a loop
			if writer.ellipsis_complete_count>=2:
				# Move to next text
				text_index += 1
				if text_index >= len(texts):
					break
				writer.set_text(texts[text_index])

	def run(self):

		while True:

			intro_index = 0
			self.intro_cap.set(cv2.CAP_PROP_POS_FRAMES, intro_index)
			while True: 
				ret, frame = self.intro_cap.read()
				frame = self.render_text_on_frame(frame)
				#print(print(writer.completed_segments))
				intro_index+=1
				if ret:
					self.current_frame = frame
					time.sleep(1001/30000)
				else:
					break


			self.body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			while True:
				ret, frame = self.body_cap.read()
				if not ret:
					if self.fading_out:
						break
					self.body_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
					ret, frame = self.body_cap.read()
				self.current_frame = frame
				time.sleep(1001/30000)
				

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

			break



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

main()