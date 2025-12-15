import cv2
import numpy as np
import time


def open_video(path):
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise Exception(f"Cannot open {path}")
	return cap


class StyledTypewriter:
	def __init__(self,
				 text="",
				 style_dict=None,
				 position=(0, 0),
				 font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
				 font_scale=1.0,
				 thickness=1,
				 chars_per_second=30,
				 enable_ellipsis=True,
				 ellipsis_speed=0.5):  # seconds per state

		self.position = position

		self.default_font = font
		self.default_scale = font_scale
		self.default_thickness = thickness
		self.default_color = (255, 255, 255)

		self.style_dict = style_dict if style_dict else {}

		self.full_segments = self._segment_text(text)
		self.total_chars = sum(len(seg["text"]) for seg in self.full_segments)

		self.chars_per_second = chars_per_second
		self.start_time = None
		self.visible_chars = 0

		# Ellipsis animation
		self.enable_ellipsis = enable_ellipsis
		self.ellipsis_states = ["", ".", "..", "..."]
		self.ellipsis_speed = ellipsis_speed

		self._prev_ellipsis_index = 0
		self.ellipsis_complete_count = 0

		self.completed_segments = 0


	# -----------------------------------------------------------
	def set_text(self, text):
		"""Break new text into styled segments and restart animation."""
		self.full_segments = self._segment_text(text)
		self.total_chars = sum(len(seg["text"]) for seg in self.full_segments)
		self.start_time = time.time()
		self.visible_chars = 0

	# -----------------------------------------------------------
	def _segment_text(self, text):
		"""Split text into segments, applying style_dict overrides per word."""
		segments = []
		words = text.split(" ")

		for i, w in enumerate(words):
			style = self.style_dict.get(w, {})

			is_last_word = i + 1 == len(words)

			word_text = w if is_last_word else w + " "
			if style.get("uppercase", False):
				word_text = word_text.upper()

			if word_text[-1] in [".", "?", "!", "-"]:
				self.enable_ellipsis = False
			else:
				self.enable_ellipsis = True

			segments.append({
				"text": word_text,
				"color": style.get("color", self.default_color),
				"bold": style.get("bold", False),
				"italic": style.get("italic", False),
				"font": self.default_font,
				"font_scale": self.default_scale,
				"thickness": self.default_thickness
			})

		return segments

	# -----------------------------------------------------------
	def update(self):
		"""Update how many total characters are visible."""
		if self.start_time is None:
			return

		elapsed = time.time() - self.start_time
		self.visible_chars = min(
			int(elapsed * self.chars_per_second),
			self.total_chars
		)

	# -----------------------------------------------------------
	def get_ellipsis(self):
		if self.visible_chars < self.total_chars:
			self.ellipsis_complete_count = 0
			return ""

		cycle_time = (time.time() - self.start_time) - (self.total_chars / self.chars_per_second)
		state_index = int(cycle_time / self.ellipsis_speed) % len(self.ellipsis_states)

		# Detect loop completion
		if state_index < self._prev_ellipsis_index:
			self.ellipsis_complete_count+=1

		self._prev_ellipsis_index = state_index

		if self.enable_ellipsis:
			return self.ellipsis_states[state_index]

		return ""
		


	# -----------------------------------------------------------
	def _putTextBold(self, frame, text, pos, font, scale, color, thickness):
		"""Simulate bold by drawing the text multiple times with pixel offsets."""
		offsets = [(0,0), (1,0), (0,1), (1,1)]
		for dx, dy in offsets:
			cv2.putText(frame, text, (pos[0]+dx, pos[1]+dy), font, scale, color, thickness, lineType=cv2.LINE_AA)

	# -----------------------------------------------------------
	def render(self, frame):
		"""Draw visible characters with segment-aware styling, including bold/italic."""
		self.update()

		x, y = self.position
		chars_left = self.visible_chars

		self.completed_segments = 0  # reset at the start of this render

		for seg in self.full_segments:
			segment_text = seg["text"]
			seg_len = len(segment_text)

			if chars_left <= 0:
				break

			draw_text = segment_text[:chars_left]
			chars_left -= seg_len

			# Draw the text
			font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX if seg.get("italic", False) else seg["font"]

			if seg.get("bold", False):
				self._putTextBold(frame, draw_text, (x, y), font, seg["font_scale"], seg["color"], seg["thickness"])
			else:
				cv2.putText(frame, draw_text, (x, y), font, seg["font_scale"], seg["color"], seg["thickness"], lineType=cv2.LINE_AA)

			# Advance cursor
			(w, h), _ = cv2.getTextSize(draw_text, font, seg["font_scale"], seg["thickness"])
			x += w

			# Track fully rendered segments
			if len(draw_text) == seg_len:
				self.completed_segments += 1

		# Ellipsis overlay
		ell = self.get_ellipsis()
		if ell:
			cv2.putText(frame, ell, (x, y), self.default_font, self.default_scale, self.default_color, self.default_thickness, lineType=cv2.LINE_AA)

		return frame



if __name__ == "__main__":

	color_dict = {
		"blue": (255, 0, 0),
		"green": (0, 255, 0),
		"red": (0, 0, 255)
	}
	colors = list(color_dict.keys())
	import random
	random.shuffle(colors)
	active_color = colors[0]

	style_dict = {k: {"color": v, "bold": True, "uppercase": True} for k, v in color_dict.items()}
	style_dict["rather"] = {"italic": True}
	style_dict["reek"] = {"bold": True}

	intro_texts = [
		#("Oh.", "It's you again"),
		#("-sniff-.", "You reek of failure"),
		("Do you enjoy wasting my time?",)
	]

	body_texts = [
		(f"Where is my {active_color} item?",)
	]

	outro_texts = [
		#("Your incompetence is disappointing, but not surprising", "Be gone.")
		("Be gone.",)
	]

	random.shuffle(intro_texts)
	random.shuffle(body_texts)
	random.shuffle(outro_texts)

	texts = intro_texts[0] + body_texts[0] + outro_texts[0]
	text_index = 0

	writer = StyledTypewriter(
		text = texts[text_index],
		style_dict=style_dict,
		position=(640//20, 480-48),
		chars_per_second=20
	)
	writer.set_text(texts[text_index])

	
	ellipsis_prev = False

	while True:
		frame = np.zeros((480, 640, 3), dtype=np.uint8)
		frame = writer.render(frame)

		print(writer.completed_segments)
		# Check if ellipsis just completed a loop
		if writer.ellipsis_complete_count>=2:
			# Move to next text
			text_index += 1
			if text_index >= len(texts):
				break
			writer.set_text(texts[text_index])

		cv2.imshow("Styled Typewriter Demo", frame)
		if cv2.waitKey(16) & 0xFF == 27:  # ESC
			break

	cv2.destroyAllWindows()

