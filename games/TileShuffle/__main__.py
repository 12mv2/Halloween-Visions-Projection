# MITCH 2026

# System Modules
from pathlib import Path
import random

# Installed Modules
import cv2
import numpy as np

# Project Modules
from config import ASSETS_DIR, DISPLAY_RES, ANIM_RES, TILE_SIZE


# --- Load all tile images ---
tile_dir = ASSETS_DIR / "tiles"
tile_paths = list(tile_dir.glob("*.*"))
if not tile_paths:
	raise FileNotFoundError("No tile images found in folder.")

noisy_background = cv2.imread(str(ASSETS_DIR/"noisy_canvas.png"))

def load_random_tile(W, H):
	path = random.choice(tile_paths)
	img = cv2.imread(str(path))
	if img is None:
		img = np.zeros((H, W, 3), dtype=np.uint8)
		img[:, :] = (0, 255, 0)
	else:
		img = cv2.resize(img, (W, H))
	return img



def generate_display_bg():
	# --- Generate full display background with random tiles ---
	display_background = np.zeros((DISPLAY_RES[1], DISPLAY_RES[0], 3), dtype=np.uint8)
	for y in range(0, DISPLAY_RES[1], TILE_SIZE):
		for x in range(0, DISPLAY_RES[0], TILE_SIZE):
			tile_img = load_random_tile(TILE_SIZE, TILE_SIZE)
			y1, y2 = y, min(y + TILE_SIZE, DISPLAY_RES[1])
			x1, x2 = x, min(x + TILE_SIZE, DISPLAY_RES[0])
			display_background[y1:y2, x1:x2] = tile_img[:y2-y1, :x2-x1]
	return display_background

# --- Sliding Tile Animation ---
import random
import numpy as np
import cv2
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"

class ImageTile:
	def __init__(self, L, T, W, H, image_dir=tile_dir):
		self.L = L
		self.T = T
		self.W = W
		self.H = H

		# Load random image if directory provided
		self.image = np.zeros((H, W, 3), dtype=np.uint8)
		self.image[:, :] = (0, 255, 0)  # default green

		if image_dir:
			images = list(Path(image_dir).glob("*.*"))
			if images:
				img_path = random.choice(images)
				img = cv2.imread(str(img_path))
				if img is not None:
					self.image = cv2.resize(img, (W, H))

		self.base_image = self.image.copy()
		self.clear_letter()

		self.cooldown = 0

	def on_moved(self, cooldown=1):
		self.cooldown = cooldown

	def tick_cooldown(self):
		if self.cooldown > 0:
			self.cooldown -= 1

	def clear_letter(self):
		self.image = self.base_image.copy()
		self.current_letter = None

	def set_letter(self, letter):
		self.clear_letter()
		self.current_letter = letter

		font = cv2.FONT_HERSHEY_SIMPLEX
		scale = 0.5
		thickness = 4

		while True:
			(w, h), _ = cv2.getTextSize(letter, font, scale, thickness)
			if w < self.W * 0.8 and h < self.H * 0.8:
				scale += 0.1
			else:
				scale -= 0.1
				break

		(w, h), _ = cv2.getTextSize(letter, font, scale, thickness)
		x = (self.W - w) // 2
		y = (self.H + h) // 2

		cv2.putText(
			self.image, letter, (x, y),
			font, scale, (255, 255, 255),
			thickness, cv2.LINE_AA
		)

	def draw(self, canvas, x=None, y=None):
		L = self.L if x is None else x
		T = self.T if y is None else y
		canvas[T:T+self.H, L:L+self.W] = self.image



class SlidingTileAnimation:
	def __init__(self, grid_size=16, canvas_size=1024, slide_steps=45):
		self.W = canvas_size
		self.H = canvas_size
		self.GRID_SIZE = grid_size
		self.TILE_SIZE = TILE_SIZE#self.W // self.GRID_SIZE
		self.slide_steps = slide_steps

		# Canvas for animation (1024x1024)
		self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

		# Create tiles
		self.tiles = []
		letters = 25*("                "+"  denhac rules  "+"                ")
		for y in range(0, self.H, self.TILE_SIZE):
			for x in range(0, self.W, self.TILE_SIZE):
				self.tiles.append(ImageTile(x, y, self.TILE_SIZE, self.TILE_SIZE))
				self.tiles[-1].set_letter(letters[len(self.tiles)-1])

		self.redraw_middle_rows()

		# Pick random empty tile
		tile = random.choice(self.tiles)
		self.empty_pos = (tile.L, tile.T)
		self.tiles.remove(tile)
		self.last_moved = None

		# Sliding state
		self.moving_tile = None
		self.start_pos = None
		self.end_pos = None
		self.current_slide = 0


	def tiles_in_row(self, row_index):
		y = row_index * self.TILE_SIZE
		row_tiles = [t for t in self.tiles if t.T == y]
		return sorted(row_tiles, key=lambda t: t.L)



	def redraw_middle_rows(self):
	
		for row_index in range(0, 16):
			row_tiles = self.tiles_in_row(row_index)
			if row_index in [1, 4, 7, 10, 13]:

				message = " "*random.randint(1,3)
				message+= "denhac rules"
				message = message.ljust(16)

				for tile, char in zip(row_tiles, message.ljust(len(row_tiles))):
					if char==" ":
						tile.clear_letter()
					else:
						tile.set_letter(char)
			else:
				for tile in row_tiles:
					tile.clear_letter()


	def get_neighbors(self, pos):
		x, y = pos
		delta = self.TILE_SIZE

		candidates = []
		for dx, dy in [(-delta,0),(delta,0),(0,-delta),(0,delta)]:
			nx, ny = x + dx, y + dy
			for tile in self.tiles:
				if (tile.L, tile.T) == (nx, ny):
					candidates.append(tile)

		if not candidates:
			return []

		# Try selecting a tile, respecting cooldown probabilistically
		for _ in range(3):  # max reselection attempts
			tile = random.choice(candidates)
			if tile.cooldown <= 0:
				return [tile]
			else:
				tile.cooldown -= 1  # penalize repeated hits

		# Fallback: allow anything if all are cooling down
		return [random.choice(candidates)]


	def read(self):
		self.canvas[:] = noisy_background
		# Start new slide if none active
		if self.moving_tile is None:
			neighbors = self.get_neighbors(self.empty_pos)
			if neighbors:
				self.moving_tile = random.choice(neighbors)
				self.start_pos = (self.moving_tile.L, self.moving_tile.T)
				self.end_pos = self.empty_pos
				self.current_slide = 0

				self.slide_steps+=random.randint(-2, 5)
				if self.slide_steps<=30:
					self.slide_steps = 30
				elif self.slide_steps>=60:
					self.slide_steps = 60
				
				#self.slide_steps = 1 # instant debug



		# Draw static tiles
		for tile in self.tiles:
			if tile != self.moving_tile:
				tile.draw(self.canvas)
			tile.tick_cooldown()

		# Animate moving tile
		if self.moving_tile:
			t = self.current_slide / self.slide_steps
			cur_x = int(self.start_pos[0] + (self.end_pos[0]-self.start_pos[0]) * t)
			cur_y = int(self.start_pos[1] + (self.end_pos[1]-self.start_pos[1]) * t)
			self.moving_tile.draw(self.canvas, cur_x, cur_y)
			self.current_slide += 1
			if self.current_slide > self.slide_steps:
				self.moving_tile.L, self.moving_tile.T = self.end_pos
				self.moving_tile.on_moved(cooldown=5)

				if self.moving_tile.current_letter is None:
					self.slide_steps-=30
				#letter = self.moving_tile.current_letter
				#if letter is not None:
				#	self.moving_tile.set_letter(letter.lower())
				
				self.last_moved = self.moving_tile
				self.empty_pos = self.start_pos
				self.moving_tile = None
				

		return self.canvas

import time

# --- Run animation ---
if __name__=="__main__":
	delay = int(1000 / 59.94)
	cv2.namedWindow("SlidingTiles", cv2.WINDOW_NORMAL)
	cv2.setWindowProperty("SlidingTiles", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	seconds_to_reset = 300  # reset after 60 seconds

	anim = SlidingTileAnimation()  # create first animation

	display_bgs = []
	for x in range(0, 10):
		display_bgs.append(generate_display_bg())

	bg_index = 0
	while True:
		start_time = time.time()

		if bg_index>=len(display_bgs):
			bg_index = 0
		display_background = display_bgs[bg_index]

		while True:
			# Get animation frame
			anim_frame = anim.read()  # 1024x1024

			# Center animation on full display background (1280x1024)
			final_canvas = display_background.copy()
			start_x = (DISPLAY_RES[0] - ANIM_RES[0]) // 2
			start_y = (DISPLAY_RES[1] - ANIM_RES[1]) // 2
			final_canvas[start_y:start_y+ANIM_RES[1], start_x:start_x+ANIM_RES[0]] = anim_frame
			
			cv2.imshow("SlidingTiles", final_canvas)
			if cv2.waitKey(delay) & 0xFF == ord('q'):
				break  # quit completely

			# check if reset time reached
			if time.time() - start_time >= seconds_to_reset:
				break  # exit inner loop to recreate animation

		anim.redraw_middle_rows()
		bg_index+=1
