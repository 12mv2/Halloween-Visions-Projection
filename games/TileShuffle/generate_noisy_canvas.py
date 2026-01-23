import cv2
import numpy as np
from pathlib import Path
import random

# --- Settings ---
image_dir = Path(r"C:\Users\Admin\Desktop\tiles")
canvas_width = 1024
canvas_height = 1024
output_file = r"C:\Users\Admin\Desktop\noisy_canvas.png"
dark_threshold = 140*3  # sum of R+G+B below this is considered too dark

# --- Step 1: Sample first column colors ---
colors = []

for img_path in image_dir.glob("*.*"):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        continue
    first_col = img[:, 0, :]  # shape: (height, 3)
    
    # Only keep bright enough pixels
    for pixel in first_col:
        if pixel.sum() > dark_threshold:
            colors.append(tuple(pixel))

print(f"Collected {len(colors)} colors from first columns (filtered).")

# --- Step 2: Generate noisy canvas ---
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

for y in range(canvas_height):
    for x in range(canvas_width):
        canvas[y, x] = random.choice(colors)

# --- Step 3: Save to file ---
cv2.imwrite(output_file, canvas)
print(f"Noisy canvas saved to {output_file}")

# Optional display
cv2.imshow("Noisy Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
