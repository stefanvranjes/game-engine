from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
ATLAS_COLS = 16
ATLAS_ROWS = 6
CHAR_WIDTH = 32  # Higher resolution for better quality, will be scaled down or used as is
CHAR_HEIGHT = 64
FONT_SIZE = 48

IMG_WIDTH = ATLAS_COLS * CHAR_WIDTH
IMG_HEIGHT = ATLAS_ROWS * CHAR_HEIGHT

# Create image with transparent background
image = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# Load a font (try to find a monospace font)
try:
    font = ImageFont.truetype("consola.ttf", FONT_SIZE)
except IOError:
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()

# Draw characters
for i in range(96):
    char_code = 32 + i
    char = chr(char_code)
    
    col = i % ATLAS_COLS
    row = i // ATLAS_COLS
    
    x = col * CHAR_WIDTH
    y = row * CHAR_HEIGHT
    
    # Center the character
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Draw white text
    draw.text((x + (CHAR_WIDTH - w) / 2, y + (CHAR_HEIGHT - h) / 2 - bbox[1]), char, font=font, fill=(255, 255, 255, 255))

# Save
output_path = "assets/font_atlas.png"
image.save(output_path)
print(f"Font atlas saved to {output_path}")
