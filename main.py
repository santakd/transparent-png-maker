#!/usr/bin/env python3

import os
import logging
from datetime import datetime
from PIL import Image
import numpy as np

# ==========================================
# Configuration & Logging Setup
# ==========================================
INPUTS = './inputs'
OUTPUTS = './outputs'
LOG_DIR = './logs'

# Ensure required directories exist before running
os.makedirs(INPUTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure timestamped logging
log_filename = datetime.now().strftime("transparent_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler() # Also print to console
    ]
)

# ==========================================
# Core Processing Functions
# ==========================================
def make_gradient_transparent(img, gradient_colors, tolerance_percent=10):
    """
    Makes pixels near any color in a gradient transparent in a PIL image.
    Optimized to process all target colors in a single NumPy pass.
    """
    img = img.convert("RGBA")
    data = np.array(img)

    # Use int32 to prevent integer overflow when squaring (16-bit caps at 32,767)
    rgb = data[:, :, :3].astype(np.int32)
    alpha = data[:, :, 3]

    max_distance = np.sqrt(255**2 * 3)
    tolerance = (tolerance_percent / 100) * max_distance

    mask = np.zeros(alpha.shape, dtype=bool)
    
    for target_color in gradient_colors:
        # Match rgb array dtype
        target = np.array(target_color, dtype=np.int32)
        distance = np.sqrt(np.sum((rgb - target) ** 2, axis=2))
        mask |= (distance <= tolerance)

    alpha[mask] = 0

    new_data = np.dstack((rgb, alpha))
    return Image.fromarray(new_data.astype(np.uint8), "RGBA")


def linear_color_gradient(low_color, high_color, steps=10):
    """
    Generate a linear list of RGB colors from low_color to high_color.
    """
    gradient = []

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0

        r = int(low_color[0] + t * (high_color[0] - low_color[0]))
        g = int(low_color[1] + t * (high_color[1] - low_color[1]))
        b = int(low_color[2] + t * (high_color[2] - low_color[2]))

        gradient.append((r, g, b))

    return gradient

# ==========================================
# Main Execution
# ==========================================
def main():
    high_color = (218, 230, 248) 
    low_color = (163, 177, 197)
    tolerance = 3 #0.65 # Adjusted tolerance for better results based on testing

    logging.info("Generating color gradient...")
    color_gradient = linear_color_gradient(
        low_color=low_color, 
        high_color=high_color, 
        steps=75
    )

    # ADDED: Explicitly add pure white to the target list to catch white backgrounds
    color_gradient.append((255, 255, 255))

    try:
        all_files = os.listdir(INPUTS)
    except Exception as e:
        logging.error(f"Failed to read input directory {INPUTS}: {e}")
        return

    if not all_files:
        logging.warning(f"No files found in {INPUTS}. Exiting.")
        return

    # Categorize and log skipped files
    png_files = []
    for f in all_files:
        if f.lower().endswith(".png"):
            png_files.append(f)
        else:
            logging.info(f"Skipped non-PNG file: {f}")

    if not png_files:
        logging.warning(f"No valid PNG files to process in {INPUTS}. Exiting.")
        return

    logging.info(f"Starting processing for {len(png_files)} files.")

    for file in png_files:
        input_path = os.path.join(INPUTS, file)
        output_path = os.path.join(OUTPUTS, file)

        try:
            img = Image.open(input_path)
            
            result = make_gradient_transparent(img, color_gradient, tolerance)
            
            result.save(output_path)
            logging.info(f"Successfully processed and saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}", exc_info=True)

    logging.info("Batch processing complete.")

if __name__ == "__main__":
    main()
