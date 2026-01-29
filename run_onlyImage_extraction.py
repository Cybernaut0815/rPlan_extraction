import argparse
import os

from PIL import Image
import numpy as np

from helpers.utils import load_image_paths
from helpers.fp import Floorplan
from helpers.info import Info

import logging
from tqdm import tqdm

from datetime import datetime
import time
from helpers.logging import TqdmLoggingHandler

# Define mapping from original room type values to continuous integers (0-based)
# Original values without walls: 0-11 (rooms), 13 (external area)
# Wall-in (12), exterior wall (14), front door (15), interior wall (16), interior door (17) are excluded
ROOM_TYPE_REMAP = {
    0: 0,   # living room
    1: 1,   # master room
    2: 2,   # kitchen
    3: 3,   # bathroom
    4: 4,   # dining room
    5: 5,   # child room
    6: 6,   # study room
    7: 7,   # second room
    8: 8,   # guest room
    9: 9,   # balcony
    10: 10, # entrance
    11: 11, # storage
    13: 12, # external area -> remapped to 12
}

# Number of room types (for visual scaling)
NUM_ROOM_TYPES = 13  # 0-12 inclusive

# Reverse mapping for reference
REMAP_TO_NAME = {
    0: "living room",
    1: "master room",
    2: "kitchen",
    3: "bathroom",
    4: "dining room",
    5: "child room",
    6: "study room",
    7: "second room",
    8: "guest room",
    9: "balcony",
    10: "entrance",
    11: "storage",
    12: "external area",
}

def remap_room_types(image_array, visual_spread=True):
    """
    Remap room type values to continuous integers.
    Input channel 0 contains original room type values.
    
    Args:
        image_array: Input image array
        visual_spread: If True, spread values across 0-255 for visual differentiation
                      If False, keep compact 0-12 values
    
    Returns remapped single-channel array.
    """
    if len(image_array.shape) == 3:
        room_channel = image_array[:, :, 0]
    else:
        room_channel = image_array
    
    remapped = np.zeros_like(room_channel, dtype=np.uint8)
    for orig_val, new_val in ROOM_TYPE_REMAP.items():
        if visual_spread:
            # Spread values across 0-255 range for visual differentiation
            # Each room type gets a distinct grayscale value
            visual_val = int(new_val * (255 / (NUM_ROOM_TYPES - 1)))
            remapped[room_channel == orig_val] = visual_val
        else:
            remapped[room_channel == orig_val] = new_val
    
    return remapped

def create_exterior_mask(image_array):
    """
    Create a mask where exterior area is 0 (transparent) and interior is 255 (opaque).
    External area is originally value 13, remapped to 12.
    """
    if len(image_array.shape) == 3:
        room_channel = image_array[:, :, 0]
    else:
        room_channel = image_array
    
    # External area is value 13 in original
    exterior_mask = (room_channel == 13)
    
    # Create alpha channel: 255 for interior, 0 for exterior
    alpha = np.where(exterior_mask, 0, 255).astype(np.uint8)
    
    return alpha

parser = argparse.ArgumentParser(description="Run floorplan extraction and visualization.")
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs')
parser.add_argument('--max_index', type=int, default=-1, help='Maximum number of images to process')
parser.add_argument('--image_size', type=int, default=64, help='Size of the image for processing')
parser.add_argument('--wall_width', type=float, default=3.0, help='Wall width for floorplan processing')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

args = parser.parse_args()

DATA_PATH = args.data_path
OUTPUT_PATH = args.output_path
LOG_DIR = args.log_dir

# R_PLAN_MeterToPixel = 16  # 16 pixel are one meter in rPlan   
# # use this to calculate the m2 scale of the rooms on the plan

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Configure logging to file in LOG_DIR and to console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"extraction_{timestamp}.log")
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
    logger = logging.getLogger("rplan_extraction")

    paths = load_image_paths(DATA_PATH)
    wall_width = args.wall_width
    logger.info(f"Found {len(paths)} images in dataset: {DATA_PATH}")

    max_index_count = args.max_index if args.max_index > 0 else len(paths)
    image_size_px = args.image_size
    logger.info(
        f"Run params -> wall_width={wall_width}, image_size_px={image_size_px}, "
        f"max_index_count={max_index_count}, output_path={OUTPUT_PATH}, log_file={log_file}, "
        f"log_level={args.log_level}"
    )

    # Initialize a tqdm progress bar that stays on the last line
    pbar = tqdm(total=max_index_count, desc="Extracting", unit="img", leave=True, dynamic_ncols=True)

    for i, path in enumerate(paths[:max_index_count]):
        if i % 10 == 0:
            logger.info(f"Progress checkpoint: {i}/{max_index_count}")

        try:
            start_time = time.perf_counter()
            logger.info(f"[{i+1}/{max_index_count}] Start processing: {path}")

            my_fp = Floorplan(os.path.join(DATA_PATH, path), wall_width=wall_width)
            resized_image = my_fp.outline_based_resize(image_size_px)

            # Remap room types to continuous integers (0-12)
            remapped_rooms = remap_room_types(resized_image)
            
            # Create exterior area mask (alpha channel)
            alpha_mask = create_exterior_mask(resized_image)

            # Save PNG with alpha channel for exterior area
            # Luminance channel: remapped room types (0-12)
            # Alpha channel: 255 for interior, 0 for exterior (transparent)
            output_file = os.path.join(OUTPUT_PATH, f"image_{i}.png")
            output_img = Image.fromarray(np.stack([remapped_rooms, alpha_mask], axis=-1), mode='LA')
            output_img.save(output_file)

            duration = time.perf_counter() - start_time
            logger.info(
                f"[{i+1}/{max_index_count}] Saved: {output_file} | shape: {remapped_rooms.shape} | elapsed: {duration:.2f}s"
            )

        except Exception as e:
            logger.exception(f"[{i+1}/{max_index_count}] Error processing {path}")
        finally:
            pbar.update(1)

    pbar.close()

