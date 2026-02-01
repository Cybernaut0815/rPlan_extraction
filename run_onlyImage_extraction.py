import argparse
import os
import json

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

# Default maximum instances per room type (fallback if stats.json not available)
# This ensures consistent color mapping across all floorplans
# Based on statistics extracted from the rPlan dataset:
#   living room: 1, master room: 5, kitchen: 2, bathroom: 3, dining room: 2,
#   child room: 3, study room: 3, second room: 4, guest room: 3, balcony: 4,
#   entrance: 1 (assumed), storage: 3, external area: 1 (not instanced)
ROOM_TYPE_MAX_INSTANCES_DEFAULT = {
    "living room": 1,
    "master room": 5,
    "kitchen": 2,
    "bathroom": 3,
    "dining room": 2,
    "child room": 3,
    "study room": 3,
    "second room": 4,
    "guest room": 3,
    "balcony": 4,
    "entrance": 1,
    "storage": 3,
    "external area": 1,  # Not instanced, always single
}

def _load_room_type_max_instances():
    """
    Load room_type_max_instances from stats.json if available.
    Falls back to ROOM_TYPE_MAX_INSTANCES_DEFAULT if stats.json is not found or missing the key.
    
    Returns:
        dict: Mapping of room_type_name -> max_instance_count
    """
    stats_path = os.path.join(os.path.dirname(__file__), 'dataset_stats', 'stats.json')
    
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            loaded_instances = stats.get('room_type_max_instances', {})
            if loaded_instances:
                # Merge with defaults to ensure all room types are covered
                # (entrance and external area may not appear in stats)
                result = ROOM_TYPE_MAX_INSTANCES_DEFAULT.copy()
                result.update(loaded_instances)
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load stats.json: {e}. Using default values.")
    
    return ROOM_TYPE_MAX_INSTANCES_DEFAULT.copy()

# Load room type max instances from stats.json (or use defaults)
ROOM_TYPE_MAX_INSTANCES = _load_room_type_max_instances()

# Create a mapping from (room_type_id, instance_id) to a unique color value
# This allows differentiating rooms of the same type (e.g., bathroom_0, bathroom_1)
# Color values are spread across 0-255 for visual differentiation
def _build_room_instance_color_map():
    """
    Build a mapping from (room_type_id, instance_id) -> unique_color_value.
    
    The color values are assigned sequentially and spread across 0-255 range.
    Room types with only 1 instance get a single color.
    Room types with multiple instances get sequential colors for each instance.
    
    Returns:
        dict: Mapping of (room_type_id, instance_id) -> color_value (0-255)
        dict: Reverse mapping of color_value -> (room_type_name, instance_id)
    """
    color_map = {}
    reverse_map = {}
    
    # Calculate total number of unique room instances
    total_instances = sum(ROOM_TYPE_MAX_INSTANCES.values())
    
    color_idx = 0
    for room_type_id, room_name in REMAP_TO_NAME.items():
        max_instances = ROOM_TYPE_MAX_INSTANCES.get(room_name, 1)
        for instance_id in range(max_instances):
            # Spread colors evenly across 0-255 range
            color_value = int(color_idx * (255 / (total_instances - 1))) if total_instances > 1 else 0
            color_map[(room_type_id, instance_id)] = color_value
            reverse_map[color_value] = (room_name, instance_id)
            color_idx += 1
    
    return color_map, reverse_map

ROOM_INSTANCE_COLOR_MAP, COLOR_TO_ROOM_INSTANCE = _build_room_instance_color_map()

# Total number of unique room instances (for reference)
NUM_ROOM_INSTANCES = sum(ROOM_TYPE_MAX_INSTANCES.values())  # 35 total

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


def remap_room_instances(image_array):
    """
    Remap room type + instance values to unique colors (35 classes).
    
    Uses channel 0 (room type) and channel 2 (instance index) from the original image.
    Channel 2 contains different integers to distinguish rooms of the same type
    (0 = non-room area, 1+ = instance index).
    
    Args:
        image_array: Input image array with at least 3 channels
    
    Returns:
        remapped: Single-channel array with unique color per (room_type, instance)
    """
    if len(image_array.shape) != 3 or image_array.shape[2] < 3:
        raise ValueError("Image must have at least 3 channels for instance remapping")
    
    room_channel = image_array[:, :, 0]      # Room type (original values)
    instance_channel = image_array[:, :, 2]  # Instance index (1-based, 0 for non-room)
    
    remapped = np.zeros_like(room_channel, dtype=np.uint8)
    
    for orig_val, new_val in ROOM_TYPE_REMAP.items():
        room_name = REMAP_TO_NAME.get(new_val, "unknown")
        max_instances = ROOM_TYPE_MAX_INSTANCES.get(room_name, 1)
        
        # Mask for this room type
        room_mask = (room_channel == orig_val)
        
        # Special case: external area (orig_val=13) has instance_channel=0
        # It's not instanced like rooms, so just use the room_mask directly
        if orig_val == 13:  # External area
            color_key = (new_val, 0)
            if color_key in ROOM_INSTANCE_COLOR_MAP:
                color_value = ROOM_INSTANCE_COLOR_MAP[color_key]
                remapped[room_mask] = color_value
            continue
        
        for instance_id in range(max_instances):
            # Instance channel is 1-based, so instance_id 0 corresponds to value 1
            # For rooms with only 1 instance, any non-zero instance value maps to instance 0
            if max_instances == 1:
                instance_mask = room_mask & (instance_channel > 0)
            else:
                instance_mask = room_mask & (instance_channel == (instance_id + 1))
            
            # Get color value from the pre-built map
            color_key = (new_val, instance_id)
            if color_key in ROOM_INSTANCE_COLOR_MAP:
                color_value = ROOM_INSTANCE_COLOR_MAP[color_key]
                remapped[instance_mask] = color_value
    
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

            # Remap room types + instances to unique colors (35 classes)
            # Uses channel 0 (room type) and channel 2 (instance index)
            remapped_rooms = remap_room_instances(resized_image)
            
            # Create exterior area mask (alpha channel)
            alpha_mask = create_exterior_mask(resized_image)

            # Save PNG with alpha channel for exterior area
            # Luminance channel: remapped room instances (35 unique colors)
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

