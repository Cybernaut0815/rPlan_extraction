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

info = Info()

parser = argparse.ArgumentParser(description="Run floorplan extraction and visualization.")
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs')
parser.add_argument('--max_index', type=int, default=-1, help='Maximum number of images to process')
parser.add_argument('--image_size', type=int, default=64, help='Size of the image for processing')
parser.add_argument('--wall_width', type=float, default=3.0, help='Wall width for floorplan processing')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
parser.add_argument('--basic_types', action='store_true', 
                    help='Use 15 basic room types (bathroom 1/2, second room 1/2 adjacent). '
                         'Balconies share color. Skips images exceeding allowed instance limits.')

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
    basic_types_mode = args.basic_types
    logger.info(
        f"Run params -> wall_width={wall_width}, image_size_px={image_size_px}, "
        f"max_index_count={max_index_count}, output_path={OUTPUT_PATH}, log_file={log_file}, "
        f"log_level={args.log_level}, basic_types={basic_types_mode}"
    )
    if basic_types_mode:
        logger.info(
            f"Basic types mode: Using {info.num_basic_room_types} room types "
            f"(13 base + bathroom 2 + second room 2), "
            f"balconies share color, skipping images exceeding instance limits"
        )

    # Initialize a tqdm progress bar that stays on the last line
    pbar = tqdm(total=max_index_count, desc="Extracting", unit="img", leave=True, dynamic_ncols=True)

    exported_count = 0  # Track successfully exported images
    skipped_count = 0   # Track skipped images
    
    for i, path in enumerate(paths):
        # Stop when we've exported enough images
        if exported_count >= max_index_count:
            break
            
        if exported_count % 10 == 0 and exported_count > 0:
            logger.info(f"Progress checkpoint: {exported_count}/{max_index_count} exported, {skipped_count} skipped")

        try:
            start_time = time.perf_counter()
            logger.debug(f"[{exported_count+1}/{max_index_count}] Processing: {path}")

            my_fp = Floorplan(os.path.join(DATA_PATH, path), wall_width=wall_width)

            # In basic_types mode, skip images with multiple instances of any room type
            # Check using the room graph (more reliable than pixel-based detection)
            if basic_types_mode and not my_fp.has_single_instances_only():
                skipped_count += 1
                logger.debug(f"Skipping {path}: has multiple instances of same room type (skipped: {skipped_count})")
                continue

            resized_image = my_fp.outline_based_resize(image_size_px)

            # Remap room types based on mode
            if basic_types_mode:
                remapped_rooms = my_fp.remap_rooms(resized_image, mode="basic_types")
            else:
                remapped_rooms = my_fp.remap_rooms(resized_image, mode="instances")
            
            # Create exterior area mask (alpha channel)
            alpha_mask = my_fp.create_exterior_mask(resized_image)

            # Save PNG with alpha channel for exterior area
            # Luminance channel: remapped room types/instances
            # Alpha channel: 255 for interior, 0 for exterior (transparent)
            output_file = os.path.join(OUTPUT_PATH, f"image_{exported_count}.png")
            output_img = Image.fromarray(np.stack([remapped_rooms, alpha_mask], axis=-1), mode='LA')
            output_img.save(output_file)

            duration = time.perf_counter() - start_time
            logger.info(
                f"[{exported_count+1}/{max_index_count}] Saved: {output_file} | shape: {remapped_rooms.shape} | elapsed: {duration:.2f}s"
            )
            
            exported_count += 1
            pbar.update(1)

        except Exception as e:
            logger.exception(f"Error processing {path}")

    pbar.close()
    logger.info(f"Extraction complete: {exported_count} images exported, {skipped_count} skipped")

