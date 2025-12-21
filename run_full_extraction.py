import argparse
import importlib
import os
import json

from PIL import Image
import numpy as np

from helpers.info import Info
from helpers.utils import load_image_paths
from helpers.fp import Floorplan
from helpers.dataset_viz import load_and_visualize_datapoint

from shapely.geometry import LineString, Polygon
import networkx as nx

from langchain_openai import ChatOpenAI

import logging
from tqdm import tqdm

from dotenv import load_dotenv
from datetime import datetime
import time
from openai import OpenAI
from helpers.logging import TqdmLoggingHandler

load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Run floorplan extraction and visualization.")
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs')
parser.add_argument('--max_index', type=int, default=-1, help='Maximum number of images to process')
parser.add_argument('--image_size', type=int, default=64, help='Size of the image for processing')
parser.add_argument('--wall_width', type=float, default=3.0, help='Wall width for floorplan processing')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

parser.add_argument('--model', type=str, default='gpt-5', help='OpenAI chat model to use')
parser.add_argument('--list_models', action='store_true', help='List available OpenAI models and continue')
parser.add_argument('--append_to_system_message', type=str, default='', help='Append custom command to the system message for the LLM')
parser.add_argument('--append_to_query', type=str, default='', help='Append custom command to the query for the LLM')

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
    
    # Optionally list available OpenAI models
    if args.list_models:
        try:
            client = OpenAI()
            model_ids = [m.id for m in client.models.list().data]
            # Log a concise list (up to 20 entries)
            shown = model_ids[:20]
            remaining = max(len(model_ids) - len(shown), 0)
            logger.info(f"Available OpenAI models ({len(model_ids)} total): {shown}" + (f" ... (+{remaining} more)" if remaining else ""))
            if args.model not in model_ids:
                logger.warning(f"Selected model '{args.model}' not found in available models.")
        except Exception as e:
            logger.warning(f"Could not list OpenAI models: {e}")

    llm = ChatOpenAI(model=args.model, temperature=0)
    system_message = "You are a helpful assistant that creates a text description of a floor plan based on the given data containing information about the room types, counts and how they are connected. Be concise and to the point, not too long or too poetic. Don't try to sell the flat and don't explain in detail what the rooms are for."
    query = "Write a short descriptions like a prompt for a flat with the given features. The prompt should be in natural language and describe the the flat based on the given count of different room types and how they are connected. This is the data: "

    if args.append_to_system_message:
        system_message += " " + args.append_to_system_message   
    if args.append_to_query:
        query += " " + args.append_to_query

    max_index_count = args.max_index if args.max_index > 0 else len(paths)
    image_size_px = args.image_size
    logger.info(
        f"Run params -> wall_width={wall_width}, image_size_px={image_size_px}, "
        f"max_index_count={max_index_count}, output_path={OUTPUT_PATH}, log_file={log_file}, "
        f"log_level={args.log_level}, model={args.model}"
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
            data = my_fp.generate_llm_descriptions(
                llm, system_message, query, pixel_based_size=image_size_px
            )

            output_file = os.path.join(OUTPUT_PATH, f"description_{i}.json")
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            duration = time.perf_counter() - start_time
            size_info = (
                len(data) if hasattr(data, "__len__") else "unknown"
            )
            logger.info(
                f"[{i+1}/{max_index_count}] Saved: {output_file} | description size: {size_info} | elapsed: {duration:.2f}s"
            )

        except Exception as e:
            logger.exception(f"[{i+1}/{max_index_count}] Error processing {path}")
        finally:
            pbar.update(1)

    pbar.close()

