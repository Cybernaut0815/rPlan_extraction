import argparse
import json
import logging
import os
import time
from datetime import datetime

from tqdm import tqdm

from helpers.fp import Floorplan
from helpers.utils import load_image_paths
from helpers.logging import TqdmLoggingHandler


def main():
	parser = argparse.ArgumentParser(description="Extract original room sizes (m2) for each floorplan image.")
	parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
	parser.add_argument('--output_path', type=str, default='./dataset_stats', help='Path to save outputs')
	parser.add_argument('--max_index', type=int, default=-1, help='Maximum number of images to process (-1 for all)')
	parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
	parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
	parser.add_argument('--room_types', type=str, default=None, help='Comma-separated room types to filter (e.g., "bedroom,bathroom"). If not specified, all rooms are included.')
	args = parser.parse_args()

	data_path = args.data_path
	output_path = args.output_path
	log_dir = args.log_dir
	
	# Parse room types filter
	room_types_filter = None
	if args.room_types:
		room_types_filter = [rt.strip() for rt in args.room_types.split(',')]

	os.makedirs(output_path, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file = os.path.join(log_dir, f"size_extraction_{timestamp}.log")
	log_level = getattr(logging, args.log_level.upper(), logging.INFO)

	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)
	console_handler = TqdmLoggingHandler()
	console_handler.setFormatter(formatter)

	logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
	logger = logging.getLogger("rplan_size_extraction")

	paths = load_image_paths(data_path)
	max_index_count = args.max_index if args.max_index > 0 else len(paths)
	filter_msg = f", room_types_filter={room_types_filter}" if room_types_filter else ""
	logger.info(
		f"Run params -> max_index_count={max_index_count}, output_path={output_path}, log_file={log_file}, log_level={args.log_level}{filter_msg}"
	)
	logger.info(f"Found {len(paths)} images in dataset: {data_path}")

	pbar = tqdm(total=max_index_count, desc="Extracting sizes", unit="img", leave=True, dynamic_ncols=True)

	all_results = []

	for i, path in enumerate(paths[:max_index_count]):
		if i % 1000 == 0:
			logger.info(f"Progress checkpoint: {i}/{max_index_count}")

		try:
			start_time = time.perf_counter()
			logger.info(f"[{i+1}/{max_index_count}] Start processing: {path}")

			fp = Floorplan(os.path.join(data_path, path))
			sizes_by_node = fp.get_original_room_sizes_by_node()
			connectivity = fp.get_room_connectivity()

			# Apply room type filter if specified
			if room_types_filter:
				# Filter sizes: keep only nodes matching specified room types
				filtered_sizes = {
					k: v for k, v in sizes_by_node.items() 
					if any(k.startswith(rt + '_') for rt in room_types_filter)
				}
				
				# Filter connectivity: keep only filtered nodes
				filtered_nodes = set(filtered_sizes.keys())
				filtered_adjacency = {
					k: [n for n in v if n in filtered_nodes]
					for k, v in connectivity['adjacency'].items()
					if k in filtered_nodes
				}
				
				# Update room counts to only include filtered types
				filtered_room_counts = {
					rt: count for rt, count in connectivity['room_counts'].items()
					if rt in room_types_filter
				}
				
				sizes_by_node = filtered_sizes
				connectivity = {
					'room_counts': filtered_room_counts,
					'adjacency': filtered_adjacency
				}

			# Ensure values are rounded to 1 decimal (defensive)
			rounded_sizes = {k: round(v, 1) for k, v in sizes_by_node.items()}

			# Inner progress bar per size extracted
			total_sizes = len(rounded_sizes)
			inner_desc = f"Sizes {os.path.basename(path)}"
			inner_pbar = tqdm(total=total_sizes, desc=inner_desc, unit="room", leave=False, dynamic_ncols=True, position=1)
			for _ in rounded_sizes:
				inner_pbar.update(1)
			inner_pbar.close()

			result = {
				"file": path,
				"room_sizes_m2": rounded_sizes,
				"connectivity": connectivity
			}

			all_results.append(result)

			duration = time.perf_counter() - start_time
			logger.info(
				f"[{i+1}/{max_index_count}] Processed: {path} | rooms: {total_sizes} | elapsed: {duration:.2f}s"
			)
		except Exception:
			logger.exception(f"[{i+1}/{max_index_count}] Error processing {path}")
		finally:
			pbar.update(1)

	pbar.close()

	# Write a single aggregated JSON file
	aggregated_path = os.path.join(output_path, "sizes.json")
	with open(aggregated_path, "w", encoding="utf-8") as f:
		json.dump({
			"dataset": data_path,
			"count": len(all_results),
			"items": all_results
		}, f, indent=2)

	logger.info(f"Aggregated sizes saved to: {aggregated_path}")


if __name__ == "__main__":
	main()