from huggingface_hub import login, HfApi
from datasets import Dataset
import dotenv
from pathlib import Path
import json
import numpy as np
from PIL import Image
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Upload floorplan dataset to HuggingFace Hub as Parquet.")
parser.add_argument('--repo_id', type=str, required=True, help='HuggingFace repository ID (e.g., "Cybernaut101/floorplans_subset_10000")')
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder containing PNG images')
parser.add_argument('--stats_path', type=str, default="dataset_stats/stats.json", 
                    help='Path to stats.json (only needed for instance mode)')
parser.add_argument('--basic_types', action='store_true', 
                    help='Use 13 basic room types instead of 35 instance-based classes')

args = parser.parse_args()

REPO_ID = args.repo_id
DATA_PATH = Path(args.data_path)
STATS_PATH = Path(args.stats_path) if args.stats_path else None
BASIC_TYPES_MODE = args.basic_types

dotenv.load_dotenv(override=True)
token = dotenv.get_key(dotenv_path=".env", key_to_get="HF_API_KEY")
login(token=token)

# Room type names in order (0-12)
ROOM_NAMES = [
    "living room", "master room", "kitchen", "bathroom", "dining room",
    "child room", "study room", "second room", "guest room", "balcony",
    "entrance", "storage", "external area"
]

NUM_ROOM_TYPES = 13  # Basic room types

if BASIC_TYPES_MODE:
    # Build simple color mapping for 13 basic room types
    def build_basic_room_color_map():
        """Build the color mapping for 13 basic room types"""
        color_map = {}
        for room_type_id, room_name in enumerate(ROOM_NAMES):
            color_value = int(room_type_id * (255 / (NUM_ROOM_TYPES - 1))) if NUM_ROOM_TYPES > 1 else 0
            color_map[room_type_id] = {
                "color_value": color_value,
                "class_name": room_name
            }
        return color_map, NUM_ROOM_TYPES

    ROOM_COLOR_MAP, NUM_CLASSES = build_basic_room_color_map()
else:
    # Load stats for instance-based mapping
    if STATS_PATH is None or not STATS_PATH.exists():
        raise ValueError("--stats_path is required when not using --basic_types mode")
    
    with open(STATS_PATH, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    # Get max instances from stats
    room_type_max_instances = stats.get('room_type_max_instances', {})
    # Add defaults for entrance and external area if not in stats
    room_type_max_instances.setdefault("entrance", 1)
    room_type_max_instances.setdefault("external area", 1)

    # Build the actual color mapping used in extraction (35 unique room instances)
    def build_room_instance_color_map():
        """Build the color mapping for 35 unique (room_type, instance) combinations"""
        total_instances = sum(room_type_max_instances.get(name, 1) for name in ROOM_NAMES)
        color_map = {}
        color_idx = 0
        for room_type_id, room_name in enumerate(ROOM_NAMES):
            max_instances = room_type_max_instances.get(room_name, 1)
            for instance_id in range(max_instances):
                color_value = int(color_idx * (255 / (total_instances - 1))) if total_instances > 1 else 0
                instance_str = f"_{instance_id}" if max_instances > 1 else ""
                color_map[color_idx] = {
                    "color_value": color_value,
                    "class_name": f"{room_name}{instance_str}"
                }
                color_idx += 1
        return color_map, total_instances

    ROOM_COLOR_MAP, NUM_CLASSES = build_room_instance_color_map()

# Generate README content with YAML frontmatter for HuggingFace
mode_description = "basic room types" if BASIC_TYPES_MODE else "room instance segmentation"
readme_content = f"""---
license: other
task_categories:
  - image-segmentation
tags:
  - floorplan
  - rplan
  - architecture
  - parquet
size_categories:
  - 1K<n<10K
---

# rPlan Floorplan Dataset (32x32)

Floorplan images from the rPlan dataset, resized to 32x32 pixels.
Stored as compressed Parquet with numpy arrays for efficient loading.

## Dataset Format

Each sample contains:
- **id**: Image identifier (string)
- **room_labels**: 32x32 uint8 numpy array - {mode_description} ({NUM_CLASSES} unique classes)
- **interior_mask**: 32x32 uint8 numpy array - interior/exterior mask (255 = interior, 0 = exterior)

## Room Color Mapping ({NUM_CLASSES} Classes)

| Class ID | Color Value | Class Name |
|----------|-------------|------------|
"""

# Add color mapping table
for class_id, info in ROOM_COLOR_MAP.items():
    readme_content += f"| {class_id} | {info['color_value']} | {info['class_name'].title()} |\n"

readme_content += f"""
## Usage

```python
from datasets import load_dataset
import numpy as np

# Load the dataset
ds = load_dataset("{REPO_ID}")

# Access a sample
sample = ds["train"][0]
room_labels = np.array(sample["room_labels"])  # 32x32 uint8
interior_mask = np.array(sample["interior_mask"])    # 32x32 uint8

# Iterate over the dataset
for sample in ds["train"]:
    room_labels = np.array(sample["room_labels"])
    interior_mask = np.array(sample["interior_mask"])
    # ... your processing here

# Color value -> class name mapping
ROOM_COLOR_MAP = {{
"""

# Add the Python dict to the usage example
for class_id, info in ROOM_COLOR_MAP.items():
    readme_content += f"    {info['color_value']}: \"{info['class_name']}\",\n"

readme_content += """}
```

## License

Please refer to the original rPlan dataset license for usage terms.
"""

# Build dataset from images as numpy arrays
print(f"Loading images from: {DATA_PATH}")
print(f"Mode: {'basic types (13 classes)' if BASIC_TYPES_MODE else 'instance-based (35 classes)'}")
rows = []
for i, png in enumerate(sorted(DATA_PATH.glob("*.png"))):
    arr = np.array(Image.open(png), dtype=np.uint8)
    rows.append({
        "id": png.stem,
        "room_labels": arr[:, :, 0],  # 32x32 uint8 array (L channel)
        "interior_mask": arr[:, :, 1],   # 32x32 uint8 array (A channel)
    })
    if (i + 1) % 1000 == 0:
        print(f"  Loaded {i + 1} images...")

print(f"Loaded {len(rows)} images total")

# Create dataset
ds = Dataset.from_list(rows)

# Push to hub with Parquet format (automatic compression)
print(f"Uploading dataset to {REPO_ID}...")
ds.push_to_hub(
    REPO_ID,
    commit_message="Upload as compressed Parquet with raw arrays",
)

# Upload README.md separately
print("Uploading README.md...")
api = HfApi()
api.upload_file(
    path_or_fileobj=readme_content.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add dataset README",
)
print(f"Dataset uploaded to: https://huggingface.co/datasets/{REPO_ID}")