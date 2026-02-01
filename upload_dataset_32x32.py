from huggingface_hub import login, HfApi
from datasets import Dataset
import dotenv
from pathlib import Path
import json
import numpy as np
from PIL import Image

dotenv.load_dotenv(override=True)
token = dotenv.get_key(dotenv_path=".env", key_to_get="HF_API_KEY")
login(token=token)

# Configuration
REPO_ID = "Cybernaut101/floorplans_subset_10000"
DATA_PATH = Path(r"E:\Datasets\rPlan\dataset\floorplan_dataset_images_10000")
STATS_PATH = Path(r"c:\Users\chris\Documents\Github\rplan_extraction\dataset_stats\stats.json")

# Load stats for dataset description
with open(STATS_PATH, 'r', encoding='utf-8') as f:
    stats = json.load(f)

# Room type names in order (0-12)
ROOM_NAMES = [
    "living room", "master room", "kitchen", "bathroom", "dining room",
    "child room", "study room", "second room", "guest room", "balcony",
    "entrance", "storage", "external area"
]

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

ROOM_INSTANCE_COLOR_MAP, NUM_ROOM_INSTANCES = build_room_instance_color_map()

# Generate README content with YAML frontmatter for HuggingFace
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

# rPlan Floorplan Dataset (32x32, 10K subset)

A subset of 10,000 floorplan images from the rPlan dataset, resized to 32x32 pixels.
Stored as compressed Parquet with numpy arrays for efficient loading.

## Dataset Format

Each sample contains:
- **id**: Image identifier (string)
- **room_instances**: 32x32 uint8 numpy array - room instance segmentation ({NUM_ROOM_INSTANCES} unique classes)
- **interior_mask**: 32x32 uint8 numpy array - interior/exterior mask (255 = interior, 0 = exterior)

## Room Instance Color Mapping ({NUM_ROOM_INSTANCES} Classes)

| Class ID | Color Value | Class Name |
|----------|-------------|------------|
"""

# Add color mapping table for all 35 classes
for class_id, info in ROOM_INSTANCE_COLOR_MAP.items():
    readme_content += f"| {class_id} | {info['color_value']} | {info['class_name'].title()} |\n"

readme_content += f"""
## Usage

```python
from datasets import load_dataset
import numpy as np

# Load the dataset
ds = load_dataset("Cybernaut101/rPlan_subset_10000")

# Access a sample
sample = ds["train"][0]
room_instances = np.array(sample["room_instances"])  # 32x32 uint8
interior_mask = np.array(sample["interior_mask"])    # 32x32 uint8

# Iterate over the dataset
for sample in ds["train"]:
    room_instances = np.array(sample["room_instances"])
    interior_mask = np.array(sample["interior_mask"])
    # ... your processing here

# Color value -> class name mapping
ROOM_INSTANCE_COLOR_MAP = {{
"""

# Add the Python dict to the usage example
for class_id, info in ROOM_INSTANCE_COLOR_MAP.items():
    readme_content += f"    {info['color_value']}: \"{info['class_name']}\",\n"

readme_content += """}
```

## License

Please refer to the original rPlan dataset license for usage terms.
"""

# Build dataset from images as numpy arrays
print("Loading images into dataset...")
rows = []
for i, png in enumerate(sorted(DATA_PATH.glob("*.png"))):
    arr = np.array(Image.open(png), dtype=np.uint8)
    rows.append({
        "id": png.stem,
        "room_instances": arr[:, :, 0],  # 32x32 uint8 array (L channel)
        "interior_mask": arr[:, :, 1],   # 32x32 uint8 array (A channel)
    })
    if (i + 1) % 1000 == 0:
        print(f"  Loaded {i + 1} images...")

print(f"Loaded {len(rows)} images total")

# Create dataset
ds = Dataset.from_list(rows)

# Push to hub with Parquet format (automatic compression)
print("Uploading dataset as Parquet...")
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