# %%
import json
import os

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from helpers.info import Info
from helpers.utils import load_image_paths
from helpers.fp import Floorplan
from helpers.dataset_viz import load_and_visualize_datapoint

from shapely.geometry import LineString, Polygon
import networkx as nx

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


# %%

DATA_PATH = r"D:\Datasets\rPlan\dataset\floorplan_dataset"
OUTPUT_PATH = r"D:\Datasets\rPlan\dataset\dataset\output_first_100"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)



#%%

paths = load_image_paths(DATA_PATH)
# open a test image
if paths:
    test_path = os.path.join(DATA_PATH, paths[3])
    img = Image.open(test_path)
else:
    print("No image files found in directory")

# %%

wall_width = 3.0
my_fp = Floorplan(test_path, wall_width=wall_width)

# %%

#my_fp.draw_connectivity_graph()
#my_fp.draw_contours(offset=False)
my_fp.draw_room_connectivity_on_plan()


# %%
print(my_fp.get_room_types_count())


# %%

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
system_message = "You are a helpful assistant that creates a text description of a floor plan based on the given data containing information about the room types, counts and how they are connected. Be concise and to the point, not too long or too poetic. Don't try to sell the flat and don't explain in detail what the rooms are for."
query = "Write a short descriptions like a prompt for a flat with the given features. The prompt should be in natural language and describe the the flat based on the given count of different room types and how they are connected. This is the data: "

# %%

data = my_fp.generate_llm_descriptions(llm, system_message, query)

for i, description in enumerate(data["descriptions"]):
    print(f"Description {i+1}:")
    print(description)
    print("\n")



# %%

# test plotting first 10 paths to check for 

for i, path in enumerate(paths[:10]):
    if i == 10:
        break
    print(f"Processing {i} of {len(paths)}")
    full_path = os.path.join(DATA_PATH, path)
    my_fp = Floorplan(full_path, wall_width=wall_width)
    my_fp.draw_room_connectivity_on_plan()


# %%

import importlib
import helpers.fp
import helpers.utils
importlib.reload(helpers.fp)
importlib.reload(helpers.utils)
from helpers.info import Info
from helpers.utils import load_image_paths
from helpers.fp import Floorplan

# test resizing and wall removal for simple diffusion model training
import random as rand

random_path = os.path.join(DATA_PATH, paths[rand.randint(0, len(paths)-1)])
test_fp = Floorplan(random_path, wall_width=wall_width)

size = 64

resized_fp_pixels = test_fp.pixel_based_resize(64)
resized_fp_outlines = test_fp.outline_based_resize(64)

test_fp.draw_room_connectivity_on_plan()


print("pixel-based shape:", resized_fp_pixels.shape, "outline-based shape:", resized_fp_outlines.shape)
plt.figure(figsize=(12,6))
ax1 = plt.subplot(1,2,1)
ax1.imshow(resized_fp_pixels[:,:,0])
ax1.set_title('Pixel-based Resize')
ax1.axis('off')

ax2 = plt.subplot(1,2,2)
ax2.imshow(resized_fp_outlines[:,:,0])
ax2.set_title('Outline-based Resize')
ax2.axis('off')

plt.tight_layout()
plt.show()

# %%

# label generation loop

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

run_label_loop = True
max_index_count = 10

image_size_px = 64

if run_label_loop:
    for i, path in tqdm(enumerate(paths[:max_index_count]), total=max_index_count):
        
        if i % 10 == 0:
            print(f"Processing {i} of {len(paths)}")
        
        try:
            my_fp = Floorplan(os.path.join(DATA_PATH, path), wall_width=wall_width)
            data = my_fp.generate_llm_descriptions(llm, system_message, query, pixel_based_size=image_size_px)
            
            # save the description
            with open(os.path.join(OUTPUT_PATH, f"description_{i}.json"), "w") as f:
                json.dump(data, f)
            logger.info(f"Processed {i} of {len(paths)}")
            
        except Exception as e:
            logger.error(f"Error processing plan {path}: {e}")


# %%

import importlib
import helpers.dataset_viz
importlib.reload(helpers.dataset_viz)
from helpers.dataset_viz import load_and_visualize_datapoint

# Now visualize
load_and_visualize_datapoint(1, OUTPUT_PATH)


# %%