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

from shapely.geometry import LineString, Polygon
import networkx as nx

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


# %%

DATA_PATH = r"G:\Datasets\rPlan\dataset\dataset\floorplan_dataset"
OUTPUT_PATH = r"G:\Datasets\rPlan\dataset\dataset\output_first_100"

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

# test plotting first 100 paths to check for 

# for i, path in enumerate(paths[:100]):
#     if i == 100:
#         break
#     print(f"Processing {i} of {len(paths)}")
#     full_path = os.path.join(DATA_PATH, path)
#     my_fp = Floorplan(full_path, wall_width=wall_width)
#     my_fp.draw_room_connectivity_on_plan()


# %%

# test resizing and wall removal for simple diffusion model training

# resized_fp = my_fp.pixel_based_resize(32)

# print(resized_fp.shape)
# plt.figure(figsize=(10,10))
# plt.imshow(resized_fp[:,:,0])
# plt.axis('off')
# plt.show()

# %%

# label generation loop

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

max_index = 100

for i, path in tqdm(enumerate(paths[:max_index]), total=max_index):
    
    if i % 10 == 0:
        print(f"Processing {i} of {len(paths)}")
    
    try:
        my_fp = Floorplan(os.path.join(DATA_PATH, path), wall_width=wall_width)
        data = my_fp.generate_llm_descriptions(llm, system_message, query)
        
        # save the description
        with open(os.path.join(OUTPUT_PATH, f"description_{i}.json"), "w") as f:
            json.dump(data, f)
        logger.info(f"Processed {i} of {len(paths)}")
        
    except Exception as e:
        logger.error(f"Error processing plan {path}: {e}")



# %%
