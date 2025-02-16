# %%

import json
import os
import networkx as nx
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from helpers.info import Info
from helpers.utils import load_image_paths
from helpers.fp import Floorplan


# %%

DATA_PATH = r"G:\Datasets\rPlan\dataset\dataset\floorplan_dataset"

#%%

paths = load_image_paths(DATA_PATH)

# open a test image
if paths:
    test_path = os.path.join(DATA_PATH, paths[7])
    img = Image.open(test_path)
else:
    print("No image files found in directory")


# %%

my_fp = Floorplan(test_path)

# %%

print(len(my_fp.contours["balcony"]))

#### So far seems to work!
#### Continue to find collisions between interior door contours and room contours to construct a graph (in the class)
#### Then, add edges between nodes that are connected by a door
#### implement the llm function in the fp class




# %%
