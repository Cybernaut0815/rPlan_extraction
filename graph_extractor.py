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

print(my_fp.contours["interior door"][0].shape)
print(my_fp.contours["interior door"][0])


# %%


def plot_contours(fp, cmap='nipy_spectral', show_plan=True):
    
    colors = fp.info.get_type_color(cmap)
    # Display the image with contours
    plt.figure(figsize=(12,8))

    # Create a blank white image with the same dimensions as the original
    blank_image = np.ones_like(fp.image) * 255
    color_idx = 0

    if show_plan:
        plt.imshow(fp.room_types_channel)
    else:
        plt.imshow(blank_image, cmap='gray')

    for room_type, contour_list in fp.contours.items():
        color = colors[color_idx % len(colors)]
        color = tuple(c/255 for c in color)
        for contours in contour_list:
            # Reshape contours to get x,y coordinates
            contours = contours.reshape(-1, 2)
            # Add first point to end to close the polygon
            x = np.append(contours[:, 0], contours[0, 0])
            y = np.append(contours[:, 1], contours[0, 1])
            # Plot the contour and add room type to legend
            plt.plot(x, y, color=color, label=room_type if contours is contour_list[0] else "")
        color_idx += 1

    plt.legend()
    plt.title("Room Contours")
    plt.axis('image')
    plt.show()


plot_contours(my_fp, show_plan=False)


# %%


# Convert interior door contours to shapely LineStrings and offset them
door_contours = my_fp.contours["interior door"]
offset_distance = 1.5  # Adjust this value based on your needs

offset_door_contours = []
for door_contour in door_contours:
    # Reshape contour to x,y coordinates
    door_coords = door_contour.copy().reshape(-1, 2)
    
    # Ensure the contour is closed by adding the first point at the end if needed
    if not np.array_equal(door_coords[0], door_coords[-1]):
        door_coords = np.vstack([door_coords, door_coords[0]])
    
    # Create a Polygon first to ensure consistent orientation
    poly = Polygon(door_coords)
    if not poly.is_valid:
        print(f"Invalid polygon: {poly}")
        continue
        
    # Convert to LineString and ensure counterclockwise orientation
    door_line = LineString(list(poly.exterior.coords))  # Include last point to keep it closed
    
    
    try:
        # Create parallel offset lines on both sides
        offset_left = door_line.parallel_offset(offset_distance, 'left')
        offset_right = door_line.parallel_offset(offset_distance, 'right')
        
        # Store the offset lines if they're valid
        for offset in [offset_left, offset_right]:
            if not offset.is_empty:
                # Handle both MultiLineString and LineString cases
                if offset.geom_type == 'MultiLineString':
                    coords = [p for line in offset.geoms for p in line.coords]
                else:
                    coords = list(offset.coords)
                
                # Ensure the line is closed
                if not np.array_equal(coords[0], coords[-1]):
                    coords.append(coords[0])
                
                offset_door_contours.append(LineString(coords))
    except Exception as e:
        print(f"Failed to create offset for contour: {e}")
        continue

# Visualize original and offset contours
plt.figure(figsize=(12,8))
plt.imshow(my_fp.room_types_channel)

# Plot original door contours in red
for door_contour in door_contours:
    door_coords = door_contour.reshape(-1, 2)
    # Ensure plotting coordinates are closed
    if not np.array_equal(door_coords[0], door_coords[-1]):
        door_coords = np.vstack([door_coords, door_coords[0]])
    x = door_coords[:, 0]
    y = door_coords[:, 1]
    plt.plot(x, y, 'r-', label='Original Door' if door_contour is door_contours[0] else "")

# Plot offset contours in blue
for offset_line in offset_door_contours:

    # Extract coordinates based on geometry type and ensure they're closed
    if offset_line.geom_type == 'LineString':
        coords = list(offset_line.coords)
    elif offset_line.geom_type == 'MultiLineString':
        coords = [list(line.coords) for line in offset_line.geoms]
        coords = [item for sublist in coords for item in sublist]
    
    # Ensure the line is closed
    if not np.array_equal(coords[0], coords[-1]):
        coords.append(coords[0])
    
    x, y = zip(*coords)
    plt.plot(x, y, 'b-', label='Offset Door' if offset_line is offset_door_contours[0] else "")

plt.legend()
plt.title("Door Contours with Offsets")
plt.axis('image')
plt.show()

# %%

room_contours = my_fp.contours.copy()
room_contours.pop("interior door")

for i, door_contour in enumerate(offset_door_contours):
    for room_type, contour_list in room_contours.items():
        for contour in contour_list:
            
            if len(contour) < 4:
                continue
            # Convert contour to closed shapely polygon
            coords = contour.copy().reshape(-1, 2)

            # Ensure the contour is closed by adding the first point at the end if needed
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            # Create a Polygon first to ensure consistent orientation
            poly = Polygon(coords)
            if not poly.is_valid:
                print(f"Invalid polygon: {poly}")
                continue

            if poly.intersects(door_contour):
                print(f"Door {i} intersects with {room_type}")
                









# %%
