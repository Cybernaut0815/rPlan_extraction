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

def offset_room_contours(fp, offset_distance=1.5):
    room_contours = fp.contours.copy()
    room_contours.pop("interior door")  # Remove interior doors since we handle them separately

    offset_contours_dict = {}
    
    # Iterate through room types and their contours
    for room_type, contour_list in room_contours.items():
        offset_contours_dict[room_type] = []
        
        for c in contour_list:
            if len(c) < 4:
                continue
            coords = c.copy().reshape(-1, 2)
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            poly = Polygon(coords)
            if not poly.is_valid:
                print(f"Invalid polygon: {poly}")
                continue
            
            # Offset the polygon by the specified distance
            try:
                offset_poly = poly.buffer(offset_distance, join_style=2)
                if offset_poly.is_valid:
                    offset_contours_dict[room_type].append(offset_poly)
            except Exception as e:
                print(f"Error offsetting polygon: {e}")
                continue

    return offset_contours_dict


room_contours_offset_dict = offset_room_contours(my_fp)

# Plot the original floorplan and offset contours
plt.figure(figsize=(12,8))
plt.imshow(my_fp.room_types_channel)

# Plot offset room contours
for offset_poly in room_contours_offset_dict.values():
    for offset_poly in offset_poly:
        # Extract coordinates from the polygon
        x, y = offset_poly.exterior.xy
        plt.plot(x, y, 'r-', linewidth=1)

# Plot offset door contours
for door_contour in my_fp.contours["interior door"]:
    # Reshape contour to x,y coordinates
    door_coords = door_contour.reshape(-1, 2)
    # Ensure plotting coordinates are closed
    if not np.array_equal(door_coords[0], door_coords[-1]):
        door_coords = np.vstack([door_coords, door_coords[0]])
    x = door_coords[:, 0]
    y = door_coords[:, 1]
    plt.plot(x, y, 'b-', linewidth=1)

plt.title("Room and Door Contours with Offsets")
plt.axis('image')
plt.show()


# %%

# Create a graph to represent room connectivity
G = nx.Graph()

# Add nodes for each room
for room_type, offset_polys in room_contours_offset_dict.items():
    for i, poly in enumerate(offset_polys):
        node_id = f"{room_type}_{i}"
        G.add_node(node_id, room_type=room_type)

# Check for intersections between offset door contours and room contours
for door_line in my_fp.contours["interior door"]:
    
    # Reshape contour to x,y coordinates
    door_coords = door_line.reshape(-1, 2)
    # Ensure plotting coordinates are closed
    if not np.array_equal(door_coords[0], door_coords[-1]):
        door_coords = np.vstack([door_coords, door_coords[0]])
        
    # Create a LineString from the door coordinates
    door_line = LineString(door_coords)
    
    intersecting_rooms = []
    
    # Check intersection with each room's offset polygon
    for room_type, offset_polys in room_contours_offset_dict.items():
        for i, room_poly in enumerate(offset_polys):
            if door_line.intersects(room_poly):
                intersecting_rooms.append(f"{room_type}_{i}")
    
    # If a door intersects exactly 2 rooms, add an edge between them
    if len(intersecting_rooms) == 2:
        G.add_edge(intersecting_rooms[0], intersecting_rooms[1])

# Print graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("\nConnections found:")
for edge in G.edges():
    print(f"{edge[0]} <-> {edge[1]}")

# Visualize the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=5000, font_size=12, font_weight='bold')
plt.title("Room Connectivity Graph")
plt.show()













# %%
