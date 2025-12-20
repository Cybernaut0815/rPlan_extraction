import os
import numpy as np
import cv2

from shapely.geometry import LineString, Polygon
import networkx as nx

from helpers.info import Info
from helpers.utils import load_image, resize_plan
from helpers.llm_utils import get_descriptions
import matplotlib.pyplot as plt


class Floorplan:
    def __init__(self, path, wall_width=2.0):
        self._image = np.array(load_image(path))
        self._outer_wall_channel = self._image[:,:,0]
        self._room_types_channel = self._image[:,:,1]
        self._distinct_rooms_channel = self._image[:,:,2]
        self._mask_channel = self._image[:,:,3]
        
        self.info = Info()
        self.wall_width = wall_width
        
        self.contours = self.get_contours()
        self.room_connectivity_graph = self.get_room_connectivity_graph(wall_width)
        self.room_types_count = self.get_room_types_count()
    
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, path):
        self._image = np.array(load_image(path))
        
        self._outer_wall_channel = self._image[:,:,0]
        self._room_types_channel = self._image[:,:,1]
        self._distinct_rooms_channel = self._image[:,:,2]
        self._mask_channel = self._image[:,:,3]
        
        self.contours = self.get_contours()
        self.room_connectivity_graph = self.get_room_connectivity_graph(self.wall_width)
        self.room_types_count = self.get_room_types_count() 
        

    @property
    def outer_wall_channel(self):
        return self._image[:,:,0]
    
    @outer_wall_channel.setter
    def outer_wall_channel(self):
        print("Cannot set outer wall channel.")
        return self._image[:,:,0]

    @property
    def room_types_channel(self):
        return self._image[:,:,1]
    
    @room_types_channel.setter
    def room_types_channel(self):
        print("Cannot set room types channel.")
        return self._image[:,:,1]
    
    @property
    def distinct_rooms_channel(self):
        return self._image[:,:,2]
    
    @distinct_rooms_channel.setter
    def distinct_rooms_channel(self):
        print("Cannot set distinct rooms channel.")
        return self._image[:,:,2]   
    
    @property
    def mask_channel(self):
        return self._image[:,:,3]
    
    @mask_channel.setter
    def mask_channel(self):
        print("Cannot set mask channel.")
        return self._image[:,:,3]

    def get_size(self):
        # Get the total number of pixels in the image
        total_pixels = self.mask_channel.shape[0] * self.mask_channel.shape[1]
        
        # Count white pixels (value 255) in mask channel
        white_pixels = np.sum(self.mask_channel == 255)
        
        # Calculate ratio
        ratio = white_pixels / total_pixels
        
        return ratio
    
    def get_room_types_count(self):
        count_dict = {}
        # Get room types from graph nodes
        for node, data in self.room_connectivity_graph.nodes(data=True):
            room_type = data['room_type']
            count_dict[room_type] = count_dict.get(room_type, 0) + 1

        return count_dict
    
    
    def get_room_types_count_pixel_based(self):
        
        data = self.pixel_based_resize(32)
        count_dict = {}
        
        for key, value in self.info.room_types.items():
            room_values = data[:,:,0]
            room_values = np.where(room_values == value)
            
        # Get the values from channel 1 where room_type matches
        masked_values = data[:,:,1][room_values]
        # Get unique values in the masked region
        unique_values = np.unique(masked_values)
        # Count number of unique values
        count_dict[key] = len(unique_values)
        return count_dict
    
    
    def get_contours(self):
        self.contours = {}
        for key, value in self.info.room_types.items():
            # Create mask for current room type
            mask = (self.room_types_channel == value).astype(np.uint8)
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours[key] = contours
        
        door_contours = self.interior_doors_outlines()
        self.contours["interior door"] = door_contours
        
        return self.contours
    
    
    def interior_doors_outlines(self):
        key = "interior door"
        value = self.info.all_types[key]
        # Create mask for current room type 
        mask = (self.room_types_channel == value).astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    
    def pixel_based_resize(self, point_count):
        x = np.linspace(0, self.image.shape[1], point_count+1)
        y = np.linspace(0, self.image.shape[0], point_count+1)
        X, Y = np.meshgrid(x, y)

        # Remove outer points by slicing the meshgrid arrays
        X = X[:-1, :-1]
        Y = Y[:-1, :-1]

        # Calculate spacing between points
        x_spacing = x[1] - x[0]
        y_spacing = y[1] - y[0]
        
        if x_spacing <= self.wall_width or y_spacing <= self.wall_width:
            # Calculate new point count to ensure spacing is slightly larger than wall width
            # Use the larger dimension to maintain aspect ratio
            width_points = int(self.image.shape[1] / (self.wall_width * 1.5))
            height_points = int(self.image.shape[0] / (self.wall_width * 1.5))
            reduced_count = min(width_points, height_points)
            
            x = np.linspace(0, self.image.shape[1], reduced_count+1)
            y = np.linspace(0, self.image.shape[0], reduced_count+1)
            X, Y = np.meshgrid(x, y)
            X = X[:-1, :-1]
            Y = Y[:-1, :-1]
            
            # Process at reduced resolution
            resized_fp = resize_plan(self.image, X, Y)
            
            # Create final array with target dimensions
            final_fp = np.zeros((point_count, point_count, 3))
            
            # Resize each channel separately using nearest neighbor interpolation
            for i in range(3):
                final_fp[:,:,i] = cv2.resize(resized_fp[:,:,i], 
                                            (point_count, point_count), 
                                            interpolation=cv2.INTER_NEAREST)
            return final_fp
            
        # Shift grid by half the spacing
        X = X + x_spacing/2
        Y = Y + y_spacing/2
        
        return resize_plan(self.image, X, Y)


    def outline_based_resize(self, target_size):
        """
        Resize floor plan using polygon-based approach:
        1. Create offset polygons (outward by half wall width)
        2. For each grid point, check which polygon(s) it's contained in
        3. For ambiguous points (in multiple polygons), select based on distance to original outline
        """
        # Create grid points
        x = np.linspace(0, self.image.shape[1], target_size + 1)
        y = np.linspace(0, self.image.shape[0], target_size + 1)
        X, Y = np.meshgrid(x, y)
        X = X[:-1, :-1]
        Y = Y[:-1, :-1]
        
        # Build mapping of original polygons (no offset at start)
        from shapely.prepared import prep
        room_entries = []  # list of dicts: {room_type, original_poly, offset_geom, prepared_offset}

        for room_type, contour_list in self.contours.items():
            if room_type == "interior door":
                continue

            for c in contour_list:
                original_poly = self.contours_to_polygons(c)
                if original_poly is None:
                    continue

                # Use original polygon without offset
                geoms = [original_poly]

                for g in geoms:
                    if g.is_empty:
                        continue
                    # Prepared geometry speeds up point-in-polygon queries
                    room_entries.append({
                        'room_type': room_type,
                        'original_poly': original_poly,
                        'offset_geom': g,
                        'prepared_offset': prep(g)
                    })
        
        # Create output array
        resized_fp = np.zeros((X.shape[0], X.shape[1], 3))
        
        # For each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point_x = X[i, j]
                point_y = Y[i, j]
                
                from shapely.geometry import Point
                point = Point(point_x, point_y)
                
                # Find which offset polygons cover this point (includes boundary)
                hits = []
                for entry in room_entries:
                    if entry['prepared_offset'].covers(point):
                        hits.append(entry)
                
                if len(hits) == 0:
                    # Outside all offset polygons: mark as external area and set mask
                    orig_y = int(np.clip(point_y, 0, self.image.shape[0] - 1))
                    orig_x = int(np.clip(point_x, 0, self.image.shape[1] - 1))
                    resized_fp[i, j, 0] = self.info.all_types["external area"]  # 13
                    resized_fp[i, j, 1] = self.image[orig_y, orig_x, 2]
                    resized_fp[i, j, 2] = 1
                
                elif len(hits) == 1:
                    # Point in exactly one room
                    room_type = hits[0]['room_type']
                    room_value = self.info.room_types[room_type]
                    resized_fp[i, j, 0] = room_value
                    
                    # Get second channel value from original image
                    orig_y = int(np.clip(point_y, 0, self.image.shape[0] - 1))
                    orig_x = int(np.clip(point_x, 0, self.image.shape[1] - 1))
                    resized_fp[i, j, 1] = self.image[orig_y, orig_x, 2]
                    resized_fp[i, j, 2] = 0
                
                else:
                    # Point in multiple rooms - use distance to original outline
                    min_dist = float('inf')
                    best_room = None
                    
                    for entry in hits:
                        room_type = entry['room_type']
                        original_poly = entry['original_poly']
                        
                        # Distance to original outline (before offset)
                        dist = point.distance(original_poly.exterior)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_room = room_type
                    
                    if best_room:
                        room_value = self.info.room_types[best_room]
                        resized_fp[i, j, 0] = room_value
                        
                        # Get second channel value from original image
                        orig_y = int(np.clip(point_y, 0, self.image.shape[0] - 1))
                        orig_x = int(np.clip(point_x, 0, self.image.shape[1] - 1))
                        resized_fp[i, j, 1] = self.image[orig_y, orig_x, 2]
                        resized_fp[i, j, 2] = 0
        
        # Post-process: iteratively fill interior-wall and interior-door gaps
        # For each iteration: for each room, first expand ALL pixels right, then expand ALL pixels down
        # Right pass happens fully, then down pass uses the updated state
        interior_wall_val = self.info.all_types.get("interior wall", 16)
        interior_door_val = self.info.all_types.get("interior door", 17)

        out_h, out_w, _ = resized_fp.shape
        
        # Pre-compute which cells are fillable (original is wall or door)
        fillable = np.zeros((out_h, out_w), dtype=bool)
        for i in range(out_h):
            for j in range(out_w):
                ty = int(np.clip(Y[i, j], 0, self.image.shape[0] - 1))
                tx = int(np.clip(X[i, j], 0, self.image.shape[1] - 1))
                orig_val = self.image[ty, tx, 1]
                fillable[i, j] = (orig_val == interior_wall_val or orig_val == interior_door_val)
        
        # Limited iterative passes
        max_passes = 5
        for pass_num in range(max_passes):
            changed = False
            
            # For each room type (0-11 are valid rooms)
            for room_val in range(12):
                # Phase 1: expand all pixels of this room to the right
                room_pixels = np.where(resized_fp[:, :, 0] == room_val)
                room_indices = list(zip(room_pixels[0], room_pixels[1]))
                
                for i, j in room_indices:
                    if j + 1 < out_w and fillable[i, j + 1] and resized_fp[i, j + 1, 0] > 11:
                        resized_fp[i, j + 1, 0] = room_val
                        resized_fp[i, j + 1, 1] = resized_fp[i, j, 1]
                        resized_fp[i, j + 1, 2] = 0
                        changed = True
                
                # Phase 2: expand all pixels of this room down (using updated state after right phase)
                room_pixels = np.where(resized_fp[:, :, 0] == room_val)
                room_indices = list(zip(room_pixels[0], room_pixels[1]))
                
                for i, j in room_indices:
                    if i + 1 < out_h and fillable[i + 1, j] and resized_fp[i + 1, j, 0] > 11:
                        resized_fp[i + 1, j, 0] = room_val
                        resized_fp[i + 1, j, 1] = resized_fp[i, j, 1]
                        resized_fp[i + 1, j, 2] = 0
                        changed = True

            if not changed:
                break

        # Post-process: 4-neighbor majority filter
        # If 2+ of the 4 neighbors have the same value and it differs from current pixel, adopt it
        # But if there's a 2-2 tie, leave unchanged
        from collections import Counter
        for i in range(out_h):
            for j in range(out_w):
                current_val = resized_fp[i, j, 0]
                
                # Get 4-neighbor values (up, down, left, right)
                neighbors = []
                if i > 0:
                    neighbors.append(resized_fp[i-1, j, 0])
                if i < out_h - 1:
                    neighbors.append(resized_fp[i+1, j, 0])
                if j > 0:
                    neighbors.append(resized_fp[i, j-1, 0])
                if j < out_w - 1:
                    neighbors.append(resized_fp[i, j+1, 0])
                
                if neighbors:
                    freq = Counter(neighbors)
                    most_common_val, most_common_count = freq.most_common(1)[0]
                    
                    # Change if: 2+ neighbors same AND different from current AND no tie
                    if most_common_count >= 2 and most_common_val != current_val:
                        # Check for 2-2 tie
                        other_with_2_plus = [v for v, c in freq.items() if c >= 2 and v != most_common_val]
                        if not other_with_2_plus:
                            resized_fp[i, j, 0] = most_common_val

        # Post-process: corner fill (8-neighbor "eaten out" corners)
        # If 3 neighbors forming a corner have the same value (different from current)
        # and the other 5 neighbors match current value, fill the pixel with corner value
        for i in range(out_h):
            for j in range(out_w):
                current_val = resized_fp[i, j, 0]
                
                # Get 8-neighbors: up, up-right, right, down-right, down, down-left, left, up-left
                neighbors_8 = {}
                if i > 0:
                    neighbors_8['up'] = resized_fp[i-1, j, 0]
                if i > 0 and j < out_w - 1:
                    neighbors_8['up-right'] = resized_fp[i-1, j+1, 0]
                if j < out_w - 1:
                    neighbors_8['right'] = resized_fp[i, j+1, 0]
                if i < out_h - 1 and j < out_w - 1:
                    neighbors_8['down-right'] = resized_fp[i+1, j+1, 0]
                if i < out_h - 1:
                    neighbors_8['down'] = resized_fp[i+1, j, 0]
                if i < out_h - 1 and j > 0:
                    neighbors_8['down-left'] = resized_fp[i+1, j-1, 0]
                if j > 0:
                    neighbors_8['left'] = resized_fp[i, j-1, 0]
                if i > 0 and j > 0:
                    neighbors_8['up-left'] = resized_fp[i-1, j-1, 0]
                
                # Define 4 corner patterns (3 neighbors each)
                corners = [
                    ['up', 'up-left', 'left'],           # top-left corner
                    ['up', 'up-right', 'right'],         # top-right corner
                    ['down', 'down-left', 'left'],       # bottom-left corner
                    ['down', 'down-right', 'right']      # bottom-right corner
                ]
                
                filled = False
                for corner_dirs in corners:
                    if filled:
                        break
                    
                    # Check if all 3 corner neighbors exist
                    if not all(d in neighbors_8 for d in corner_dirs):
                        continue
                    
                    corner_vals = [neighbors_8[d] for d in corner_dirs]
                    # Check if all 3 corner neighbors have the same value
                    if len(set(corner_vals)) != 1:
                        continue
                    
                    corner_val = corner_vals[0]
                    if corner_val == current_val:
                        continue  # corner value must differ from current
                    
                    # Check if all other neighbors match current value
                    other_dirs = [d for d in neighbors_8 if d not in corner_dirs]
                    other_vals = [neighbors_8[d] for d in other_dirs]
                    
                    if all(v == current_val for v in other_vals):
                        # Found eaten-out corner, fill it
                        resized_fp[i, j, 0] = corner_val
                        filled = True

        return resized_fp


    def contours_to_polygons(self, contour):
        if len(contour) < 4:
            return None
        coords = contour.copy().reshape(-1, 2)
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        poly = Polygon(coords)
        
        if not poly.is_valid:
            # Try to fix the invalid polygon
            try:
                # Clean the coordinates by removing duplicate points
                _, unique_indices = np.unique(coords, axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                coords = coords[unique_indices]
                
                # Ensure we still have enough points for a valid polygon
                if len(coords) < 4:
                    return None
                    
                # Make sure polygon is closed
                if not np.array_equal(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])
                
                # Try to create a valid polygon with cleaned coordinates
                poly = Polygon(coords).buffer(0)
                
                if poly.is_valid:
                    return poly
            except:
                pass
        
        if not poly.is_valid:
            print(f"Invalid polygon: {poly}")
            return None
        
        # Ensure counterclockwise orientation
        if not poly.exterior.is_ccw:
            coords = np.flip(coords[:-1], axis=0)  # Remove last point and reverse
            coords = np.vstack([coords, coords[0]])  # Re-add closing point
            poly = Polygon(coords)
            
        return poly


    def offset_room_contours(self, offset_distance=2.0):
        room_contours = self.contours.copy()
        room_contours.pop("interior door")  # Remove interior doors since we handle them separately

        offset_contours_dict = {}
        
        # Iterate through room types and their contours
        for room_type, contour_list in room_contours.items():
            offset_contours_dict[room_type] = []
            
            for c in contour_list:
                poly = self.contours_to_polygons(c)
                
                if poly is None:
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
                
                
    def get_room_connectivity_graph(self, offset_distance=3.0):
        G = nx.Graph()
        offset_contours_dict = self.offset_room_contours(offset_distance)
        for room_type, offset_polys in offset_contours_dict.items():
            for i, poly in enumerate(offset_polys):
                node_id = f"{room_type}_{i}"
                # Get centroid coordinates
                centroid = poly.centroid
                G.add_node(node_id, 
                            room_type=room_type,
                            centroid=(centroid.x, centroid.y))
                
        for door_line in self.contours["interior door"]:
            door_poly = self.contours_to_polygons(door_line)
            if door_poly is None:
                continue
            
            intersecting_rooms = []
            for room_type, offset_polys in offset_contours_dict.items():
                for i, room_poly in enumerate(offset_polys):
                    if door_poly.intersects(room_poly):
                        intersection = door_poly.intersection(room_poly)
                        intersection_area = intersection.area
                        intersection_centroid = intersection.centroid
                        intersecting_rooms.append({
                            'id': f"{room_type}_{i}",
                            'area': intersection_area,
                            'centroid': (intersection_centroid.x, intersection_centroid.y)
                        })
            
            if len(intersecting_rooms) == 2:
                G.add_edge(intersecting_rooms[0]['id'], intersecting_rooms[1]['id'])
            elif len(intersecting_rooms) > 2:
                # Sort rooms by intersection area in descending order
                intersecting_rooms.sort(key=lambda x: x['area'], reverse=True)
                # Get the second largest intersection area as threshold
                threshold = intersecting_rooms[1]['area'] * 0.5
                # Connect room with largest intersection to others that meet threshold
                largest_room = intersecting_rooms[0]['id']
                for room in intersecting_rooms[1:]:
                    if room['area'] >= threshold:
                        G.add_edge(largest_room, room['id'])
            
        # Find isolated nodes and try to connect them
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
        
        if isolated_nodes:
            # Get larger offset for isolated rooms to find potential connections
            larger_offset = offset_distance * 2
            for isolated_node in isolated_nodes:
                room_type, idx = isolated_node.rsplit('_', 1)
                idx = int(idx)
                
                # Get the original room polygon and create larger offset
                original_poly = offset_contours_dict[room_type][idx]
                try:
                    larger_poly = original_poly.buffer(larger_offset, join_style=2)
                except Exception as e:
                    print(f"Error creating larger offset for isolated room: {e}")
                    continue
                
                # Check intersections with all other room polygons
                intersecting_rooms = []
                for other_type, other_polys in offset_contours_dict.items():
                    for i, other_poly in enumerate(other_polys):
                        other_node = f"{other_type}_{i}"
                        if other_node != isolated_node:  # Don't check against self
                            if larger_poly.intersects(other_poly):
                                intersection = larger_poly.intersection(other_poly)
                                intersection_centroid = intersection.centroid
                                intersecting_rooms.append({
                                    'id': other_node,
                                    'area': intersection.area,
                                    'centroid': (intersection_centroid.x, intersection_centroid.y)
                                })
                
                # Connect to room with largest intersection area if any found
                if intersecting_rooms:
                    intersecting_rooms.sort(key=lambda x: x['area'], reverse=True)
                    G.add_edge(isolated_node, intersecting_rooms[0]['id'])
        
        # Check for disconnected components
        components = list(nx.connected_components(G))
        if len(components) > 1:
            # Sort components by size to find the smaller one(s)
            components.sort(key=len)
            
            # For each smaller component
            for small_component in components[:-1]:  # All except the largest component
                # Find the largest room in the smaller component
                largest_room = None
                largest_area = 0
                
                for node in small_component:
                    room_type, idx = node.rsplit('_', 1)
                    idx = int(idx)
                    room_poly = offset_contours_dict[room_type][idx]
                    area = room_poly.area
                    
                    if area > largest_area:
                        largest_area = area
                        largest_room = node
                
                if largest_room:
                    # Create larger offset for the largest room
                    room_type, idx = largest_room.rsplit('_', 1)
                    idx = int(idx)
                    original_poly = offset_contours_dict[room_type][idx]
                    larger_poly = original_poly.buffer(offset_distance * 3, join_style=2)
                    
                    # Find intersections with rooms in the main component
                    best_connection = None
                    max_intersection = 0
                    
                    for other_node in components[-1]:  # Check against largest component
                        other_type, other_idx = other_node.rsplit('_', 1)
                        other_idx = int(other_idx)
                        other_poly = offset_contours_dict[other_type][other_idx]
                        
                        if larger_poly.intersects(other_poly):
                            intersection = larger_poly.intersection(other_poly)
                            if intersection.area > max_intersection:
                                max_intersection = intersection.area
                                best_connection = other_node
                    
                    if best_connection:
                        G.add_edge(largest_room, best_connection)

        return G
    
    
    def draw_connectivity_graph(self):
        G = self.room_connectivity_graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=5000, font_size=12, font_weight='bold')
        plt.title("Room Connectivity Graph")
        plt.show()
    
    
    def draw_room_connectivity_on_plan(self):
        G = self.room_connectivity_graph
        
        # Create figure and axis
        plt.figure(figsize=(10, 10))
        
        # Display the room types channel as background
        plt.imshow(self.room_types_channel, cmap='viridis')
        
        # Get positions from node centroids
        pos = {node: data['centroid'] for node, data in G.nodes(data=True)}
        
        # Draw the graph
        nx.draw(G, pos=pos, 
                node_color='red',
                node_size=100,
                edge_color='yellow',
                width=2,
                with_labels=True,
                font_size=8,
                font_color='white',
                font_weight='bold')
            
        plt.title("Room Connectivity Graph Overlaid on Floorplan")
        plt.axis('on')
        plt.show()
    
    
    def get_room_connectivity_matrix(self):
        G = self.room_connectivity_graph
        return nx.to_numpy_array(G)
    
    
    def draw_contours(self):
        for room_type, contours in self.contours.items():
            polygon = self.contours_to_polygons(contours)
            if polygon is None:
                continue
            plt.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], 'r-')
        plt.show()


    def draw_contours(self, offset=False):
        
        if offset:
            contours = self.offset_room_contours(offset_distance=self.wall_width)
        else:
            contours = self.offset_room_contours(offset_distance=0)

        door_contours = [self.contours_to_polygons(door_contours) for door_contours in self.contours["interior door"]]
        
        plt.figure(figsize=(10,10))
        colors = self.info.get_type_color()

        # Draw offset room contours
        for (room_type, polys), color in zip(contours.items(), colors):
            for poly in polys:
                x, y = poly.exterior.xy
                plt.plot(x, y, color=color, label=room_type)

        # Draw door contours
        for door_contour in door_contours:
            if door_contour is not None:
                x, y = door_contour.exterior.xy
                plt.plot(x, y, color='black', label='interior door', linewidth=2)

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.title("Offset Room Contours with Doors")
        plt.axis('equal')
        plt.show()
        

    def graph_to_string(self):
        G = self.room_connectivity_graph
        
        rStrings = []

        edges = G.edges()
        for edge in edges:
            room1, room2 = edge
            rStrings.append(f'"{room1}" is next to "{room2}"')

        return "\n".join(rStrings)

    ### here llm descriptions ###
    
    def generate_llm_descriptions(self, llm, system_message, query, description_count=3, pixel_based_size=16):
        
        resized_fp = self.pixel_based_resize(pixel_based_size)
        room_types_count = self.get_room_types_count()     
        graph_string = self.graph_to_string()
        
        data = {
            "room_counts": room_types_count,
            "dimensions": [resized_fp.shape[0], resized_fp.shape[1]],
            "functions": resized_fp[:,:,0].tolist(),
            "mask": resized_fp[:,:,2].tolist(),
            "graph": self.get_room_connectivity_matrix().tolist(),
            "graph_string": graph_string
        }
        
        descriptions = get_descriptions(data, llm, system_message, query, description_count)
        data["descriptions"] = descriptions
        
        return data
        


