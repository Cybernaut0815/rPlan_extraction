import numpy as np
import cv2
from skimage.morphology import skeletonize

from shapely.geometry import Polygon
import networkx as nx

from helpers.info import Info
from helpers.utils import load_image, resize_plan, apply_room_postprocess, expand_rooms_right_down
from helpers.llm_utils import get_descriptions
import matplotlib.pyplot as plt


class Floorplan:
    def __init__(self, path, wall_width=2.0, meter_to_pixel=16):
        self._image = np.array(load_image(path))
        self._outer_wall_channel = self._image[:,:,0]
        self._room_types_channel = self._image[:,:,1]
        self._distinct_rooms_channel = self._image[:,:,2]
        self._mask_channel = self._image[:,:,3]
        
        self.info = Info()
        self.wall_width = wall_width
        self.meter_to_pixel = meter_to_pixel  # pixels per meter (e.g., 16 pixels = 1 meter)
        
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
        # Get room types from graph nodes (exclude entrance nodes)
        for node, data in self.room_connectivity_graph.nodes(data=True):
            room_type = data['room_type']
            # Skip entrance nodes (exterior doors)
            if room_type == 'entrance':
                continue
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
        
        # Add exterior doors (front door)
        exterior_door_contours = self.exterior_doors_outlines()
        self.contours["exterior door"] = exterior_door_contours
        
        return self.contours
    
    
    def interior_doors_outlines(self):
        key = "interior door"
        value = self.info.all_types[key]
        # Create mask for current room type 
        mask = (self.room_types_channel == value).astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    
    def exterior_doors_outlines(self):
        key = "front door"
        value = self.info.all_types[key]
        # Create mask for exterior doors
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
            width_points = int(self.image.shape[1] / (self.wall_width * 1.5))
            height_points = int(self.image.shape[0] / (self.wall_width * 1.5))
            reduced_count = min(width_points, height_points)
            
            x = np.linspace(0, self.image.shape[1], reduced_count+1)
            y = np.linspace(0, self.image.shape[0], reduced_count+1)
            X, Y = np.meshgrid(x, y)
            X = X[:-1, :-1]
            Y = Y[:-1, :-1]
            base_fp = resize_plan(self.image, X, Y)
            final_fp = np.zeros((point_count, point_count, 3))
            for i in range(3):
                final_fp[:,:,i] = cv2.resize(base_fp[:,:,i], (point_count, point_count), interpolation=cv2.INTER_NEAREST)
            resized_fp = final_fp
        else:
            X = X + x_spacing/2
            Y = Y + y_spacing/2
            resized_fp = resize_plan(self.image, X, Y)

        # Mixed postprocessing: expand like outline-based and run kernels
        interior_wall_val = self.info.all_types.get("interior wall", 16)
        interior_door_val = self.info.all_types.get("interior door", 17)
        x_fill = np.linspace(0, self.image.shape[1], point_count+1)[:-1]
        y_fill = np.linspace(0, self.image.shape[0], point_count+1)[:-1]
        X_fill, Y_fill = np.meshgrid(x_fill, y_fill)
        X_fill = X_fill + x_spacing/2
        Y_fill = Y_fill + y_spacing/2
        ty = np.clip(Y_fill, 0, self.image.shape[0] - 1).astype(int)
        tx = np.clip(X_fill, 0, self.image.shape[1] - 1).astype(int)
        orig_vals = self.image[ty, tx, 1]
        fillable = (orig_vals == interior_wall_val) | (orig_vals == interior_door_val)
        resized_fp = expand_rooms_right_down(resized_fp, fillable, num_rooms=12, max_passes=5)
        
        # Expand to 4 channels before postprocessing:
        # resize_plan returns [room_type, per-type_instance, mask_placeholder]
        # Convert to [room_type, placeholder, per-type_instance, mask]
        resized_4ch = np.zeros((resized_fp.shape[0], resized_fp.shape[1], 4), dtype=resized_fp.dtype)
        resized_4ch[:, :, 0] = resized_fp[:, :, 0]  # room type
        resized_4ch[:, :, 1] = 0                      # will be filled by _postprocess_and_mask
        resized_4ch[:, :, 2] = resized_fp[:, :, 1]   # per-type instance indices (from original ch2)
        resized_4ch[:, :, 3] = 0                      # will be filled by _postprocess_and_mask
        
        return self._postprocess_and_mask(resized_4ch)

    def _extract_wall_skeleton(self):
        """
        Extract skeleton from wall pixels (exterior walls, interior walls, doors).
        Returns a binary mask where True = wall skeleton.
        """
        room_types = self._room_types_channel
        
        # Get wall-related pixel values
        exterior_wall = self.info.all_types.get("exterior wall", 14)
        interior_wall = self.info.all_types.get("interior wall", 16)
        interior_door = self.info.all_types.get("interior door", 17)
        front_door = self.info.all_types.get("front door", 15)
        
        # Create wall mask
        wall_mask = (
            (room_types == exterior_wall) | 
            (room_types == interior_wall) | 
            (room_types == interior_door) |
            (room_types == front_door)
        ).astype(np.uint8)
        
        # Skeletonize to get 1-pixel thick boundaries
        skeleton = skeletonize(wall_mask > 0)
        
        return skeleton.astype(np.uint8)

    def outline_based_resize(self, target_size, buffer_distance=1.0, debug=False):
        """
        Resize floor plan using vector-based approach:
        1. Extract skeleton and find room regions
        2. Convert each region to a Shapely polygon  
        3. For each output pixel, check polygon containment
        
        Args:
            target_size: Output size (target_size x target_size)
            buffer_distance: Distance to expand room polygons outward (default 0.0).
                            Set to 0 for polygons that exactly match regions.
            debug: Show debug visualization
        """
        from shapely.geometry import Polygon as ShapelyPolygon, Point
        from shapely import prepare
        
        room_types = self._room_types_channel.copy()
        external_area = self.info.all_types.get("external area", 13)
        
        # Extract skeleton and find connected regions
        skeleton = self._extract_wall_skeleton()
        inverted = (skeleton == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(inverted, connectivity=4)
        
        # Build Shapely polygons for each room region
        room_polygons = []
        for label_id in range(1, num_labels):
            mask = (labels == label_id).astype(np.uint8)
            region_room_types = room_types[mask > 0]
            valid_types = region_room_types[region_room_types <= 11]
            
            if len(valid_types) == 0:
                continue
            
            room_value = np.bincount(valid_types).argmax()
            region_instances = self._distinct_rooms_channel[mask > 0]
            instance_value = np.bincount(region_instances[region_instances > 0]).argmax() if np.any(region_instances > 0) else 0
            
            # Extract contour - use CHAIN_APPROX_NONE for full boundary coverage
            contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            
            # Light simplification to reduce points while keeping shape
            # Use smaller epsilon for better accuracy
            simplified = cv2.approxPolyDP(contour, 1.0, True)
            points = simplified.reshape(-1, 2).tolist()
            
            if len(points) < 3:
                continue
            
            # Create polygon directly from contour points (no manual straightening)
            try:
                poly = ShapelyPolygon(points)
                if not poly.is_valid:
                    poly = poly.buffer(0)  # Fix self-intersections
                if poly.is_valid and poly.area > 0:
                    # Only apply buffer if requested (default 0 means no expansion)
                    if buffer_distance > 0:
                        poly = poly.buffer(buffer_distance, cap_style=3, join_style=2)
                    prepare(poly)
                    room_polygons.append((poly, room_value, instance_value))
            except:
                continue
        
        # Sample grid points and check polygon containment
        # Use covers() instead of contains() to include points on the boundary
        resized_room_map = np.full((target_size, target_size), external_area, dtype=np.uint8)
        resized_instance_map = np.zeros((target_size, target_size), dtype=np.uint8)
        scale_y = self.image.shape[0] / target_size
        scale_x = self.image.shape[1] / target_size
        
        for i in range(target_size):
            for j in range(target_size):
                sample_point = Point((j + 0.5) * scale_x, (i + 0.5) * scale_y)
                for poly, room_value, instance_value in room_polygons:
                    # covers() returns True for points inside OR on the boundary
                    if poly.covers(sample_point):
                        resized_room_map[i, j] = room_value
                        resized_instance_map[i, j] = instance_value
                        break
        
        # Build output array (4 channels)
        # ch0: room type (original values 0-11, 13)
        # ch1: global instance ID (from connected components)
        # ch2: per-type instance index (from original distinct_rooms_channel, 1-based)
        # ch3: interior mask (1 = room, 0 = non-room/external)
        resized_fp = np.zeros((target_size, target_size, 4), dtype=np.float32)
        resized_fp[:, :, 0] = resized_room_map
        resized_fp[:, :, 1] = self._label_instances(resized_room_map.astype(np.int32))
        resized_fp[:, :, 2] = resized_instance_map  # original per-type instance indices
        resized_fp[:, :, 3] = (resized_room_map <= 11).astype(np.uint8)
        
        if debug:
            self._debug_outline_resize(skeleton, labels, num_labels, room_polygons, resized_room_map)
        
        return resized_fp

    def _debug_outline_resize(self, skeleton, labels, num_labels, room_polygons, resized_room_map):
        """Debug visualization for outline_based_resize."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(skeleton, cmap='gray')
        axes[0].set_title('Skeleton')
        axes[0].axis('off')
        
        axes[1].imshow(labels, cmap='tab20')
        axes[1].set_title(f'Regions ({num_labels - 1})')
        axes[1].axis('off')
        
        poly_img = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
        for poly, rtype, _ in room_polygons:
            coords = np.array(poly.exterior.coords).astype(np.int32)
            color = (int((rtype * 50 + 100) % 255), int((rtype * 80 + 50) % 255), int((rtype * 30 + 150) % 255))
            cv2.polylines(poly_img, [coords], True, color, 1)
        axes[2].imshow(poly_img)
        axes[2].set_title(f'Polygons ({len(room_polygons)})')
        axes[2].axis('off')
        
        axes[3].imshow(resized_room_map, cmap='tab20')
        axes[3].set_title('Resized Room Map')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()

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
        # Remove exterior doors from room contour offsetting (handled separately as entrance node)
        room_contours.pop("exterior door", None)

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
            if room_type in {"exterior door", "entrance"}:
                continue
            for i, poly in enumerate(offset_polys):
                node_id = f"{room_type}_{i}"
                # Get centroid coordinates
                centroid = poly.centroid
                G.add_node(node_id, 
                            room_type=room_type,
                            centroid=(centroid.x, centroid.y))
        
        # Add a single entrance node (aggregating all exterior doors)
        entrance_polys = []
        for door_line in self.contours["exterior door"]:
            door_poly = self.contours_to_polygons(door_line)
            if door_poly is not None:
                entrance_polys.append(door_poly)

        if entrance_polys:
            cx = sum(p.centroid.x for p in entrance_polys) / len(entrance_polys)
            cy = sum(p.centroid.y for p in entrance_polys) / len(entrance_polys)
            self.entrance_centroid = (cx, cy)
            entrance_node = "entrance_0"
            G.add_node(
                entrance_node,
                room_type="entrance",
                centroid=(cx, cy)
            )

            # Process exterior doors - connect to room with largest intersection area
            for door_poly in entrance_polys:
                intersecting_rooms = []
                for room_type, offset_polys in offset_contours_dict.items():
                    for j, room_poly in enumerate(offset_polys):
                        if door_poly.intersects(room_poly):
                            intersection = door_poly.intersection(room_poly)
                            intersecting_rooms.append({
                                'id': f"{room_type}_{j}",
                                'area': intersection.area
                            })
                
                # Connect to room with largest intersection area
                if intersecting_rooms:
                    intersecting_rooms.sort(key=lambda x: x['area'], reverse=True)
                    G.add_edge(entrance_node, intersecting_rooms[0]['id'])
                
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
                # Parse room_type and index from node_id
                parts = isolated_node.rsplit('_', 1)
                if len(parts) != 2:
                    # Skip nodes that can't be parsed (shouldn't happen now)
                    continue
                room_type, idx_str = parts
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                
                # Get the original room polygon and create larger offset
                if room_type not in offset_contours_dict:
                    continue
                # Check bounds before accessing
                if idx < 0 or idx >= len(offset_contours_dict[room_type]):
                    continue
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
                    # Parse node_id to get room_type and index
                    parts = node.rsplit('_', 1)
                    if len(parts) != 2:
                        continue
                    room_type, idx_str = parts
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    if room_type not in offset_contours_dict:
                        continue
                    # Check bounds before accessing
                    if idx < 0 or idx >= len(offset_contours_dict[room_type]):
                        continue
                    room_poly = offset_contours_dict[room_type][idx]
                    area = room_poly.area
                    
                    if area > largest_area:
                        largest_area = area
                        largest_room = node
                
                if largest_room:
                    # Create larger offset for the largest room
                    parts = largest_room.rsplit('_', 1)
                    if len(parts) != 2:
                        largest_room = None
                    else:
                        room_type, idx_str = parts
                        try:
                            idx = int(idx_str)
                        except ValueError:
                            largest_room = None
                        else:
                            if room_type not in offset_contours_dict:
                                largest_room = None
                            else:
                                original_poly = offset_contours_dict[room_type][idx]
                                larger_poly = original_poly.buffer(offset_distance * 3, join_style=2)
                    
                    # Find intersections with rooms in the main component
                    best_connection = None
                    max_intersection = 0
                    
                    if largest_room:
                        # Find intersections with rooms in the main component
                        best_connection = None
                        max_intersection = 0
                        
                        for other_node in components[-1]:  # Check against largest component
                            # Parse other_node to get room_type and index
                            parts = other_node.rsplit('_', 1)
                            if len(parts) != 2:
                                continue
                            other_type, other_idx_str = parts
                            try:
                                other_idx = int(other_idx_str)
                            except ValueError:
                                continue
                            if other_type not in offset_contours_dict:
                                continue
                            # Check bounds before accessing
                            if other_idx < 0 or other_idx >= len(offset_contours_dict[other_type]):
                                continue
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
    
    
    def draw_contours(self, offset=False):
        """Draw room contours with optional offset and door markers."""
        contours_dict = self.offset_room_contours(offset_distance=self.wall_width if offset else 0)
        interior_door_contours = [self.contours_to_polygons(c) for c in self.contours["interior door"]]
        exterior_door_contours = [self.contours_to_polygons(c) for c in self.contours["exterior door"]]
        
        plt.figure(figsize=(10, 10))
        colors = self.info.get_type_color()

        for (room_type, polys), color in zip(contours_dict.items(), colors):
            for poly in polys:
                x, y = poly.exterior.xy
                plt.plot(x, y, color=color, label=room_type)

        for door_contour in interior_door_contours:
            if door_contour is not None:
                x, y = door_contour.exterior.xy
                plt.plot(x, y, color='black', label='interior door', linewidth=2)
        
        # Draw entrance centroid as a single marker (no polygon outline)
        if hasattr(self, "entrance_centroid") and self.entrance_centroid is not None:
            cx, cy = self.entrance_centroid
            plt.scatter([cx], [cy], color='red', label='entrance', s=50, marker='o')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(f"Room Contours {'(Offset)' if offset else ''}")
        plt.axis('equal')
        plt.show()
        

    
    def calculate_room_sizes(self, resized_fp):
        """
        Calculate room sizes in square meters independently for both original and resized floorplans.
        Each size is calculated directly from the actual pixels in its corresponding image.
        
        Args:
            resized_fp: The resized floorplan array with shape (height, width, channels)
                       where channel 0 contains room types and channel 1 contains instance IDs
        
        Returns:
            dict: Contains 'original_sizes' and 'resized_sizes', each mapping instance_id to area in m2
        """
        original_sizes = {}
        resized_sizes = {}
        
        # Get instance map from resized floorplan
        final_instances_resized = resized_fp[:, :, 1].astype(np.int32)
        room_types_resized = resized_fp[:, :, 0].astype(np.int32)
        unique_instances = np.unique(final_instances_resized)
        
        # Pixel area to m2 conversion factors
        pixel_area_per_m2_original = self.meter_to_pixel ** 2
        
        # Calculate adjusted meter_to_pixel for resized image
        scale_x = resized_fp.shape[1] / self.image.shape[1]
        scale_y = resized_fp.shape[0] / self.image.shape[0]
        adjusted_meter_to_pixel_x = self.meter_to_pixel * scale_x
        adjusted_meter_to_pixel_y = self.meter_to_pixel * scale_y
        pixel_area_per_m2_resized = adjusted_meter_to_pixel_x * adjusted_meter_to_pixel_y
        
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip background/invalid instances
                continue
            
            # ===== RESIZED SIZE =====
            # Count pixels directly from the resized image for this instance
            resized_mask = (final_instances_resized == instance_id)
            resized_pixel_count = np.sum(resized_mask)
            resized_size_m2 = resized_pixel_count / pixel_area_per_m2_resized
            
            # ===== ORIGINAL SIZE =====
            # Get the room type for this instance from the resized image
            room_type_value = room_types_resized[resized_mask][0]  # Get first occurrence of this room type
            
            # Map this instance back to the original image and count matching pixels
            # For each resized pixel belonging to this instance, find the corresponding region in original image
            original_pixel_count = 0
            resized_coords = np.argwhere(resized_mask)
            
            for ry, rx in resized_coords:
                # Map resized pixel to original image region
                orig_y_start = int(ry / scale_y)
                orig_y_end = int((ry + 1) / scale_y)
                orig_x_start = int(rx / scale_x)
                orig_x_end = int((rx + 1) / scale_x)
                
                # Clamp to image boundaries
                orig_y_start = max(0, orig_y_start)
                orig_y_end = min(self.image.shape[0], orig_y_end)
                orig_x_start = max(0, orig_x_start)
                orig_x_end = min(self.image.shape[1], orig_x_end)
                
                # Count only pixels in original image that have the matching room type
                # Use channel 1 (room_types_channel) which contains the room type values
                region = self.image[orig_y_start:orig_y_end, orig_x_start:orig_x_end, 1]
                matching_pixels = np.sum(region == room_type_value)
                original_pixel_count += matching_pixels
            
            original_size_m2 = original_pixel_count / pixel_area_per_m2_original
            
            original_sizes[int(instance_id)] = round(original_size_m2, 1)
            resized_sizes[int(instance_id)] = round(resized_size_m2, 1)
        
        return {
            "original_sizes": original_sizes,
            "resized_sizes": resized_sizes,
            "pixel_area_per_m2_original": pixel_area_per_m2_original,
            "pixel_area_per_m2_resized": round(pixel_area_per_m2_resized, 6),
            "meter_to_pixel": self.meter_to_pixel,
            "adjusted_meter_to_pixel": round(np.sqrt(pixel_area_per_m2_resized), 6)
        }

    def calculate_room_sizes_original(self):
        """Calculate per-instance areas (m2) directly on the original image without resizing."""
        instances = self.distinct_rooms_channel.astype(np.int32)
        room_types = self.room_types_channel.astype(np.int32)
        unique_instances = np.unique(instances)

        pixel_area_per_m2_original = self.meter_to_pixel ** 2
        original_sizes = {}

        for instance_id in unique_instances:
            if instance_id == 0:
                continue  # background

            mask = instances == instance_id
            if not np.any(mask):
                continue

            room_type_vals = room_types[mask]
            room_type_value = np.bincount(room_type_vals).argmax()

            # Skip non-room areas (labels above 11 are structural/external)
            if room_type_value > 11:
                continue

            pixel_count = int(np.sum(mask))
            size_m2 = pixel_count / pixel_area_per_m2_original
            original_sizes[int(instance_id)] = round(size_m2, 1)

        return {
            "original_sizes": original_sizes,
            "pixel_area_per_m2_original": pixel_area_per_m2_original,
            "meter_to_pixel": self.meter_to_pixel,
        }

    def get_original_room_sizes(self):
        """Convenience wrapper returning only the original instance sizes (m2)."""
        return self.calculate_room_sizes_original().get("original_sizes", {})

    def get_original_room_sizes_by_node(self):
        """Return original sizes keyed by node ID (matching connectivity graph): {node_id: size_m2}."""
        sizes = self.calculate_room_sizes_original().get("original_sizes", {})
        instances = self.distinct_rooms_channel.astype(np.int32)
        room_types = self.room_types_channel.astype(np.int32)
        
        # Build mapping from instance_id to node_id from the connectivity graph
        instance_to_node = {}
        for node_id, data in self.room_connectivity_graph.nodes(data=True):
            # Skip entrance/door nodes (no instance index)
            room_type = data.get('room_type')
            if room_type == 'entrance' or '_' not in node_id:
                continue

            # Extract instance index from node_id (e.g., "living room_0" -> 0)
            idx = int(node_id.rsplit('_', 1)[1])
            
            # Find which original instance this corresponds to
            # Get all instances of this room type
            instances_of_type = []
            for inst_id, size in sizes.items():
                mask = instances == int(inst_id)
                if not np.any(mask):
                    continue
                room_type_vals = room_types[mask]
                room_type_value = np.bincount(room_type_vals).argmax()
                
                # Check if this instance matches the room type
                for rt_name, rt_val in self.info.room_types.items():
                    if rt_val == room_type_value and rt_name == room_type:
                        instances_of_type.append((inst_id, size))
                        break
            
            # Map by index
            if idx < len(instances_of_type):
                instance_to_node[instances_of_type[idx][0]] = node_id
        
        # Build result with node IDs
        result = {}
        for inst_id, size in sizes.items():
            if inst_id in instance_to_node:
                result[instance_to_node[inst_id]] = size
        
        return result

    def get_original_room_sizes_by_type(self):
        """Return original sizes grouped by room type name: {room_type: [sizes...]}."""
        sizes = self.calculate_room_sizes_original().get("original_sizes", {})
        instances = self.distinct_rooms_channel.astype(np.int32)
        room_types = self.room_types_channel.astype(np.int32)
        result = {}

        for instance_id, size_m2 in sizes.items():
            mask = instances == int(instance_id)
            if not np.any(mask):
                continue
            room_type_vals = room_types[mask]
            room_type_value = np.bincount(room_type_vals).argmax()
            # Map numeric type to name; skip if unknown
            name = None
            for rt_name, rt_val in self.info.room_types.items():
                if rt_val == room_type_value:
                    name = rt_name
                    break
            if name is None:
                continue

            result.setdefault(name, []).append(size_m2)

        return result

    def get_room_connectivity(self):
        """Extract room connectivity as JSON-serializable dict with adjacency list, room counts, and entrance info."""
        G = self.room_connectivity_graph
        adjacency = {}
        
        for node, data in G.nodes(data=True):
            neighbors = list(G.neighbors(node))
            if neighbors:  # Only include nodes with connections
                adjacency[node] = neighbors
        
        # Count entrance nodes
        entrance_count = sum(1 for node, data in G.nodes(data=True) if data.get("room_type") == "entrance")
        
        return {
            "room_counts": self.room_types_count,
            "entrance_count": entrance_count,
            "adjacency": adjacency
        }

    def graph_to_string(self):
        G = self.room_connectivity_graph
        
        rStrings = []

        edges = G.edges()
        for edge in edges:
            room1, room2 = edge
            rStrings.append(f'"{room1}" is next to "{room2}"')

        return "\n".join(rStrings)

    def _label_instances(self, room_grid):
        """Create instance map from connected components per room type."""
        inst = np.zeros(room_grid.shape, dtype=np.int32)
        next_id = 1
        for rv in np.unique(room_grid):
            if rv > 11:
                continue
            mask = (room_grid == rv).astype(np.uint8)
            if mask.sum() == 0:
                continue
            num, labels = cv2.connectedComponents(mask, connectivity=4)
            for comp_id in range(1, num):
                inst[labels == comp_id] = next_id
                next_id += 1
        return inst

    def _postprocess_and_mask(self, resized_fp):
        """Apply postprocessing and calculate mask based on final room grid.
        
        Expects and returns a 4-channel array:
            ch0: room type, ch1: global instance ID, ch2: per-type instance idx, ch3: mask
        """
        resized_fp[:, :, 0] = apply_room_postprocess(resized_fp[:, :, 0].astype(np.int32))
        resized_fp[:, :, 1] = self._label_instances(resized_fp[:, :, 0].astype(np.int32))
        # ch2 (per-type instance indices) is preserved from the resize step
        resized_fp[:, :, 3] = (resized_fp[:, :, 0] <= 11).astype(np.uint8)
        return resized_fp

    # ── Room remapping ────────────────────────────────────────────────

    def remap_rooms(self, resized_fp, mode="instances"):
        """
        Remap a resized floorplan array into a single-channel color-coded image.
        
        Uses the 4-channel resized array produced by outline_based_resize / pixel_based_resize:
            ch0: room type (original values 0-11, 13)
            ch1: global instance ID
            ch2: per-type instance index (1-based, 0 for non-room)
            ch3: interior mask
        
        Args:
            resized_fp: 4-channel array from a resize method
            mode: One of:
                - "types"       → 13 classes (one per room type, spread 0-255)
                - "basic_types" → 15 classes (bathroom/second room split by instance)
                - "instances"   → 35 classes (unique color per room_type × instance)
        
        Returns:
            np.ndarray: Single-channel uint8 array with color-coded room values
        """
        room_channel = resized_fp[:, :, 0].astype(np.int32)
        instance_channel = resized_fp[:, :, 2].astype(np.int32)
        
        remapped = np.zeros(room_channel.shape, dtype=np.uint8)
        
        if mode == "types":
            num = self.info.num_room_types
            for orig_val, new_val in self.info.room_type_remap.items():
                visual_val = int(new_val * (255 / (num - 1)))
                remapped[room_channel == orig_val] = visual_val
        
        elif mode == "basic_types":
            BASIC_CLASS_MAP = {
                0: 0, 1: 1, 2: 2,
                3: (3, 4),        # bathroom 1 / 2
                4: 5, 5: 6, 6: 7,
                7: (8, 9),        # second room 1 / 2
                8: 10,
                9: 11,            # balcony (all instances share color)
                10: 12, 11: 13, 13: 14,
            }
            num = self.info.num_basic_room_types
            
            def _val(cid):
                return int(cid * (255 / (num - 1)))
            
            for orig_val, class_ids in BASIC_CLASS_MAP.items():
                room_mask = (room_channel == orig_val)
                if isinstance(class_ids, tuple):
                    id_1, id_2 = class_ids
                    mask_2 = room_mask & (instance_channel == 2)
                    mask_1 = room_mask & ~mask_2
                    remapped[mask_1] = _val(id_1)
                    remapped[mask_2] = _val(id_2)
                else:
                    remapped[room_mask] = _val(class_ids)
        
        elif mode == "instances":
            cmap = self.info.room_instance_color_map
            max_inst = self.info.room_type_max_instances
            remap = self.info.room_type_remap
            remap_name = self.info.remap_to_name
            
            for orig_val, new_val in remap.items():
                room_name = remap_name.get(new_val, "unknown")
                mi = max_inst.get(room_name, 1)
                room_mask = (room_channel == orig_val)
                
                # External area: instance_channel is 0, not instanced
                if orig_val == 13:
                    key = (new_val, 0)
                    if key in cmap:
                        remapped[room_mask] = cmap[key]
                    continue
                
                for inst_id in range(mi):
                    if mi == 1:
                        inst_mask = room_mask & (instance_channel > 0)
                    else:
                        inst_mask = room_mask & (instance_channel == (inst_id + 1))
                    key = (new_val, inst_id)
                    if key in cmap:
                        remapped[inst_mask] = cmap[key]
        else:
            raise ValueError(f"Unknown remap mode: {mode!r}. Use 'types', 'basic_types', or 'instances'.")
        
        return remapped

    def create_exterior_mask(self, resized_fp):
        """
        Create an alpha mask from a resized floorplan array.
        
        Returns:
            np.ndarray: uint8 array — 255 for interior, 0 for exterior
        """
        room_channel = resized_fp[:, :, 0].astype(np.int32)
        exterior_mask = (room_channel == 13)
        return np.where(exterior_mask, 0, 255).astype(np.uint8)

    def create_daylight_mask(self, resized_fp):
        """
        Create a daylight-need mask from a resized floorplan array.
        
        Marks rooms that need daylight (based on info.remap_needs_daylight)
        AND the immediately neighbouring exterior pixels — i.e. the building
        perimeter positions where windows could or should be placed.
        
        Pixel values in the returned mask:
            0   — not relevant (interior rooms that don't need light, or far exterior)
            128 — exterior pixel adjacent to a daylight-needing room (potential window)
            255 — room pixel that needs daylight
        
        Args:
            resized_fp: 4-channel array from a resize method
        
        Returns:
            np.ndarray: uint8 single-channel mask
        """
        room_channel = resized_fp[:, :, 0].astype(np.int32)
        remap = self.info.room_type_remap
        needs_daylight = self.info.remap_needs_daylight
        external_area = 13

        # 1. Build binary mask of rooms that need daylight
        daylight_rooms = np.zeros(room_channel.shape, dtype=np.uint8)
        for orig_val, remapped_val in remap.items():
            if needs_daylight.get(remapped_val, False):
                daylight_rooms[room_channel == orig_val] = 255

        # 2. Dilate by 1 pixel (cross kernel: only up/down/left/right, no diagonals)
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        dilated = cv2.dilate(daylight_rooms, kernel, iterations=1)

        # 3. The exterior boundary ring = dilated pixels that are in the exterior
        exterior_mask = (room_channel == external_area)
        boundary_ring = (dilated > 0) & exterior_mask

        # 4. Output only the exterior neighbour pixels (the dilated ring)
        result = np.zeros(room_channel.shape, dtype=np.uint8)
        result[boundary_ring] = 255

        return result

    def has_single_instances_only(self):
        """
        Check if this floorplan has at most one instance of each room type,
        with exceptions for allowed multi-instance types (balcony, bathroom, second room).
        """
        allowed = self.info.basic_types_multi_instance_allowed
        for room_type, count in self.room_types_count.items():
            max_allowed = allowed.get(room_type, 1)
            if count > max_allowed:
                return False
        return True

    ### here llm descriptions ###
    
    def generate_llm_descriptions(self, llm, system_message, query, description_count=3, pixel_based_size=16, use_outline_based=True):
        
        if use_outline_based:
            resized_fp = self.outline_based_resize(pixel_based_size)
        else:   
            resized_fp = self.pixel_based_resize(pixel_based_size)
        room_types_count = self.get_room_types_count()     
        graph_string = self.graph_to_string()
        
        # Calculate room sizes
        room_sizes = self.calculate_room_sizes(resized_fp)
        
        # Build node metadata to preserve per-room identity and provide stable centroids
        nodes = []
        # Scale factors so centroids can be mapped onto the resized grid for visualization
        scale_x = resized_fp.shape[1] / self.image.shape[1]
        scale_y = resized_fp.shape[0] / self.image.shape[0]

        # Use the final instance grid to map centroids to instance ids
        final_instances = resized_fp[:, :, 1]

        for node_id, data in self.room_connectivity_graph.nodes(data=True):
            cx, cy = data.get("centroid", (None, None))

            instance_id = None
            if cx is not None and cy is not None:
                # Map centroid into resized grid and sample the final instance map
                gx = int(np.clip(cx * scale_x, 0, resized_fp.shape[1] - 1))
                gy = int(np.clip(cy * scale_y, 0, resized_fp.shape[0] - 1))
                instance_id = int(final_instances[gy, gx])

            nodes.append({
                "id": node_id,
                "room_type": data.get("room_type"),
                "centroid": [cx, cy] if cx is not None and cy is not None else None,
                # Centroid rescaled to the resized grid for plotting
                "centroid_resized": [cx * scale_x, cy * scale_y] if cx is not None and cy is not None else None,
                "instance_id": instance_id
            })
        
        data = {
            "room_counts": room_types_count,
            "original_dimensions": [self.image.shape[0], self.image.shape[1]],
            "dimensions": [resized_fp.shape[0], resized_fp.shape[1]],
            "functions": resized_fp[:,:,0].tolist(),
            "instances": resized_fp[:,:,1].tolist(),
            "per_type_instances": resized_fp[:,:,2].tolist(),
            "mask": resized_fp[:,:,3].tolist(),
            "graph": self.get_room_connectivity_matrix().tolist(),
            "nodes": nodes,
            "graph_string": graph_string,
            "room_sizes_m2": room_sizes
        }
        
        descriptions = get_descriptions(data, llm, system_message, query, description_count)
        data["descriptions"] = descriptions
        
        return data
        


