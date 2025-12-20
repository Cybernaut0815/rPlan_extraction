import json
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


def process_grid_points(room_type_channel, X, Y):
    # Extract values at grid points
    values = room_type_channel[Y.astype(int), X.astype(int)]

    def check_neighbors(room_type_channel, start_y, start_x):
        from collections import deque
        queue = deque([(start_y, start_x)])
        visited = set()
        
        # Define all 8 directions (up, down, left, right, and diagonals)
        directions = [
            (-1,-1), (-1,0), (-1,1),  # up-left, up, up-right
            (0,-1),          (0,1),   # left, right
            (1,-1),  (1,0),  (1,1)    # down-left, down, down-right
        ]
        
        while queue:
            y, x = queue.popleft()
            
            # If point is out of bounds or already visited, continue
            if (y >= room_type_channel.shape[0] or x >= room_type_channel.shape[1] or 
                y < 0 or x < 0 or (y,x) in visited):
                continue
            
            visited.add((y,x))
            value = room_type_channel[y,x]
            
            # If value <= 11, return it
            if value <= 11:
                return value
            
            # Add all neighbors to queue
            for dy, dx in directions:
                queue.append((y + dy, x + dx))
                
        # If no valid value found, return original value
        return room_type_channel[start_y, start_x]

    # Process each grid point
    processed_values = np.zeros_like(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            y = Y[i,j].astype(int)
            x = X[i,j].astype(int)
            if room_type_channel[y,x] == 13:
                processed_values[i,j] = 13
            else:
                processed_values[i,j] = check_neighbors(room_type_channel, y, x)
            
    return processed_values



def resize_plan(img, X, Y):
    # Extract values at grid points for both channels
    room_type_channel = img[:,:,1]
    second_channel = img[:,:,2]
    
    values_ch1 = room_type_channel[Y.astype(int), X.astype(int)]
    def check_neighbors_weighted(room_type_channel, second_channel, start_y, start_x, window_size=7, max_radius=50):
        # Use distance-weighted voting to find values for both channels
        # Returns the winning room type value and corresponding second channel value
        h, w = room_type_channel.shape
        
        # Try expanding windows until we find valid values
        for k in range(window_size, max_radius + 1, 2):
            half_k = k // 2
            y_min = max(0, start_y - half_k)
            y_max = min(h, start_y + half_k + 1)
            x_min = max(0, start_x - half_k)
            x_max = min(w, start_x + half_k + 1)
            
            # Extract windows for both channels
            window_ch1 = room_type_channel[y_min:y_max, x_min:x_max]
            window_ch2 = second_channel[y_min:y_max, x_min:x_max]
            
            # Get valid mask (room types <= 11)
            valid_mask = window_ch1 <= 11
            
            if np.any(valid_mask):
                # Create coordinate grids for the window
                y_coords, x_coords = np.meshgrid(
                    np.arange(y_min, y_max),
                    np.arange(x_min, x_max),
                    indexing='ij'
                )
                
                # Calculate squared distances from start point
                dy = y_coords - start_y
                dx = x_coords - start_x
                dist2 = dy * dy + dx * dx
                
                # Weight by inverse distance (closer = higher weight)
                weights = 1.0 / (dist2 + 1.0)
                
                # Only consider valid room types
                weights = weights * valid_mask
                
                # Get valid values and their weights
                valid_values_ch1 = window_ch1[valid_mask].astype(int)
                valid_values_ch2 = window_ch2[valid_mask].astype(int)
                valid_weights = weights[valid_mask]
                
                # Weighted voting for channel 1: accumulate weights for each room type
                weighted_votes = {}
                for val, weight in zip(valid_values_ch1, valid_weights):
                    weighted_votes[val] = weighted_votes.get(val, 0.0) + weight
                
                # Find room type with highest weighted vote
                winner_ch1 = max(weighted_votes.items(), key=lambda x: x[1])[0]
                
                # For channel 2: weighted average among pixels with the winning room type
                winner_mask = (window_ch1 == winner_ch1) & valid_mask
                winner_weights = weights[winner_mask]
                winner_ch2_values = window_ch2[winner_mask]
                
                # Weighted average for channel 2
                if np.sum(winner_weights) > 0:
                    winner_ch2 = int(np.round(np.average(winner_ch2_values, weights=winner_weights)))
                else:
                    winner_ch2 = int(winner_ch2_values[0])
                
                return winner_ch1, winner_ch2
        
        # If no valid value found, return original values
        return int(room_type_channel[start_y, start_x]), int(second_channel[start_y, start_x])

    # Create a 3D array to hold the results including mask
    processed_values = np.zeros((values_ch1.shape[0], values_ch1.shape[1], 3))
    
    for i in range(values_ch1.shape[0]):
        for j in range(values_ch1.shape[1]):
            y = Y[i,j].astype(int)
            x = X[i,j].astype(int)
            
            if room_type_channel[y,x] == 13:
                processed_values[i,j,0] = 13
                processed_values[i,j,1] = second_channel[y,x]
                processed_values[i,j,2] = 1  # mask value
            else:
                # Get the values directly from weighted voting
                val_ch1, val_ch2 = check_neighbors_weighted(room_type_channel, second_channel, y, x)
                processed_values[i,j,0] = val_ch1
                processed_values[i,j,1] = val_ch2
                processed_values[i,j,2] = 0  # mask value
    
    return processed_values


def load_image_paths(path):
    files = sorted(os.listdir(path), key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[-1].split('.')[0]))))
    return [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]

def load_image(path):
    return Image.open(path)


def _majority_filter_4(room_channel, num_classes=18):
    """4-neighbor majority without 2-2 tie. Returns filtered channel."""
    h, w = room_channel.shape
    counts = np.zeros((h, w, num_classes), dtype=np.int16)
    # Up
    if h > 1:
        up = room_channel[:-1, :]
        counts[1:, :, :] += np.eye(num_classes, dtype=np.int16)[up]
    # Down
    if h > 1:
        down = room_channel[1:, :]
        counts[:-1, :, :] += np.eye(num_classes, dtype=np.int16)[down]
    # Left
    if w > 1:
        left = room_channel[:, :-1]
        counts[:, 1:, :] += np.eye(num_classes, dtype=np.int16)[left]
    # Right
    if w > 1:
        right = room_channel[:, 1:]
        counts[:, :-1, :] += np.eye(num_classes, dtype=np.int16)[right]

    max_counts = counts.max(axis=-1)
    modes = counts.argmax(axis=-1)
    tie_mask = (counts == max_counts[..., None]).sum(axis=-1) > 1
    update_mask = (max_counts >= 2) & (~tie_mask) & (modes != room_channel)
    return np.where(update_mask, modes, room_channel)


def _corner_fill_8(room_channel):
    """Fill eaten-out corners: if 3-corner neighbors match and others match current, fill pixel."""
    h, w = room_channel.shape
    padded = np.pad(room_channel, 1, mode='edge')
    # Neighbor slices
    up = padded[0:h, 1:w+1]
    up_right = padded[0:h, 2:w+2]
    right = padded[1:h+1, 2:w+2]
    down_right = padded[2:h+2, 2:w+2]
    down = padded[2:h+2, 1:w+1]
    down_left = padded[2:h+2, 0:w]
    left = padded[1:h+1, 0:w]
    up_left = padded[0:h, 0:w]

    current = room_channel
    result = room_channel.copy()

    corners = [
        (up, up_left, left),          # top-left
        (up, up_right, right),        # top-right
        (down, down_left, left),      # bottom-left
        (down, down_right, right),    # bottom-right
    ]

    all_neighbors = [up, up_right, right, down_right, down, down_left, left, up_left]
    for c1, c2, c3 in corners:
        same_corner = (c1 == c2) & (c2 == c3)
        corner_val = c1
        # other neighbors (exclude the corner triple)
        other_equal = np.ones_like(current, dtype=bool)
        for neigh in all_neighbors:
            if neigh is c1 or neigh is c2 or neigh is c3:
                continue
            other_equal &= (neigh == current)
        mask = same_corner & (corner_val != current) & other_equal
        result = np.where(mask, corner_val, result)
    return result


def apply_room_postprocess(room_channel):
    """Apply 4-neighbor majority then corner fill to room_channel."""
    filtered = _majority_filter_4(room_channel)
    filled = _corner_fill_8(filtered)
    return filled


def expand_rooms_right_down(resized_fp, fillable, num_rooms=12, max_passes=5):
    """Iteratively expand each room right then down into fillable cells (in-place)."""
    h, w, _ = resized_fp.shape
    for _ in range(max_passes):
        changed = False
        for room_val in range(num_rooms):
            yx = np.where(resized_fp[:, :, 0] == room_val)
            if yx[0].size == 0:
                continue

            # Right
            tgt_j = yx[1] + 1
            mask = (tgt_j < w) & fillable[yx[0], tgt_j] & (resized_fp[yx[0], tgt_j, 0] > 11)
            if mask.any():
                ry, rj = yx[0][mask], tgt_j[mask]
                resized_fp[ry, rj, 0] = room_val
                resized_fp[ry, rj, 1] = resized_fp[yx[0][mask], yx[1][mask], 1]
                resized_fp[ry, rj, 2] = 0
                changed = True

            # Down (recompute sources after right phase for this room)
            yx = np.where(resized_fp[:, :, 0] == room_val)
            tgt_i = yx[0] + 1
            mask = (tgt_i < h) & fillable[tgt_i, yx[1]] & (resized_fp[tgt_i, yx[1], 0] > 11)
            if mask.any():
                di, dj = tgt_i[mask], yx[1][mask]
                resized_fp[di, dj, 0] = room_val
                resized_fp[di, dj, 1] = resized_fp[yx[0][mask], yx[1][mask], 1]
                resized_fp[di, dj, 2] = 0
                changed = True

        if not changed:
            break
    return resized_fp





    



