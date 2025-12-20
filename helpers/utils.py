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





    



