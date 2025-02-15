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

    def check_neighbors(room_type_channel, start_y, start_x):
        from collections import deque
        queue = deque([(start_y, start_x)])
        visited = set()
        
        # Define all 8 directions
        directions = [
            (-1,-1), (-1,0), (-1,1),
            (0,-1),          (0,1),
            (1,-1),  (1,0),  (1,1)
        ]
        
        while queue:
            y, x = queue.popleft()
            
            if (y >= room_type_channel.shape[0] or x >= room_type_channel.shape[1] or 
                y < 0 or x < 0 or (y,x) in visited):
                continue
            
            visited.add((y,x))
            value = room_type_channel[y,x]
            
            # If value <= 11, return the position where we found it
            if value <= 11:
                return y, x
            
            # Add all neighbors to queue
            for dy, dx in directions:
                queue.append((y + dy, x + dx))
                
        # If no valid value found, return original position
        return start_y, start_x

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
                # Get the position of the valid value
                final_y, final_x = check_neighbors(room_type_channel, y, x)
                # Use the same position for both channels
                processed_values[i,j,0] = room_type_channel[final_y, final_x]
                processed_values[i,j,1] = second_channel[final_y, final_x]
                processed_values[i,j,2] = 0  # mask value
    
    return processed_values
