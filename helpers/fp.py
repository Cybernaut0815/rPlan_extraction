import os
import numpy as np
import cv2

from helpers.info import Info
from helpers.utils import load_image, resize_plan


class Floorplan:
    def __init__(self, path):
        self._image = np.array(load_image(path))
        self._outer_wall_channel = self._image[:,:,0]
        self._room_types_channel = self._image[:,:,1]
        self._distinct_rooms_channel = self._image[:,:,2]
        self._mask_channel = self._image[:,:,3]
        
        self.info = Info()
        
        self.contours = self.get_contours()
        self.room_types_count = self.room_types_count()
    
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
        self.types_count = self.room_types_count() 

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


    
    def room_types_count(self):
        count_dict = {}
        for key, value in self.info.all_types.items():
            room_values = np.where(self.image[:,:,0] == value)    
            # Get the values from channel 1 where room_type matches
            masked_values = self.image[:,:,1][room_values]
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

        # Shift grid by half the spacing
        X = X + x_spacing/2
        Y = Y + y_spacing/2
        
        return resize_plan(self.image, X, Y)




