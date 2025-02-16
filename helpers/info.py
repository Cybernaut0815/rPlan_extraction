from dataclasses import dataclass
import matplotlib.pyplot as plt

#@dataclass
class Info:
        def __init__(self):
                self.all_types = {"living room": 0, 
                        "master room": 1, 
                        "kitchen": 2, 
                        "bathroom": 3, 
                        "dining room": 4, 
                        "child room": 5,
                        "study room": 6,
                        "second room": 7,
                        "guest room": 8,
                        "balcony": 9,
                        "entrance": 10,
                        "storage": 11, 
                        "wall-in": 12,
                        "external area": 13,
                        "exterior wall": 14,
                        "front door": 15,
                        "interior wall": 16,
                        "interior door": 17,
                        }

                self.room_types = {"living room": 0, 
                        "master room": 1, 
                        "kitchen": 2, 
                        "bathroom": 3, 
                        "dining room": 4, 
                        "child room": 5,
                        "study room": 6,
                        "second room": 7,
                        "guest room": 8,
                        "balcony": 9,
                        "entrance": 10,
                        "storage": 11, 
                        }

                self.room_types_colors = {"living room": (0,0,255), 
                        "master room": (0,255,0), 
                        "kitchen": (255,0,0), 
                        "bathroom": (0,255,255), 
                        "dining room": (255,255,0), 
                        "child room": (255,0,255), 
                        "study room": (0,255,255), 
                        }
                
                self.bedroom_synonyms = ["child room", "master room", "second room", "guest room", "study room"]
        
        def get_type_color(self, colormap="nipy_spectral"):
                # Create colormap object using newer matplotlib syntax
                cmap = plt.colormaps[colormap]
                
                # Calculate colors for each type based on their position
                n_types = len(self.all_types)
                # Create a list to store colors in order
                colors = [None] * n_types
                for room_type, idx in self.all_types.items():
                        # Normalize index to [0, 1] range for colormap
                        normalized_idx = idx / (n_types - 1)
                        # Get RGB values (multiply by 255 to get 0-255 range)
                        rgb = tuple(int(x * 255) for x in cmap(normalized_idx)[:3])
                        # Store color at the corresponding index
                        colors[idx] = rgb
                
                return colors

        

