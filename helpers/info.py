from dataclasses import dataclass
import os
import json
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

                # ── Remap tables ──────────────────────────────────────────────
                # Maps original rPlan room-type values → contiguous 0-12
                # (walls / doors excluded)
                self.room_type_remap = {
                        0: 0,   # living room
                        1: 1,   # master room
                        2: 2,   # kitchen
                        3: 3,   # bathroom
                        4: 4,   # dining room
                        5: 5,   # child room
                        6: 6,   # study room
                        7: 7,   # second room
                        8: 8,   # guest room
                        9: 9,   # balcony
                        10: 10, # entrance
                        11: 11, # storage
                        13: 12, # external area
                }
                self.num_room_types = 13  # 0-12 inclusive

                # Remapped ID → human name
                self.remap_to_name = {v: k for k, v in self.room_type_remap.items()
                                       if True}  # placeholder, built properly below
                self.remap_to_name = {
                        0: "living room",
                        1: "master room",
                        2: "kitchen",
                        3: "bathroom",
                        4: "dining room",
                        5: "child room",
                        6: "study room",
                        7: "second room",
                        8: "guest room",
                        9: "balcony",
                        10: "entrance",
                        11: "storage",
                        12: "external area",
                }
                
                self.remap_needs_daylight = {
                        0: True,
                        1: True,
                        2: False,
                        3: False,
                        4: True,
                        5: True,
                        6: True,
                        7: True,
                        8: True,
                        9: True,
                        10: False,
                        11: False,
                        12: False,
                }


                # ── Basic-types mode (15 classes) ─────────────────────────────
                self.basic_types_multi_instance_allowed = {
                        "balcony": 2,
                        "bathroom": 2,
                        "second room": 2,
                }
                self.num_basic_room_types = 15
                self.basic_remap_to_name = {
                        0: "living room",
                        1: "master room",
                        2: "kitchen",
                        3: "bathroom 1",
                        4: "bathroom 2",
                        5: "dining room",
                        6: "child room",
                        7: "study room",
                        8: "second room 1",
                        9: "second room 2",
                        10: "guest room",
                        11: "balcony",
                        12: "entrance",
                        13: "storage",
                        14: "external area",
                }

                # ── Max instances per room type ───────────────────────────────
                self.room_type_max_instances_default = {
                        "living room": 1,
                        "master room": 5,
                        "kitchen": 2,
                        "bathroom": 3,
                        "dining room": 2,
                        "child room": 3,
                        "study room": 3,
                        "second room": 4,
                        "guest room": 3,
                        "balcony": 4,
                        "entrance": 1,
                        "storage": 3,
                        "external area": 1,
                }
                self.room_type_max_instances = self._load_room_type_max_instances()

                # ── Instance color map (35 classes) ───────────────────────────
                (self.room_instance_color_map,
                 self.color_to_room_instance) = self._build_room_instance_color_map()
                self.num_room_instances = sum(self.room_type_max_instances.values())

        # ── helpers ───────────────────────────────────────────────────────

        def _load_room_type_max_instances(self):
                """Load room_type_max_instances from stats.json, fallback to defaults."""
                stats_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..', 'dataset_stats', 'stats.json')
                if os.path.exists(stats_path):
                        try:
                                with open(stats_path, 'r', encoding='utf-8') as f:
                                        stats = json.load(f)
                                loaded = stats.get('room_type_max_instances', {})
                                if loaded:
                                        result = self.room_type_max_instances_default.copy()
                                        result.update(loaded)
                                        return result
                        except (json.JSONDecodeError, IOError):
                                pass
                return self.room_type_max_instances_default.copy()

        def _build_room_instance_color_map(self):
                """Build (room_type_id, instance_id) → color_value and reverse map."""
                color_map = {}
                reverse_map = {}
                total = sum(self.room_type_max_instances.values())
                idx = 0
                for rt_id, rt_name in self.remap_to_name.items():
                        max_inst = self.room_type_max_instances.get(rt_name, 1)
                        for inst in range(max_inst):
                                val = int(idx * (255 / (total - 1))) if total > 1 else 0
                                color_map[(rt_id, inst)] = val
                                reverse_map[val] = (rt_name, inst)
                                idx += 1
                return color_map, reverse_map
        
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

        

