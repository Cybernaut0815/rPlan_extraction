from dataclasses import dataclass

@dataclass
class Info:
    types = {"living room": 0, 
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
    
    room_types = {"living room": 0, 
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
    
    bedroom_synonyms = ["child room", "master room", "second room", "guest room", "study room"]

