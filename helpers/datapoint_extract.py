import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt


class FloorplanDatapoint:
    """
    A class to extract and manage floorplan data from JSON exports.
    Compatible with PyTorch Dataset interface.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize a datapoint from a JSON file.
        
        Args:
            json_path: Path to the exported floorplan JSON file
        """
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    # ===== Basic Properties =====
    
    @property
    def original_dimensions(self) -> Tuple[int, int]:
        """Get original image dimensions (height, width)."""
        h, w = self.data["original_dimensions"]
        return h, w
    
    @property
    def resized_dimensions(self) -> Tuple[int, int]:
        """Get resized image dimensions (height, width)."""
        h, w = self.data["dimensions"]
        return h, w
    
    @property
    def room_counts(self) -> Dict[str, int]:
        """Get count of each room type."""
        return self.data["room_counts"]
    
    @property
    def graph_string(self) -> str:
        """Get room connectivity as text description."""
        return self.data.get("graph_string", "")
    
    # ===== Grid Data =====
    
    @property
    def functions(self) -> np.ndarray:
        """Get room type grid from resized image (height, width)."""
        return np.array(self.data["functions"], dtype=np.uint8)
    
    @property
    def instances(self) -> np.ndarray:
        """Get room instance grid from resized image (height, width)."""
        return np.array(self.data["instances"], dtype=np.uint8)
    
    @property
    def mask(self) -> np.ndarray:
        """Get valid room mask from resized image (height, width)."""
        return np.array(self.data["mask"], dtype=np.uint8)
    
    @property
    def connectivity_matrix(self) -> np.ndarray:
        """Get room connectivity adjacency matrix."""
        return np.array(self.data["graph"], dtype=np.float32)
    
    # ===== Size Information =====
    
    def get_room_sizes(self) -> Dict[str, Dict[int, float]]:
        """
        Get room sizes in both original and resized scales.
        
        Returns:
            dict with keys:
                - 'original_sizes': {instance_id: size_m2}
                - 'resized_sizes': {instance_id: size_m2}
                - 'meter_to_pixel': original pixels per meter
                - 'adjusted_meter_to_pixel': adjusted pixels per meter after resize
        """
        return self.data["room_sizes_m2"]
    
    def get_instance_size(self, instance_id: int, scale: str = "original") -> Optional[float]:
        """
        Get size of a specific instance.
        
        Args:
            instance_id: The instance ID to look up
            scale: Either "original" or "resized"
        
        Returns:
            Size in m2, or None if not found
        """
        sizes = self.get_room_sizes()
        size_key = f"{scale}_sizes"
        if size_key in sizes:
            return sizes[size_key].get(str(instance_id))
        return None
    
    def get_total_area(self, scale: str = "original") -> float:
        """
        Get total area of all rooms.
        
        Args:
            scale: Either "original" or "resized"
        
        Returns:
            Total area in m2
        """
        sizes = self.get_room_sizes()
        size_key = f"{scale}_sizes"
        if size_key in sizes:
            return sum(sizes[size_key].values())
        return 0.0
    
    # ===== Node Information =====
    
    @property
    def nodes(self) -> List[Dict]:
        """Get all room nodes with their properties."""
        return self.data["nodes"]
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Get a specific node by its ID."""
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None
    
    def get_nodes_by_room_type(self, room_type: str) -> List[Dict]:
        """Get all nodes of a specific room type."""
        return [node for node in self.nodes if node["room_type"] == room_type]
    
    # ===== Descriptions =====
    
    @property
    def descriptions(self) -> List[str]:
        """Get LLM-generated descriptions."""
        return self.data.get("descriptions", [])
    
    # ===== Utility Methods =====
    
    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert all grid data to numpy arrays for ML pipelines.
        
        Returns:
            dict with keys: 'functions', 'instances', 'mask', 'connectivity'
        """
        return {
            "functions": self.functions,
            "instances": self.instances,
            "mask": self.mask,
            "connectivity": self.connectivity_matrix
        }
    
    def get_room_stats(self) -> Dict:
        """
        Get comprehensive statistics about all rooms.
        
        Returns:
            dict with room-wise statistics including counts and sizes
        """
        sizes = self.get_room_sizes()
        stats = {
            "room_counts": self.room_counts,
            "total_area_original_m2": self.get_total_area("original"),
            "total_area_resized_m2": self.get_total_area("resized"),
            "meter_to_pixel_original": sizes.get("meter_to_pixel"),
            "meter_to_pixel_adjusted": sizes.get("adjusted_meter_to_pixel"),
            "original_dimensions": self.original_dimensions,
            "resized_dimensions": self.resized_dimensions,
        }
        return stats
    
    # ===== Graph and Visualization =====
    
    def build_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from the connectivity matrix and nodes.
        
        Returns:
            NetworkX Graph with nodes labeled by room IDs
        """
        G = nx.from_numpy_array(self.connectivity_matrix)
        
        # Relabel nodes with their actual IDs
        node_id_map = {i: node["id"] for i, node in enumerate(self.nodes)}
        G = nx.relabel_nodes(G, node_id_map)
        
        # Add node attributes
        for node in self.nodes:
            node_id = node["id"]
            if G.has_node(node_id):
                G.nodes[node_id].update({
                    "room_type": node["room_type"],
                    "centroid": node.get("centroid"),
                    "instance_id": node.get("instance_id")
                })
        
        return G
    
    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate 2D positions for nodes based on instance centroids in the resized grid.
        
        Returns:
            dict mapping node_id to (x, y) coordinates
        """
        pos = {}
        instances = self.instances
        
        for node in self.nodes:
            node_id = node["id"]
            instance_id = node.get("instance_id")
            
            # Try to find position from instance mask
            if instance_id is not None:
                node_mask = instances == instance_id
                if node_mask.any():
                    y_coords, x_coords = np.where(node_mask)
                    pos[node_id] = (np.mean(x_coords), np.mean(y_coords))
                    continue
            
            # Fall back to centroid_resized
            if node.get("centroid_resized"):
                cx, cy = node["centroid_resized"]
                pos[node_id] = (cx, cy)
                continue
            
            # Fall back to original centroid scaled
            if node.get("centroid"):
                cx, cy = node["centroid"]
                scale_x = self.resized_dimensions[1] / self.original_dimensions[1]
                scale_y = self.resized_dimensions[0] / self.original_dimensions[0]
                pos[node_id] = (cx * scale_x, cy * scale_y)
        
        return pos
    
    def visualize(self, figsize: Tuple[int, int] = (20, 7), show: bool = True) -> plt.Figure:
        """
        Visualize the floorplan with mask, instances, and graph overlay.
        
        Args:
            figsize: Figure size (width, height)
            show: Whether to call plt.show()
        
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        mask = self.mask
        instances = self.instances
        functions = self.functions
        
        # Subplot 1: Mask
        ax1.imshow(np.stack([mask, mask, mask], axis=-1))
        ax1.set_title('Mask (Valid Room Areas)')
        ax1.axis('on')
        
        # Subplot 2: Instances
        im2 = ax2.imshow(instances, cmap='tab20')
        ax2.set_title('Instances (Individual Rooms)')
        ax2.axis('on')
        plt.colorbar(im2, ax=ax2, label='Instance ID')
        
        # Subplot 3: Functions + Graph
        ax3.imshow(functions, cmap='viridis', alpha=0.7)
        
        # Build graph and get positions
        G = self.build_graph()
        pos = self.get_node_positions()
        
        # Draw edges
        for u, v in G.edges():
            if u in pos and v in pos:
                ax3.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                        color='yellow', linewidth=3, alpha=0.8, zorder=5)
        
        # Draw nodes
        if pos:
            node_pos = np.array([pos[n] for n in G.nodes() if n in pos])
            ax3.scatter(node_pos[:, 0], node_pos[:, 1], c='red', s=200, zorder=10, 
                       edgecolors='white', linewidth=2)
        
        # Draw labels
        for node, (x, y) in pos.items():
            room_type = node.rsplit('_', 1)[0]
            ax3.text(x, y, room_type, fontsize=9, fontweight='bold', ha='center', va='center',
                    color='white', zorder=15, bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                    alpha=0.7, edgecolor='white', linewidth=1))
        
        ax3.set_title('Room Functions + Connectivity Graph')
        ax3.axis('on')
        grid_h, grid_w = self.resized_dimensions
        ax3.set_xlim(-1, grid_w + 1)
        ax3.set_ylim(grid_h + 1, -1)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def print_info(self):
        """Print comprehensive information about the datapoint."""
        print(f"\n{'='*50}")
        print(f"Floorplan Datapoint: {self.json_path}")
        print(f"{'='*50}")
        print(f"\nDimensions:")
        print(f"  Original: {self.original_dimensions[0]}x{self.original_dimensions[1]}")
        print(f"  Resized:  {self.resized_dimensions[0]}x{self.resized_dimensions[1]}")
        
        print(f"\nRoom Counts:")
        for room_type, count in self.room_counts.items():
            print(f"  {room_type}: {count}")
        
        sizes = self.get_room_sizes()
        print(f"\nArea Information:")
        print(f"  Total original area: {self.get_total_area('original'):.2f} m²")
        print(f"  Total resized area:  {self.get_total_area('resized'):.2f} m²")
        print(f"  Meter-to-pixel (original): {sizes.get('meter_to_pixel')}")
        print(f"  Meter-to-pixel (adjusted): {sizes.get('adjusted_meter_to_pixel'):.4f}")
        
        print(f"\nRooms ({len(self.nodes)} total):")
        for node in self.nodes:
            inst_id = node.get("instance_id", "?")
            room_type = node["room_type"]
            size_orig = self.get_instance_size(inst_id, "original")
            size_resized = self.get_instance_size(inst_id, "resized")
            print(f"  {node['id']}")
            print(f"    Type: {room_type}, Instance: {inst_id}")
            if size_orig is not None:
                print(f"    Size: {size_orig:.2f} m² (orig) / {size_resized:.4f} m² (resized)")
        
        print(f"\nConnectivity Graph:")
        G = self.build_graph()
        print(f"  Edges: {list(G.edges())}")
        
        if self.descriptions:
            print(f"\nGenerated Descriptions ({len(self.descriptions)}):")
            for i, desc in enumerate(self.descriptions, 1):
                print(f"  [{i}] {desc[:100]}...")
        
        print(f"{'='*50}\n")


class FloorplanDataset:
    """
    PyTorch-compatible Dataset class for floorplan data.
    """
    
    def __init__(self, json_dir: str):
        """
        Initialize dataset from directory of JSON files.
        
        Args:
            json_dir: Directory containing exported floorplan JSON files
        """
        self.json_dir = Path(json_dir)
        self.json_files = sorted(self.json_dir.glob("description_*.json"))
        
        if not self.json_files:
            raise ValueError(f"No JSON files found in {json_dir}")
    
    def __len__(self) -> int:
        """Get number of datapoints."""
        return len(self.json_files)
    
    def __getitem__(self, idx: int) -> FloorplanDatapoint:
        """
        Get a datapoint by index.
        
        Args:
            idx: Index of the datapoint
        
        Returns:
            FloorplanDatapoint object
        """
        return FloorplanDatapoint(str(self.json_files[idx]))
    
    def get_by_path(self, json_path: str) -> FloorplanDatapoint:
        """Get a datapoint by file path."""
        return FloorplanDatapoint(json_path)
    
    def get_all_room_stats(self) -> List[Dict]:
        """Get statistics for all datapoints."""
        return [self[i].get_room_stats() for i in range(len(self))]
    
    def get_average_stats(self) -> Dict:
        """Get average statistics across all datapoints."""
        stats_list = self.get_all_room_stats()
        
        if not stats_list:
            return {}
        
        avg_stats = {
            "num_datapoints": len(stats_list),
            "avg_total_area_original_m2": np.mean([s["total_area_original_m2"] for s in stats_list]),
            "avg_total_area_resized_m2": np.mean([s["total_area_resized_m2"] for s in stats_list]),
        }
        
        # Average room counts across dataset
        all_room_types = set()
        for s in stats_list:
            all_room_types.update(s["room_counts"].keys())
        
        avg_room_counts = {}
        for room_type in all_room_types:
            counts = [s["room_counts"].get(room_type, 0) for s in stats_list]
            avg_room_counts[room_type] = np.mean(counts)
        
        avg_stats["avg_room_counts"] = avg_room_counts
        
        return avg_stats


# Example usage:
if __name__ == "__main__":
    # Single datapoint
    dp = FloorplanDatapoint("./output/description_0.json")
    print(f"Original dimensions: {dp.original_dimensions}")
    print(f"Resized dimensions: {dp.resized_dimensions}")
    print(f"Room counts: {dp.room_counts}")
    print(f"Total area (original): {dp.get_total_area('original'):.2f} m2")
    print(f"Total area (resized): {dp.get_total_area('resized'):.2f} m2")
    
    # Print comprehensive info
    dp.print_info()
    
    # Visualize
    # dp.visualize()
    
    # Dataset
    # dataset = FloorplanDataset("./output")
    # print(f"\nDataset size: {len(dataset)}")
    # dp = dataset[0]
    # print(f"First datapoint: {dp.original_dimensions}")
