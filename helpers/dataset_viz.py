import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from helpers.info import Info


def load_and_visualize_datapoint(index, output_dir):
    """Load a saved datapoint JSON and visualize it with the graph overlaid on the room types grid."""
    info = Info()
    
    # Load the JSON
    json_path = os.path.join(output_dir, f"description_{index}.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data
    functions = np.array(data["functions"])
    instances = np.array(data["instances"])
    mask = np.array(data["mask"])
    dimensions = data["dimensions"]
    nodes = data["nodes"]
    graph_adj = np.array(data["graph"])
    
    # Build graph and positions
    G = nx.from_numpy_array(graph_adj)
    node_id_map = {i: node["id"] for i, node in enumerate(nodes)}
    G = nx.relabel_nodes(G, node_id_map)
    
    pos = {}
    grid_h, grid_w = dimensions
    
    for idx, node in enumerate(nodes):
        node_id = node["id"]
        room_type = node.get("room_type")

        # For entrance, always use centroid-based position (do not use instance mask)
        if room_type == "entrance":
            if node.get("centroid_resized"):
                cx, cy = node["centroid_resized"]
                pos[node_id] = (cx, cy)
            elif node.get("centroid"):
                cx, cy = node["centroid"]
                scale_x = grid_w / data["original_dimensions"][1]
                scale_y = grid_h / data["original_dimensions"][0]
                pos[node_id] = (cx * scale_x, cy * scale_y)
            continue

        # Default: use instance mask, then fall back to centroid-based position
        instance_id = node.get("instance_id", idx)
        node_mask = instances == instance_id

        if node_mask.any():
            y_coords, x_coords = np.where(node_mask)
            pos[node_id] = (np.mean(x_coords), np.mean(y_coords))
        elif node.get("centroid_resized"):
            cx, cy = node["centroid_resized"]
            pos[node_id] = (cx, cy)
        elif node.get("centroid"):
            cx, cy = node["centroid"]
            scale_x = grid_w / data["original_dimensions"][1]
            scale_y = grid_h / data["original_dimensions"][0]
            pos[node_id] = (cx * scale_x, cy * scale_y)
    
    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    ax1.imshow(np.stack([mask, mask, mask], axis=-1))
    ax1.set_title(f'Datapoint {index}: Mask')
    ax1.axis('on')
    
    im2 = ax2.imshow(instances, cmap='tab20')
    ax2.set_title(f'Datapoint {index}: Instances')
    ax2.axis('on')
    plt.colorbar(im2, ax=ax2, label='Instance ID')
    
    ax3.imshow(functions, cmap='viridis', alpha=0.7)
    
    # Draw edges
    for u, v in G.edges():
        if u in pos and v in pos:
            ax3.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                    color='yellow', linewidth=3, alpha=0.8, zorder=5)
    
    # Draw nodes and labels
    if pos:
        node_pos = np.array([pos[n] for n in G.nodes() if n in pos])
        ax3.scatter(node_pos[:, 0], node_pos[:, 1], c='red', s=200, zorder=10, 
                   edgecolors='white', linewidth=2)
    
    for node, (x, y) in pos.items():
        room_type = node.rsplit('_', 1)[0]
        ax3.text(x, y, room_type, fontsize=9, fontweight='bold', ha='center', va='center',
                color='white', zorder=15, bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                alpha=0.7, edgecolor='white', linewidth=1))
    
    ax3.set_title(f'Datapoint {index}: Functions + Graph')
    ax3.axis('on')
    ax3.set_xlim(-1, grid_w + 1)
    ax3.set_ylim(grid_h + 1, -1)
    
    plt.tight_layout()
    plt.show()
    
    # Print metadata
    print(f"\n=== Datapoint {index} ===")
    print(f"Dimensions: {dimensions}")
    print(f"Room Counts: {data['room_counts']}")
    print(f"Number of Rooms: {len(nodes)}")
    print("\nRooms:")
    for node in nodes:
        print(f"  {node['id']}: {node['room_type']}")
    print(f"\nGraph Connections: {list(G.edges())}")
