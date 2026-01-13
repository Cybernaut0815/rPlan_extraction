import numpy as np
import json 
import os

import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx

from typing import Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

SIZES_PATH = r".\dataset_stats\sizes.json"
# Rooms with size <= this threshold (m²) are excluded from stats
MIN_VALID_ROOM_SIZE = 1.0
# Connectivity graphs are filtered using statistical threshold (mean occurrence by default)
GRAPH_FILTER_MODE = 'mean'  # 'mean', 'median', or a specific integer value


def _base_room_type(name: str) -> str:
    """Strip index suffix from room name (e.g., bedroom_1 -> bedroom)."""
    return name.split('_')[0] if '_' in name else name


def build_graph_signature(connectivity: Dict) -> Optional[Dict[str, object]]:
    """Create a canonical signature for a connectivity graph (type-level, undirected).

    The signature ignores per-room indices and counts edges by room types, so graphs
    that only differ by numbering share the same signature.
    """
    adjacency = connectivity.get('adjacency') or {}
    if not adjacency:
        return None

    # Include entrance nodes as their own type
    node_counts = defaultdict(int, connectivity.get('room_counts') or {})
    node_counts['entrance'] += connectivity.get('entrance_count', 0)

    edge_counts = defaultdict(int)
    seen_edges = set()

    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            type_a = _base_room_type(edge[0])
            type_b = _base_room_type(edge[1])
            edge_key: Tuple[str, str] = tuple(sorted((type_a, type_b)))
            edge_counts[edge_key] += 1

    nodes_part = sorted(node_counts.items())
    edges_part = sorted(
        ({"edge": f"{a}--{b}", "count": count} for (a, b), count in edge_counts.items()),
        key=lambda item: item["edge"],
    )

    signature_payload = {"nodes": nodes_part, "edges": edges_part}
    signature_str = json.dumps(signature_payload, sort_keys=True)

    return {
        "signature": signature_str,
        "nodes": nodes_part,
        "edges": edges_part,
    }

def extract_dataset_stats(json_path: Path) -> dict:
    """
    Extract statistics from the dataset including room type distributions,
    total sizes, and room counts.
    
    Args:
        json_path: Path to the sizes.json file
        
    Returns:
        Dictionary containing comprehensive dataset statistics
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize structures for collecting statistics (valid rooms only)
    room_type_sizes = defaultdict(list)  # {room_type: [sizes]}
    total_sizes_list = []  # List of total sizes per floorplan (valid rooms only)
    room_counts_list = []  # List of total valid room counts per floorplan
    room_type_counts = defaultdict(int)  # {room_type: total_valid_count_across_dataset}
    graph_signature_counts = defaultdict(int)
    graph_signature_examples: Dict[str, Dict] = {}
    graphs_processed = 0
    
    # Process each floorplan (excluding zero/small rooms from all stats)
    for item in data['items']:
        room_sizes = item['room_sizes_m2']

        # Build connectivity graph signature (independent of size filtering)
        connectivity = item.get('connectivity')
        graph_sig_data = build_graph_signature(connectivity) if connectivity else None
        if graph_sig_data:
            graphs_processed += 1
            sig = graph_sig_data['signature']
            graph_signature_counts[sig] += 1
            if sig not in graph_signature_examples:
                graph_signature_examples[sig] = {
                    'example_file': item.get('file'),
                    'nodes': graph_sig_data['nodes'],
                    'edges': graph_sig_data['edges']
                }

        # Keep only valid rooms
        valid_items = [(rn, sz) for rn, sz in room_sizes.items() if sz > MIN_VALID_ROOM_SIZE]
        if not valid_items:
            # Skip floorplans with no valid rooms to avoid zeros in totals
            continue

        # Calculate total size for this floorplan (valid rooms only)
        total_size = sum(sz for _, sz in valid_items)
        total_sizes_list.append(total_size)

        # Count total valid rooms in this floorplan
        room_counts_list.append(len(valid_items))

        # Process each valid room
        for room_name, size in valid_items:
            # Extract room type (before the underscore)
            room_type = room_name.split('_')[0] if '_' in room_name else room_name
            room_type_sizes[room_type].append(size)
            room_type_counts[room_type] += 1
    
    # Calculate statistics for each room type
    room_type_stats = {}
    for room_type, sizes in room_type_sizes.items():
        if len(sizes) == 0:
            continue
        room_type_stats[room_type] = {
            'sizes': sizes,
            'min': float(np.min(sizes)),
            'max': float(np.max(sizes)),
            'mean': float(np.mean(sizes)),
            'median': float(np.median(sizes)),
            'std': float(np.std(sizes)),
            'count': len(sizes)
        }
    
    # Calculate overall statistics
    # Guard against empty aggregates after filtering
    if len(total_sizes_list) == 0:
        total_sizes_block = {
            'values': [],
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None
        }
    else:
        total_sizes_block = {
            'values': total_sizes_list,
            'min': float(np.min(total_sizes_list)),
            'max': float(np.max(total_sizes_list)),
            'mean': float(np.mean(total_sizes_list)),
            'median': float(np.median(total_sizes_list)),
            'std': float(np.std(total_sizes_list))
        }

    if len(room_counts_list) == 0:
        room_counts_block = {
            'values': [],
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None
        }
    else:
        room_counts_block = {
            'values': room_counts_list,
            'min': int(np.min(room_counts_list)),
            'max': int(np.max(room_counts_list)),
            'mean': float(np.mean(room_counts_list)),
            'median': float(np.median(room_counts_list)),
            'std': float(np.std(room_counts_list))
        }

    sorted_graphs = sorted(graph_signature_counts.items(), key=lambda kv: kv[1], reverse=True)
    
    # Calculate filter threshold based on mode
    occurrence_values = list(graph_signature_counts.values())
    if GRAPH_FILTER_MODE == 'mean':
        filter_threshold = int(np.mean(occurrence_values)) if occurrence_values else 1
        threshold_label = f'mean ({filter_threshold})'
    elif GRAPH_FILTER_MODE == 'median':
        filter_threshold = int(np.median(occurrence_values)) if occurrence_values else 1
        threshold_label = f'median ({filter_threshold})'
    else:
        filter_threshold = int(GRAPH_FILTER_MODE)
        threshold_label = f'{filter_threshold}'
    
    # Filter graphs by occurrence threshold
    filtered_sorted_graphs = [(sig, count) for sig, count in sorted_graphs if count >= filter_threshold]
    
    top_graphs = [
        {
            'signature': sig,
            'count': count,
            'example_file': graph_signature_examples.get(sig, {}).get('example_file'),
            'nodes': graph_signature_examples.get(sig, {}).get('nodes'),
            'edges': graph_signature_examples.get(sig, {}).get('edges')
        }
        for sig, count in filtered_sorted_graphs[:20]
    ]

    connectivity_stats = {
        'total_with_connectivity': graphs_processed,
        'unique_graphs': len(graph_signature_counts),
        'unique_graphs_filtered': len(filtered_sorted_graphs),
        'filter_threshold': filter_threshold,
        'filter_threshold_label': threshold_label,
        'top_graphs': top_graphs,
        'counts': list(graph_signature_counts.values()),
        'filtered_counts': [count for count in graph_signature_counts.values() if count >= filter_threshold]
    }

    stats = {
        'dataset_info': {
            'source': data.get('dataset', 'unknown'),
            'floorplan_count': data['count']
        },
        'total_sizes': total_sizes_block,
        'room_counts': room_counts_block,
        'room_type_stats': room_type_stats,
        'room_type_total_counts': dict(room_type_counts),
        'connectivity_graphs': connectivity_stats
    }
    
    # Save statistics to stats.json
    stats_path = os.path.join(os.path.dirname(json_path), 'stats.json')
    
    # Create a saveable version (without the large 'values' arrays)
    stats_to_save = {
        'dataset_info': stats['dataset_info'],
        'total_sizes': {k: v for k, v in stats['total_sizes'].items() if k != 'values'},
        'room_counts': {k: v for k, v in stats['room_counts'].items() if k != 'values'},
        'room_type_stats': {
            room_type: {k: v for k, v in room_stats.items() if k != 'sizes'}
            for room_type, room_stats in stats['room_type_stats'].items()
        },
        'room_type_total_counts': stats['room_type_total_counts'],
        'connectivity_graphs': {
            'total_with_connectivity': stats['connectivity_graphs']['total_with_connectivity'],
            'unique_graphs': stats['connectivity_graphs']['unique_graphs'],
            'top_graphs': stats['connectivity_graphs']['top_graphs']
        }
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")

    graph_stats_path = os.path.join(os.path.dirname(json_path), 'graph_stats.json')
    graph_full_dump = [
        {
            'signature': sig,
            'count': count,
            'example_file': graph_signature_examples.get(sig, {}).get('example_file'),
            'nodes': graph_signature_examples.get(sig, {}).get('nodes'),
            'edges': graph_signature_examples.get(sig, {}).get('edges')
        }
        for sig, count in sorted_graphs
    ]

    with open(graph_stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_with_connectivity': stats['connectivity_graphs']['total_with_connectivity'],
            'unique_graphs': stats['connectivity_graphs']['unique_graphs'],
            'graphs': graph_full_dump
        }, f, indent=2)

    print(f"Graph statistics saved to: {graph_stats_path}")
    
    return stats

def plot_distributions(stats: Dict) -> None:
    """
    Create and save distribution plots for room type sizes and total sizes.
    
    Args:
        stats: Dictionary containing dataset statistics from extract_dataset_stats
    """
    # Create plots directory relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Plot total sizes distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['total_sizes']['values'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Total Floorplan Size (m²)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Total Floorplan Sizes\n'
              f'Min: {stats["total_sizes"]["min"]:.2f} m², '
              f'Max: {stats["total_sizes"]["max"]:.2f} m², '
              f'Mean: {stats["total_sizes"]["mean"]:.2f} m², '
              f'Median: {stats["total_sizes"]["median"]:.2f} m²')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'total_sizes_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: total_sizes_distribution.png")
    
    # 2. Plot room count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['room_counts']['values'], bins=range(
        stats['room_counts']['min'], 
        stats['room_counts']['max'] + 2
    ), edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Room Counts per Floorplan\n'
              f'Min: {stats["room_counts"]["min"]}, '
              f'Max: {stats["room_counts"]["max"]}, '
              f'Mean: {stats["room_counts"]["mean"]:.2f}, '
              f'Median: {stats["room_counts"]["median"]:.2f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'room_counts_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: room_counts_distribution.png")
    
    # 3. Plot distributions for each room type
    room_types = sorted(stats['room_type_stats'].keys())
    
    # Generate different colors for each room type
    colors = plt.cm.tab20(np.linspace(0, 1, len(room_types)))
    
    # Create individual plots for each room type
    for i, room_type in enumerate(room_types):
        room_stats = stats['room_type_stats'][room_type]
        sizes = room_stats['sizes']
        
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=30, edgecolor='black', alpha=0.7, color=colors[i])
        plt.xlabel('Room Size (m²)')
        plt.ylabel('Frequency')
        plt.title(f'{room_type.capitalize()} Size Distribution\n'
                  f'Min: {room_stats["min"]:.2f} m², '
                  f'Max: {room_stats["max"]:.2f} m², '
                  f'Mean: {room_stats["mean"]:.2f} m², '
                  f'Median: {room_stats["median"]:.2f} m², '
                  f'Count: {room_stats["count"]}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{room_type}_size_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: {room_type}_size_distribution.png")
    
    # 4. Create a combined subplot with all room types (box plot at top + histograms below)
    num_rooms = len(room_types)
    cols = 2
    histogram_rows = (num_rooms + cols - 1) // cols  # Ceiling division
    total_rows = 1 + histogram_rows  # 1 for box plot + histogram rows
    
    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=(16, 9 * total_rows / 2.5))  # 16:9 aspect ratio
    gs = fig.add_gridspec(total_rows, 2, height_ratios=[1] + [1] * histogram_rows, hspace=0.4, wspace=0.3)
    
    # 4a. Create box plot at the top (spanning both columns)
    ax_box = fig.add_subplot(gs[0, :])
    room_data = [stats['room_type_stats'][rt]['sizes'] for rt in room_types]
    bp = ax_box.boxplot(room_data, labels=[rt.capitalize() for rt in room_types], patch_artist=True)
    
    # Color each box with the same colors as histograms
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax_box.set_xlabel('Room Type')
    ax_box.set_ylabel('Room Size (m²)')
    ax_box.set_title('Room Size Distributions by Type (Box Plot)')
    ax_box.tick_params(axis='x', rotation=45)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # 4b. Create histograms below (2 per row)
    for i, room_type in enumerate(room_types):
        row = 1 + (i // cols)
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        room_stats = stats['room_type_stats'][room_type]
        sizes = room_stats['sizes']
        
        ax.hist(sizes, bins=30, edgecolor='black', alpha=0.7, color=colors[i])
        ax.set_xlabel('Room Size (m²)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{room_type.capitalize()}\n'
                     f'Min: {room_stats["min"]:.2f} m², Max: {room_stats["max"]:.2f} m², '
                     f'Mean: {room_stats["mean"]:.2f} m², Median: {room_stats["median"]:.2f} m²\n'
                     f'Count: {room_stats["count"]}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_rooms, (total_rows - 1) * cols):
        row = 1 + (i // cols)
        col = i % cols
        if row < total_rows:
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    fig.suptitle('All Room Type Size Distributions', fontsize=16, y=0.98)
    plt.savefig(os.path.join(plots_dir, 'all_room_types_combined.png'), dpi=150)
    plt.close()
    print(f"Saved: all_room_types_combined.png")

    # 6. Connectivity graph occurrence plot (sorted, with horizontal threshold)
    graph_counts = stats.get('connectivity_graphs', {}).get('counts') or []
    if graph_counts:
        unique_graphs = stats['connectivity_graphs']['unique_graphs']
        unique_graphs_filtered = stats['connectivity_graphs']['unique_graphs_filtered']
        total_graphs = stats['connectivity_graphs']['total_with_connectivity']
        singleton_graphs = sum(1 for count in graph_counts if count == 1)
        threshold_label = stats['connectivity_graphs']['filter_threshold_label']
        filter_threshold = stats['connectivity_graphs']['filter_threshold']
        
        # Sort counts in descending order
        sorted_counts = sorted(graph_counts, reverse=True)
        
        plt.figure(figsize=(14, 6))
        x = range(len(sorted_counts))
        
        # Color bars based on whether they're above/below threshold
        colors = ['seagreen' if c >= filter_threshold else 'lightgray' for c in sorted_counts]
        plt.bar(x, sorted_counts, color=colors, edgecolor='none', width=1.0)
        
        # Horizontal threshold line
        plt.axhline(filter_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold_label}')
        
        plt.xlabel('Unique Connectivity Graphs (sorted by occurrence)')
        plt.ylabel('Occurrence Count')
        plt.title('Connectivity Graph Occurrences (Sorted)')
        plt.legend(fontsize=11, loc='upper right')
        
        # Add text annotation
        plt.text(0.02, 0.98, f'Total Unique: {unique_graphs:,}\nAbove Threshold: {unique_graphs_filtered:,}\nBelow Threshold: {unique_graphs - unique_graphs_filtered:,}\nSingletons: {singleton_graphs:,}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'connectivity_graph_frequency_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: connectivity_graph_frequency_distribution.png")
        
        # 6b. Connectivity graph occurrence plot - only graphs above threshold
        threshold_value = stats['connectivity_graphs']['filter_threshold']
        filtered_counts = [count for count in graph_counts if count >= threshold_value]
        if filtered_counts:
            threshold_label = stats['connectivity_graphs']['filter_threshold_label']
            
            # Sort in descending order
            sorted_filtered = sorted(filtered_counts, reverse=True)
            
            plt.figure(figsize=(14, 6))
            x = range(len(sorted_filtered))
            plt.bar(x, sorted_filtered, color='teal', edgecolor='none', width=1.0)
            
            # Horizontal threshold line
            plt.axhline(threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold_label}')
            
            plt.xlabel('Unique Connectivity Graphs (sorted by occurrence, filtered)')
            plt.ylabel('Occurrence Count')
            plt.title(f'Connectivity Graph Occurrences (Only ≥{threshold_label})')
            plt.legend(fontsize=11, loc='upper right')
            
            # Add text annotation
            plt.text(0.02, 0.98, f'Graphs Above Threshold: {len(filtered_counts):,}\nExcluded (below): {len(graph_counts) - len(filtered_counts):,}\nTotal Unique: {unique_graphs:,}',
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'connectivity_graph_frequency_distribution_filtered.png'), dpi=150)
            plt.close()
            print(f"Saved: connectivity_graph_frequency_distribution_filtered.png")

    # 7. Top connectivity graphs bar chart
    top_graphs = stats.get('connectivity_graphs', {}).get('top_graphs') or []
    if top_graphs:
        labels = []
        counts = []
        for idx, g in enumerate(top_graphs, 1):
            label = g.get('example_file') or f"G{idx}"
            if len(label) > 18:
                label = label[:15] + '...'
            labels.append(label)
            counts.append(g.get('count', 0))

        unique_graphs = stats['connectivity_graphs']['unique_graphs']
        unique_graphs_filtered = stats['connectivity_graphs']['unique_graphs_filtered']
        total_graphs = stats['connectivity_graphs']['total_with_connectivity']
        graph_counts = stats.get('connectivity_graphs', {}).get('counts') or []
        threshold_label = stats['connectivity_graphs']['filter_threshold_label']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(counts)), counts, color='slateblue', alpha=0.85, edgecolor='black')
        plt.xlabel('Top Unique Connectivity Graphs (example file)')
        plt.ylabel('Occurrences')
        plt.title('Top Connectivity Graphs by Frequency')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.25, axis='y')

        # Annotate bars with counts for quick read
        for bar, c in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + max(counts) * 0.01, f"{int(c)}",
                     ha='center', va='bottom', fontsize=8)
        
        # Add text annotation with unique graph count
        plt.text(0.98, 0.98, f'Total Unique: {unique_graphs:,}\nFiltered (≥{threshold_label}): {unique_graphs_filtered:,}\nTotal Graphs: {total_graphs:,}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'connectivity_graph_top_counts.png'), dpi=150)
        plt.close()
        print(f"Saved: connectivity_graph_top_counts.png")

    # 8. Visualize top 10 connectivity graphs as network diagrams
    # Need to load the actual adjacency data from sizes.json
    top_graphs = stats.get('connectivity_graphs', {}).get('top_graphs') or []
    if top_graphs:
        # Load sizes.json to get actual connectivity data
        sizes_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizes.json')
        with open(sizes_json_path, 'r', encoding='utf-8') as f:
            sizes_data = json.load(f)
        
        # Create a mapping from filename to connectivity
        file_to_connectivity = {}
        for item in sizes_data['items']:
            file_to_connectivity[item['file']] = item.get('connectivity', {})
        
        # Create a combined figure with 2 rows x 5 columns for top 10 graphs
        num_to_plot = min(10, len(top_graphs))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx in range(num_to_plot):
            graph_info = top_graphs[idx]
            ax = axes[idx]
            
            # Get the example file and its connectivity data
            example_file = graph_info.get('example_file', '')
            connectivity = file_to_connectivity.get(example_file, {})
            adjacency = connectivity.get('adjacency', {})
            
            if not adjacency:
                ax.axis('off')
                continue
            
            # Build NetworkX graph from adjacency data (same as in Floorplan class)
            G = nx.Graph()
            
            # Add edges from adjacency list
            for node, neighbors in adjacency.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            
            # Use spring layout for positioning
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
            
            # Color nodes by room type
            node_colors = []
            color_map = {
                'living room': '#FF6B6B',
                'master room': '#4ECDC4',
                'bedroom': '#45B7D1',
                'kitchen': '#FFA07A',
                'bathroom': '#98D8C8',
                'balcony': '#F7DC6F',
                'storage': '#BB8FCE',
                'entrance': '#85C1E2',
                'dining room': '#F8B88B',
                'child room': '#A8E6CF',
                'second room': '#FFD3B6',
                'guest room': '#FFAAA5',
            }
            
            for node in G.nodes():
                # Extract room type (before underscore)
                room_type = node.split('_')[0] if '_' in node else node
                node_colors.append(color_map.get(room_type, '#D3D3D3'))
            
            # Draw the graph
            nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=300,
                   with_labels=True, font_size=5, font_weight='bold', 
                   edge_color='gray', width=1.5, alpha=0.9)
            
            # Title with rank and count
            title_file = example_file[:12] + '...' if len(example_file) > 12 else example_file
            ax.set_title(f"#{idx+1}: {graph_info.get('count', 0)} occurrences\n{title_file}",
                        fontsize=8, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_to_plot, 10):
            axes[idx].axis('off')
        
        unique_graphs = stats['connectivity_graphs']['unique_graphs']
        unique_graphs_filtered = stats['connectivity_graphs']['unique_graphs_filtered']
        total_graphs = stats['connectivity_graphs']['total_with_connectivity']
        graph_counts = stats.get('connectivity_graphs', {}).get('counts') or []
        threshold_label = stats['connectivity_graphs']['filter_threshold_label']
        
        fig.suptitle(f'Top 10 Most Common Connectivity Graph Patterns (≥{threshold_label})\n({unique_graphs_filtered:,} filtered out of {unique_graphs:,} unique graphs)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'connectivity_graph_top10_visualized.png'), dpi=150)
        plt.close()
        print(f"Saved: connectivity_graph_top10_visualized.png")
    
    # 5. Create a summary plot with all room types (box plot)
    # Note: This is now integrated into the combined plot above
    
    print(f"\nAll plots saved to: {plots_dir}")

def search_by_room_size(json_path: Path, min_size: float, max_size: float, exclude_room_types: list = None) -> list:
    """
    Search for floorplans containing rooms with sizes in the specified range.
    
    Args:
        json_path: Path to the sizes.json file
        min_size: Minimum room size (m²)
        max_size: Maximum room size (m²)
        room_type: Optional room type filter (e.g., 'bedroom', 'bathroom')
        
    Returns:
        List of dictionaries with matching floorplans and room details
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # Search through all floorplans
    for item in data['items']:
        file_path = item['file']
        room_sizes = item['room_sizes_m2']
        
        matching_rooms = []
        
        # Check each room in the floorplan
        for room_name, size in room_sizes.items():
            # Extract room type
            current_room_type = room_name.split('_')[0] if '_' in room_name else room_name
            
            # Check if size is in range
            if min_size <= size <= max_size:
                # If exclude_room_types filter is specified, check it does not match
                if exclude_room_types is None or current_room_type not in exclude_room_types:
                    matching_rooms.append({
                        'room_name': room_name,
                        'room_type': current_room_type,
                        'size': size
                    })
        
        # If this floorplan has matching rooms, add it to results
        if matching_rooms:
            results.append({
                'file': file_path,
                'matching_rooms': matching_rooms,
                'total_matching': len(matching_rooms)
            })
    
    return results


if __name__ == "__main__":
    # Path to the sizes.json file
    sizes_json_path = os.path.join(os.path.dirname(__file__), 'sizes.json')
    
    if not os.path.exists(sizes_json_path):
        print(f"Error: sizes.json not found at {sizes_json_path}")
        print("Please run the size extraction script first.")
        exit(1)
    
    print("Extracting dataset statistics...")
    stats = extract_dataset_stats(sizes_json_path)
    
    print(f"\nDataset Statistics Summary:")
    print(f"  Total floorplans: {stats['dataset_info']['floorplan_count']}")
    print(f"  Total size range: {stats['total_sizes']['min']:.2f} - {stats['total_sizes']['max']:.2f} m²")
    print(f"  Average total size: {stats['total_sizes']['mean']:.2f} m²")
    print(f"  Room count range: {stats['room_counts']['min']} - {stats['room_counts']['max']}")
    print(f"  Average room count: {stats['room_counts']['mean']:.2f}")
    print(f"  Room types found: {len(stats['room_type_stats'])}")
    print(f"  Connectivity graphs: {stats['connectivity_graphs']['total_with_connectivity']} total / {stats['connectivity_graphs']['unique_graphs']} unique")
    print(f"  Graphs filtered by threshold ({stats['connectivity_graphs']['filter_threshold_label']}): {stats['connectivity_graphs']['unique_graphs_filtered']} remaining")
    
    print("\n" + "="*60)
    print("Creating distribution plots...")
    plot_distributions(stats)
    
    print("\n" + "="*60)
    print("Searching for edge cases (rooms with very small sizes)...")
    
    # Search for rooms with sizes between 0 and 1 m² (suspiciously small)
    small_rooms = search_by_room_size(sizes_json_path, min_size=0, max_size=1.0,  exclude_room_types=["bathroom", "storage", "balcony"])
    
    if small_rooms:
        print(f"\nFound {len(small_rooms)} floorplan(s) with rooms ≤ 1 m²:")
        for i, result in enumerate(small_rooms[:10], 1):  # Show first 10
            print(f"  {i}. {result['file']}")
            for room in result['matching_rooms']:
                print(f"     - {room['room_name']}: {room['size']:.2f} m²")
            if i == 10 and len(small_rooms) > 10:
                print(f"  ... and {len(small_rooms) - 10} more")

    else:
        print("  No rooms found with size ≤ 1 m²")
    
    # Search for rooms with size exactly 0
    zero_rooms = search_by_room_size(sizes_json_path, min_size=0, max_size=0)
    
    if zero_rooms:
        print(f"\nFound {len(zero_rooms)} floorplan(s) with rooms of size 0:")
        for i, result in enumerate(zero_rooms[:10], 1):
            print(f"  {i}. {result['file']}")
            for room in result['matching_rooms']:
                print(f"     - {room['room_name']}: {room['size']:.2f} m²")
            if i == 10 and len(zero_rooms) > 10:
                print(f"  ... and {len(zero_rooms) - 10} more")
    else:
        print("  No rooms found with size = 0 m²")
    
    # Save both results in a single JSON file
    edge_cases_path = os.path.join(os.path.dirname(sizes_json_path), 'edge_cases.json')
    edge_cases_data = {
        'small_rooms': {
            'description': 'Floorplans with rooms ≤ 1 m²',
            'count': len(small_rooms),
            'results': small_rooms
        },
        'zero_rooms': {
            'description': 'Floorplans with rooms of size 0 m²',
            'count': len(zero_rooms),
            'results': zero_rooms
        }
    }
    
    with open(edge_cases_path, 'w', encoding='utf-8') as f:
        json.dump(edge_cases_data, f, indent=2)
    print(f"\nEdge cases saved to: {edge_cases_path}")
    
    print("\n" + "="*60)
    print("Done!")