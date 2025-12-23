import numpy as np
import json 
import os

import matplotlib.pyplot as plt
from PIL import Image

from typing import Dict
from pathlib import Path
from collections import defaultdict

SIZES_PATH = r".\dataset_stats\sizes.json"
# Rooms with size <= this threshold (m²) are excluded from stats
MIN_VALID_ROOM_SIZE = 1.0

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
    
    # Process each floorplan (excluding zero/small rooms from all stats)
    for item in data['items']:
        room_sizes = item['room_sizes_m2']

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

    stats = {
        'dataset_info': {
            'source': data.get('dataset', 'unknown'),
            'floorplan_count': data['count']
        },
        'total_sizes': total_sizes_block,
        'room_counts': room_counts_block,
        'room_type_stats': room_type_stats,
        'room_type_total_counts': dict(room_type_counts)
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
        'room_type_total_counts': stats['room_type_total_counts']
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    
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