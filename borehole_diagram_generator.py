#!/usr/bin/env python3
"""
Borehole Diagram Generator
Converts borehole JSON data into visual diagram for ControlNet input
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from PIL import Image
import io

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_borehole_diagram(boreholes_data, section_boreholes=None, output_size=(1024, 768), 
                            style='schematic'):
    """
    Create a borehole diagram showing positions, depths, and layers
    
    Args:
        boreholes_data: Full borehole data from JSON
        section_boreholes: Boreholes in the cross section (with x positions)
        output_size: (width, height) of output image
        style: 'schematic' or 'detailed'
    
    Returns:
        PIL Image of borehole diagram
    """
    fig, ax = plt.subplots(figsize=(output_size[0]/100, output_size[1]/100), dpi=100)
    ax.set_xlim(0, output_size[0])
    ax.set_ylim(0, output_size[1])
    ax.axis('off')
    
    # Create mapping from borehole name to full data
    bh_name_to_data = {bh.get('th_title', ''): bh for bh in boreholes_data}
    
    # Get boreholes for this section
    if section_boreholes is None:
        # Use all boreholes, estimate x positions
        section_boreholes = []
        for i, bh in enumerate(boreholes_data):
            section_boreholes.append({
                'name': bh.get('th_title', f'BH-{i}'),
                'x': i * (output_size[0] / max(len(boreholes_data), 1)),
                'elevation': 600,  # Default elevation
                'depth': bh.get('th_depth', 200)
            })
    
    # Find max depth and elevation range
    max_depth = max([bh.get('depth', 200) for bh in section_boreholes] + 
                   [bh.get('th_depth', 200) for bh in boreholes_data])
    min_elevation = min([bh.get('elevation', 500) for bh in section_boreholes])
    max_elevation = max([bh.get('elevation', 600) for bh in section_boreholes])
    elevation_range = max_elevation - min_elevation + max_depth
    
    # Scale factors
    max_x = max([bh.get('x', 0) for bh in section_boreholes] + [1000])
    x_scale = output_size[0] / max(max_x, 1)
    y_scale = (output_size[1] * 0.8) / max(elevation_range, 1)
    
    # Draw each borehole
    for bh in section_boreholes:
        bh_name = bh.get('name', 'Unknown')
        x_pos = bh.get('x', 0) * x_scale
        elevation = bh.get('elevation', 500)
        depth = bh.get('depth', 200)
        
        # Get full borehole data
        full_bh_data = bh_name_to_data.get(bh_name)
        if not full_bh_data:
            continue
        
        # Calculate y positions (inverted - top is higher elevation)
        y_top = output_size[1] * 0.9 - (elevation - min_elevation) * y_scale
        y_bottom = y_top + depth * y_scale
        
        # Draw borehole line
        ax.plot([x_pos, x_pos], [y_top, y_bottom], 
               'k-', linewidth=4, alpha=0.9, zorder=10)
        
        # Draw layers
        layers = full_bh_data.get('th_layers', [])
        current_y = y_top
        
        for layer in layers:
            layer_from = layer.get('layer_from', 0)
            layer_to = layer.get('layer_to', layer_from + 10)
            layer_thickness = layer_to - layer_from
            
            # Get color
            color_hex = layer.get('layer_backcolor', '#FFFFFF')
            if color_hex == '#FFFFFF':
                color_hex = layer.get('layer_forecolor', '#808080')
            color_rgb = hex_to_rgb(color_hex)
            
            # Calculate layer position
            layer_top_y = y_top + layer_from * y_scale
            layer_bottom_y = y_top + layer_to * y_scale
            
            # Draw layer rectangle
            rect_width = 30
            rect = patches.Rectangle(
                (x_pos - rect_width/2, layer_top_y),
                rect_width,
                layer_bottom_y - layer_top_y,
                facecolor=color_rgb,
                edgecolor='black',
                linewidth=1,
                alpha=0.8,
                zorder=9
            )
            ax.add_patch(rect)
            
            # Add layer symbol label
            symbol = layer.get('layer_symbol', '') or 'N/A'
            ax.text(x_pos, (layer_top_y + layer_bottom_y) / 2,
                   symbol, ha='center', va='center',
                   fontsize=8, fontweight='bold', zorder=11)
        
        # Add borehole label
        ax.text(x_pos, y_top - 20, bh_name,
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                        alpha=0.8, edgecolor='black'),
               zorder=11)
        
        # Add depth label
        ax.text(x_pos, y_bottom + 10, f'{depth:.1f}ft',
               ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
               zorder=11)
    
    # Add title
    ax.text(output_size[0]/2, output_size[1] - 30,
           'Borehole Log Diagram',
           ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Add scale indicators
    ax.text(20, output_size[1] - 60, 'Elevation (ft)', fontsize=10)
    ax.text(output_size[0] - 100, 20, 'Distance (ft)', fontsize=10)
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize(output_size, Image.Resampling.LANCZOS)
    plt.close()
    
    return img

def load_borehole_data(json_file):
    """Load borehole data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_diagram_from_section(json_file, section_index=0):
    """
    Create borehole diagram for a specific cross section
    
    Args:
        json_file: Path to JSON file
        section_index: Index of cross section in polygonsBySection
    
    Returns:
        PIL Image of borehole diagram
    """
    data = load_borehole_data(json_file)
    
    # Get cross section
    section = data['polygonsBySection'][section_index]
    section_boreholes = section.get('boreholes', [])
    boreholes_data = data.get('boreholesData', [])
    
    # Create diagram
    diagram = create_borehole_diagram(boreholes_data, section_boreholes)
    
    return diagram

if __name__ == "__main__":
    # Test with easier dataset
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    
    print("Creating borehole diagram for easier dataset...")
    diagram = create_diagram_from_section(easier_file, section_index=0)
    
    # Save diagram
    output_path = base_path / "borehole_diagram_easier.png"
    diagram.save(output_path)
    print(f"Borehole diagram saved to: {output_path}")
    
    # Display
    diagram.show()

