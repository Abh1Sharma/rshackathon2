#!/usr/bin/env python3
"""
Visualize all boreholes from the Easier RSLog section file
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hex2color
import numpy as np
from pathlib import Path

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def visualize_all_boreholes_in_section(section_data, boreholes_data, title="All Boreholes"):
    """Visualize cross section with all boreholes and their detailed layers"""
    fig = plt.figure(figsize=(20, 12))
    
    # Main cross section view
    ax_main = plt.subplot(2, 1, 1)
    
    # Draw polygons
    for polygon in section_data.get('polygons', []):
        points_2d = polygon.get('points2D', [])
        if len(points_2d) < 3:
            continue
        
        vertices = np.array([[v['vertex'][0], v['vertex'][1]] for v in points_2d])
        color_hex = polygon.get('color', '000000')
        color_rgb = hex_to_rgb(color_hex)
        
        poly_patch = patches.Polygon(vertices, closed=True, 
                                     facecolor=color_rgb, 
                                     edgecolor='black', 
                                     linewidth=0.5,
                                     alpha=0.7)
        ax_main.add_patch(poly_patch)
    
    # Draw ground water lines
    for gw_line in section_data.get('groundWaterLines', []):
        points = gw_line.get('points2D', [])
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax_main.plot(x_coords, y_coords, 'b--', linewidth=2, alpha=0.6, label='Water Table')
    
    # Draw ground surface profile
    surface_profile = section_data.get('groundSurfaceProfile', [])
    if surface_profile:
        x_coords = [p.get('xDistance', 0) for p in surface_profile]
        y_coords = [p.get('elevation', 0) for p in surface_profile]
        ax_main.plot(x_coords, y_coords, 'brown', linewidth=2, alpha=0.8, label='Ground Surface')
    
    # Create mapping from borehole ID to full data
    bh_id_to_data = {bh['th_id']: bh for bh in boreholes_data}
    
    # Draw all boreholes in the section
    for bh in section_data.get('boreholes', []):
        x_pos = bh.get('x', 0)
        elevation = bh.get('elevation', 0)
        depth = bh.get('depth', 0)
        bh_id = bh.get('boreholeId', '')
        bh_name = bh.get('name', 'Unknown')
        
        # Get full borehole data
        full_bh_data = None
        for bid, data in bh_id_to_data.items():
            if data.get('th_title', '') == bh_name or bid == bh_id:
                full_bh_data = data
                break
        
        # Draw borehole as vertical line
        ax_main.plot([x_pos, x_pos], [elevation - depth, elevation], 
                    'k-', linewidth=4, alpha=0.9, zorder=10)
        
        # Draw layers if we have full data
        if full_bh_data:
            layers = full_bh_data.get('th_layers', [])
            for layer in layers:
                layer_from = layer.get('layer_from', 0)
                layer_to = layer.get('layer_to', layer_from + 10)
                
                # Get color
                color_hex = layer.get('layer_backcolor', '#FFFFFF')
                if color_hex == '#FFFFFF':
                    color_hex = layer.get('layer_forecolor', '#808080')
                color_rgb = hex_to_rgb(color_hex)
                
                # Draw layer segment
                layer_top = elevation - layer_from
                layer_bottom = elevation - layer_to
                ax_main.plot([x_pos-5, x_pos+5], [layer_top, layer_top], 
                           color=color_rgb, linewidth=6, alpha=0.8, zorder=9)
                ax_main.plot([x_pos-5, x_pos+5], [layer_bottom, layer_bottom], 
                           color=color_rgb, linewidth=6, alpha=0.8, zorder=9)
        
        # Add borehole label
        ax_main.text(x_pos, elevation + 15, bh_name, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'),
                    zorder=11)
    
    ax_main.set_xlabel('Distance (ft)', fontsize=12)
    ax_main.set_ylabel('Elevation (ft)', fontsize=12)
    ax_main.set_title(f"{title}\n{section_data.get('sectionName', 'Unknown')} - {len(section_data.get('boreholes', []))} boreholes", 
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right')
    
    # Individual borehole detail views
    num_boreholes = len(section_data.get('boreholes', []))
    cols = min(5, num_boreholes)
    rows = (num_boreholes + cols - 1) // cols
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    
    # Create subplots for each borehole
    gs = ax2.get_gridspec()
    fig.delaxes(ax2)
    
    for idx, bh in enumerate(section_data.get('boreholes', [])):
        bh_id = bh.get('boreholeId', '')
        bh_name = bh.get('name', 'Unknown')
        
        # Find matching borehole in full data
        full_bh_data = None
        for bid, data in bh_id_to_data.items():
            if data.get('th_title', '') == bh_name or bid == bh_id:
                full_bh_data = data
                break
        
        if not full_bh_data:
            continue
        
        # Create subplot for this borehole
        ax_bh = plt.subplot(rows, cols, idx + 1)
        
        layers = full_bh_data.get('th_layers', [])
        depth = full_bh_data.get('th_depth', 0)
        
        current_depth = 0
        for layer in layers:
            layer_from = layer.get('layer_from', current_depth)
            layer_to = layer.get('layer_to', layer_from + 10)
            layer_thickness = layer_to - layer_from
            
            # Get color
            color_hex = layer.get('layer_backcolor', '#FFFFFF')
            if color_hex == '#FFFFFF':
                color_hex = layer.get('layer_forecolor', '#808080')
            color_rgb = hex_to_rgb(color_hex)
            
            # Draw layer rectangle
            rect = patches.Rectangle((0, -layer_to), 1, layer_thickness,
                                    facecolor=color_rgb, edgecolor='black', linewidth=1)
            ax_bh.add_patch(rect)
            
            # Add layer info
            symbol = layer.get('layer_symbol', '') or 'N/A'
            title = layer.get('layer_title', '') or ''
            if title:
                title = title[:20]  # Truncate
            
            # Add text annotation
            mid_depth = (layer_from + layer_to) / 2
            ax_bh.text(0.5, -mid_depth, f"{symbol}\n{title}",
                      ha='center', va='center', fontsize=6, rotation=0,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax_bh.set_xlim(-0.2, 1.2)
        ax_bh.set_ylim(-depth - 5, 5)
        ax_bh.set_xlabel('', fontsize=8)
        ax_bh.set_ylabel('Depth (ft)', fontsize=8)
        ax_bh.set_title(f"{bh_name}\n{depth}ft deep", fontsize=9, fontweight='bold')
        ax_bh.grid(True, alpha=0.3, axis='y')
        ax_bh.set_xticks([])
        ax_bh.invert_yaxis()
    
    plt.tight_layout()
    return fig

def main():
    """Main visualization function"""
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    
    print("Loading Easier RSLog section file...")
    easier_data = load_json(easier_file)
    
    print(f"Found {len(easier_data['polygonsBySection'])} cross section(s)")
    print(f"Found {len(easier_data.get('boreholesData', []))} boreholes in data")
    
    # Get the cross section
    section = easier_data['polygonsBySection'][0]
    boreholes_data = easier_data.get('boreholesData', [])
    
    print(f"\nCross Section: {section['sectionName']}")
    print(f"Boreholes in section: {len(section.get('boreholes', []))}")
    
    # Visualize
    fig = visualize_all_boreholes_in_section(section, boreholes_data, 
                                            title="Easier File - All Boreholes")
    
    # Save
    output_file = 'easier_all_boreholes.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as '{output_file}'")
    
    # Print borehole summary
    print("\n" + "="*60)
    print("BOREHOLE SUMMARY")
    print("="*60)
    for i, bh in enumerate(section.get('boreholes', []), 1):
        print(f"\n{i}. {bh.get('name', 'Unknown')}")
        print(f"   Position (x): {bh.get('x', 0):.2f} ft")
        print(f"   Elevation: {bh.get('elevation', 0):.2f} ft")
        print(f"   Depth: {bh.get('depth', 0):.2f} ft")
        print(f"   Water Depth: {bh.get('waterDepth', 'N/A')}")
        
        # Find matching full data
        for full_bh in boreholes_data:
            if full_bh.get('th_title', '') == bh.get('name', ''):
                print(f"   Layers: {len(full_bh.get('th_layers', []))}")
                print(f"   Coordinates: {full_bh.get('th_coordinates', 'N/A')}")
                break
    
    plt.show()

if __name__ == "__main__":
    main()

