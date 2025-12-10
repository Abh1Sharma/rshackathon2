#!/usr/bin/env python3
"""
Quick visualization script for Project 2: Borehole/Soil Layer Generation
Shows existing cross sections and borehole data
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

def visualize_cross_section(section_data, title="Cross Section", ax=None):
    """Visualize a single cross section with polygons and boreholes"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw polygons
    for polygon in section_data.get('polygons', []):
        points_2d = polygon.get('points2D', [])
        if len(points_2d) < 3:
            continue
        
        # Extract vertices
        vertices = np.array([[v['vertex'][0], v['vertex'][1]] for v in points_2d])
        
        # Get color
        color_hex = polygon.get('color', '000000')
        color_rgb = hex_to_rgb(color_hex)
        
        # Create polygon patch
        poly_patch = patches.Polygon(vertices, closed=True, 
                                     facecolor=color_rgb, 
                                     edgecolor='black', 
                                     linewidth=0.5,
                                     alpha=0.7,
                                     label=f"{polygon.get('symbol', 'N/A')} - {polygon.get('symbolDescription', '')}")
        
        ax.add_patch(poly_patch)
    
    # Draw boreholes
    for bh in section_data.get('boreholes', []):
        x_pos = bh.get('x', 0)
        elevation = bh.get('elevation', 0)
        depth = bh.get('depth', 0)
        
        # Draw borehole as vertical line
        ax.plot([x_pos, x_pos], [elevation - depth, elevation], 
               'k-', linewidth=3, alpha=0.8)
        
        # Add borehole label
        ax.text(x_pos, elevation + 10, bh.get('name', 'BH'), 
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Draw ground water lines
    for gw_line in section_data.get('groundWaterLines', []):
        points = gw_line.get('points2D', [])
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            ax.plot(x_coords, y_coords, 'b--', linewidth=2, alpha=0.6, label='Water Table')
    
    # Draw ground surface profile
    surface_profile = section_data.get('groundSurfaceProfile', [])
    if surface_profile:
        x_coords = [p.get('xDistance', 0) for p in surface_profile]
        y_coords = [p.get('elevation', 0) for p in surface_profile]
        ax.plot(x_coords, y_coords, 'brown', linewidth=2, alpha=0.8, label='Ground Surface')
    
    ax.set_xlabel('Distance (ft)', fontsize=12)
    ax.set_ylabel('Elevation (ft)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax

def visualize_borehole_layers(borehole_data, ax=None):
    """Visualize a single borehole with its layers"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))
    
    layers = borehole_data.get('th_layers', [])
    depth = borehole_data.get('th_depth', 0)
    
    current_depth = 0
    for layer in layers:
        layer_from = layer.get('layer_from', current_depth)
        layer_to = layer.get('layer_to', layer_from + 10)
        layer_thickness = layer_to - layer_from
        
        # Get color
        color_hex = layer.get('layer_backcolor', '#FFFFFF')
        if color_hex == '#FFFFFF':
            color_hex = layer.get('layer_forecolor', '#000000')
        color_rgb = hex_to_rgb(color_hex)
        
        # Draw layer rectangle
        rect = patches.Rectangle((0, -layer_to), 1, layer_thickness,
                                facecolor=color_rgb, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add layer label
        symbol = layer.get('layer_symbol', '') or 'N/A'
        title = layer.get('layer_title', '') or ''
        if title:
            title = title[:30]  # Truncate long titles
        ax.text(0.5, -(layer_from + layer_to)/2, f"{symbol}\n{title}",
               ha='center', va='center', fontsize=7, rotation=0)
        
        current_depth = layer_to
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-depth - 10, 10)
    ax.set_xlabel('Borehole', fontsize=10)
    ax.set_ylabel('Depth (ft)', fontsize=10)
    ax.set_title(f"Borehole {borehole_data.get('th_title', 'Unknown')}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])
    
    return ax

def main():
    """Main visualization function"""
    # File paths
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    complex_file = base_path / "Complex RSLog section (1) 1 (1).json"
    
    print("Loading JSON files...")
    easier_data = load_json(easier_file)
    complex_data = load_json(complex_file)
    
    print(f"\nEasier file: {len(easier_data['polygonsBySection'])} cross section(s)")
    print(f"Complex file: {len(complex_data['polygonsBySection'])} cross section(s)")
    print(f"Easier file: {len(easier_data.get('boreholesData', []))} boreholes in data")
    print(f"Complex file: {len(complex_data.get('boreholesData', []))} boreholes in data")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Easier file cross section
    ax1 = plt.subplot(2, 3, 1)
    easier_section = easier_data['polygonsBySection'][0]
    visualize_cross_section(easier_section, 
                           title=f"Easier: {easier_section['sectionName']}\n({len(easier_section['polygons'])} polygons, {len(easier_section['boreholes'])} boreholes)",
                           ax=ax1)
    
    # Plot 2: Complex file - Cross Section Slide2 A
    ax2 = plt.subplot(2, 3, 2)
    complex_section1 = complex_data['polygonsBySection'][0]
    visualize_cross_section(complex_section1,
                           title=f"Complex: {complex_section1['sectionName']}\n({len(complex_section1['polygons'])} polygons, {len(complex_section1['boreholes'])} boreholes)",
                           ax=ax2)
    
    # Plot 3: Complex file - Cross Section 1 (empty polygons)
    ax3 = plt.subplot(2, 3, 3)
    if len(complex_data['polygonsBySection']) > 1:
        complex_section2 = complex_data['polygonsBySection'][1]
        visualize_cross_section(complex_section2,
                               title=f"Complex: {complex_section2['sectionName']}\n({len(complex_section2['polygons'])} polygons, {len(complex_section2['boreholes'])} boreholes) - TARGET",
                               ax=ax3)
    
    # Plot 4-6: Sample borehole visualizations
    if easier_data.get('boreholesData'):
        ax4 = plt.subplot(2, 3, 4)
        visualize_borehole_layers(easier_data['boreholesData'][0], ax=ax4)
    
    if len(easier_data.get('boreholesData', [])) > 1:
        ax5 = plt.subplot(2, 3, 5)
        visualize_borehole_layers(easier_data['boreholesData'][1], ax=ax5)
    
    if complex_data.get('boreholesData'):
        ax6 = plt.subplot(2, 3, 6)
        visualize_borehole_layers(complex_data['boreholesData'][0], ax=ax6)
    
    plt.tight_layout()
    plt.savefig('cross_sections_overview.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'cross_sections_overview.png'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print("\nEasier File Cross Sections:")
    for i, section in enumerate(easier_data['polygonsBySection']):
        print(f"  {i+1}. {section['sectionName']}")
        print(f"     - Polygons: {len(section.get('polygons', []))}")
        print(f"     - Boreholes: {len(section.get('boreholes', []))}")
        print(f"     - Image Preview: {section.get('imagePreview', 'N/A')[:80]}...")
    
    print("\nComplex File Cross Sections:")
    for i, section in enumerate(complex_data['polygonsBySection']):
        print(f"  {i+1}. {section['sectionName']}")
        print(f"     - Polygons: {len(section.get('polygons', []))}")
        print(f"     - Boreholes: {len(section.get('boreholes', []))}")
        print(f"     - Image Preview: {section.get('imagePreview', 'N/A')[:80]}...")
    
    print(f"\nTotal Boreholes (Easier): {len(easier_data.get('boreholesData', []))}")
    print(f"Total Boreholes (Complex): {len(complex_data.get('boreholesData', []))}")

if __name__ == "__main__":
    main()

