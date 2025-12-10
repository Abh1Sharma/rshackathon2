#!/usr/bin/env python3
"""
3D Model Generator from Cross Section Polygons
Directly uses points3D coordinates from JSON data
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from scipy.interpolate import griddata, LinearNDInterpolator
import colorsys
import struct

# Try to import intelligent layer matching (optional, for ML-based layer matching)
try:
    from intelligent_layer_matching import IntelligentLayerMatcher
    INTELLIGENT_MATCHING_AVAILABLE = True
except ImportError:
    INTELLIGENT_MATCHING_AVAILABLE = False
    IntelligentLayerMatcher = None

# Global matcher instance (lazy-loaded)
_global_matcher = None

def get_intelligent_matcher(data=None, force_retrain=False):
    """
    Get or create a trained IntelligentLayerMatcher instance.
    Uses a global singleton to avoid retraining.
    """
    global _global_matcher
    
    if not INTELLIGENT_MATCHING_AVAILABLE:
        print("⚠ Intelligent matching not available (missing dependencies)")
        return None
    
    if _global_matcher is not None and not force_retrain:
        return _global_matcher
    
    if data is None:
        print("⚠ No data provided to train matcher")
        return None
    
    print("\n🧠 Initializing Intelligent Layer Matcher...")
    matcher = IntelligentLayerMatcher(use_openai=True)
    
    # Try to load existing model
    model_path = Path(__file__).parent / "layer_matching_model.pkl"
    if model_path.exists() and not force_retrain:
        try:
            matcher.load_model(model_path)
            print("  Loaded pre-trained model")
            _global_matcher = matcher
            return matcher
        except Exception as e:
            print(f"  Could not load model: {e}")
    
    # Train new model
    print("  Training new model on dataset...")
    matcher.train(data)
    
    # Save for future use
    try:
        matcher.save_model(model_path)
    except Exception as e:
        print(f"  Warning: Could not save model: {e}")
    
    _global_matcher = matcher
    return matcher

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


class STLExporter:
    """Export 3D meshes to STL/OBJ format with layer support"""
    
    def __init__(self):
        self.triangles = []  # List of (v1, v2, v3, normal, layer_name) tuples
        self.layers = {}  # layer_name -> list of triangle indices
        self.current_layer = "default"
    
    def set_layer(self, layer_name):
        """Set the current layer for subsequent triangles"""
        self.current_layer = layer_name
        if layer_name not in self.layers:
            self.layers[layer_name] = []
    
    def add_triangle(self, v1, v2, v3):
        """Add a triangle with vertices v1, v2, v3 (each is [x, y, z])"""
        v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
        # Calculate normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 0:
            normal = normal / norm_length
        else:
            normal = np.array([0, 0, 1])
        
        tri_idx = len(self.triangles)
        self.triangles.append((v1, v2, v3, normal, self.current_layer))
        
        if self.current_layer not in self.layers:
            self.layers[self.current_layer] = []
        self.layers[self.current_layer].append(tri_idx)
    
    def add_quad(self, v1, v2, v3, v4):
        """Add a quad as two triangles (v1-v2-v3 and v1-v3-v4)"""
        self.add_triangle(v1, v2, v3)
        self.add_triangle(v1, v3, v4)
    
    def add_box(self, corners_top, corners_bottom):
        """
        Add a 3D box defined by 4 top corners and 4 bottom corners
        corners_top/bottom: list of 4 [x, y, z] vertices
        """
        t0, t1, t2, t3 = corners_top
        b0, b1, b2, b3 = corners_bottom
        
        # Top face
        self.add_quad(t0, t1, t2, t3)
        
        # Bottom face (reversed winding)
        self.add_quad(b0, b3, b2, b1)
        
        # Front face
        self.add_quad(t0, t1, b1, b0)
        
        # Back face
        self.add_quad(t2, t3, b3, b2)
        
        # Left face
        self.add_quad(t3, t0, b0, b3)
        
        # Right face
        self.add_quad(t1, t2, b2, b1)
    
    def write_ascii(self, filepath, name="model"):
        """Write ASCII STL file"""
        with open(filepath, 'w') as f:
            f.write(f"solid {name}\n")
            for v1, v2, v3, normal, layer in self.triangles:
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")
        print(f"✓ Exported ASCII STL: {filepath} ({len(self.triangles)} triangles)")
    
    def write_binary(self, filepath):
        """Write binary STL file (more compact)"""
        with open(filepath, 'wb') as f:
            # 80-byte header
            header = b'Binary STL exported from geological model' + b'\0' * 40
            f.write(header[:80])
            
            # Number of triangles (4 bytes, unsigned int)
            f.write(struct.pack('<I', len(self.triangles)))
            
            # Each triangle: normal (3 floats) + 3 vertices (9 floats) + attribute (2 bytes)
            for v1, v2, v3, normal, layer in self.triangles:
                f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))
                f.write(struct.pack('<fff', v1[0], v1[1], v1[2]))
                f.write(struct.pack('<fff', v2[0], v2[1], v2[2]))
                f.write(struct.pack('<fff', v3[0], v3[1], v3[2]))
                f.write(struct.pack('<H', 0))  # Attribute byte count
        
        print(f"✓ Exported binary STL: {filepath} ({len(self.triangles)} triangles)")
    
    def write_obj(self, filepath, name="geological_model"):
        """Write OBJ file with separate objects for each layer (selectable in 3D software)"""
        with open(filepath, 'w') as f:
            f.write(f"# OBJ file exported from geological 3D model\n")
            f.write(f"# {len(self.triangles)} triangles in {len(self.layers)} layers\n")
            f.write(f"# Each layer is a separate selectable object\n\n")
            
            # Collect ALL unique vertices first
            all_vertices = []
            vertex_map = {}
            
            for v1, v2, v3, normal, layer in self.triangles:
                for v in [v1, v2, v3]:
                    key = (round(v[0], 4), round(v[1], 4), round(v[2], 4))
                    if key not in vertex_map:
                        vertex_map[key] = len(all_vertices) + 1  # OBJ is 1-indexed
                        all_vertices.append(v)
            
            # Write ALL vertices at the top
            f.write(f"# {len(all_vertices)} vertices\n")
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write ALL normals
            f.write(f"\n# {len(self.triangles)} normals\n")
            for i, (v1, v2, v3, normal, layer) in enumerate(self.triangles):
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            # Write faces grouped by layer (each layer as separate object)
            f.write(f"\n# ========== LAYER OBJECTS ==========\n")
            f.write(f"# {len(self.layers)} selectable layers\n\n")
            
            for layer_name in sorted(self.layers.keys()):
                tri_indices = self.layers[layer_name]
                
                # Create a clean object name (replace spaces and special chars)
                obj_name = layer_name.replace(' ', '_').replace('-', '_').replace(':', '_')
                
                f.write(f"\n# Layer: {layer_name} ({len(tri_indices)} triangles)\n")
                f.write(f"o {obj_name}\n")
                f.write(f"g {obj_name}\n")
                
                for tri_idx in tri_indices:
                    v1, v2, v3, normal, _ = self.triangles[tri_idx]
                    
                    key1 = (round(v1[0], 4), round(v1[1], 4), round(v1[2], 4))
                    key2 = (round(v2[0], 4), round(v2[1], 4), round(v2[2], 4))
                    key3 = (round(v3[0], 4), round(v3[1], 4), round(v3[2], 4))
                    
                    idx1 = vertex_map[key1]
                    idx2 = vertex_map[key2]
                    idx3 = vertex_map[key3]
                    normal_idx = tri_idx + 1  # 1-indexed
                    
                    f.write(f"f {idx1}//{normal_idx} {idx2}//{normal_idx} {idx3}//{normal_idx}\n")
        
        print(f"✓ Exported OBJ: {filepath} ({len(all_vertices)} vertices, {len(self.triangles)} faces, {len(self.layers)} layers)")

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_3d_polygons(data):
    """
    Extract all 3D polygons from JSON data
    
    Returns list of dicts with:
        - points: np.array of 3D vertices
        - color: RGB tuple
        - symbol: soil symbol
        - description: soil description
    """
    polygons = []
    
    for section in data.get('polygonsBySection', []):
        section_name = section.get('sectionName', 'Unknown')
        
        for poly in section.get('polygons', []):
            points_3d = poly.get('points3D', [])
            if not points_3d:
                continue
            
            # Extract vertices [x, y, z] where z is elevation
            vertices = np.array([p['vertex'] for p in points_3d])
            
            # Get color
            color_hex = poly.get('color', 'cccccc')
            if not color_hex.startswith('#'):
                color_hex = '#' + color_hex
            color_rgb = hex_to_rgb(color_hex)
            
            polygons.append({
                'vertices': vertices,
                'color': color_rgb,
                'symbol': poly.get('symbol', 'Unknown'),
                'description': poly.get('symbolDescription', ''),
                'section': section_name
            })
    
    return polygons

def create_3d_mesh_traces(polygons):
    """
    Create Plotly mesh traces from polygons
    """
    traces = []
    
    for i, poly in enumerate(polygons):
        vertices = poly['vertices']
        color = poly['color']
        symbol = poly['symbol']
        desc = poly['description']
        
        if len(vertices) < 3:
            continue
        
        # Extract x, y, z coordinates
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]  # Elevation
        
        # Create triangulation for polygon face
        if len(vertices) == 3:
            # Single triangle
            i_faces = [0]
            j_faces = [1]
            k_faces = [2]
        elif len(vertices) == 4:
            # Quad -> 2 triangles
            i_faces = [0, 0]
            j_faces = [1, 2]
            k_faces = [2, 3]
        else:
            # General polygon - fan triangulation from first vertex
            i_faces = [0] * (len(vertices) - 2)
            j_faces = list(range(1, len(vertices) - 1))
            k_faces = list(range(2, len(vertices)))
        
        # Convert color to plotly format
        color_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
        
        # Create mesh trace
        trace = go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces, j=j_faces, k=k_faces,
            color=color_str,
            opacity=0.8,
            name=f'{symbol}: {desc}',
            hovertemplate=f'<b>{symbol}</b><br>{desc}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Elevation: %{{z:.1f}}<extra></extra>',
            showlegend=True,
            flatshading=True
        )
        traces.append(trace)
        
        # Add wireframe edges for visibility
        edge_trace = go.Scatter3d(
            x=np.append(x, x[0]),
            y=np.append(y, y[0]),
            z=np.append(z, z[0]),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(edge_trace)
    
    return traces

def create_cylinder_mesh(x_center, y_center, z_top, z_bottom, radius, n_sides=12):
    """
    Create a cylinder mesh for borehole layer visualization
    
    Returns vertices and face indices for a cylinder
    """
    # Create circle points
    theta = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    
    # Top circle
    x_top = x_center + radius * np.cos(theta)
    y_top = y_center + radius * np.sin(theta)
    z_top_arr = np.full(n_sides, z_top)
    
    # Bottom circle
    x_bottom = x_center + radius * np.cos(theta)
    y_bottom = y_center + radius * np.sin(theta)
    z_bottom_arr = np.full(n_sides, z_bottom)
    
    # Combine vertices
    x = np.concatenate([x_top, x_bottom])
    y = np.concatenate([y_top, y_bottom])
    z = np.concatenate([z_top_arr, z_bottom_arr])
    
    # Create faces for cylinder sides
    i_faces = []
    j_faces = []
    k_faces = []
    
    for idx in range(n_sides):
        next_idx = (idx + 1) % n_sides
        # Two triangles per side
        # Triangle 1
        i_faces.append(idx)
        j_faces.append(next_idx)
        k_faces.append(idx + n_sides)
        # Triangle 2
        i_faces.append(next_idx)
        j_faces.append(next_idx + n_sides)
        k_faces.append(idx + n_sides)
    
    # Top cap
    for idx in range(1, n_sides - 1):
        i_faces.append(0)
        j_faces.append(idx)
        k_faces.append(idx + 1)
    
    # Bottom cap
    for idx in range(1, n_sides - 1):
        i_faces.append(n_sides)
        j_faces.append(n_sides + idx + 1)
        k_faces.append(n_sides + idx)
    
    return x, y, z, i_faces, j_faces, k_faces

def create_borehole_traces(data, radius=15):
    """
    Create 3D cylinder traces for boreholes with colored soil layers
    
    Args:
        data: JSON data
        radius: Radius of borehole cylinders in feet
    """
    traces = []
    
    boreholes = data.get('boreholesData', [])
    
    for bh in boreholes:
        name = bh.get('th_title', 'Unknown')
        easting = bh.get('th_easting', 0)
        northing = bh.get('th_northing', 0)
        
        # Parse elevation from coordinates
        coords = bh.get('th_coordinates', '0,0,0')
        try:
            elevation = float(coords.split(',')[2].strip())
        except:
            elevation = 500  # Default
        
        layers = bh.get('th_layers', [])
        
        if not layers:
            continue
        
        # Create cylinder for each layer
        for layer in layers:
            layer_from = layer.get('layer_from', 0)
            layer_to = layer.get('layer_to', layer_from + 10)
            symbol = layer.get('layer_symbol', 'Unknown')
            description = layer.get('layer_descr', '')
            
            # Get color - prefer forecolor over backcolor
            color_hex = layer.get('layer_forecolor', '#808080')
            if color_hex == '#FFFFFF' or color_hex == '#ffffff':
                color_hex = layer.get('layer_backcolor', '#808080')
            if not color_hex.startswith('#'):
                color_hex = '#' + color_hex
            
            try:
                color_rgb = hex_to_rgb(color_hex)
            except:
                color_rgb = (0.5, 0.5, 0.5)
            
            # Calculate z positions (elevation - depth)
            z_top = elevation - layer_from
            z_bottom = elevation - layer_to
            
            # Create cylinder mesh
            x, y, z, i_f, j_f, k_f = create_cylinder_mesh(
                easting, northing, z_top, z_bottom, radius
            )
            
            color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
            
            trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=i_f, j=j_f, k=k_f,
                color=color_str,
                opacity=0.95,
                name=f'{name}: {symbol}',
                hovertemplate=f'<b>{name}</b><br>{symbol}: {description}<br>Depth: {layer_from}-{layer_to} ft<extra></extra>',
                showlegend=False,
                flatshading=True
            )
            traces.append(trace)
        
        # Add label above borehole
        label_trace = go.Scatter3d(
            x=[easting],
            y=[northing],
            z=[elevation + 20],
            mode='text+markers',
            text=[name],
            textposition='top center',
            textfont=dict(size=12, color='black', family='Arial Black'),
            marker=dict(size=8, color='darkblue', symbol='diamond'),
            name=f'Label: {name}',
            showlegend=False,
            hovertemplate=f'<b>{name}</b><br>Elevation: {elevation:.1f} ft<br>Total Depth: {bh.get("th_depth", 0)} ft<extra></extra>'
        )
        traces.append(label_trace)
    
    return traces

def export_obj(polygons, filepath):
    """Export 3D model as OBJ file"""
    with open(filepath, 'w') as f:
        f.write("# 3D Geological Model from Cross Section Data\n")
        f.write(f"# Generated from RSLog JSON data\n\n")
        
        vertex_offset = 0
        
        for poly in polygons:
            vertices = poly['vertices']
            symbol = poly['symbol']
            
            if len(vertices) < 3:
                continue
            
            f.write(f"# Soil layer: {symbol} - {poly['description']}\n")
            f.write(f"g {symbol}_{vertex_offset}\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write face (1-indexed in OBJ format)
            face_indices = ' '.join(str(vertex_offset + i + 1) for i in range(len(vertices)))
            f.write(f"f {face_indices}\n\n")
            
            vertex_offset += len(vertices)
    
    print(f"✓ Exported OBJ to: {filepath}")

def create_cross_section_between_boreholes(data, borehole1_name, borehole2_name, num_segments=10, layer_matcher=None):
    """
    Create a cross-section plane between two boreholes showing interpolated soil layers.
    
    RULES:
    1. Same material in both boreholes: Connect directly (top-to-top, bottom-to-bottom)
    2. Different materials: Layer boundaries connect - one layer's bottom connects to next layer's top
    3. NO GAPS - continuous filled cross-section from surface to bottom
    
    Uses POSITIONAL matching - layers are matched by their order in the sequence,
    ensuring layer boundaries align and create continuous surfaces.
    
    If layer_matcher is provided (IntelligentLayerMatcher), uses ML-based matching
    to intelligently pair layers based on soil properties, colors, and descriptions.
    """
    traces = []
    
    # Find the two boreholes
    boreholes = data.get('boreholesData', [])
    bh1 = None
    bh2 = None
    
    for bh in boreholes:
        name = bh.get('th_title', '')
        if name == borehole1_name:
            bh1 = bh
        elif name == borehole2_name:
            bh2 = bh
    
    if not bh1 or not bh2:
        print(f"Warning: Could not find boreholes {borehole1_name} and/or {borehole2_name}")
        return traces
    
    def get_borehole_position(bh):
        easting = bh.get('th_easting', 0)
        northing = bh.get('th_northing', 0)
        coords = bh.get('th_coordinates', '0,0,0')
        try:
            elevation = float(coords.split(',')[2].strip())
        except:
            elevation = 500
        return easting, northing, elevation
    
    x1, y1, elev1 = get_borehole_position(bh1)
    x2, y2, elev2 = get_borehole_position(bh2)
    
    # Get layers sorted by depth
    layers1 = sorted(bh1.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    layers2 = sorted(bh2.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    
    if not layers1 or not layers2:
        return traces
    
    def get_layer_info(layer):
        symbol = layer.get('layer_symbol', 'Unknown')
        color_hex = layer.get('layer_forecolor', '#808080')
        if color_hex == '#FFFFFF' or color_hex == '#ffffff':
            color_hex = layer.get('layer_backcolor', '#808080')
        if not color_hex.startswith('#'):
            color_hex = '#' + color_hex
        description = layer.get('layer_descr', '')
        return symbol, color_hex, description
    
    # Build boundary lists - each layer has top and bottom depth
    # We'll create boundary points that form continuous lines across the section
    
    # Collect all boundary depths from both boreholes
    boundaries1 = [0]  # Start at surface
    boundaries2 = [0]
    
    for layer in layers1:
        boundaries1.append(layer.get('layer_to', 0))
    for layer in layers2:
        boundaries2.append(layer.get('layer_to', 0))
    
    # The number of layers
    n_layers1 = len(layers1)
    n_layers2 = len(layers2)
    max_layers = max(n_layers1, n_layers2)
    
    # Normalize boundaries - interpolate to have same number of boundary points
    # This ensures each layer boundary in BH1 has a corresponding point in BH2
    
    def interpolate_boundaries(boundaries, target_count):
        """Interpolate boundary list to have target_count+1 points"""
        if len(boundaries) == target_count + 1:
            return boundaries
        
        result = []
        for i in range(target_count + 1):
            # Map index to position in original list
            t = i / target_count if target_count > 0 else 0
            orig_idx = t * (len(boundaries) - 1)
            low_idx = int(orig_idx)
            high_idx = min(low_idx + 1, len(boundaries) - 1)
            frac = orig_idx - low_idx
            
            interp_val = boundaries[low_idx] + frac * (boundaries[high_idx] - boundaries[low_idx])
            result.append(interp_val)
        
        return result
    
    # Interpolate to have same number of boundaries
    norm_boundaries1 = interpolate_boundaries(boundaries1, max_layers)
    norm_boundaries2 = interpolate_boundaries(boundaries2, max_layers)
    
    # Helper to get layer info at a specific index
    def get_layer_at_index(layers, idx):
        if idx < len(layers):
            return layers[idx]
        return layers[-1] if layers else None  # Return last layer if out of range
    
    # Create a polygon for each layer (between consecutive boundaries)
    for i in range(max_layers):
        # Get boundary depths
        depth1_top = norm_boundaries1[i]
        depth1_bottom = norm_boundaries1[i + 1]
        depth2_top = norm_boundaries2[i]
        depth2_bottom = norm_boundaries2[i + 1]
        
        # Get layer info from both boreholes
        layer1 = get_layer_at_index(layers1, i)
        layer2 = get_layer_at_index(layers2, i)
        
        symbol1, color1, desc1 = get_layer_info(layer1) if layer1 else ('Unknown', '#808080', '')
        symbol2, color2, desc2 = get_layer_info(layer2) if layer2 else ('Unknown', '#808080', '')
        
        # Convert depths to z coordinates (elevation - depth)
        z1_top = elev1 - depth1_top
        z1_bottom = elev1 - depth1_bottom
        z2_top = elev2 - depth2_top
        z2_bottom = elev2 - depth2_bottom
        
        # Determine color - use layer1's color if same material, otherwise blend
        if symbol1 == symbol2:
            # Same material - use single color
            try:
                color_rgb = hex_to_rgb(color1)
            except:
                color_rgb = (0.5, 0.5, 0.5)
            color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
            
            # Create single quad
            x = np.array([x1, x2, x2, x1])
            y = np.array([y1, y2, y2, y1])
            z = np.array([z1_top, z2_top, z2_bottom, z1_bottom])
            
            trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color_str,
                opacity=0.92,
                name=f'{symbol1}',
                hovertemplate=f'<b>{symbol1}</b>: {desc1}<extra></extra>',
                showlegend=False,
                flatshading=True
            )
            traces.append(trace)
        else:
            # Different materials - BH1's material extends to BH2, blend at BH2 endpoint
            # This creates a diagonal transition where BH1's layer tapers as it approaches BH2
            
            try:
                color_rgb1 = hex_to_rgb(color1)
            except:
                color_rgb1 = (0.5, 0.5, 0.5)
            color_str1 = f'rgb({int(color_rgb1[0]*255)},{int(color_rgb1[1]*255)},{int(color_rgb1[2]*255)})'
            
            try:
                color_rgb2 = hex_to_rgb(color2)
            except:
                color_rgb2 = (0.5, 0.5, 0.5)
            color_str2 = f'rgb({int(color_rgb2[0]*255)},{int(color_rgb2[1]*255)},{int(color_rgb2[2]*255)})'
            
            # BH1's material - full band that tapers at BH2
            # Top triangle: BH1 top -> BH2 top -> BH2 mid
            # Bottom triangle: BH1 top -> BH2 mid -> BH1 bottom
            z2_mid = (z2_top + z2_bottom) / 2
            
            # Upper portion - BH1's material extends and tapers
            x_upper = np.array([x1, x2, x2, x1])
            y_upper = np.array([y1, y2, y2, y1])
            z_upper = np.array([z1_top, z2_top, z2_mid, z1_bottom])
            
            trace1 = go.Mesh3d(
                x=x_upper, y=y_upper, z=z_upper,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color_str1,
                opacity=0.92,
                name=f'{symbol1}',
                hovertemplate=f'<b>{symbol1}</b>: {desc1} (extends from {borehole1_name})<extra></extra>',
                showlegend=False,
                flatshading=True
            )
            traces.append(trace1)
            
            # Lower portion at BH2 - BH2's material fills remaining space
            x_lower = np.array([x1, x2, x2])
            y_lower = np.array([y1, y2, y2])
            z_lower = np.array([z1_bottom, z2_mid, z2_bottom])
            
            trace2 = go.Mesh3d(
                x=x_lower, y=y_lower, z=z_lower,
                i=[0], j=[1], k=[2],
                color=color_str2,
                opacity=0.92,
                name=f'{symbol2}',
                hovertemplate=f'<b>{symbol2}</b>: {desc2} (at {borehole2_name})<extra></extra>',
                showlegend=False,
                flatshading=True
            )
            traces.append(trace2)
        
        # Add boundary line at top of this layer
        top_line = go.Scatter3d(
            x=[x1, x2], y=[y1, y2], z=[z1_top, z2_top],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', width=2),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(top_line)
    
    # Add bottom boundary line
    z1_bottom_final = elev1 - norm_boundaries1[-1]
    z2_bottom_final = elev2 - norm_boundaries2[-1]
    bottom_line = go.Scatter3d(
        x=[x1, x2], y=[y1, y2], z=[z1_bottom_final, z2_bottom_final],
        mode='lines',
        line=dict(color='rgba(0,0,0,0.5)', width=2),
        showlegend=False,
        hoverinfo='skip'
    )
    traces.append(bottom_line)
    
    # Add outline
    outline_x = [x1, x2, x2, x1, x1]
    outline_y = [y1, y2, y2, y1, y1]
    outline_z = [elev1, elev2, z2_bottom_final, z1_bottom_final, elev1]
    
    outline_trace = go.Scatter3d(
        x=outline_x, y=outline_y, z=outline_z,
        mode='lines',
        line=dict(color='black', width=4),
        name=f'Section: {borehole1_name}-{borehole2_name}',
        showlegend=True,
        hoverinfo='skip'
    )
    traces.append(outline_trace)
    
    # Add label
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    mid_z = max(elev1, elev2) + 30
    
    label_trace = go.Scatter3d(
        x=[mid_x], y=[mid_y], z=[mid_z],
        mode='text',
        text=[f'{borehole1_name} ↔ {borehole2_name}'],
        textfont=dict(size=14, color='darkgreen', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    )
    traces.append(label_trace)
    
    print(f"  Created cross-section between {borehole1_name} and {borehole2_name} ({max_layers} layers, no gaps)")
    
    return traces


def create_intelligent_cross_section(data, borehole1_name, borehole2_name, matcher=None):
    """
    Create a cross-section using ML-based intelligent layer matching.
    
    Uses the IntelligentLayerMatcher to determine optimal layer connections
    based on soil properties, colors, descriptions, and geological context.
    
    Args:
        data: JSON data with borehole info
        borehole1_name, borehole2_name: Names of boreholes to connect
        matcher: Trained IntelligentLayerMatcher instance (optional)
    
    Returns:
        List of Plotly traces for the cross-section
    """
    traces = []
    
    # Find the two boreholes
    boreholes = data.get('boreholesData', [])
    bh1 = None
    bh2 = None
    
    for bh in boreholes:
        name = bh.get('th_title', '')
        if name == borehole1_name:
            bh1 = bh
        elif name == borehole2_name:
            bh2 = bh
    
    if not bh1 or not bh2:
        print(f"  Warning: Could not find boreholes {borehole1_name} and/or {borehole2_name}")
        return create_cross_section_between_boreholes(data, borehole1_name, borehole2_name)
    
    # Get or create matcher
    if matcher is None:
        matcher = get_intelligent_matcher(data)
    
    if matcher is None:
        print("  Falling back to positional matching")
        return create_cross_section_between_boreholes(data, borehole1_name, borehole2_name)
    
    # Get layers
    layers1 = sorted(bh1.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    layers2 = sorted(bh2.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    
    if not layers1 or not layers2:
        return traces
    
    # Use ML to find best matches
    matches = matcher.find_best_matches(layers1, layers2, bh1=bh1, bh2=bh2, threshold=0.3)
    
    print(f"  🧠 ML Matched {len(matches)} layer pairs for {borehole1_name} ↔ {borehole2_name}")
    for i, j, prob in matches[:5]:  # Show first 5
        sym1 = layers1[i].get('layer_symbol', '?')
        sym2 = layers2[j].get('layer_symbol', '?')
        print(f"     {sym1} ↔ {sym2} ({prob:.1%})")
    
    # Now create the cross-section using the matched layers
    # For unmatched layers, we'll blend them appropriately
    
    def get_borehole_position(bh):
        easting = bh.get('th_easting', 0)
        northing = bh.get('th_northing', 0)
        coords = bh.get('th_coordinates', '0,0,0')
        try:
            elevation = float(coords.split(',')[2].strip())
        except:
            elevation = 500
        return easting, northing, elevation
    
    x1, y1, elev1 = get_borehole_position(bh1)
    x2, y2, elev2 = get_borehole_position(bh2)
    
    def get_layer_info(layer):
        symbol = layer.get('layer_symbol', 'Unknown')
        color_hex = layer.get('layer_forecolor', '#808080')
        if color_hex == '#FFFFFF' or color_hex == '#ffffff':
            color_hex = layer.get('layer_backcolor', '#808080')
        if not color_hex or not color_hex.startswith('#'):
            color_hex = '#808080'
        description = layer.get('layer_descr', '')
        return symbol, color_hex, description
    
    # Create traces for each matched pair
    for i, j, prob in matches:
        layer1 = layers1[i]
        layer2 = layers2[j]
        
        d1_top = layer1.get('layer_from', 0)
        d1_bottom = layer1.get('layer_to', d1_top + 10)
        d2_top = layer2.get('layer_from', 0)
        d2_bottom = layer2.get('layer_to', d2_top + 10)
        
        z1_top = elev1 - d1_top
        z1_bottom = elev1 - d1_bottom
        z2_top = elev2 - d2_top
        z2_bottom = elev2 - d2_bottom
        
        symbol1, color1, desc1 = get_layer_info(layer1)
        symbol2, color2, desc2 = get_layer_info(layer2)
        
        # Use confidence to adjust opacity
        opacity = 0.7 + 0.25 * prob
        
        # For high-confidence matches, use solid connection
        # For lower confidence, show gradient blend
        try:
            color_rgb = hex_to_rgb(color1)
        except:
            color_rgb = (0.5, 0.5, 0.5)
        
        color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
        
        # Create the connecting polygon
        x = np.array([x1, x2, x2, x1])
        y = np.array([y1, y2, y2, y1])
        z = np.array([z1_top, z2_top, z2_bottom, z1_bottom])
        
        match_indicator = "✓" if symbol1 == symbol2 else "≈"
        
        trace = go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=opacity,
            name=f'{symbol1} {match_indicator} {symbol2}',
            hovertemplate=f'<b>ML Match: {prob:.0%}</b><br>{symbol1}: {desc1}<br>↔<br>{symbol2}: {desc2}<extra></extra>',
            showlegend=False,
            flatshading=True
        )
        traces.append(trace)
        
        # Add boundary lines
        boundary_line = go.Scatter3d(
            x=[x1, x2], y=[y1, y2], z=[z1_top, z2_top],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.4)', width=2),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(boundary_line)
    
    # Add label with ML indicator
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    mid_z = max(elev1, elev2) + 30
    
    label_trace = go.Scatter3d(
        x=[mid_x], y=[mid_y], z=[mid_z],
        mode='text',
        text=[f'🧠 {borehole1_name} ↔ {borehole2_name}'],
        textfont=dict(size=14, color='purple', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    )
    traces.append(label_trace)
    
    print(f"  ✓ Intelligent cross-section: {borehole1_name} ↔ {borehole2_name} ({len(matches)} ML-matched layers)")
    
    return traces


def create_3d_visualization(data, output_html=None, cross_section_pairs=None, use_intelligent_matching=True):
    """
    Create interactive 3D visualization
    
    Args:
        data: Parsed JSON data
        output_html: Path to save HTML file
        cross_section_pairs: List of tuples of borehole names to create cross-sections between
                            e.g., [('B-57', 'B-58'), ('B-53', 'B-54')]
        use_intelligent_matching: If True, use ML-based layer matching
    
    Returns:
        Plotly figure
    """
    print("Extracting 3D polygons...")
    polygons = extract_3d_polygons(data)
    print(f"  Found {len(polygons)} polygons")
    
    print("Creating 3D mesh traces...")
    mesh_traces = create_3d_mesh_traces(polygons)
    
    print("Creating borehole traces...")
    borehole_traces = create_borehole_traces(data)
    
    # Initialize intelligent matcher if requested
    matcher = None
    if use_intelligent_matching and INTELLIGENT_MATCHING_AVAILABLE:
        matcher = get_intelligent_matcher(data)
        if matcher:
            print("🧠 Using Intelligent Layer Matching (ML-based)")
    
    # Create cross-section planes between borehole pairs
    cross_section_traces = []
    if cross_section_pairs:
        print("Creating cross-section planes...")
        for bh1, bh2 in cross_section_pairs:
            if matcher:
                cs_traces = create_intelligent_cross_section(data, bh1, bh2, matcher)
            else:
                cs_traces = create_cross_section_between_boreholes(data, bh1, bh2)
            cross_section_traces.extend(cs_traces)
    
    # Combine all traces
    all_traces = mesh_traces + borehole_traces + cross_section_traces
    
    # Create figure
    fig = go.Figure(data=all_traces)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>3D Geological Model</b><br><sub>Generated from Cross Section Polygon Data</sub>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Easting (ft)',
            yaxis_title='Northing (ft)',
            zaxis_title='Elevation (ft)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    if output_html:
        fig.write_html(output_html)
        print(f"✓ Saved interactive HTML to: {output_html}")
    
    return fig, polygons

def generate_3d_between_cross_sections(data, section_pair1, section_pair2, output_html=None, output_stl=None):
    """
    Generate a 3D model by connecting two cross-section planes.
    Creates a clean 3D volume between 4 boreholes forming 2 cross-section pairs.
    
    Args:
        data: JSON data with borehole information
        section_pair1: Tuple of (borehole1_name, borehole2_name) for first cross-section
        section_pair2: Tuple of (borehole1_name, borehole2_name) for second cross-section
        output_html: Path to save HTML visualization
    
    Returns:
        Plotly figure with 3D model
    """
    print("\n" + "=" * 60)
    print("3D MODEL BETWEEN CROSS-SECTIONS")
    print(f"  Section 1: {section_pair1[0]} ↔ {section_pair1[1]}")
    print(f"  Section 2: {section_pair2[0]} ↔ {section_pair2[1]}")
    print("=" * 60)
    
    boreholes_data = data.get('boreholesData', [])
    
    # Get borehole info by name
    def get_borehole(name):
        for bh in boreholes_data:
            if bh.get('th_title', '') == name:
                return bh
        return None
    
    def get_borehole_position(bh):
        easting = bh.get('th_easting', 0)
        northing = bh.get('th_northing', 0)
        coords = bh.get('th_coordinates', '0,0,0')
        try:
            elevation = float(coords.split(',')[2].strip())
        except:
            elevation = 500
        return easting, northing, elevation
    
    def get_layer_info(layer):
        symbol = layer.get('layer_symbol', 'Unknown')
        color_hex = layer.get('layer_forecolor', '#808080')
        if color_hex == '#FFFFFF' or color_hex == '#ffffff':
            color_hex = layer.get('layer_backcolor', '#808080')
        if not color_hex.startswith('#'):
            color_hex = '#' + color_hex
        return symbol, color_hex
    
    # Get all 4 boreholes
    bh1a = get_borehole(section_pair1[0])
    bh1b = get_borehole(section_pair1[1])
    bh2a = get_borehole(section_pair2[0])
    bh2b = get_borehole(section_pair2[1])
    
    if not all([bh1a, bh1b, bh2a, bh2b]):
        print("Error: Could not find all boreholes!")
        return None
    
    # Get positions
    x1a, y1a, elev1a = get_borehole_position(bh1a)
    x1b, y1b, elev1b = get_borehole_position(bh1b)
    x2a, y2a, elev2a = get_borehole_position(bh2a)
    x2b, y2b, elev2b = get_borehole_position(bh2b)
    
    print(f"\nBorehole positions:")
    print(f"  {section_pair1[0]}: ({x1a}, {y1a}, {elev1a})")
    print(f"  {section_pair1[1]}: ({x1b}, {y1b}, {elev1b})")
    print(f"  {section_pair2[0]}: ({x2a}, {y2a}, {elev2a})")
    print(f"  {section_pair2[1]}: ({x2b}, {y2b}, {elev2b})")
    
    # Get layers for all boreholes
    layers1a = sorted(bh1a.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    layers1b = sorted(bh1b.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    layers2a = sorted(bh2a.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    layers2b = sorted(bh2b.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
    
    # Find the maximum number of layers across all boreholes
    max_layers = max(len(layers1a), len(layers1b), len(layers2a), len(layers2b))
    
    # Get max depth
    max_depth = max(
        bh1a.get('th_depth', 0), bh1b.get('th_depth', 0),
        bh2a.get('th_depth', 0), bh2b.get('th_depth', 0)
    )
    
    # Normalize layer boundaries for all boreholes
    def get_normalized_boundaries(layers, max_count):
        boundaries = [0]
        for layer in layers:
            boundaries.append(layer.get('layer_to', 0))
        
        if len(boundaries) == max_count + 1:
            return boundaries
        
        result = []
        for i in range(max_count + 1):
            t = i / max_count if max_count > 0 else 0
            orig_idx = t * (len(boundaries) - 1)
            low_idx = int(orig_idx)
            high_idx = min(low_idx + 1, len(boundaries) - 1)
            frac = orig_idx - low_idx
            interp_val = boundaries[low_idx] + frac * (boundaries[high_idx] - boundaries[low_idx])
            result.append(interp_val)
        return result
    
    def get_layer_at_index(layers, idx):
        if idx < len(layers):
            return layers[idx]
        return layers[-1] if layers else None
    
    norm1a = get_normalized_boundaries(layers1a, max_layers)
    norm1b = get_normalized_boundaries(layers1b, max_layers)
    norm2a = get_normalized_boundaries(layers2a, max_layers)
    norm2b = get_normalized_boundaries(layers2b, max_layers)
    
    traces = []
    
    print(f"\nCreating 3D volume with {max_layers} layer bands...")
    
    # Initialize STL exporter
    stl_exporter = STLExporter()
    
    # For each layer, create 4 separate filled surfaces (front, back, left, right walls)
    # This creates a solid-looking volume instead of a hollow box
    for i in range(max_layers):
        # Get layer info from each borehole
        layer1a = get_layer_at_index(layers1a, i)
        layer1b = get_layer_at_index(layers1b, i)
        layer2a = get_layer_at_index(layers2a, i)
        layer2b = get_layer_at_index(layers2b, i)
        
        # Get depths at each corner
        d1a_top, d1a_bot = norm1a[i], norm1a[i+1]
        d1b_top, d1b_bot = norm1b[i], norm1b[i+1]
        d2a_top, d2a_bot = norm2a[i], norm2a[i+1]
        d2b_top, d2b_bot = norm2b[i], norm2b[i+1]
        
        # Convert to z coordinates
        z1a_top, z1a_bot = elev1a - d1a_top, elev1a - d1a_bot
        z1b_top, z1b_bot = elev1b - d1b_top, elev1b - d1b_bot
        z2a_top, z2a_bot = elev2a - d2a_top, elev2a - d2a_bot
        z2b_top, z2b_bot = elev2b - d2b_top, elev2b - d2b_bot
        
        # Get color (use first borehole's color)
        symbol, color_hex = get_layer_info(layer1a) if layer1a else ('Unknown', '#808080')
        try:
            color_rgb = hex_to_rgb(color_hex)
        except:
            color_rgb = (0.5, 0.5, 0.5)
        
        # Slightly darken alternate layers for better distinction
        if i % 2 == 1:
            color_rgb = tuple(c * 0.85 for c in color_rgb)
        
        color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
        
        # Create 4 wall surfaces (quads) instead of a single hollow box
        # This makes the volume look solid
        
        # FRONT WALL (Section 1: B-55 to B-56)
        front_x = np.array([x1a, x1b, x1b, x1a])
        front_y = np.array([y1a, y1b, y1b, y1a])
        front_z = np.array([z1a_top, z1b_top, z1b_bot, z1a_bot])
        
        front_trace = go.Mesh3d(
            x=front_x, y=front_y, z=front_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            name=f'{symbol}',
            hovertemplate=f'<b>{symbol}</b><br>Layer {i+1}<extra></extra>',
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2)
        )
        traces.append(front_trace)
        
        # BACK WALL (Section 2: B-54 to B-54B)
        back_x = np.array([x2a, x2b, x2b, x2a])
        back_y = np.array([y2a, y2b, y2b, y2a])
        back_z = np.array([z2a_top, z2b_top, z2b_bot, z2a_bot])
        
        back_trace = go.Mesh3d(
            x=back_x, y=back_y, z=back_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2),
            hoverinfo='skip'
        )
        traces.append(back_trace)
        
        # LEFT WALL (connecting B-55 to B-54)
        left_x = np.array([x1a, x2a, x2a, x1a])
        left_y = np.array([y1a, y2a, y2a, y1a])
        left_z = np.array([z1a_top, z2a_top, z2a_bot, z1a_bot])
        
        left_trace = go.Mesh3d(
            x=left_x, y=left_y, z=left_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2),
            hoverinfo='skip'
        )
        traces.append(left_trace)
        
        # RIGHT WALL (connecting B-56 to B-54B)
        right_x = np.array([x1b, x2b, x2b, x1b])
        right_y = np.array([y1b, y2b, y2b, y1b])
        right_z = np.array([z1b_top, z2b_top, z2b_bot, z1b_bot])
        
        right_trace = go.Mesh3d(
            x=right_x, y=right_y, z=right_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2),
            hoverinfo='skip'
        )
        traces.append(right_trace)
        
        # TOP SURFACE (horizontal cap)
        top_x = np.array([x1a, x1b, x2b, x2a])
        top_y = np.array([y1a, y1b, y2b, y2a])
        top_z = np.array([z1a_top, z1b_top, z2b_top, z2a_top])
        
        top_trace = go.Mesh3d(
            x=top_x, y=top_y, z=top_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.4, specular=0.3),
            hoverinfo='skip'
        )
        traces.append(top_trace)
        
        # BOTTOM SURFACE (horizontal cap)
        bottom_x = np.array([x1a, x1b, x2b, x2a])
        bottom_y = np.array([y1a, y1b, y2b, y2a])
        bottom_z = np.array([z1a_bot, z1b_bot, z2b_bot, z2a_bot])
        
        bottom_trace = go.Mesh3d(
            x=bottom_x, y=bottom_y, z=bottom_z,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color_str,
            opacity=0.95,
            showlegend=False,
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=0.5, specular=0.2),
            hoverinfo='skip'
        )
        traces.append(bottom_trace)
        
        # Add THICK edge lines between layers for clear delineation
        edge_color = 'black' if i % 3 == 0 else 'rgba(50,50,50,0.8)'
        edge_width = 4 if i % 3 == 0 else 2
        
        # Section 1 top boundary
        traces.append(go.Scatter3d(
            x=[x1a, x1b], y=[y1a, y1b], z=[z1a_top, z1b_top],
            mode='lines', line=dict(color=edge_color, width=edge_width),
            showlegend=False, hoverinfo='skip'
        ))
        
        # Section 2 top boundary
        traces.append(go.Scatter3d(
            x=[x2a, x2b], y=[y2a, y2b], z=[z2a_top, z2b_top],
            mode='lines', line=dict(color=edge_color, width=edge_width),
            showlegend=False, hoverinfo='skip'
        ))
        
        # Connecting edges (left and right)
        traces.append(go.Scatter3d(
            x=[x1a, x2a], y=[y1a, y2a], z=[z1a_top, z2a_top],
            mode='lines', line=dict(color=edge_color, width=edge_width),
            showlegend=False, hoverinfo='skip'
        ))
        
        traces.append(go.Scatter3d(
            x=[x1b, x2b], y=[y1b, y2b], z=[z1b_top, z2b_top],
            mode='lines', line=dict(color=edge_color, width=edge_width),
            showlegend=False, hoverinfo='skip'
        ))
        
        # Add layer box to STL exporter with layer name for selection
        layer_name = f"Layer_{i+1:02d}_{symbol}"
        stl_exporter.set_layer(layer_name)
        
        corners_top = [
            [x1a, y1a, z1a_top],
            [x1b, y1b, z1b_top],
            [x2b, y2b, z2b_top],
            [x2a, y2a, z2a_top]
        ]
        corners_bottom = [
            [x1a, y1a, z1a_bot],
            [x1b, y1b, z1b_bot],
            [x2b, y2b, z2b_bot],
            [x2a, y2a, z2a_bot]
        ]
        stl_exporter.add_box(corners_top, corners_bottom)
        
        print(f"  Layer {i+1}: {symbol}")
    
    # Add borehole columns for these 4 boreholes only
    print("\nAdding borehole columns...")
    
    for bh_name in [section_pair1[0], section_pair1[1], section_pair2[0], section_pair2[1]]:
        bh = get_borehole(bh_name)
        if not bh:
            continue
        
        x, y, elev = get_borehole_position(bh)
        layers = sorted(bh.get('th_layers', []), key=lambda l: l.get('layer_from', 0))
        
        for layer in layers:
            layer_from = layer.get('layer_from', 0)
            layer_to = layer.get('layer_to', 0)
            symbol, color_hex = get_layer_info(layer)
            
            try:
                color_rgb = hex_to_rgb(color_hex)
            except:
                color_rgb = (0.5, 0.5, 0.5)
            
            z_top = elev - layer_from
            z_bottom = elev - layer_to
            
            # Create cylinder for layer
            cyl_x, cyl_y, cyl_z, i_f, j_f, k_f = create_cylinder_mesh(x, y, z_top, z_bottom, radius=12)
            
            color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
            
            trace = go.Mesh3d(
                x=cyl_x, y=cyl_y, z=cyl_z,
                i=i_f, j=j_f, k=k_f,
                color=color_str,
                opacity=0.95,
                name=f'{bh_name}: {symbol}',
                hovertemplate=f'<b>{bh_name}</b><br>{symbol}<br>Depth: {layer_from:.1f}-{layer_to:.1f} ft<extra></extra>',
                showlegend=False
            )
            traces.append(trace)
        
        # Add label
        label = go.Scatter3d(
            x=[x], y=[y], z=[elev + 20],
            mode='text+markers',
            text=[bh_name],
            textfont=dict(size=12, color='black', family='Arial Black'),
            marker=dict(size=6, color='darkblue'),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.append(label)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(
            text=f'<b>3D Model: {section_pair1[0]}-{section_pair1[1]} to {section_pair2[0]}-{section_pair2[1]}</b>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='Easting (ft)',
            yaxis_title='Northing (ft)',
            zaxis_title='Elevation (ft)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    if output_html:
        fig.write_html(output_html)
        print(f"\n✓ Saved 3D model to: {output_html}")
    
    # Export STL and OBJ
    if output_stl:
        stl_exporter.write_binary(output_stl)
        # Also export OBJ with same base name
        obj_path = str(output_stl).replace('.stl', '.obj')
        stl_exporter.write_obj(obj_path, name="geological_3d_volume")
    
    return fig, stl_exporter


def list_available_boreholes(data):
    """List all available boreholes in the data"""
    boreholes = data.get('boreholesData', [])
    print("\nAvailable boreholes:")
    for bh in boreholes:
        name = bh.get('th_title', 'Unknown')
        easting = bh.get('th_easting', 0)
        northing = bh.get('th_northing', 0)
        depth = bh.get('th_depth', 0)
        layers = len(bh.get('th_layers', []))
        print(f"  {name}: pos=({easting}, {northing}), depth={depth}ft, layers={layers}")
    return [bh.get('th_title', '') for bh in boreholes]


def main():
    """Generate 3D model from easier dataset"""
    import sys
    
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    
    print("=" * 60)
    print("3D Model Generation from Cross Section Data")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {easier_file.name}")
    data = load_json(easier_file)
    
    # List available boreholes
    available = list_available_boreholes(data)
    
    # Check for command line arguments for custom pairs
    # Usage: python generate_3d_model.py B-55 B-56 B-57 B-58
    # This would create cross-sections: B-55↔B-56 and B-57↔B-58
    
    if len(sys.argv) > 2:
        # Parse pairs from command line (every 2 arguments = one pair)
        args = sys.argv[1:]
        cross_section_pairs = []
        for i in range(0, len(args) - 1, 2):
            pair = (args[i], args[i + 1])
            if args[i] in available and args[i + 1] in available:
                cross_section_pairs.append(pair)
            else:
                print(f"Warning: Skipping invalid pair {pair}")
        print(f"\nUsing custom cross-section pairs from command line:")
        for p in cross_section_pairs:
            print(f"  {p[0]} ↔ {p[1]}")
    else:
        # Default cross-section pairs - comprehensive set
        cross_section_pairs = [
            ('B-55', 'B-56'),
            ('B-57', 'B-59'),
            ('B-58', 'B-60'),
            ('B-64', 'B-65'),
        ]
        print(f"\nUsing default cross-section pairs:")
        for p in cross_section_pairs:
            print(f"  {p[0]} ↔ {p[1]}")
        print("\n(Use CLI args for custom: python generate_3d_model.py B-55 B-56 B-57 B-58)")
    
    # Create visualization with cross-sections
    output_html = base_path / "3d_model_easier.html"
    fig, polygons = create_3d_visualization(data, output_html, cross_section_pairs)
    
    # Export OBJ
    output_obj = base_path / "3d_model_easier.obj"
    export_obj(polygons, output_obj)
    
    print("\n" + "=" * 60)
    print("✓ Cross-Section Model Complete!")
    print(f"  - Interactive viewer: {output_html}")
    print(f"  - OBJ export: {output_obj}")
    print(f"  - Cross-sections created: {len(cross_section_pairs)}")
    print("=" * 60)
    
    # Generate focused 3D model between two cross-section planes
    section_pair1 = ('B-55', 'B-56')
    section_pair2 = ('B-54', 'B-54B')
    
    output_3d = base_path / "3d_model_between_sections.html"
    output_stl = base_path / "3d_model_between_sections.stl"
    fig_3d, stl_exporter = generate_3d_between_cross_sections(
        data, section_pair1, section_pair2, 
        output_html=output_3d,
        output_stl=output_stl
    )
    
    print("\n" + "=" * 60)
    print("✓ ALL MODELS COMPLETE!")
    print(f"  - Cross-sections: {output_html}")
    print(f"  - 3D model (HTML): {output_3d}")
    print(f"  - 3D model (STL): {output_stl}")
    print("=" * 60)
    
    # Show the 3D model
    if fig_3d:
        fig_3d.show()
    
    # Test intelligent layer matching if available
    if INTELLIGENT_MATCHING_AVAILABLE:
        print("\n" + "=" * 60)
        print("🧠 INTELLIGENT LAYER MATCHING AVAILABLE")
        print("=" * 60)
        print("To use ML-based layer matching, run:")
        print("  python intelligent_layer_matching.py")
        print("This trains a model on soil properties for better layer matching.")
        print("=" * 60)


if __name__ == "__main__":
    main()

