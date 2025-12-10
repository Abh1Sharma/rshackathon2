#!/usr/bin/env python3
"""
Prepare Training Data Pairs
Creates training pairs: (borehole diagram, cross section image)
"""

import json
from pathlib import Path
from PIL import Image
from borehole_diagram_generator import create_diagram_from_section, load_borehole_data

def prepare_training_pairs(json_file, output_dir=None, section_index=0):
    """
    Prepare training pair: borehole diagram + cross section image
    
    Args:
        json_file: Path to JSON file
        output_dir: Directory to save training pairs
        section_index: Index of cross section
    
    Returns:
        Tuple of (borehole_diagram_path, cross_section_path)
    """
    json_path = Path(json_file)
    if output_dir is None:
        output_dir = json_path.parent / "training_pairs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data = load_borehole_data(json_file)
    section = data['polygonsBySection'][section_index]
    section_name = section.get('sectionName', f'section_{section_index}')
    
    # Create borehole diagram
    print(f"Creating borehole diagram for {section_name}...")
    borehole_diagram = create_diagram_from_section(json_file, section_index)
    
    # Save borehole diagram
    diagram_path = output_dir / f"{section_name}_borehole_diagram.png"
    borehole_diagram.save(diagram_path)
    print(f"  Saved borehole diagram: {diagram_path}")
    
    # Load corresponding cross section image
    cross_section_dir = json_path.parent / "cross_section_images"
    cross_section_path = cross_section_dir / f"{section_name}_preview.png"
    
    if not cross_section_path.exists():
        cross_section_path = cross_section_dir / f"{section_name}_report.png"
    
    if cross_section_path.exists():
        print(f"  Found cross section image: {cross_section_path}")
        
        # Copy to training pairs directory
        training_cross_section_path = output_dir / f"{section_name}_cross_section.png"
        img = Image.open(cross_section_path)
        img.save(training_cross_section_path)
        print(f"  Saved cross section: {training_cross_section_path}")
        
        return (diagram_path, training_cross_section_path)
    else:
        print(f"  ⚠ Cross section image not found for {section_name}")
        return (diagram_path, None)

def prepare_all_training_pairs(json_file, output_dir=None):
    """
    Prepare all training pairs from a JSON file
    
    Args:
        json_file: Path to JSON file
        output_dir: Directory to save training pairs
    
    Returns:
        List of tuples (borehole_diagram_path, cross_section_path)
    """
    data = load_borehole_data(json_file)
    pairs = []
    
    for i, section in enumerate(data.get('polygonsBySection', [])):
        pair = prepare_training_pairs(json_file, output_dir, section_index=i)
        pairs.append(pair)
    
    return pairs

if __name__ == "__main__":
    base_path = Path(__file__).parent
    
    print("=" * 60)
    print("Preparing Training Data Pairs")
    print("=" * 60)
    
    # Prepare from easier dataset
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    print("\nProcessing easier dataset...")
    easier_pairs = prepare_all_training_pairs(easier_file)
    
    # Prepare from complex dataset
    complex_file = base_path / "Complex RSLog section (1) 1 (1).json"
    print("\nProcessing complex dataset...")
    complex_pairs = prepare_all_training_pairs(complex_file)
    
    print("\n" + "=" * 60)
    print(f"Total training pairs prepared: {len(easier_pairs) + len(complex_pairs)}")
    print("=" * 60)

