#!/usr/bin/env python3
"""
DEMO: Intelligent 3D Geological Model Generation

This script demonstrates the full AI-enhanced pipeline:
1. Load borehole data from JSON
2. Train intelligent layer matcher (ML + OpenAI embeddings)
3. Generate 3D cross-sections with ML-based layer matching
4. Create interactive visualization
5. Export to OBJ/STL

Usage:
    export OPENAI_API_KEY="your-key-here"
    python demo_intelligent_3d.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_3d_model import (
    load_json,
    create_3d_visualization,
    generate_3d_between_cross_sections,
    get_intelligent_matcher,
    INTELLIGENT_MATCHING_AVAILABLE
)
from intelligent_layer_matching import (
    IntelligentLayerMatcher,
    analyze_layer_distributions
)


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_intelligent_matching():
    """Run the full intelligent 3D generation demo"""
    
    print_header("🎯 INTELLIGENT 3D GEOLOGICAL MODEL - DEMO")
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print("✓ OpenAI API key found")
    else:
        print("⚠ No OpenAI API key - using fallback similarity")
        print("  Set: export OPENAI_API_KEY='your-key-here'")
    
    # Load data
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    
    print(f"\n📁 Loading: {easier_file.name}")
    data = load_json(easier_file)
    
    # Show data statistics
    boreholes = data.get('boreholesData', [])
    sections = data.get('polygonsBySection', [])
    total_layers = sum(len(bh.get('th_layers', [])) for bh in boreholes)
    
    print(f"  • Boreholes: {len(boreholes)}")
    print(f"  • Cross-sections: {len(sections)}")
    print(f"  • Total layers: {total_layers}")
    
    # Analyze layer distributions
    print_header("📊 LAYER DISTRIBUTION ANALYSIS")
    analyze_layer_distributions(data)
    
    # Train intelligent matcher
    print_header("🧠 TRAINING INTELLIGENT LAYER MATCHER")
    
    matcher = IntelligentLayerMatcher(use_openai=True)
    matcher.train(data)
    
    # Save model
    model_path = base_path / "layer_matching_model.pkl"
    try:
        matcher.save_model(model_path)
    except Exception as e:
        print(f"  Note: Could not save model ({e})")
    
    # Test matching on specific pairs
    print_header("🔬 TESTING ML LAYER MATCHING")
    
    bh_lookup = {bh.get('th_title', ''): bh for bh in boreholes}
    test_pairs = [
        ('B-55', 'B-56'),
        ('B-57', 'B-58'),
        ('B-54', 'B-54B'),
    ]
    
    for bh1_name, bh2_name in test_pairs:
        bh1 = bh_lookup.get(bh1_name)
        bh2 = bh_lookup.get(bh2_name)
        
        if not bh1 or not bh2:
            continue
        
        layers1 = bh1.get('th_layers', [])
        layers2 = bh2.get('th_layers', [])
        
        matches = matcher.find_best_matches(layers1, layers2, bh1=bh1, bh2=bh2, threshold=0.3)
        
        print(f"\n{bh1_name} ({len(layers1)} layers) ↔ {bh2_name} ({len(layers2)} layers)")
        print(f"  ML found {len(matches)} optimal matches:")
        
        for i, j, prob in matches[:5]:
            sym1 = layers1[i].get('layer_symbol', '?')
            sym2 = layers2[j].get('layer_symbol', '?')
            d1 = f"{layers1[i].get('layer_from', 0):.0f}-{layers1[i].get('layer_to', 0):.0f}ft"
            d2 = f"{layers2[j].get('layer_from', 0):.0f}-{layers2[j].get('layer_to', 0):.0f}ft"
            match_type = "✓" if sym1 == sym2 else "≈"
            print(f"    {match_type} {sym1} ({d1}) ↔ {sym2} ({d2}) [{prob:.0%}]")
    
    # Generate 3D visualization with intelligent matching
    print_header("🎨 GENERATING 3D VISUALIZATION")
    
    cross_section_pairs = [
        ('B-55', 'B-56'),
        ('B-57', 'B-58'),
        ('B-57', 'B-59'),
        ('B-58', 'B-60'),
    ]
    
    output_html = base_path / "demo_intelligent_3d.html"
    
    print(f"Creating cross-sections:")
    for p in cross_section_pairs:
        print(f"  • {p[0]} ↔ {p[1]}")
    
    fig, polygons = create_3d_visualization(
        data, 
        output_html=output_html, 
        cross_section_pairs=cross_section_pairs,
        use_intelligent_matching=True
    )
    
    # Generate 3D volume between cross-sections
    print_header("📦 GENERATING 3D VOLUME MODEL")
    
    section_pair1 = ('B-55', 'B-56')
    section_pair2 = ('B-54', 'B-54B')
    
    output_3d = base_path / "demo_3d_volume.html"
    output_stl = base_path / "demo_3d_volume.stl"
    
    fig_3d, stl_exporter = generate_3d_between_cross_sections(
        data, section_pair1, section_pair2,
        output_html=output_3d,
        output_stl=output_stl
    )
    
    # Summary
    print_header("✅ DEMO COMPLETE")
    
    print("\nGenerated files:")
    print(f"  📊 Cross-section viewer: {output_html}")
    print(f"  📦 3D volume viewer:     {output_3d}")
    print(f"  🖨️  3D volume (STL):      {output_stl}")
    print(f"  🧠 ML model:             {model_path}")
    
    print("\n🚀 AI Features Used:")
    print("  • OpenAI text-embedding-3-small for description similarity")
    print("  • Geological context (water table, depth ratios)")
    print("  • GradientBoosting classifier for layer matching")
    print("  • 93-dimension feature vectors")
    
    print("\n📖 To view results:")
    print(f"  open {output_html}")
    print(f"  open {output_3d}")
    
    # Open in browser
    print("\nOpening 3D viewer...")
    fig.show()
    
    return fig, fig_3d


if __name__ == "__main__":
    demo_intelligent_matching()

