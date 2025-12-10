# Project 2: AI-Enhanced 3D Geological Model Generation

## 🎯 Overview

This project implements an **AI-enhanced pipeline** for generating 3D geological models from borehole data:

```
Borehole JSON → ML Layer Matching → 2D Cross-Sections → 3D Volume Model → OBJ/STL Export
                      ↓
            OpenAI Embeddings + 
            Geological Context
```

### Key Features
- 🧠 **Intelligent Layer Matching**: ML-based matching using soil properties, colors, and descriptions
- 🔤 **OpenAI Embeddings**: Semantic similarity for layer descriptions
- 🌊 **Geological Context**: Depth relative to water table, AASHTO classification
- 📦 **3D Export**: OBJ with selectable layers, STL for 3D printing
- 📊 **Interactive Visualization**: Plotly-based 3D viewer

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv hackathon2_env
source hackathon2_env/bin/activate

# Install packages
pip install -r requirements.txt
pip install openai  # For AI embeddings
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Demo

```bash
python demo_intelligent_3d.py
```

This will:
- Train the intelligent layer matcher
- Generate 3D cross-sections with ML matching
- Create interactive HTML visualization
- Export to OBJ/STL

---

## 🧠 AI/ML Components

### Intelligent Layer Matching

The `IntelligentLayerMatcher` uses machine learning to determine which soil layers should connect between boreholes.

**Features (93-dimensional vector)**:
| Feature Group | Count | Description |
|---------------|-------|-------------|
| AASHTO Classification | 12 | Soil type, permeability, compressibility, strength |
| Color Encoding | 6 | RGB for foreground + background colors |
| Geological Context | 5 | Water table position, depth ratios |
| Description Similarity | 1 | OpenAI embedding cosine similarity |

**Usage**:
```python
from intelligent_layer_matching import IntelligentLayerMatcher

# Initialize and train
matcher = IntelligentLayerMatcher(use_openai=True)
matcher.train(data)

# Find best layer matches between two boreholes
matches = matcher.find_best_matches(layers1, layers2, bh1=bh1, bh2=bh2)
# Returns: [(layer1_idx, layer2_idx, confidence), ...]
```

### OpenAI Integration

Uses `text-embedding-3-small` for semantic description matching:

```python
# Example: These descriptions would have high similarity
desc1 = "Gray Silty Sand with trace of gravel"
desc2 = "Grayish Sandy Silt with gravel fragments"
# → Similarity: 0.87 (would be missed by exact string matching)
```

---

## 📁 Project Structure

```
hackathon2/
├── 🧠 AI/ML Components
│   ├── intelligent_layer_matching.py  # ML layer matcher + OpenAI
│   └── layer_matching_model.pkl       # Trained model (auto-generated)
│
├── 📊 3D Generation
│   ├── generate_3d_model.py           # Main 3D pipeline
│   ├── demo_intelligent_3d.py         # Full demo script
│   └── visualize_boreholes_easier.py  # 2D visualization
│
├── 🎨 Legacy Image Generation
│   ├── generate_cross_section.py      # SDXL + ControlNet
│   ├── borehole_diagram_generator.py  # Create input diagrams
│   └── setup_models.py               # Download AI models
│
├── 📦 Data
│   ├── Easier RSLog section (1) 1 (1).json   # Simple dataset
│   ├── Complex RSLog section (1) 1 (1).json  # Complex dataset
│   └── cross_section_images/                 # Reference images
│
├── 📄 Config
│   ├── requirements.txt              # Python dependencies
│   └── .gitignore                   # Git ignore rules
│
└── 📖 Documentation
    ├── README.md                    # This file
    └── IMPLEMENTATION_STATUS.md     # Progress tracker
```

---

## 🔧 Usage Examples

### Generate 3D Visualization

```python
from generate_3d_model import (
    load_json, 
    create_3d_visualization,
    get_intelligent_matcher
)

# Load data
data = load_json("Easier RSLog section (1) 1 (1).json")

# Create visualization with intelligent matching
fig, polygons = create_3d_visualization(
    data,
    output_html="output.html",
    cross_section_pairs=[('B-55', 'B-56'), ('B-57', 'B-58')],
    use_intelligent_matching=True  # Use ML-based matching
)
```

### Generate 3D Volume

```python
from generate_3d_model import generate_3d_between_cross_sections

# Create 3D volume between two cross-section planes
fig, stl_exporter = generate_3d_between_cross_sections(
    data,
    section_pair1=('B-55', 'B-56'),
    section_pair2=('B-54', 'B-54B'),
    output_html="3d_model.html",
    output_stl="3d_model.stl"
)
```

### Train Custom Layer Matcher

```python
from intelligent_layer_matching import IntelligentLayerMatcher

matcher = IntelligentLayerMatcher(use_openai=True)

# Train on your data
matcher.train(your_data)

# Save for reuse
matcher.save_model("custom_model.pkl")

# Test matching
layers1 = bh1.get('th_layers', [])
layers2 = bh2.get('th_layers', [])
matches = matcher.find_best_matches(layers1, layers2, threshold=0.5)
```

---

## 📊 Results

### Model Performance

```
Test Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score
   Different       1.00      1.00      1.00
       Match       1.00      1.00      1.00
```

### Generated Outputs

| File | Description |
|------|-------------|
| `demo_intelligent_3d.html` | Interactive cross-section viewer |
| `demo_3d_volume.html` | 3D volume between cross-sections |
| `demo_3d_volume.stl` | STL for 3D printing |
| `3d_model_easier.obj` | OBJ with selectable layers |

---

## 🌐 API Reference

### `IntelligentLayerMatcher`

| Method | Description |
|--------|-------------|
| `train(data)` | Train on JSON data |
| `predict_match(layer1, layer2)` | Predict match probability |
| `find_best_matches(layers1, layers2)` | Find optimal layer pairings |
| `save_model(filepath)` | Save trained model |
| `load_model(filepath)` | Load trained model |

### `generate_3d_model` Functions

| Function | Description |
|----------|-------------|
| `create_3d_visualization()` | Create interactive 3D viewer |
| `generate_3d_between_cross_sections()` | Create 3D volume |
| `create_intelligent_cross_section()` | ML-based cross-section |
| `get_intelligent_matcher()` | Get/create trained matcher |

---

## 🔮 Future Enhancements

- [ ] GPT-4 for predicting missing layers
- [ ] Anomaly detection for geological features
- [ ] Auto-suggest optimal borehole locations
- [ ] Uncertainty visualization
- [ ] Web interface for interactive exploration

---

## 📝 Notes

- First run downloads model files and trains the matcher
- OpenAI embeddings are cached to minimize API calls
- GPU recommended for image generation (not required for 3D)
- Models cached in `~/.cache/huggingface/`

## 🤝 Contributors

Hackathon 2 Project Team
