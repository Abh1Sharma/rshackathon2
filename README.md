# Project 2: Borehole Data → Cross Section → 3D Model

## Overview

This project implements a two-phase pipeline for generating 3D geological models from borehole data:

1. **Phase 1**: Generate 2D cross sections from borehole data using Stable Diffusion XL + ControlNet
2. **Phase 2**: Create 3D models from cross sections using surface interpolation and mesh generation

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script:

```bash
python3 setup_models.py
```

### Verify GPU (Optional but Recommended)

The pipeline works on CPU but is much faster on GPU. Check GPU availability:

```python
import torch
print(torch.cuda.is_available())
```

## Usage

### Step 1: Prepare Data

1. **Create borehole diagrams**:
```bash
python3 borehole_diagram_generator.py
```

2. **Download cross-section images**:
```bash
python3 download_cross_sections.py
```

3. **Prepare training pairs**:
```bash
python3 prepare_training_data.py
```

### Step 2: Generate Cross Sections

```bash
python3 generate_cross_section.py
```

Or use programmatically:

```python
from generate_cross_section import CrossSectionGenerator

generator = CrossSectionGenerator()
generated = generator.generate_from_json(
    "Easier RSLog section (1) 1 (1).json",
    section_index=0,
    output_path="output.png"
)
```

## Project Structure

```
hackathon2/
├── borehole_diagram_generator.py    # Creates borehole diagrams from JSON
├── download_cross_sections.py       # Downloads cross-section images
├── prepare_training_data.py         # Prepares training pairs
├── setup_models.py                  # Sets up AI models
├── generate_cross_section.py        # Main generation pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── cross_section_images/            # Downloaded cross-section images
├── training_pairs/                  # Training data pairs
└── *.json                          # Input data files
```

## Model Details

- **Base Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **ControlNet**: `lllyasviel/sd-controlnet-canny`
- **Input**: Borehole diagram (schematic showing positions, depths, layers)
- **Output**: Geological cross section image

## Next Steps

- [ ] Implement polygon extraction from generated images
- [ ] Fine-tune model on geological cross sections
- [ ] Implement Phase 2: 3D model generation
- [ ] Add validation and metrics

## Notes

- First run will download large model files (~10GB)
- GPU recommended for faster generation
- Models are cached in `~/.cache/huggingface/`

