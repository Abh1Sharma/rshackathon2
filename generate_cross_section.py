#!/usr/bin/env python3
"""
Cross Section Generation Pipeline
Uses Stable Diffusion XL + ControlNet to generate cross sections from borehole diagrams
"""

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from borehole_diagram_generator import create_diagram_from_section, load_borehole_data

class CrossSectionGenerator:
    """Generate cross sections from borehole data using Stable Diffusion XL + ControlNet"""
    
    def __init__(self, device=None, dtype=None):
        """
        Initialize the generator
        
        Args:
            device: 'cuda' or 'cpu' (auto-detect if None)
            dtype: torch.float16 or torch.float32 (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.device = device
        self.dtype = dtype
        
        print(f"Initializing pipeline on {device}...")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load Stable Diffusion XL + ControlNet pipeline"""
        # Load ControlNet (SDXL-compatible version)
        print("Loading ControlNet (SDXL-compatible)...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=self.dtype
        )
        
        # Load Stable Diffusion XL
        print("Loading Stable Diffusion XL...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        
        # Use faster scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        
        print("✓ Pipeline loaded successfully")
    
    def prepare_control_image(self, borehole_diagram, low_threshold=100, high_threshold=200):
        """
        Convert borehole diagram to Canny edge image for ControlNet
        
        Args:
            borehole_diagram: PIL Image of borehole diagram
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
        
        Returns:
            PIL Image of Canny edges
        """
        # Convert to numpy array
        img_array = np.array(borehole_diagram.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL Image
        control_image = Image.fromarray(edges)
        
        return control_image
    
    def generate(self, borehole_diagram, prompt=None, negative_prompt=None, 
                 num_inference_steps=20, guidance_scale=7.5, seed=None):
        """
        Generate cross section from borehole diagram
        
        Args:
            borehole_diagram: PIL Image of borehole diagram
            prompt: Text prompt (default: geological cross section description)
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            seed: Random seed for reproducibility
        
        Returns:
            PIL Image of generated cross section
        """
        if prompt is None:
            prompt = (
                "geological cross section, soil layers, borehole data, "
                "professional engineering diagram, clean lines, accurate geology, "
                "color-coded soil types, elevation profile"
            )
        
        if negative_prompt is None:
            negative_prompt = (
                "blurry, distorted, unrealistic, cartoon, artistic, "
                "watercolor, painting, abstract"
            )
        
        # Prepare control image (Canny edges)
        control_image = self.prepare_control_image(borehole_diagram)
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        print("Generating cross section...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=1.0,
        )
        
        generated_image = result.images[0]
        print("✓ Generation complete")
        
        return generated_image
    
    def generate_from_json(self, json_file, section_index=0, output_path=None, **kwargs):
        """
        Generate cross section directly from JSON file
        
        Args:
            json_file: Path to JSON file
            section_index: Index of cross section
            output_path: Path to save generated image
            **kwargs: Additional arguments for generate()
        
        Returns:
            PIL Image of generated cross section
        """
        # Create borehole diagram
        print("Creating borehole diagram...")
        borehole_diagram = create_diagram_from_section(json_file, section_index)
        
        # Generate cross section
        generated = self.generate(borehole_diagram, **kwargs)
        
        # Save if output path provided
        if output_path:
            generated.save(output_path)
            print(f"Saved to: {output_path}")
        
        return generated

def main():
    """Test the generation pipeline"""
    base_path = Path(__file__).parent
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    
    print("=" * 60)
    print("Cross Section Generation Test")
    print("=" * 60)
    
    # Initialize generator
    generator = CrossSectionGenerator()
    
    # Generate cross section
    output_path = base_path / "generated_cross_section_easier.png"
    generated = generator.generate_from_json(
        easier_file,
        section_index=0,
        output_path=output_path,
        num_inference_steps=20,
        guidance_scale=7.5
    )
    
    print(f"\n✓ Generated cross section saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

