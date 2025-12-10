#!/usr/bin/env python3
"""
Setup Stable Diffusion XL + ControlNet Pipeline
Installs dependencies and sets up the model pipeline
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    # Install numpy first with compatible version (<2.0)
    print("Installing numpy (compatible version)...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy<2.0'])
        print("✓ Installed numpy<2.0")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install numpy: {e}")
        return False
    
    # Install other packages
    packages = [
        'diffusers',
        'transformers',
        'torch',
        'torchvision',
        'accelerate',
        'controlnet-aux',
        'pillow',
        'opencv-python',
        'scikit-image',
        'matplotlib',
        'requests'
    ]
    
    print("\nInstalling other dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    # Reinstall packages that depend on numpy to fix binary compatibility
    print("\nFixing binary compatibility...")
    numpy_dependent_packages = ['opencv-python==4.8.1.78', 'scikit-image', 'scipy']
    for package in numpy_dependent_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-cache-dir', '--quiet', package])
            print(f"✓ Reinstalled {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Could not reinstall {package}: {e}")
    
    print("\nAll dependencies installed!")
    return True

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected. Will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False

def setup_model_pipeline():
    """Set up and test the Stable Diffusion XL + ControlNet pipeline"""
    try:
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
        from diffusers import UniPCMultistepScheduler
        import torch
        
        print("\nSetting up Stable Diffusion XL + ControlNet pipeline...")
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Using device: {device}")
        print(f"Using dtype: {dtype}")
        
        # Load ControlNet (SDXL-compatible version)
        print("Loading ControlNet model (SDXL-compatible)...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=dtype
        )
        print("✓ ControlNet loaded")
        
        # Load Stable Diffusion XL
        print("Loading Stable Diffusion XL base model...")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        # Use faster scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        
        print("✓ Pipeline setup complete!")
        print("\nPipeline ready for use:")
        print(f"  - Base model: stabilityai/stable-diffusion-xl-base-1.0")
        print(f"  - ControlNet: lllyasviel/sd-controlnet-canny")
        print(f"  - Device: {device}")
        
        return pipe
        
    except Exception as e:
        print(f"✗ Error setting up pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check internet connection (models download from Hugging Face)")
        print("3. Ensure sufficient disk space (~10GB for models)")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Stable Diffusion XL + ControlNet Setup")
    print("=" * 60)
    
    # Install dependencies
    success = install_dependencies()
    if not success:
        print("\n" + "=" * 60)
        print("Failed to install dependencies. Please fix errors above.")
        print("=" * 60)
        sys.exit(1)
    
    # Check GPU
    print("\n" + "=" * 60)
    check_gpu()
    
    # Setup pipeline
    print("\n" + "=" * 60)
    pipe = setup_model_pipeline()
    
    if pipe:
        print("\n" + "=" * 60)
        print("Setup complete! Ready to generate cross sections.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Setup incomplete. Please check errors above.")
        print("=" * 60)

