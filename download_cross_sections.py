#!/usr/bin/env python3
"""
Download Cross Section Images
Downloads cross section images from RSLog URLs for training data
"""

import json
import requests
from pathlib import Path
from PIL import Image
import io

def download_image(url, output_path):
    """Download image from URL and save to file"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        img.save(output_path)
        print(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_cross_section_images(json_file, output_dir=None):
    """
    Download all cross section images from JSON file
    
    Args:
        json_file: Path to JSON file
        output_dir: Directory to save images (default: same as JSON file)
    """
    json_path = Path(json_file)
    if output_dir is None:
        output_dir = json_path.parent / "cross_section_images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Download images for each cross section
    downloaded = []
    for i, section in enumerate(data.get('polygonsBySection', [])):
        section_name = section.get('sectionName', f'section_{i}')
        section_id = section.get('sectionId', f'id_{i}')
        
        # Download preview image
        preview_url = section.get('imagePreview')
        if preview_url:
            preview_path = output_dir / f"{section_name}_preview.png"
            if download_image(preview_url, preview_path):
                downloaded.append(('preview', preview_path, section_name))
        
        # Download report image (higher resolution)
        report_url = section.get('imageReport')
        if report_url:
            report_path = output_dir / f"{section_name}_report.png"
            if download_image(report_url, report_path):
                downloaded.append(('report', report_path, section_name))
    
    print(f"\nDownloaded {len(downloaded)} images to {output_dir}")
    return downloaded

if __name__ == "__main__":
    base_path = Path(__file__).parent
    
    # Download from easier dataset
    easier_file = base_path / "Easier RSLog section (1) 1 (1).json"
    print("Downloading cross section images from easier dataset...")
    downloaded_easier = download_cross_section_images(easier_file)
    
    # Download from complex dataset
    complex_file = base_path / "Complex RSLog section (1) 1 (1).json"
    print("\nDownloading cross section images from complex dataset...")
    downloaded_complex = download_cross_section_images(complex_file)
    
    print(f"\nTotal images downloaded: {len(downloaded_easier) + len(downloaded_complex)}")

