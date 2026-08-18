#!/usr/bin/env python3
"""
Create a GIF from Generation_###_airfoils.png files in the current directory.
Includes options for resolution reduction and automatic looping.



# Create GIF with half resolution and faster frame rate
python create_generation_gif.py --scale 0.2 --duration 50 --output 00_reduced.gif

# Create GIF with custom filename and slower frame rate
python create_generation_gif.py --output my_airfoil_evolution.gif --duration 800

# Create very small GIF for web use
python create_generation_gif.py --scale 0.3 --duration 200 --output small_evolution.gif

"""

import os
import glob
import re
from PIL import Image
import argparse

def natural_sort_key(text):
    """
    Sort key function for natural sorting (e.g., Generation_1, Generation_2, ..., Generation_10, Generation_11)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

def create_generation_gif(output_filename="generation_evolution.gif", 
                         duration=500, 
                         scale_factor=1.0, 
                         quality=85):
    """
    Create a GIF from Generation_###_airfoils.png files.
    
    Parameters:
    - output_filename: Name of the output GIF file
    - duration: Duration of each frame in milliseconds
    - scale_factor: Factor to scale images (1.0 = original size, 0.5 = half size, etc.)
    - quality: JPEG quality when resizing (0-100, higher is better quality)
    """
    
    # Find all Generation_###_airfoils.png files
    pattern = "Generation_*_airfoils*.png"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Sort files naturally (Generation_1, Generation_2, ..., Generation_10, etc.)
    files.sort(key=natural_sort_key)
    
    print(f"Found {len(files)} generation files:")
    for f in files[:5]:  # Show first 5 files
        print(f"  {f}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")
    
    # Load and process images
    images = []
    original_size = None
    
    for i, file_path in enumerate(files):
        try:
            # Load image
            img = Image.open(file_path)
            
            # Store original size from first image
            if original_size is None:
                original_size = img.size
                print(f"Original image size: {original_size[0]}x{original_size[1]} pixels")
            
            # Resize if scale_factor is not 1.0
            if scale_factor != 1.0:
                new_width = int(img.size[0] * scale_factor)
                new_height = int(img.size[1] * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (GIF requires RGB or P mode)
            if img.mode not in ['RGB', 'P']:
                img = img.convert('RGB')
            
            images.append(img)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(files)} images...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not images:
        print("No images could be processed!")
        return
    
    # Calculate final size
    final_size = images[0].size
    if scale_factor != 1.0:
        print(f"Scaled image size: {final_size[0]}x{final_size[1]} pixels (scale factor: {scale_factor})")
    
    # Create GIF
    print(f"\nCreating GIF: {output_filename}")
    print(f"Frame duration: {duration}ms")
    print(f"Total duration: {duration * len(images) / 1000:.1f} seconds")
    print(f"Looping: Infinite")
    
    try:
        # Save as GIF with infinite loop
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,  # Duration of each frame in milliseconds
            loop=0,  # 0 means infinite loop
            optimize=True  # Optimize the GIF file size
        )
        
        # Get file size
        file_size = os.path.getsize(output_filename)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\nGIF created successfully!")
        print(f"Output file: {output_filename}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Frames: {len(images)}")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create a GIF from Generation_###_airfoils.png files")
    
    parser.add_argument('-o', '--output', 
                       default='generation_evolution.gif',
                       help='Output GIF filename (default: generation_evolution.gif)')
    
    parser.add_argument('-d', '--duration', 
                       type=int, 
                       default=500,
                       help='Duration of each frame in milliseconds (default: 500)')
    
    parser.add_argument('-s', '--scale', 
                       type=float, 
                       default=1.0,
                       help='Scale factor for images (1.0=original, 0.5=half size, etc.) (default: 1.0)')
    
    parser.add_argument('-q', '--quality', 
                       type=int, 
                       default=85,
                       help='Quality when resizing (0-100, higher=better) (default: 85)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scale <= 0:
        print("Error: Scale factor must be positive")
        return
    
    if not (0 <= args.quality <= 100):
        print("Error: Quality must be between 0 and 100")
        return
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        return
    
    print("=== Generation Airfoils GIF Creator ===")
    print(f"Output file: {args.output}")
    print(f"Frame duration: {args.duration}ms")
    print(f"Scale factor: {args.scale}")
    print(f"Quality: {args.quality}")
    print()
    
    create_generation_gif(
        output_filename=args.output,
        duration=args.duration,
        scale_factor=args.scale,
        quality=args.quality
    )

if __name__ == "__main__":
    main()
