#!/usr/bin/env python3
"""
ORM Texture Creator
Combines separate Ambient Occlusion, Roughness, and Metallic maps into a single ORM texture.
Channel packing: R=AO, G=Roughness, B=Metallic
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Common naming patterns for PBR maps
AO_SUFFIXES = ['_ao', '_ambient_occlusion', '_ambientocclusion', '_AO']
ROUGHNESS_SUFFIXES = ['_roughness', '_rough', '_Roughness']
METALLIC_SUFFIXES = ['_metallic', '_metalness', '_metal', '_Metallic']

def find_map(base_path, suffixes, extensions=['.png', '.jpg', '.jpeg', '.tga']):
    """Find a texture map with common naming patterns."""
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    # Remove common suffixes from base name
    for suffix in AO_SUFFIXES + ROUGHNESS_SUFFIXES + METALLIC_SUFFIXES:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    # Try to find the map
    for suffix in suffixes:
        for ext in extensions:
            candidate = os.path.join(base_dir, base_name + suffix + ext)
            if os.path.exists(candidate):
                return candidate
    return None

def load_grayscale_map(path, target_size=None):
    """Load a grayscale map and return as numpy array."""
    if path is None or not os.path.exists(path):
        return None
    
    img = Image.open(path).convert('L')
    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)

def create_orm(ao_path=None, roughness_path=None, metallic_path=None, output_path=None, 
               default_ao=255, default_roughness=128, default_metallic=0):
    """
    Create an ORM texture from separate maps.
    
    Args:
        ao_path: Path to ambient occlusion map
        roughness_path: Path to roughness map
        metallic_path: Path to metallic map
        output_path: Output path for ORM texture
        default_ao: Default AO value (0-255) if map missing
        default_roughness: Default roughness value (0-255) if map missing
        default_metallic: Default metallic value (0-255) if map missing
    
    Returns:
        True if successful, False otherwise
    """
    
    # Determine target size from first available map
    target_size = None
    for path in [ao_path, roughness_path, metallic_path]:
        if path and os.path.exists(path):
            with Image.open(path) as img:
                target_size = img.size
                break
    
    if target_size is None:
        print("Error: No valid input maps found")
        return False
    
    # Load maps
    ao = load_grayscale_map(ao_path, target_size)
    roughness = load_grayscale_map(roughness_path, target_size)
    metallic = load_grayscale_map(metallic_path, target_size)
    
    # Use defaults for missing maps
    if ao is None:
        ao = np.full(target_size[::-1], default_ao, dtype=np.uint8)
        print(f"  Using default AO value: {default_ao}")
    
    if roughness is None:
        roughness = np.full(target_size[::-1], default_roughness, dtype=np.uint8)
        print(f"  Using default Roughness value: {default_roughness}")
    
    if metallic is None:
        metallic = np.full(target_size[::-1], default_metallic, dtype=np.uint8)
        print(f"  Using default Metallic value: {default_metallic}")
    
    # Stack into RGB
    orm = np.stack([ao, roughness, metallic], axis=2)
    
    # Save
    Image.fromarray(orm).save(output_path)
    print(f"✓ Created: {output_path}")
    return True

def process_material(base_path, output_dir=None, output_format='png'):
    """Process a single material and create ORM texture."""
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    # Remove common suffixes
    for suffix in AO_SUFFIXES + ROUGHNESS_SUFFIXES + METALLIC_SUFFIXES:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    print(f"\nProcessing: {base_name}")
    
    # Find maps
    ao_path = find_map(base_path, AO_SUFFIXES)
    roughness_path = find_map(base_path, ROUGHNESS_SUFFIXES)
    metallic_path = find_map(base_path, METALLIC_SUFFIXES)
    
    # Report found maps
    print(f"  AO: {os.path.basename(ao_path) if ao_path else 'Not found'}")
    print(f"  Roughness: {os.path.basename(roughness_path) if roughness_path else 'Not found'}")
    print(f"  Metallic: {os.path.basename(metallic_path) if metallic_path else 'Not found'}")
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_orm.{output_format}")
    else:
        output_path = os.path.join(base_dir, f"{base_name}_orm.{output_format}")
    
    # Create ORM
    return create_orm(ao_path, roughness_path, metallic_path, output_path)

def batch_process(directory, output_dir=None, output_format='png'):
    """Batch process all materials in a directory."""
    directory = Path(directory)
    
    # Find all unique material base names
    materials = set()
    for suffix_list in [AO_SUFFIXES, ROUGHNESS_SUFFIXES, METALLIC_SUFFIXES]:
        for suffix in suffix_list:
            for ext in ['.png', '.jpg', '.jpeg', '.tga']:
                for file in directory.glob(f'*{suffix}{ext}'):
                    base_name = file.stem
                    for s in AO_SUFFIXES + ROUGHNESS_SUFFIXES + METALLIC_SUFFIXES:
                        if base_name.endswith(s):
                            base_name = base_name[:-len(s)]
                            break
                    materials.add(base_name)
    
    if not materials:
        print("No PBR materials found in directory")
        return
    
    print(f"Found {len(materials)} material(s)")
    
    success_count = 0
    for material in sorted(materials):
        # Use any map as base path
        base_path = None
        for suffix in AO_SUFFIXES + ROUGHNESS_SUFFIXES + METALLIC_SUFFIXES:
            for ext in ['.png', '.jpg', '.jpeg', '.tga']:
                candidate = directory / f"{material}{suffix}{ext}"
                if candidate.exists():
                    base_path = str(candidate)
                    break
            if base_path:
                break
        
        if base_path and process_material(base_path, output_dir, output_format):
            success_count += 1
    
    print(f"\n✓ Successfully created {success_count}/{len(materials)} ORM textures")

def main():
    parser = argparse.ArgumentParser(
        description='Create ORM (Occlusion-Roughness-Metallic) textures from separate maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single material
  python create_orm.py material_ao.png
  
  # Batch process directory
  python create_orm.py --batch assets/textures
  
  # Specify output directory
  python create_orm.py --batch assets/textures --output assets/orm
  
  # Change output format
  python create_orm.py material_ao.png --format jpg
        """
    )
    
    parser.add_argument('input', help='Input file or directory (for batch mode)')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--output', '-o', help='Output directory (optional)')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'jpg', 'tga'],
                       help='Output format (default: png)')
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return 1
        batch_process(args.input, args.output, args.format)
    else:
        if not os.path.exists(args.input):
            print(f"Error: {args.input} not found")
            return 1
        if not process_material(args.input, args.output, args.format):
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
