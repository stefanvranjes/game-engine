# ORM Texture Creator

A Python utility to create ORM (Occlusion-Roughness-Metallic) combined textures from separate PBR maps.

## What is ORM?

ORM textures pack three grayscale PBR maps into a single RGB texture:
- **R channel**: Ambient Occlusion
- **G channel**: Roughness
- **B channel**: Metallic

This reduces texture count, saves memory, and improves rendering performance.

## Requirements

```bash
pip install Pillow numpy
```

## Usage

### Single Material

Process a single material by providing any of its maps:

```bash
python create_orm.py material_ao.png
```

The script will automatically find the corresponding roughness and metallic maps.

### Batch Processing

Process an entire directory of materials:

```bash
python create_orm.py --batch assets/textures
```

### Output Directory

Specify a custom output directory:

```bash
python create_orm.py --batch assets/textures --output assets/orm
```

### Output Format

Choose output format (PNG, JPG, or TGA):

```bash
python create_orm.py material_ao.png --format jpg
```

## Supported Naming Patterns

The script automatically detects common PBR naming conventions:

**Ambient Occlusion**:
- `material_ao.png`
- `material_ambient_occlusion.png`
- `material_ambientocclusion.png`
- `material_AO.png`

**Roughness**:
- `material_roughness.png`
- `material_rough.png`
- `material_Roughness.png`

**Metallic**:
- `material_metallic.png`
- `material_metalness.png`
- `material_metal.png`
- `material_Metallic.png`

## Missing Maps

If a map is missing, the script uses default values:
- **AO**: 255 (white, no occlusion)
- **Roughness**: 128 (mid-range)
- **Metallic**: 0 (non-metallic)

## Examples

### Example 1: Complete Material Set
```
Input:
  wood_ao.png
  wood_roughness.png
  wood_metallic.png

Output:
  wood_orm.png
```

### Example 2: Missing Metallic Map
```
Input:
  concrete_ao.png
  concrete_roughness.png

Output:
  concrete_orm.png (with metallic = 0)
```

### Example 3: Batch Process
```bash
# Process all materials in textures/
python create_orm.py --batch textures/

# Output to separate directory
python create_orm.py --batch textures/ --output textures/orm/
```

## Command-Line Options

```
usage: create_orm.py [-h] [--batch] [--output OUTPUT] [--format {png,jpg,tga}] input

positional arguments:
  input                 Input file or directory (for batch mode)

optional arguments:
  -h, --help            show this help message and exit
  --batch               Batch process directory
  --output OUTPUT, -o OUTPUT
                        Output directory (optional)
  --format {png,jpg,tga}, -f {png,jpg,tga}
                        Output format (default: png)
```

## Integration with Game Engine

After creating ORM textures, use them in your materials:

```cpp
auto material = std::make_shared<Material>();
material->texture = textureManager->LoadTexture("assets/wood_albedo.png");
material->normalMap = textureManager->LoadTexture("assets/wood_normal.png");
material->ormMap = textureManager->LoadTexture("assets/wood_orm.png");  // Combined!
```

The engine will automatically extract AO, Roughness, and Metallic from the ORM texture.
