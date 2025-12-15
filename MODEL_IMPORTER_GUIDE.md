# Model Importer Improvements & glTF Enhancements

## Overview

This document covers comprehensive improvements to the model importing system, including a unified `ModelLoader` interface, Assimp integration, and advanced glTF extension support.

## New Components

### 1. Unified ModelLoader

The new `ModelLoader` class provides a consistent interface for loading 3D models in multiple formats with automatic format detection and unified error handling.

**File:** [include/ModelLoader.h](../include/ModelLoader.h)

#### Supported Formats

| Format | Extension | Library | Skeletal Animation | Materials | Notes |
|--------|-----------|---------|------------------|-----------|-------|
| Wavefront OBJ | `.obj` | Native | ✓ | ✓ | Traditional format with MTL |
| glTF 2.0 | `.gltf` | tinygltf | ✓ | ✓ | Text format, commonly used |
| glTF Binary | `.glb` | tinygltf | ✓ | ✓ | Optimized binary format |
| Autodesk FBX | `.fbx` | Assimp | ✓ | ✓ | Industry standard |
| COLLADA | `.dae` | Assimp | ✓ | ✓ | Legacy but well-supported |
| Blender | `.blend` | Assimp | ✓ | ✓ | Direct Blender export |
| Doom 3 MD5 | `.md5mesh/.md5anim` | Assimp | ✓ | ✓ | Game engine format |
| Inter-Quake Model | `.iqm` | Assimp | ✓ | ✓ | Modern game format |
| STL | `.stl` | Assimp | ✗ | ✗ | CAD/3D printing |
| USD | `.usd/.usdz` | Assimp | ✓ | ✓ | Pixar Universal Scene Description |

#### Quick Start

```cpp
#include "ModelLoader.h"

// Load with automatic format detection
auto result = ModelLoader::Load("assets/models/character.fbx", textureManager);

if (result) {
    auto gameObject = result.root;
    std::cout << "Loaded " << result.meshCount << " meshes, "
              << result.materialCount << " materials, "
              << result.animationCount << " animations" << std::endl;
    
    // Add to scene
    scene->AddChild(gameObject);
} else {
    std::cerr << "Failed to load model: " << result.errorMessage << std::endl;
}
```

#### Advanced Options

```cpp
ModelLoader::LoadOptions options;
options.loadAnimations = true;
options.generateNormalsIfMissing = true;
options.generateTangents = true;  // For normal mapping
options.optimizeMeshes = true;
options.verbose = true;

auto result = ModelLoader::Load("assets/models/scene.gltf", texManager, options);
```

#### Format-Specific Loading

```cpp
// Load explicitly as FBX (skips detection)
auto result = ModelLoader::LoadAs(path, ModelLoader::Format::FBX, texManager);

// Load from memory buffer
auto result = ModelLoader::LoadFromMemory(buffer, size, ModelLoader::Format::GLB, texManager);

// Validate file before loading
if (ModelLoader::ValidateFile("model.fbx")) {
    auto result = ModelLoader::Load("model.fbx");
}
```

### 2. Assimp Integration

**Assimp v5.3.1** is now integrated into the build system, providing support for multiple industry-standard formats.

#### Features

- **Automatic format detection** from file extensions and magic bytes
- **Post-processing pipeline** with multiple optimization passes:
  - Normal generation (smooth normals if missing)
  - Tangent/bitangent calculation for normal mapping
  - Mesh optimization and graph optimization
  - Redundant material removal
- **Skeletal animation support** for rigged characters
- **Material extraction** from embedded material definitions
- **Mesh hierarchy** preservation with transform preservation

#### Configuration

Edit `CMakeLists.txt` to customize Assimp features:

```cmake
# Currently enabled importers:
set(ASSIMP_BUILD_FBX_IMPORTER ON)          # Autodesk FBX
set(ASSIMP_BUILD_COLLADA_IMPORTER ON)      # COLLADA DAE
set(ASSIMP_BUILD_BLEND_IMPORTER ON)        # Blender native
set(ASSIMP_BUILD_IQM_IMPORTER ON)          # Inter-Quake Model
set(ASSIMP_BUILD_MD5_IMPORTER ON)          # Doom 3 MD5
```

Additional importers can be enabled in CMakeLists.txt:
- `ASSIMP_BUILD_3DS_IMPORTER` - 3D Studio MAX
- `ASSIMP_BUILD_X_IMPORTER` - DirectX
- `ASSIMP_BUILD_LWO_IMPORTER` - LightWave
- `ASSIMP_BUILD_OBJ_IMPORTER` - OBJ (Assimp version, slower than native)

### 3. glTF 2.0 Extensions Support

**File:** [include/GLTFExtensions.h](../include/GLTFExtensions.h)

The engine now supports advanced glTF 2.0 extensions for enhanced material rendering and geometry optimization.

#### Supported Extensions

##### Material Extensions (Fully Supported)

- **KHR_materials_unlit** - Emissive-only materials without lighting
  ```cpp
  if (GLTFExtensions::IsUnlit(materialJson)) {
      material->DisableLighting();
  }
  ```

- **KHR_materials_emissive_strength** - Intensity control for emissive textures
  ```cpp
  auto strength = GLTFExtensions::ParseEmissiveStrength(materialJson);
  material->SetEmissiveIntensity(strength);
  ```

- **KHR_texture_transform** - UV coordinate transforms (rotation, scale, offset)
  ```cpp
  auto transform = GLTFExtensions::ParseTextureTransform(json);
  material->SetTextureTransform(transform.offsetU, transform.offsetV, 
                                transform.rotationZ, transform.scaleU, transform.scaleV);
  ```

##### Material Extensions (Partial Support)

- **KHR_materials_pbrSpecularGlossiness** - Legacy PBR workflow (converts to metallic-roughness)
- **KHR_materials_clearcoat** - Clear coat layer for car paints, varnish
- **KHR_materials_sheen** - Fabric/cloth surface characteristics
- **KHR_materials_transmission** - Glass and transparent materials
- **KHR_materials_ior** - Index of refraction for accurate glass rendering
- **KHR_materials_volume** - Volume rendering for thick translucent materials

##### Lighting Extensions

- **KHR_lights_punctual** - Advanced punctual light definitions
  ```cpp
  auto light = GLTFExtensions::ParsePunctualLight(lightsJson);
  switch (light.type) {
      case GLTFExtensions::PunctualLight::Type::Directional:
      case GLTFExtensions::PunctualLight::Type::Point:
      case GLTFExtensions::PunctualLight::Type::Spot:
  }
  ```

##### Geometry Extensions

- **KHR_mesh_quantization** - Vertex position quantization for storage reduction
  ```cpp
  if (GLTFExtensions::IsQuantized(primitiveJson)) {
      // Geometry uses quantized positions (handled transparently by tinygltf)
  }
  ```

- **KHR_draco_mesh_compression** - Advanced geometry compression
  - Status: Requires additional `draco` library (planned for v2)
  - Current workaround: Pre-decompress assets

##### Animation Extensions

- **KHR_animation_pointer** - Advanced animation targeting for complex properties

##### Planned Enhancements

- **EXT_meshopt_compression** - Alternative mesh optimization codec
- **KHR_materials_variants** - Material LOD/variants per mesh
- **EXT_texture_webp** - WebP texture format support

#### Extension Usage Example

```cpp
#include "GLTFExtensions.h"

// Check if file uses supported extensions
auto extensions = GLTFExtensions::ParseExtensions(assetJson);
for (auto ext : extensions) {
    if (!GLTFExtensions::IsExtensionSupported(ext)) {
        std::cout << "Unsupported extension: " 
                  << GLTFExtensions::GetExtensionName(ext) << std::endl;
    }
}

// Handle specific extension
if (GLTFExtensions::HasClearCoat(materialJson)) {
    // Material has clear coat layer - apply special shader
    material->EnableClearCoat();
}
```

## Architecture

### Class Hierarchy

```
ModelLoader (Facade)
├── LoadOBJ() → Model (existing)
├── LoadGLTF() → GLTFLoader::Load()
└── LoadWithAssimp() → Assimp::Importer

GLTFExtensions (Utility)
├── ParseExtensions()
├── IsExtensionSupported()
└── ParseMaterial/Light/etc()
```

### Error Handling

All loading functions return a `LoadResult` structure with detailed status:

```cpp
struct LoadResult {
    std::shared_ptr<GameObject> root;
    bool success;
    std::string errorMessage;
    std::vector<std::string> warnings;
    Format detectedFormat;
    int meshCount;
    int materialCount;
    int animationCount;
    int textureCount;
};

// Usage
auto result = ModelLoader::Load(path);
if (result) {
    // Success
} else {
    std::cerr << "Error: " << result.errorMessage << std::endl;
    for (const auto& warn : result.warnings) {
        std::cerr << "Warning: " << warn << std::endl;
    }
}
```

## Integration with Asset Pipeline

The ModelLoader integrates seamlessly with the existing asset pipeline:

```cpp
// Through AssetDatabase
auto asset = AssetDatabase::LoadAsset("character.fbx");

// Through ResourceStreamingManager
auto resource = std::make_shared<ModelResource>("models/player.gltf");
resource->Load();  // Async loading with progress callback

// Through VirtualFileSystem
auto data = vfs.ReadFile("/assets/models/statue.blend");
auto result = ModelLoader::LoadFromMemory(data.data(), data.size(), 
                                         ModelLoader::Format::BLEND);
```

## Performance Considerations

### Memory Usage

- **OBJ**: Uncompressed vertex data, fast loading
- **glTF**: Compact binary with embedded or separate assets
- **FBX**: Slower parse time, rich feature set
- **Draco-compressed**: Best compression, requires decompression overhead

### Loading Time Estimates (at 1M triangles)

| Format | Time | Notes |
|--------|------|-------|
| OBJ | ~50ms | Simple format, fast parse |
| glTF | ~30ms | Optimized format |
| GLB | ~20ms | Binary, parallel decompression |
| FBX | ~200ms | Complex format, lots of metadata |
| BLEND | ~300ms | Most data to process |

### Optimization Tips

```cpp
// Parallel loading with resource manager
resourceMgr.QueueResource("character.fbx", 
                         ResourceStreamingManager::Priority::High);

// Reduce texture memory
LoadOptions opts;
opts.loadTextures = false;  // Load meshes only
opts.optimizeMeshes = true;  // Merge small meshes

// Pre-optimize with external tools
// - Use glTF-Transform to compress
// - Use assimp-cmd to pre-process FBX files
// - Use draco encoder for geometry compression
```

## Best Practices

### 1. Format Selection

- **For Tools/Editor**: Use **glTF** (human-readable, standardized)
- **For Shipping**: Use **GLB** (optimized, self-contained)
- **For Animation-Heavy**: Use **FBX** (rich skeletal support)
- **For Procedural**: Use **OBJ** (simple, widely compatible)

### 2. Validation

```cpp
// Always validate before use in critical paths
if (!ModelLoader::ValidateFile(path)) {
    // Log error, provide fallback
    return LoadPlaceholderModel();
}

auto result = ModelLoader::Load(path, texManager);
if (!result.success) {
    std::cerr << "Loading failed: " << result.errorMessage << std::endl;
}
```

### 3. Extension Handling

```cpp
// Check required extensions
std::vector<std::string> required = { "KHR_materials_unlit" };
for (const auto& ext : required) {
    if (!GLTFExtensions::ValidateExtension(ext, assetRequired)) {
        // Use fallback material system
        fallback = true;
    }
}
```

### 4. Memory Management

```cpp
// Use smart pointers for automatic cleanup
auto model = ModelLoader::Load(path);

// Models are shared, safe to use across scenes
scene1->AddChild(model.root);
scene2->AddChild(model.root);  // Shared reference

// Clean up when no longer needed
scene1->RemoveChild(model.root);
// Only deleted when all references released
```

## Future Enhancements

### Phase 2
- [ ] Draco mesh decompression support
- [ ] WebP texture loading
- [ ] Material variants LOD system
- [ ] Async loading with progress callbacks

### Phase 3
- [ ] EXT_meshopt_compression support
- [ ] USD/USDZ full support
- [ ] Custom extension framework
- [ ] Material baking pipeline

### Phase 4
- [ ] Runtime import API (load models at runtime)
- [ ] Model editor UI
- [ ] Batch import tools
- [ ] Model optimization suite

## API Reference

### ModelLoader

#### Static Methods

```cpp
// Main loading functions
static LoadResult Load(const std::string& path, 
                      TextureManager* texManager = nullptr,
                      const LoadOptions& options = LoadOptions());

static LoadResult LoadAs(const std::string& path,
                        Format format,
                        TextureManager* texManager = nullptr,
                        const LoadOptions& options = LoadOptions());

static LoadResult LoadFromMemory(const uint8_t* data,
                                size_t size,
                                Format format,
                                TextureManager* texManager = nullptr,
                                const LoadOptions& options = LoadOptions());

// Utility functions
static Format DetectFormat(const std::string& path);
static std::string GetFormatName(Format format);
static bool IsFormatSupported(Format format);
static std::vector<std::string> GetSupportedExtensions();
static bool ValidateFile(const std::string& path);
static std::string GetVersionInfo();
```

### GLTFExtensions

```cpp
// Extension detection
static std::vector<Extension> ParseExtensions(const nlohmann::json& json);
static bool IsExtensionSupported(Extension ext);
static std::string GetExtensionName(Extension ext);

// Material extensions
static bool IsUnlit(const nlohmann::json& materialJson);
static bool IsSpecularGlossiness(const nlohmann::json& materialJson);
static bool HasClearCoat(const nlohmann::json& materialJson);

// Texture extensions
static TextureTransform ParseTextureTransform(const nlohmann::json& json);

// Lighting extensions
static PunctualLight ParsePunctualLight(const nlohmann::json& json);

// Geometry extensions
static bool IsDracoCompressed(const nlohmann::json& primitiveJson);
static bool IsQuantized(const nlohmann::json& primitiveJson);
```

## Troubleshooting

### Common Issues

**Issue**: "Format not supported: Unknown"
```
Solution: Check file extension and validate file integrity
- Use ModelLoader::ValidateFile() to check
- Ensure file is not corrupted
- Try explicit format with LoadAs()
```

**Issue**: Slow loading times for FBX/BLEND
```
Solution: Use background loading or pre-process
- Use ResourceStreamingManager for async loading
- Pre-process with Assimp tools: `assimp export input.blend output.glb`
- Reduce mesh complexity before export
```

**Issue**: Materials not loading correctly
```
Solution: Check for unsupported extensions
- Use GLTFExtensions::ParseExtensions() to inspect
- Enable verbose mode to see warnings
- Pre-bake complex extensions offline
```

**Issue**: Skeleton/animation not loading
```
Solution: Verify skeletal data in source file
- Check LoadOptions.loadAnimations is true
- Verify rigging in source application
- Use GLTFLoader verbose output for debugging
```

## References

- [glTF 2.0 Specification](https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html)
- [glTF Extensions Registry](https://github.com/KhronosGroup/glTF/tree/main/extensions)
- [Assimp Documentation](http://assimp.sourceforge.net/)
- [tinygltf GitHub](https://github.com/syoyo/tinygltf)

