# Model Importer Quick Reference

## Basic Usage

### Load Any Format (Auto-Detect)
```cpp
#include "ModelLoader.h"

auto result = ModelLoader::Load("assets/model.fbx", textureManager);
if (result) {
    scene->AddChild(result.root);
}
```

### Load Specific Format
```cpp
auto result = ModelLoader::LoadAs("model.dae", ModelLoader::Format::DAE, texMgr);
```

### Load with Options
```cpp
ModelLoader::LoadOptions opts;
opts.generateTangents = true;
opts.optimizeMeshes = true;
opts.verbose = true;

auto result = ModelLoader::Load("model.fbx", texMgr, opts);
```

## Supported Formats

| Format | Extension | Status | Notes |
|--------|-----------|--------|-------|
| OBJ | `.obj` | ✅ Stable | + MTL support |
| glTF 2.0 | `.gltf` | ✅ Stable | PBR materials |
| glTF Binary | `.glb` | ✅ Stable | Optimized |
| FBX | `.fbx` | ✅ Stable | Industry standard |
| COLLADA | `.dae` | ✅ Stable | Legacy support |
| Blender | `.blend` | ✅ Stable | Native Blender |
| MD5 | `.md5mesh` | ✅ Stable | Doom 3 format |
| IQM | `.iqm` | ✅ Stable | Modern game format |
| STL | `.stl` | ✅ Supported | CAD format |
| USD | `.usd`/`.usdz` | ✅ Supported | Pixar format |

## Error Handling

```cpp
auto result = ModelLoader::Load(path);

if (!result) {
    std::cerr << "Failed: " << result.errorMessage << std::endl;
    
    for (const auto& warn : result.warnings) {
        std::cerr << "Warning: " << warn << std::endl;
    }
} else {
    std::cout << "Loaded: " << result.meshCount << " meshes\n";
    std::cout << "Format: " << ModelLoader::GetFormatName(result.detectedFormat) << "\n";
}
```

## glTF Extensions

### Check for Extensions
```cpp
#include "GLTFExtensions.h"

auto exts = GLTFExtensions::ParseExtensions(assetJson);

for (auto ext : exts) {
    if (GLTFExtensions::IsExtensionSupported(ext)) {
        std::cout << "Supported: " << GLTFExtensions::GetExtensionName(ext) << std::endl;
    }
}
```

### Handle Common Extensions

**Unlit Materials**
```cpp
if (GLTFExtensions::IsUnlit(materialJson)) {
    material->DisableLighting();
    material->SetEmissiveOnly();
}
```

**Texture Transforms**
```cpp
auto transform = GLTFExtensions::ParseTextureTransform(textureJson);
material->SetTextureOffset(transform.offsetU, transform.offsetV);
material->SetTextureScale(transform.scaleU, transform.scaleV);
material->SetTextureRotation(transform.rotationZ);
```

**Punctual Lights (Custom Lighting)**
```cpp
auto light = GLTFExtensions::ParsePunctualLight(lightJson);

Light* engineLight = scene->CreateLight();
switch (light.type) {
    case GLTFExtensions::PunctualLight::Type::Directional:
        engineLight->SetType(LightType::Directional);
        break;
    case GLTFExtensions::PunctualLight::Type::Point:
        engineLight->SetType(LightType::Point);
        engineLight->SetRange(light.range);
        break;
    case GLTFExtensions::PunctualLight::Type::Spot:
        engineLight->SetType(LightType::Spot);
        engineLight->SetOuterAngle(light.outerConeAngle);
        break;
}
engineLight->SetColor(light.colorR, light.colorG, light.colorB);
engineLight->SetIntensity(light.intensity);
```

## Validation

### Validate Before Loading
```cpp
if (!ModelLoader::ValidateFile("model.fbx")) {
    std::cerr << "Invalid or corrupted file" << std::endl;
    return;
}

auto result = ModelLoader::Load("model.fbx");
```

### Detect Format
```cpp
auto fmt = ModelLoader::DetectFormat("model.glb");

if (fmt == ModelLoader::Format::GLB) {
    std::cout << "Detected glTF Binary" << std::endl;
}
```

## Advanced Usage

### Load from Memory
```cpp
std::vector<uint8_t> data = LoadFileIntoMemory("model.glb");

auto result = ModelLoader::LoadFromMemory(
    data.data(),
    data.size(),
    ModelLoader::Format::GLB,
    textureManager
);
```

### Batch Loading
```cpp
std::vector<std::string> modelPaths = {
    "assets/character.fbx",
    "assets/environment.glb",
    "assets/items/sword.obj"
};

std::vector<ModelLoader::LoadResult> results;

for (const auto& path : modelPaths) {
    results.push_back(ModelLoader::Load(path, texMgr));
}

// Add all to scene
for (auto& result : results) {
    if (result) {
        scene->AddChild(result.root);
    }
}
```

### Check Version Info
```cpp
std::cout << ModelLoader::GetVersionInfo() << std::endl;

// Output:
// ModelLoader v1.0
// - tinygltf v2.8.13 (glTF/GLB support)
// - Assimp v5.3.1 (FBX, DAE, BLEND, IQM, MD5 support)
// - OBJ support (native implementation)
```

## LoadOptions Reference

```cpp
struct LoadOptions {
    bool loadAnimations = true;              // Load skeletal animations
    bool loadMaterials = true;               // Load material definitions
    bool loadTextures = true;                // Load texture files
    bool generateNormalsIfMissing = true;    // Generate smooth normals
    bool generateTangents = false;           // Generate tangent/bitangent
    bool optimizeMeshes = false;             // Optimize mesh data
    bool mergeVertexBones = true;            // Combine bone influences
    float meshOptimizationThreshold = 0.001f;// Merge threshold
    bool verbose = false;                    // Print debug output
};
```

## Performance Tips

### For Fast Loading
```cpp
LoadOptions opts;
opts.generateTangents = false;  // Skip if no normal mapping
opts.optimizeMeshes = true;
opts.verbose = false;           // Disable debug output

auto result = ModelLoader::Load(path, texMgr, opts);
```

### For Quality
```cpp
LoadOptions opts;
opts.generateTangents = true;   // For normal mapping quality
opts.generateNormalsIfMissing = true;
opts.optimizeMeshes = false;    // Preserve original data

auto result = ModelLoader::Load(path, texMgr, opts);
```

### For Memory
```cpp
LoadOptions opts;
opts.loadTextures = false;      // Skip texture loading
opts.generateTangents = false;  // Skip extra vertex data
opts.optimizeMeshes = true;     // Reduce vertex count

auto result = ModelLoader::Load(path, texMgr, opts);

// Load textures separately later
// texMgr->LoadTexture("texture.png");
```

## Format Recommendations

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Shipping game | GLB | Optimized, single file, standard |
| Asset distribution | glTF | Human-readable, widely compatible |
| Animation-heavy | FBX | Rich skeletal support, industry standard |
| CAD models | STL | Native CAD format |
| Editor/tools | glTF | Editable, debuggable |
| Legacy projects | OBJ | Maximum compatibility |

## Troubleshooting

### "Format not supported"
```cpp
// Check if format is supported
if (!ModelLoader::IsFormatSupported(fmt)) {
    std::cerr << "Use a different format" << std::endl;
}

// Or convert first:
// assimp export model.blend model.glb
```

### Slow Loading
```cpp
// Enable optimization
LoadOptions opts;
opts.optimizeMeshes = true;
opts.verbose = true;  // See what's slow

auto result = ModelLoader::Load(path, texMgr, opts);
```

### Materials Not Loaded
```cpp
// Make sure materials are enabled
LoadOptions opts;
opts.loadMaterials = true;
opts.verbose = true;

auto result = ModelLoader::Load(path, texMgr, opts);

// Check loaded material count
if (result.materialCount == 0) {
    std::cerr << "No materials in source file" << std::endl;
}
```

### Skeletal Animation Not Working
```cpp
// Verify animations loaded
auto result = ModelLoader::Load(path);
std::cout << "Animations: " << result.animationCount << std::endl;

// Check if Animator attached
auto animator = result.root->GetAnimator();
if (animator) {
    animator->PlayAnimation(0);
} else {
    std::cerr << "No animator attached" << std::endl;
}
```

## Code Examples

### Complete Model Loading Example
```cpp
#include "ModelLoader.h"
#include "TextureManager.h"

int main() {
    auto texManager = std::make_unique<TextureManager>();
    
    // Load multiple models
    struct {
        std::string path;
        std::string name;
    } models[] = {
        {"character.fbx", "Player"},
        {"environment.glb", "World"},
        {"items/sword.obj", "Weapon"}
    };
    
    auto scene = std::make_shared<GameObject>("Scene");
    
    for (const auto& model : models) {
        auto result = ModelLoader::Load(model.path, texManager.get());
        
        if (result) {
            result.root->SetName(model.name);
            scene->AddChild(result.root);
            
            std::cout << "✓ Loaded " << model.name 
                      << " (" << result.meshCount << " meshes)" << std::endl;
        } else {
            std::cerr << "✗ Failed: " << result.errorMessage << std::endl;
        }
    }
    
    return 0;
}
```

### glTF Extension Handling
```cpp
#include "GLTFExtensions.h"

void ProcessGLTFMaterial(const nlohmann::json& matJson, Material* material) {
    // Check extensions
    auto exts = GLTFExtensions::ParseExtensions(matJson);
    
    for (auto ext : exts) {
        if (!GLTFExtensions::IsExtensionSupported(ext)) {
            std::cout << "Warning: Unsupported extension: " 
                      << GLTFExtensions::GetExtensionName(ext) << std::endl;
            continue;
        }
        
        switch (ext) {
            case GLTFExtensions::Extension::KHR_Materials_Unlit:
                material->SetUnlit(true);
                break;
                
            case GLTFExtensions::Extension::KHR_Texture_Transform:
                if (matJson["extensions"].contains("KHR_texture_transform")) {
                    auto tf = GLTFExtensions::ParseTextureTransform(
                        matJson["extensions"]["KHR_texture_transform"]
                    );
                    material->SetTextureTransform(tf.offsetU, tf.offsetV,
                                                 tf.rotationZ, tf.scaleU, tf.scaleV);
                }
                break;
                
            case GLTFExtensions::Extension::KHR_Materials_ClearCoat:
                material->EnableClearCoat();
                break;
                
            default:
                // Other extensions...
                break;
        }
    }
}
```

## See Also

- [MODEL_IMPORTER_GUIDE.md](MODEL_IMPORTER_GUIDE.md) - Full documentation
- [MODEL_IMPORTER_IMPLEMENTATION.md](MODEL_IMPORTER_IMPLEMENTATION.md) - Technical details
- [glTF 2.0 Spec](https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html)
- [Assimp Docs](http://assimp.sourceforge.net/)

