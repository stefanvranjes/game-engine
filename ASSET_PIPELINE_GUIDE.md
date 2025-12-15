# Asset Pipeline System - Implementation Guide

## Overview

The Asset Pipeline system provides **automatic asset conversion, hashing, and incremental builds** for the game engine. It eliminates manual asset preprocessing and enables fast iteration with only changed assets being reprocessed.

### Key Features

- **Asset Hashing**: SHA256 + xxHash64 for integrity verification and change detection
- **Incremental Builds**: Only reprocess changed assets
- **Asset Conversion**: Format conversion and optimization (textures, models, materials)
- **Dependency Tracking**: Automatically handle asset dependencies
- **Multi-threaded Processing**: Parallel conversion with configurable thread count
- **Metadata Database**: Track all asset state in persistent JSON database
- **Progress Reporting**: Real-time conversion progress callbacks
- **Integrity Validation**: Detect and handle corrupted assets

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Asset Pipeline                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ AssetPipeline (Main Orchestrator)                │   │
│  │ - Config management                             │   │
│  │ - Job scheduling                                │   │
│  │ - Progress tracking                             │   │
│  └──────────────────────────────────────────────────┘   │
│          │              │              │                │
│          ▼              ▼              ▼                │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ AssetHash   │ │ AssetDatabase│ │AssetConverter│    │
│  │             │ │              │ │              │    │
│  │ • SHA256    │ │ • Track meta │ │ • Convert    │    │
│  │ • xxHash64  │ │ • Dependencies
│ │ • Optimize  │    │
│  │ • Chunked   │ │ • Validate   │ │ • Validate   │    │
│  └─────────────┘ └──────────────┘ └──────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Worker Threads (Parallel Conversion)             │   │
│  │ - Configurable thread pool                       │   │
│  │ - Job queue processing                           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. AssetHash (Cryptographic Hashing)

Provides multiple hashing strategies for different use cases.

**File**: [include/AssetHash.h](include/AssetHash.h)

```cpp
// Compute full hash with SHA256 (integrity checking)
AssetHash::Hash hash = AssetHash::ComputeHash("assets/textures/player.png");

// Quick hash for fast comparisons (xxHash64)
uint64_t quickHash = AssetHash::ComputeQuickHash("assets/models/scene.gltf");

// Detect changes
if (AssetHash::HasFileChanged("assets/material.json", previousHash)) {
    // Asset changed, reprocess
}

// Verify integrity
if (!AssetHash::VerifyIntegrity("assets/audio.wav", storedHash)) {
    // Asset corrupted, reprocess
}

// Chunked hashing for streaming validation
auto chunks = AssetHash::ComputeChunkedHash("assets/video.mp4", 1024*1024);
```

**Hash Structure**:
```cpp
struct Hash {
    std::string sha256;      // Full cryptographic hash
    uint64_t xxHash64;       // Fast hash for comparisons
    std::string timestamp;   // Modification time
    size_t fileSize;         // Asset file size
};
```

### 2. AssetDatabase (Metadata & State Tracking)

Persistent JSON database tracking all asset metadata and processing state.

**File**: [include/AssetDatabase.h](include/AssetDatabase.h)

```cpp
// Initialize database
AssetDatabase db;
db.Initialize("asset_database.json");

// Check if asset needs reprocessing
if (db.IsAssetDirty("models/character.glb")) {
    // Reprocess this asset
}

// Get all dirty assets of specific type
auto dirtyTextures = db.GetDirtyAssets("texture");

// Find dependent assets
auto dependents = db.FindDependents("textures/base_color.png");
for (const auto& dep : dependents) {
    // Mark dependent material as dirty
    db.MarkAssetDirty(dep);
}

// Save state
db.Save();
```

**Asset Entry**:
```cpp
struct AssetEntry {
    std::string path;                           // Relative path
    std::string type;                           // texture, model, shader, etc.
    AssetHash::Hash sourceHash;                 // Source file hash
    AssetHash::Hash processedHash;              // Output file hash
    std::vector<std::string> dependencies;      // Referenced assets
    std::string lastProcessedTime;              // Timestamp
    bool isDirty;                               // Needs reprocessing
    std::map<std::string, std::string> metadata;// Custom data
};
```

### 3. AssetConverter (Format Conversion & Optimization)

Converts assets between formats with optional compression and optimization.

**File**: [include/AssetConverter.h](include/AssetConverter.h)

```cpp
// Convert texture with options
AssetConverter::TextureConversionOptions texOpts;
texOpts.compress = true;
texOpts.maxMipLevels = 12;
texOpts.targetFormat = "dds";  // Use DXT compression
texOpts.maxWidth = 4096;

auto result = AssetConverter::ConvertTexture(
    "assets/textures/raw/diffuse.png",
    "assets_processed/textures/diffuse.dds",
    texOpts);

if (result.success) {
    std::cout << "Compression: " << result.compressionRatio << "x\n";
    std::cout << "Time: " << result.conversionTimeMs << "ms\n";
}

// Convert mesh with optimization
AssetConverter::MeshConversionOptions meshOpts;
meshOpts.optimize = true;
meshOpts.mergeSubmeshes = true;
meshOpts.quantizationBits = 16;

auto meshResult = AssetConverter::ConvertMesh(
    "assets/models/character.fbx",
    "assets_processed/models/character.glb",
    meshOpts);

// Detect asset type from extension
std::string type = AssetConverter::DetectAssetType("model.gltf");  // Returns "model"

// Validate asset integrity
if (!AssetConverter::ValidateAsset("texture.png")) {
    std::cerr << "Asset is corrupted!\n";
}

// Estimate memory when loaded
size_t estimatedMem = AssetConverter::EstimateMemorySize("model.glb");
```

**Conversion Result**:
```cpp
struct ConversionResult {
    bool success;
    std::string outputPath;
    std::string errorMessage;
    size_t inputSize;           // Original file size
    size_t outputSize;          // Processed file size
    float compressionRatio;     // Output/Input
    double conversionTimeMs;    // Processing time
};
```

### 4. AssetPipeline (Main Orchestrator)

High-level pipeline managing the complete build process.

**File**: [include/AssetPipeline.h](include/AssetPipeline.h)

## Usage Examples

### Basic Setup

```cpp
#include "AssetPipeline.h"

int main() {
    // Configure pipeline
    AssetPipeline::Config config;
    config.assetSourceDir = "assets";
    config.assetOutputDir = "assets_processed";
    config.databasePath = "asset_database.json";
    config.maxThreads = 4;
    config.enableCompression = true;
    config.incrementalBuild = true;
    config.verbose = true;

    // Initialize
    auto& pipeline = AssetPipeline::GetInstance();
    if (!pipeline.Initialize(config)) {
        std::cerr << "Failed to initialize asset pipeline\n";
        return 1;
    }

    // Setup progress callback
    pipeline.SetProgressCallback([](float progress, const std::string& desc) {
        printf("Progress: %.1f%% - %s\n", progress * 100, desc.c_str());
    });

    // Scan assets
    if (!pipeline.ScanAssetDirectory(config.assetSourceDir)) {
        std::cerr << "Failed to scan assets\n";
        return 1;
    }

    // Process (only changed assets)
    if (!pipeline.ProcessAssets(true)) {
        std::cerr << "Pipeline processing failed\n";
    }

    // Get statistics
    const auto& stats = pipeline.GetStatistics();
    printf("Processed: %zu/%zu assets\n", stats.processedAssets, stats.totalAssets);
    printf("Compression ratio: %.2fx\n", stats.GetCompressionRatio());

    pipeline.Shutdown();
    return 0;
}
```

### Incremental Build

```cpp
// Only reprocess changed assets
if (pipeline.ProcessAssets(true)) {  // true = incremental
    const auto& stats = pipeline.GetStatistics();
    std::cout << "Skipped: " << stats.skippedAssets << " unchanged assets\n";
}
```

### Force Full Rebuild

```cpp
// Clear cache and rebuild everything
pipeline.ForceFullRebuild();
pipeline.ProcessAssets(false);  // false = full rebuild
```

### Handle Dependencies

```cpp
// When material texture changes, mark material as dirty
auto dependents = pipeline.GetDatabase().FindDependents("textures/base_color.png");
for (const auto& dependent : dependents) {
    pipeline.GetDatabase().MarkAssetDirty(dependent);
}

// Reprocess affected materials
pipeline.ProcessAssets(true);
```

### Check Asset Validity

```cpp
// Validate all assets
auto failedAssets = pipeline.GetDatabase().VerifyIntegrity();
if (!failedAssets.empty()) {
    std::cout << "Corrupted assets detected:\n";
    for (const auto& asset : failedAssets) {
        std::cout << "  - " << asset << "\n";
    }
}
```

## Database Format

The asset database is stored as JSON for easy debugging and manual editing:

```json
{
    "version": "1.0",
    "assets": {
        "textures/diffuse.png": {
            "path": "textures/diffuse.png",
            "type": "texture",
            "sourceHash": "a1b2c3d4e5f6...",
            "processedHash": "f6e5d4c3b2a1...",
            "processedPath": "assets_processed/textures/diffuse.dds",
            "dependencies": ["materials/character.json"],
            "lastProcessedTime": "2025-12-15 10:30:45",
            "isDirty": false,
            "metadata": {
                "maxWidth": "4096",
                "format": "dds"
            }
        },
        "models/character.glb": {
            "path": "models/character.glb",
            "type": "model",
            "sourceHash": "b2c3d4e5f6a1...",
            "processedHash": "e5d4c3b2a1f6...",
            "processedPath": "assets_processed/models/character.glb",
            "dependencies": ["textures/diffuse.png"],
            "lastProcessedTime": "2025-12-15 10:30:50",
            "isDirty": false,
            "metadata": {}
        }
    }
}
```

## Supported Asset Types

| Type | Input Formats | Output Formats | Compression |
|------|---|---|---|
| **Texture** | PNG, JPG, TGA, BMP, HDR, EXR, DDS | PNG, DDS, KTX2, JPG | DXT (BC) |
| **Model** | OBJ, FBX, glTF, GLB, DAE, USDZ | GLB, glTF, OBJ | meshopt |
| **Shader** | GLSL, VERT, FRAG, GEOM, COMP | SPV, GLSL | - |
| **Material** | JSON, YAML | JSON, YAML | - |
| **Audio** | WAV, MP3, FLAC, OGG | WAV, OGG | Vorbis |
| **Scene** | TXT, SCENE, PREFAB | JSON | - |

## Optimization Techniques

### Texture Optimization

```cpp
AssetConverter::TextureConversionOptions opts;
opts.compress = true;              // Enable DXT compression
opts.maxWidth = 2048;              // Limit resolution
opts.maxHeight = 2048;
opts.maxMipLevels = 10;            // Generate mipmaps
opts.generateNormalMap = false;    // Or true to generate from height
opts.sRGB = true;                  // For color data
```

### Mesh Optimization

```cpp
AssetConverter::MeshConversionOptions opts;
opts.optimize = true;              // Optimize vertex data
opts.mergeSubmeshes = true;        // Reduce draw calls
opts.removeDuplicates = true;      // Remove redundant vertices
opts.quantizationBits = 16;        // Reduce precision (16-bit positions)
opts.stripUnusedData = true;       // Remove unused attributes
```

## Performance Considerations

### Thread Count

- **1-2 threads**: Safe for background processing
- **4 threads**: Default, balanced approach
- **8+ threads**: Suitable for high-end hardware with many assets

```cpp
config.maxThreads = std::thread::hardware_concurrency();
```

### Chunk Size for Large Files

```cpp
// Hash large files in chunks to avoid loading into memory
auto chunks = AssetHash::ComputeChunkedHash("video.mp4", 10 * 1024 * 1024);  // 10MB chunks
```

### Cache Strategy

```cpp
// Keep processed assets in cache
config.enableCaching = true;

// Only reprocess if source has changed
config.incrementalBuild = true;
```

## Build Integration

### CMakeLists.txt

The asset pipeline is automatically compiled when including:

```cmake
# Already added to CMakeLists.txt
target_sources(GameEngine PRIVATE
    src/AssetHash.cpp
    src/AssetDatabase.cpp
    src/AssetConverter.cpp
    src/AssetPipeline.cpp
)
```

### Runtime Usage

```cpp
// In Application.cpp or similar
void Application::InitializeAssetPipeline() {
    AssetPipeline::Config config;
    config.assetSourceDir = "assets";
    config.assetOutputDir = "assets/.processed";
    config.databasePath = "assets/.database.json";
    config.maxThreads = 4;
    config.verbose = true;

    auto& pipeline = AssetPipeline::GetInstance();
    pipeline.Initialize(config);
    pipeline.ScanAssetDirectory(config.assetSourceDir);
    
    // Process during loading screen or startup
    pipeline.ProcessAssets(true);  // Incremental
}
```

## Common Patterns

### Full Build on First Run

```cpp
// Check if database exists
if (!std::filesystem::exists("asset_database.json")) {
    pipeline.ProcessAssets(false);  // Full rebuild
}
```

### Watch for Changes (Live Development)

```cpp
// In game loop
if (assetChangeDetected) {
    pipeline.ProcessAssets(true);  // Incremental - very fast
}
```

### Pre-ship Optimization

```cpp
// Before shipping, force full rebuild with max compression
config.enableCompression = true;
config.assetOutputDir = "assets_shipping";
pipeline.Clean();
pipeline.ForceFullRebuild();
pipeline.ProcessAssets(false);
```

## Troubleshooting

### Asset Not Processing

1. Check `AssetDatabase::IsAssetDirty()` - may be clean
2. Verify asset file exists at expected path
3. Check asset type is recognized: `AssetConverter::DetectAssetType()`
4. Force processing: `pipeline.ProcessAsset(path, true)`

### Corruption Detected

```cpp
// Verify all assets
auto failed = pipeline.GetDatabase().VerifyIntegrity();
for (const auto& path : failed) {
    // Reprocess or restore from source control
    pipeline.ProcessAsset(path, true);
}
```

### Out of Memory

- Reduce `maxThreads` (fewer concurrent conversions)
- Process assets in batches: `CleanAssetType()` then process type by type
- Reduce texture resolution with `TextureConversionOptions::maxWidth`

### Slow Builds

- Enable incremental builds: `incrementalBuild = true`
- Increase thread count: `maxThreads = hardware_concurrency()`
- Skip compression for development: `enableCompression = false`

## Future Enhancements

- [ ] Parallel hashing with multiple cores
- [ ] Resume interrupted conversions
- [ ] Distributed builds (network-based processing)
- [ ] GPU-accelerated texture compression
- [ ] Real-time asset reimport in editor
- [ ] Asset profiling and memory tracking
- [ ] Differential compression (delta updates)
- [ ] Cloud asset caching

## References

- [AssetHash.h](include/AssetHash.h) - Hashing implementation
- [AssetDatabase.h](include/AssetDatabase.h) - Metadata storage
- [AssetConverter.h](include/AssetConverter.h) - Format conversion
- [AssetPipeline.h](include/AssetPipeline.h) - Main orchestrator
