# Asset Pipeline - Quick Reference

## Quick Start

```cpp
#include "AssetPipeline.h"

// Initialize pipeline
AssetPipeline::Config config;
config.assetSourceDir = "assets";
config.assetOutputDir = "assets_processed";
config.databasePath = "asset_database.json";
config.maxThreads = 4;

auto& pipeline = AssetPipeline::GetInstance();
pipeline.Initialize(config);

// Scan and process
pipeline.ScanAssetDirectory(config.assetSourceDir);
pipeline.ProcessAssets(true);  // true = incremental only

// Check results
const auto& stats = pipeline.GetStatistics();
std::cout << "Processed: " << stats.processedAssets << " assets\n";

pipeline.Shutdown();
```

## Common Operations

### Check if Asset Changed
```cpp
auto hash1 = AssetHash::ComputeHash("texture.png");
// ... later ...
if (AssetHash::HasFileChanged("texture.png", hash1)) {
    // Reprocess
}
```

### Get Only Dirty Assets
```cpp
auto dirtyAssets = pipeline.GetDatabase().GetDirtyAssets();
auto dirtyTextures = pipeline.GetDatabase().GetDirtyAssets("texture");
```

### Mark Asset for Reprocessing
```cpp
pipeline.GetDatabase().MarkAssetDirty("textures/diffuse.png");
```

### Find Dependent Assets
```cpp
auto deps = pipeline.GetDatabase().FindDependents("textures/base_color.png");
```

### Rebuild Everything
```cpp
pipeline.Clean();
pipeline.ProcessAssets(false);  // false = full rebuild
```

### Convert Single Texture
```cpp
auto result = AssetConverter::ConvertTexture(
    "source.png",
    "output.dds",
    AssetConverter::TextureConversionOptions()
);
```

### Validate Asset Integrity
```cpp
auto corrupted = pipeline.GetDatabase().VerifyIntegrity();
for (const auto& asset : corrupted) {
    // Reprocess
}
```

## Configuration Options

```cpp
AssetPipeline::Config config;
config.assetSourceDir = "assets";          // Input directory
config.assetOutputDir = "assets_processed";// Output directory
config.databasePath = "asset_database.json";// Metadata file
config.maxThreads = 4;                     // Conversion threads
config.enableCompression = true;           // Compress assets
config.enableCaching = true;               // Cache processed assets
config.validateAssets = true;              // Verify integrity
config.incrementalBuild = true;            // Only changed assets
config.verbose = true;                     // Debug output
```

## Asset Types

- **texture**: PNG, JPG, TGA, BMP, HDR, EXR
- **model**: OBJ, FBX, glTF, GLB, DAE
- **shader**: GLSL, VERT, FRAG, GEOM, COMP
- **material**: JSON, YAML
- **audio**: WAV, MP3, FLAC, OGG
- **scene**: TXT, SCENE, PREFAB

## Texture Options

```cpp
AssetConverter::TextureConversionOptions opts;
opts.compress = true;              // Enable DXT compression
opts.maxMipLevels = 12;            // Generate mipmaps
opts.maxWidth = 4096;              // Max resolution
opts.maxHeight = 4096;
opts.generateNormalMap = false;    // Generate from height
opts.sRGB = true;                  // Color space
opts.targetFormat = "dds";         // Output format
opts.compressionQuality = 0.95f;   // Quality (0.0-1.0)
```

## Mesh Options

```cpp
AssetConverter::MeshConversionOptions opts;
opts.optimize = true;              // Optimize vertices
opts.mergeSubmeshes = true;        // Merge meshes
opts.removeDuplicates = true;      // Remove duplicates
opts.generateNormals = false;      // Generate if missing
opts.generateTangents = true;      // Generate tangents
opts.quantizationBits = 16;        // Vertex precision
opts.stripUnusedData = true;       // Remove unused data
opts.targetFormat = "glb";         // Output format
```

## Statistics

```cpp
const auto& stats = pipeline.GetStatistics();
stats.totalAssets;           // Total assets
stats.processedAssets;       // Processed count
stats.failedAssets;          // Failed count
stats.skippedAssets;         // Skipped (unchanged)
stats.totalInputSize;        // Input bytes
stats.totalOutputSize;       // Output bytes
stats.GetCompressionRatio(); // Output/Input ratio
stats.GetProgress();          // 0.0-1.0
```

## Hash Functions

```cpp
// Full hash (SHA256)
auto hash = AssetHash::ComputeHash("file.png");
std::string hashStr = AssetHash::ToString(hash);

// Quick hash (xxHash64)
uint64_t quick = AssetHash::ComputeQuickHash("file.png");

// Chunked hash (for large files)
auto chunks = AssetHash::ComputeChunkedHash("large.bin", 1024*1024);

// Verify integrity
bool ok = AssetHash::VerifyIntegrity("file.png", previousHash);

// Check if changed
bool changed = AssetHash::HasFileChanged("file.png", oldHash);
```

## Progress Callback

```cpp
pipeline.SetProgressCallback([](float progress, const std::string& desc) {
    printf("%.0f%% - %s\n", progress * 100, desc.c_str());
});
```

## Database Operations

```cpp
auto& db = pipeline.GetDatabase();

// Get asset info
auto entry = db.GetAssetEntry("textures/diffuse.png");

// Check if dirty
bool dirty = db.IsAssetDirty("models/character.glb");

// Mark clean/dirty
db.MarkAssetClean("textures/normal.png");
db.MarkAssetDirty("textures/diffuse.png");

// Get by type
auto textures = db.GetAssetsByType("texture");
auto models = db.GetAssetsByType("model");

// Find dependents
auto deps = db.FindDependents("textures/base.png");

// Add dependency
db.AddDependency("material.json", "textures/base.png");

// Verify integrity
auto failed = db.VerifyIntegrity();

// Save/Load
db.Initialize("path.json");
db.Save();
```

## Pipeline Workflow

### Startup (First Run)
```cpp
pipeline.Initialize(config);
pipeline.ScanAssetDirectory(config.assetSourceDir);
pipeline.ProcessAssets(false);  // Full rebuild
```

### Development (Iteration)
```cpp
pipeline.ProcessAssets(true);   // Only changed assets (very fast)
```

### Pre-ship (Optimization)
```cpp
config.enableCompression = true;
pipeline.ProcessAssets(false);  // Full rebuild with compression
```

### Cleanup
```cpp
pipeline.Clean();               // Clear all processed assets
pipeline.CleanAssetType("texture");  // Clear specific type
```

## Error Handling

```cpp
// Check initialization
if (!pipeline.Initialize(config)) {
    // Handle error
}

// Check scan result
if (!pipeline.ScanAssetDirectory(dir)) {
    // Handle error
}

// Check conversion result
auto result = AssetConverter::ConvertTexture(src, dst);
if (!result.success) {
    std::cerr << result.errorMessage << "\n";
}

// Check integrity
auto failed = pipeline.GetDatabase().VerifyIntegrity();
if (!failed.empty()) {
    // Handle corrupted assets
}
```

## Performance Tips

1. **Use incremental builds** in development
2. **Increase thread count** for faster processing
3. **Disable compression** during development
4. **Enable caching** to skip unchanged assets
5. **Use smaller textures** in development
6. **Batch process** large asset collections
7. **Validate integrity** before shipping

## Debugging

```cpp
// Enable verbose logging
config.verbose = true;

// Get detailed statistics
const auto& stats = pipeline.GetStatistics();
std::cout << "Total time: " << stats.totalTimeMs << "ms\n";
std::cout << "Compression: " << stats.GetCompressionRatio() << "x\n";

// Check what's dirty
auto dirty = pipeline.GetDatabase().GetDirtyAssets();
for (const auto& path : dirty) {
    std::cout << "Dirty: " << path << "\n";
}

// Verify specific asset
if (!AssetConverter::ValidateAsset("asset.png")) {
    std::cout << "Asset is invalid!\n";
}
```
