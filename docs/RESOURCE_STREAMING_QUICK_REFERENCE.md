# Resource Streaming & Virtual Filesystem - Quick Reference

## Core Classes

### VirtualFileSystem
```cpp
VirtualFileSystem vfs;

// Mount providers
vfs.Mount("/assets", std::make_shared<PhysicalFileSystemProvider>("./assets"));
vfs.Mount("/pak0", std::make_shared<AssetPackageProvider>(package));

// Read files
std::vector<uint8_t> data = vfs.ReadFile("/assets/models/player.gltf");
bool exists = vfs.FileExists("/assets/texture.png");
size_t size = vfs.GetFileSize("/assets/texture.png");

// Directory operations
auto files = vfs.ListDirectory("/assets");
auto mounts = vfs.GetMountPoints();
```

### ResourceStreamingManager
```cpp
ResourceStreamingManager streaming;
streaming.Initialize(&vfs, 4);  // 4 worker threads
streaming.SetMemoryBudget(512 * 1024 * 1024);  // 512 MB

// Request loads
auto resource = std::make_shared<MyResource>("path.txt");
streaming.RequestLoad(resource, ResourcePriority::High, 
    [](bool success) { /* handle result */ });

// Unload
streaming.RequestUnload(resource);
streaming.UnloadAll();

// Update (call once per frame)
streaming.Update(deltaTime);

// Monitor
auto stats = streaming.GetStatistics();
```

### AssetPackage
```cpp
// Build package
{
    AssetPackageBuilder builder;
    builder.AddDirectory("./assets", "/", "*");
    builder.Build("game.pak");
}

// Load and mount
{
    auto pkg = std::make_shared<AssetPackage>();
    pkg->Load("game.pak");
    vfs.Mount("/pak", std::make_shared<AssetPackageProvider>(pkg));
}

// Direct access
auto data = pkg->ExtractAsset("/models/player.gltf");
bool has = pkg->HasAsset("/models/player.gltf");
auto assets = pkg->ListAssets();
```

## File System Providers

### PhysicalFileSystemProvider
Maps physical directory to virtual path:
```cpp
auto phys = std::make_shared<PhysicalFileSystemProvider>("./assets");
vfs.Mount("/assets", phys);
```

### MemoryFileSystemProvider
In-memory filesystem for testing:
```cpp
auto mem = std::make_shared<MemoryFileSystemProvider>();
mem->AddFile("/test.txt", data, size);
vfs.Mount("/memory", mem);
```

### AssetPackageProvider
Access packaged assets:
```cpp
auto pkg = std::make_shared<AssetPackage>();
pkg->Load("game.pak");
auto provider = std::make_shared<AssetPackageProvider>(pkg);
vfs.Mount("/pak", provider);
```

## Priority Levels

```cpp
enum class ResourcePriority {
    Critical = 0,   // Load immediately
    High = 1,       // Load soon
    Normal = 2,     // Standard
    Low = 3,        // Low priority
    Deferred = 4    // Background
};
```

## Resource States

```cpp
enum class ResourceState {
    Unloaded = 0,
    Loading = 1,
    Loaded = 2,
    Unloading = 3,
    Failed = 4
};
```

## Creating Custom Resources

```cpp
class MyResource : public Resource {
public:
    MyResource(const std::string& path) : Resource(path) {}
    
    bool OnLoadComplete(const std::vector<uint8_t>& data) override {
        // Parse data, create GPU resources
        m_State = ResourceState::Loaded;
        m_MemoryUsage = CalculateSize();
        return true;
    }
    
    void OnUnload() override {
        // Free GPU resources
        m_State = ResourceState::Unloaded;
    }
    
    // Your API
    MyData* GetData() { return m_Data; }
    
private:
    MyData* m_Data = nullptr;
};
```

## Common Patterns

### Load Single Resource
```cpp
auto resource = std::make_shared<ModelResource>("player.gltf");
streaming.RequestLoad(resource, ResourcePriority::Critical,
    [resource](bool success) {
        if (success) {
            auto model = resource->GetData();
            // Use model...
        }
    });
```

### Batch Load Related Assets
```cpp
std::vector<std::string> assets = {
    "player.gltf",
    "player_anim.anim",
    "player_texture.png"
};
streaming.PreloadResources(assets, ResourcePriority::High);
```

### Monitor Memory
```cpp
auto stats = streaming.GetStatistics();
float usage_pct = 100.0f * stats.totalLoadedMemory / stats.memoryBudget;
std::cout << "Memory: " << usage_pct << "%" << std::endl;
```

### Create Package
```cpp
AssetPackageBuilder builder;
builder.AddDirectory("./assets/models", "/models", "*.gltf");
builder.AddDirectory("./assets/textures", "/textures", "*.png");
builder.SetCompression(AssetPackage::CompressionType::LZ4);
builder.Build("game.pak");
```

## Memory Management

```cpp
// Set budget (auto-evicts LRU when exceeded)
streaming.SetMemoryBudget(256 * 1024 * 1024);  // 256 MB

// Set frame time limit
streaming.SetFrameTimeLimit(5.0f);  // Max 5ms per frame

// Disable/enable loading
streaming.SetLoadingEnabled(false);
streaming.SetLoadingEnabled(true);

// Unload everything
streaming.UnloadAll();
```

## Statistics

```cpp
struct StreamingStatistics {
    size_t totalLoadedMemory;      // Current usage
    size_t peakMemoryUsage;        // Highest seen
    size_t memoryBudget;           // Limit
    uint32_t resourcesLoaded;      // Success count
    uint32_t resourcesFailed;      // Failure count
    uint32_t pendingRequests;      // Queue size
    float averageLoadTime;         // ms
    float bytesPerSecond;          // Throughput
};
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Files not found | Check mount points: `vfs.GetMountPoints()` |
| Memory exceeded | Reduce `SetMemoryBudget()` or increase RAM |
| Slow loading | Increase workers: `Initialize(&vfs, 8)` |
| Long frame times | Reduce `SetFrameTimeLimit()` |
| Package corrupted | Verify with `pkg->GetAssetCount()` |
| Missing resources | List directory: `vfs.ListDirectory(dir)` |

## Platform Considerations

### Windows
```cpp
vfs.Mount("/assets", 
    std::make_shared<PhysicalFileSystemProvider>("C:\\game\\assets"));
```

### Linux/macOS
```cpp
vfs.Mount("/assets",
    std::make_shared<PhysicalFileSystemProvider>("./assets"));
```

### Mobile (Assets in APK/IPA)
```cpp
// Mount package instead of directory
auto pkg = std::make_shared<AssetPackage>();
pkg->Load("/assets/game.pak");
vfs.Mount("/", std::make_shared<AssetPackageProvider>(pkg));
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Load latency | < 100ms |
| Frame time impact | < 5ms |
| Throughput | > 50 MB/s |
| Memory efficiency | > 85% utilization |
| Package overhead | < 2% |

## See Also

- [RESOURCE_STREAMING_GUIDE.md](RESOURCE_STREAMING_GUIDE.md) - Detailed guide
- [TextureManager.h](../include/TextureManager.h) - Integration example
- [VirtualFileSystem.h](../include/VirtualFileSystem.h) - API reference
- [ResourceStreamingManager.h](../include/ResourceStreamingManager.h) - API reference
- [AssetPackage.h](../include/AssetPackage.h) - API reference
