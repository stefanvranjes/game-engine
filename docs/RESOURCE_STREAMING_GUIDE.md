# Resource Streaming & Virtual Filesystem Guide

## Overview

The game engine now includes a comprehensive resource streaming and virtual filesystem system designed for efficient asset management in large-scale games. This system provides:

- **Virtual Filesystem (VFS)**: Unified interface for multiple file storage backends
- **Resource Streaming Manager**: Priority-based async loading with memory budgeting
- **Asset Packages**: Binary container format for efficient asset distribution
- **Memory Management**: LRU eviction and frame-time limiting
- **Extensibility**: Pluggable file system providers for custom backends

## Architecture

### Virtual Filesystem

The VFS abstracts away file storage details, allowing your game to load assets from multiple sources transparently:

```
┌─────────────────────────────────────────┐
│      Application Code                    │
│   vfs.ReadFile("/models/player.gltf")   │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼──────────────┐
        │  VirtualFileSystem  │
        │  (Path resolution)  │
        └──────┬──────────────┘
               │
     ┌─────────┼─────────┬──────────────┐
     │         │         │              │
┌────▼─┐  ┌────▼─┐  ┌───▼────┐  ┌─────▼───┐
│/     │  │/pak0 │  │/cache  │  │/memory  │
│      │  │      │  │        │  │         │
│Phys. │  │Asset │  │Physical│  │Memory   │
│Files │  │Pkg   │  │Files   │  │Files    │
└──────┘  └──────┘  └────────┘  └─────────┘
```

### Resource Streaming Pipeline

```
Request → Priority Queue → Worker Threads → VFS I/O → Load Completion → Register → Cache
   ↓                            ↓                                           ↓
Priority based            Multi-threaded                            Memory budgeting
```

## Usage Examples

### Basic VFS Setup

```cpp
#include "VirtualFileSystem.h"

// Create and configure VFS
VirtualFileSystem vfs;

// Mount physical directory
vfs.Mount("/assets", std::make_shared<PhysicalFileSystemProvider>("./assets"));

// Mount in-memory filesystem (for testing)
auto memFS = std::make_shared<MemoryFileSystemProvider>();
memFS->AddFile("/test.txt", (uint8_t*)"Hello", 5);
vfs.Mount("/memory", memFS);

// Access files transparently
if (vfs.FileExists("/assets/textures/brick.png")) {
    auto data = vfs.ReadFile("/assets/textures/brick.png");
    // Process data...
}

// List directory
auto files = vfs.ListDirectory("/assets/models");
for (const auto& file : files) {
    std::cout << "Found: " << file << std::endl;
}
```

### Asset Packages

```cpp
#include "AssetPackage.h"
#include "VirtualFileSystem.h"

// Build a package from directory
{
    AssetPackageBuilder builder;
    builder.AddDirectory("./assets/models", "/models", "*.gltf");
    builder.AddDirectory("./assets/textures", "/textures", "*.png");
    builder.SetCompression(AssetPackage::CompressionType::LZ4);
    builder.Build("./dist/game.pak");
    
    auto stats = builder.GetStats();
    std::cout << "Packaged " << stats.fileCount << " files" << std::endl;
    std::cout << "Compression ratio: " << stats.compressionRatio << std::endl;
}

// Mount and use package
{
    auto package = std::make_shared<AssetPackage>();
    if (package->Load("./dist/game.pak")) {
        VirtualFileSystem vfs;
        vfs.Mount("/pak0", std::make_shared<AssetPackageProvider>(package));
        
        // Transparent access to packaged assets
        auto modelData = vfs.ReadFile("/pak0/models/player.gltf");
    }
}
```

### Resource Streaming

Custom resource types inherit from `Resource`:

```cpp
#include "ResourceStreamingManager.h"
#include "VirtualFileSystem.h"

// Define custom resource type
class ModelResource : public Resource {
public:
    ModelResource(const std::string& path)
        : Resource(path), m_Model(nullptr) {}
    
    bool OnLoadComplete(const std::vector<uint8_t>& data) override {
        // Parse data and create GPU resources
        m_Model = LoadGLTF(data.data(), data.size());
        m_State = m_Model ? ResourceState::Loaded : ResourceState::Failed;
        m_MemoryUsage = EstimateMemoryUsage();
        return m_State == ResourceState::Loaded;
    }
    
    void OnUnload() override {
        if (m_Model) {
            FreeGLTFModel(m_Model);
            m_Model = nullptr;
        }
        m_State = ResourceState::Unloaded;
    }
    
    Model* GetModel() const { return m_Model; }

private:
    Model* m_Model;
    
    size_t EstimateMemoryUsage() const {
        if (!m_Model) return 0;
        // Calculate actual GPU memory used
        return 1024 * 1024; // 1 MB estimate for demo
    }
};

// Usage in game
class GameScene {
private:
    ResourceStreamingManager m_StreamingMgr;
    VirtualFileSystem m_VFS;
    
public:
    void Init() {
        m_StreamingMgr.Initialize(&m_VFS, 4); // 4 worker threads
        m_StreamingMgr.SetMemoryBudget(512 * 1024 * 1024); // 512 MB
        m_StreamingMgr.SetFrameTimeLimit(5.0f); // 5ms per frame for loading
        
        // Mount default assets
        m_VFS.Mount("/assets", 
            std::make_shared<PhysicalFileSystemProvider>("./assets"));
    }
    
    void LoadPlayerModel() {
        auto playerModel = std::make_shared<ModelResource>("player.gltf");
        
        m_StreamingMgr.RequestLoad(
            playerModel,
            ResourcePriority::Critical,
            [this, playerModel](bool success) {
                if (success) {
                    auto* model = dynamic_cast<ModelResource*>(playerModel.get());
                    SpawnPlayer(model->GetModel());
                } else {
                    std::cerr << "Failed to load player model" << std::endl;
                }
            }
        );
    }
    
    void LoadNearbyObjects(const glm::vec3& position, float radius) {
        // Preload objects nearby with normal priority
        std::vector<std::string> nearbyAssets = {
            "rock_1.gltf",
            "rock_2.gltf",
            "tree.gltf"
        };
        
        m_StreamingMgr.PreloadResources(nearbyAssets, ResourcePriority::Normal);
    }
    
    void Update(float deltaTime) {
        m_StreamingMgr.Update(deltaTime);
        
        // Monitor memory usage
        auto stats = m_StreamingMgr.GetStatistics();
        if (stats.totalLoadedMemory > stats.memoryBudget * 0.9f) {
            std::cout << "Memory warning: " 
                      << stats.totalLoadedMemory / (1024*1024) << " MB / "
                      << stats.memoryBudget / (1024*1024) << " MB" << std::endl;
        }
        
        // Print stats every 60 frames
        static int frameCount = 0;
        if (++frameCount % 60 == 0) {
            std::cout << "Loaded resources: " << stats.resourcesLoaded << std::endl;
            std::cout << "Failed: " << stats.resourcesFailed << std::endl;
            std::cout << "Average load time: " << stats.averageLoadTime << "ms" << std::endl;
            std::cout << "Pending: " << stats.pendingRequests << std::endl;
        }
    }
    
    void Shutdown() {
        m_StreamingMgr.UnloadAll();
    }
};
```

### Integration with TextureManager

```cpp
// Extend TextureManager to use VFS
class TextureManager {
private:
    VirtualFileSystem* m_VFS;
    
public:
    void Initialize(VirtualFileSystem* vfs) {
        m_VFS = vfs;
    }
    
    std::shared_ptr<Texture> LoadTexture(const std::string& path) {
        auto texture = std::make_shared<Texture>();
        
        // Load via VFS instead of direct file access
        auto data = m_VFS->ReadFile(path);
        if (!data.empty()) {
            texture->LoadFromData(data.data(), 0, 0, 0);
        }
        
        return texture;
    }
};
```

## Memory Management

The streaming manager enforces memory budgets through LRU (Least Recently Used) eviction:

```cpp
// Set memory budget
streaming.SetMemoryBudget(512 * 1024 * 1024); // 512 MB

// Set per-frame load time limit
streaming.SetFrameTimeLimit(5.0f); // Max 5ms of I/O per frame

// When budget exceeded, oldest unused resources are evicted
streaming.Update(deltaTime); // Call once per frame
```

### Statistics

```cpp
auto stats = streaming.GetStatistics();

// Available metrics:
stats.totalLoadedMemory;      // Current memory used
stats.peakMemoryUsage;        // Highest usage seen
stats.memoryBudget;           // Configured limit
stats.resourcesLoaded;        // Count of successful loads
stats.resourcesFailed;        // Count of failed loads
stats.pendingRequests;        // Currently queued requests
stats.averageLoadTime;        // Avg load time in ms
stats.bytesPerSecond;         // Throughput metric
```

## Priority Levels

Control loading order with priority levels:

```cpp
enum class ResourcePriority {
    Critical = 0,   // Player character, active UI
    High = 1,       // Nearby objects, active audio
    Normal = 2,     // Standard priority
    Low = 3,        // Distant objects
    Deferred = 4    // Background loading, preemptable
};

// Critical resources load first
auto player = std::make_shared<ModelResource>("player.gltf");
streaming.RequestLoad(player, ResourcePriority::Critical);

// Distant objects can wait
auto distant = std::make_shared<ModelResource>("mountain.gltf");
streaming.RequestLoad(distant, ResourcePriority::Low);
```

## Performance Tips

### 1. Preload Critical Assets
```cpp
// Load essential assets early
std::vector<std::string> criticalAssets = {
    "player.gltf",
    "ui_background.png",
    "menu_font.fnt"
};
streaming.PreloadResources(criticalAssets, ResourcePriority::Critical);
```

### 2. Use Asset Packages for Distribution
- Reduces file count and download size
- Faster enumeration and loading
- Easier version management

### 3. Set Appropriate Memory Budgets
```cpp
// Estimate based on target platform
size_t platformRAM = GetSystemRAM();
size_t vramBudget = platformRAM * 0.25; // 25% for streaming
streaming.SetMemoryBudget(vramBudget);
```

### 4. Monitor Frame Time
```cpp
// Never let asset loading stall frames
streaming.SetFrameTimeLimit(3.0f); // 3ms per frame
```

### 5. Use Priority Appropriately
```cpp
// Load path-critical resources first
streaming.RequestLoad(uiResource, ResourcePriority::Critical);

// Background loads don't block gameplay
streaming.RequestLoad(fxResource, ResourcePriority::Deferred);
```

## Extending VFS

Create custom providers for specialized storage:

```cpp
class CloudStorageProvider : public IFileSystemProvider {
public:
    bool ReadFile(const std::string& path, uint8_t*& data, size_t& size) override {
        // Fetch from cloud storage (S3, Azure, etc.)
        auto response = FetchFromCloud(path);
        size = response.size();
        data = new uint8_t[size];
        std::memcpy(data, response.data(), size);
        return true;
    }
    
    void ReadFileAsync(
        const std::string& path,
        std::function<void(uint8_t*, size_t, bool)> callback) override {
        // Async cloud fetch
        std::thread([this, path, callback]() {
            uint8_t* data = nullptr;
            size_t size = 0;
            bool success = ReadFile(path, data, size);
            callback(data, size, success);
        }).detach();
    }
    
    // ... implement other IFileSystemProvider methods
};

// Use custom provider
VirtualFileSystem vfs;
vfs.Mount("/cloud", std::make_shared<CloudStorageProvider>());
```

## Troubleshooting

### Assets not loading
```cpp
// Check VFS mount points
auto mounts = vfs.GetMountPoints();
for (const auto& mount : mounts) {
    std::cout << "Mounted: " << mount << std::endl;
}

// Verify paths
if (!vfs.FileExists("/assets/models/player.gltf")) {
    std::cout << "File not found!" << std::endl;
    auto files = vfs.ListDirectory("/assets/models");
    std::cout << "Available files:" << std::endl;
    for (const auto& f : files) std::cout << "  " << f << std::endl;
}
```

### Memory budget exceeded
```cpp
// Reduce budget or increase platform resources
auto stats = streaming.GetStatistics();
if (stats.totalLoadedMemory > stats.memoryBudget) {
    std::cout << "Memory overcommitted!" << std::endl;
    
    // Reduce loading rate
    streaming.SetFrameTimeLimit(2.0f); // Load slower
    
    // Disable non-critical loading
    streaming.SetLoadingEnabled(false);
}
```

### Slow loading
```cpp
// Increase worker threads
streaming.Initialize(&vfs, 8); // More threads

// Use asset packages for faster access
auto pkg = std::make_shared<AssetPackage>();
pkg->Load("game.pak");
vfs.Mount("/pak0", std::make_shared<AssetPackageProvider>(pkg));

// Increase time budget
streaming.SetFrameTimeLimit(10.0f); // 10ms per frame
```

## Best Practices

1. **Mount filesystems in order of priority** - Higher priority mounts first
2. **Use appropriate compression** - LZ4 for fast decompression, Deflate for size
3. **Batch preloads** - Load multiple related assets together
4. **Profile memory usage** - Monitor statistics during gameplay
5. **Stream LOD models** - Load detailed models only when needed
6. **Cache hot assets** - Keep frequently accessed assets in memory
7. **Async load everything** - Never block the render thread on I/O
8. **Provide user feedback** - Show loading progress to players

## Integration Checklist

- [ ] Add VirtualFileSystem to Application initialization
- [ ] Update AssetManager/TextureManager to use VFS
- [ ] Configure ResourceStreamingManager with appropriate budgets
- [ ] Mount physical asset directories
- [ ] Create asset packages for production
- [ ] Set up logging for streaming statistics
- [ ] Add loading progress UI
- [ ] Profile memory usage on target platforms
- [ ] Test edge cases (out of memory, missing files)
- [ ] Document custom resource types
