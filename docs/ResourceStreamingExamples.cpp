/**
 * @file ResourceStreamingExamples.cpp
 * @brief Comprehensive examples for Resource Streaming & VFS
 * 
 * This file demonstrates various use cases for the resource streaming
 * and virtual filesystem systems.
 */

#include "VirtualFileSystem.h"
#include "ResourceStreamingManager.h"
#include "AssetPackage.h"
#include "Texture.h"
#include "GLTFLoader.h"

// ============================================================================
// Example 1: Basic VFS Setup
// ============================================================================

void Example_BasicVFSSetup() {
    // Create virtual filesystem
    VirtualFileSystem vfs;
    
    // Mount physical directories with different priorities
    vfs.Mount("/assets", 
        std::make_shared<PhysicalFileSystemProvider>("./assets"));
    vfs.Mount("/userdata", 
        std::make_shared<PhysicalFileSystemProvider>("./userdata"));
    vfs.Mount("/cache", 
        std::make_shared<PhysicalFileSystemProvider>("./cache"));
    
    // Mount in-memory filesystem for temporary data
    auto memFS = std::make_shared<MemoryFileSystemProvider>();
    vfs.Mount("/temp", memFS);
    
    // Read files transparently
    if (vfs.FileExists("/assets/models/player.gltf")) {
        auto data = vfs.ReadFile("/assets/models/player.gltf");
        std::cout << "Loaded " << data.size() << " bytes" << std::endl;
    }
    
    // List available files
    auto textures = vfs.ListDirectory("/assets/textures");
    for (const auto& tex : textures) {
        std::cout << "Texture: " << tex << " (" 
                  << vfs.GetFileSize("/assets/textures/" + tex) << " bytes)" << std::endl;
    }
}

// ============================================================================
// Example 2: Asset Package Creation and Usage
// ============================================================================

void Example_AssetPackageWorkflow() {
    std::cout << "=== Asset Package Example ===" << std::endl;
    
    // Step 1: Build package from game assets
    {
        std::cout << "Building package..." << std::endl;
        
        AssetPackageBuilder builder;
        builder.AddDirectory("./assets/models", "/models", "*.gltf");
        builder.AddDirectory("./assets/textures", "/textures", "*.png");
        builder.AddDirectory("./assets/audio", "/audio", "*.wav");
        builder.AddDirectory("./assets/shaders", "/shaders", "*.glsl");
        
        builder.SetCompression(AssetPackage::CompressionType::LZ4);
        
        if (builder.Build("./dist/game.pak")) {
            auto stats = builder.GetStats();
            std::cout << "Package created:" << std::endl;
            std::cout << "  Files: " << stats.fileCount << std::endl;
            std::cout << "  Size: " << stats.compressedSize / (1024*1024) << " MB" << std::endl;
            std::cout << "  Compression: " << (stats.compressionRatio * 100) << "%" << std::endl;
        }
    }
    
    // Step 2: Load and mount package
    {
        std::cout << "Loading package..." << std::endl;
        
        auto package = std::make_shared<AssetPackage>();
        if (package->Load("./dist/game.pak")) {
            // Display contents
            auto assets = package->ListAssets();
            std::cout << "Package contains " << assets.size() << " assets" << std::endl;
            
            // Mount it in VFS
            VirtualFileSystem vfs;
            vfs.Mount("/game", std::make_shared<AssetPackageProvider>(package));
            
            // Access assets
            if (vfs.FileExists("/game/models/player.gltf")) {
                auto data = vfs.ReadFile("/game/models/player.gltf");
                std::cout << "Loaded player model: " << data.size() << " bytes" << std::endl;
            }
        }
    }
}

// ============================================================================
// Example 3: Custom Resource Type
// ============================================================================

class TextureResource : public Resource {
public:
    TextureResource(const std::string& path)
        : Resource(path), m_Texture(nullptr) {}
    
    ~TextureResource() override {
        OnUnload();
    }
    
    bool OnLoadComplete(const std::vector<uint8_t>& data) override {
        std::cout << "Loading texture: " << m_Path << std::endl;
        
        // Create texture
        m_Texture = new Texture();
        
        // Load from data (would be implemented in Texture class)
        // bool success = m_Texture->LoadFromData(data.data(), 0, 0, 0);
        
        if (m_Texture) {
            m_State = ResourceState::Loaded;
            m_MemoryUsage = data.size() + 1024 * 1024; // Rough estimate
            return true;
        }
        
        m_State = ResourceState::Failed;
        return false;
    }
    
    void OnUnload() override {
        if (m_Texture) {
            delete m_Texture;
            m_Texture = nullptr;
        }
        m_State = ResourceState::Unloaded;
    }
    
    Texture* GetTexture() const { return m_Texture; }

private:
    Texture* m_Texture;
};

class ModelResource : public Resource {
public:
    ModelResource(const std::string& path)
        : Resource(path), m_GameObject(nullptr) {}
    
    ~ModelResource() override {
        OnUnload();
    }
    
    bool OnLoadComplete(const std::vector<uint8_t>& data) override {
        std::cout << "Loading model: " << m_Path << std::endl;
        
        // Would load glTF or other format
        // m_GameObject = GLTFLoader::Load(m_Path, nullptr);
        
        if (m_GameObject) {
            m_State = ResourceState::Loaded;
            m_MemoryUsage = data.size() + 5 * 1024 * 1024; // Estimate with GPU memory
            return true;
        }
        
        m_State = ResourceState::Failed;
        return false;
    }
    
    void OnUnload() override {
        if (m_GameObject) {
            // Cleanup
            m_GameObject = nullptr;
        }
        m_State = ResourceState::Unloaded;
    }
    
    std::shared_ptr<class GameObject> GetGameObject() const { return m_GameObject; }

private:
    std::shared_ptr<class GameObject> m_GameObject;
};

// ============================================================================
// Example 4: Resource Streaming in Game Scene
// ============================================================================

class GameScene {
public:
    void Initialize(const std::string& assetDirectory) {
        // Setup VFS
        m_VFS = std::make_unique<VirtualFileSystem>();
        m_VFS->Mount("/assets", 
            std::make_shared<PhysicalFileSystemProvider>(assetDirectory));
        
        // Mount package if it exists
        auto package = std::make_shared<AssetPackage>();
        if (package->Load(assetDirectory + "/data.pak")) {
            m_VFS->Mount("/pak", 
                std::make_shared<AssetPackageProvider>(package));
        }
        
        // Setup streaming manager
        m_StreamingManager = std::make_unique<ResourceStreamingManager>();
        m_StreamingManager->Initialize(m_VFS.get(), 4);
        m_StreamingManager->SetMemoryBudget(512 * 1024 * 1024);
        m_StreamingManager->SetFrameTimeLimit(5.0f);
        
        // Load critical assets
        LoadCriticalAssets();
    }
    
    void LoadCriticalAssets() {
        std::cout << "Loading critical assets..." << std::endl;
        
        // Load player model
        auto playerModel = std::make_shared<ModelResource>("player.gltf");
        m_StreamingManager->RequestLoad(
            playerModel,
            ResourcePriority::Critical,
            [this, playerModel](bool success) {
                if (success) {
                    std::cout << "Player model loaded!" << std::endl;
                    // m_Player = playerModel->GetGameObject();
                } else {
                    std::cerr << "Failed to load player model!" << std::endl;
                }
            }
        );
        
        // Load UI textures
        std::vector<std::string> uiTextures = {
            "ui_button.png",
            "ui_panel.png",
            "ui_font.png"
        };
        
        for (const auto& tex : uiTextures) {
            auto resource = std::make_shared<TextureResource>(tex);
            m_StreamingManager->RequestLoad(resource, ResourcePriority::High);
        }
    }
    
    void LoadNearbyAssets(const glm::vec3& playerPos) {
        std::cout << "Streaming nearby assets..." << std::endl;
        
        // Load assets near player with appropriate priority
        std::vector<std::string> nearbyModels = {
            "rock_1.gltf",
            "tree_oak.gltf",
            "grass_clump.gltf"
        };
        
        for (const auto& model : nearbyModels) {
            auto resource = std::make_shared<ModelResource>(model);
            m_StreamingManager->RequestLoad(resource, ResourcePriority::Normal);
        }
    }
    
    void Update(float deltaTime) {
        // Update streaming system
        m_StreamingManager->Update(deltaTime);
        
        // Monitor memory usage
        static float statsTimer = 0.0f;
        statsTimer += deltaTime;
        
        if (statsTimer > 1.0f) {
            statsTimer = 0.0f;
            PrintStatistics();
        }
    }
    
    void PrintStatistics() {
        auto stats = m_StreamingManager->GetStatistics();
        
        float memUsageMB = stats.totalLoadedMemory / (1024.0f * 1024.0f);
        float budgetMB = stats.memoryBudget / (1024.0f * 1024.0f);
        float usage = 100.0f * stats.totalLoadedMemory / stats.memoryBudget;
        
        std::cout << "Streaming Stats:" << std::endl;
        std::cout << "  Memory: " << memUsageMB << " MB / " 
                  << budgetMB << " MB (" << usage << "%)" << std::endl;
        std::cout << "  Loaded: " << stats.resourcesLoaded << std::endl;
        std::cout << "  Failed: " << stats.resourcesFailed << std::endl;
        std::cout << "  Pending: " << stats.pendingRequests << std::endl;
        std::cout << "  Avg Load Time: " << stats.averageLoadTime << " ms" << std::endl;
    }
    
    void Shutdown() {
        std::cout << "Shutting down scene..." << std::endl;
        m_StreamingManager->UnloadAll();
    }

private:
    std::unique_ptr<VirtualFileSystem> m_VFS;
    std::unique_ptr<ResourceStreamingManager> m_StreamingManager;
    // std::shared_ptr<GameObject> m_Player;
};

// ============================================================================
// Example 5: Async File Loading
// ============================================================================

void Example_AsyncFileLoading() {
    VirtualFileSystem vfs;
    vfs.Mount("/assets", 
        std::make_shared<PhysicalFileSystemProvider>("./assets"));
    
    // Read file asynchronously
    vfs.ReadFileAsync("/assets/config.json",
        [](const std::vector<uint8_t>& data, bool success) {
            if (success) {
                std::cout << "Config loaded: " << data.size() << " bytes" << std::endl;
            } else {
                std::cerr << "Failed to load config" << std::endl;
            }
        }
    );
    
    // Process other things while loading happens in background
}

// ============================================================================
// Example 6: Memory Management
// ============================================================================

void Example_MemoryManagement() {
    VirtualFileSystem vfs;
    vfs.Mount("/assets", 
        std::make_shared<PhysicalFileSystemProvider>("./assets"));
    
    ResourceStreamingManager streaming;
    streaming.Initialize(&vfs);
    
    // Set tight memory budget (will trigger LRU eviction)
    streaming.SetMemoryBudget(64 * 1024 * 1024); // 64 MB
    
    // Set frame time limit to prevent stuttering
    streaming.SetFrameTimeLimit(3.0f); // 3ms per frame
    
    // Simulate frame updates
    for (int frame = 0; frame < 100; ++frame) {
        // ... game logic ...
        
        // Update streaming (calls LRU eviction if needed)
        streaming.Update(0.016f); // 60 FPS
        
        // Check if memory is being managed
        auto stats = streaming.GetStatistics();
        if (frame % 10 == 0) {
            std::cout << "Frame " << frame 
                      << " - Memory: " << (stats.totalLoadedMemory / (1024*1024)) 
                      << " MB" << std::endl;
        }
    }
}

// ============================================================================
// Main - Run examples
// ============================================================================

void RunResourceStreamingExamples() {
    std::cout << "Resource Streaming Examples\n" << std::endl;
    
    std::cout << "1. Basic VFS Setup\n";
    Example_BasicVFSSetup();
    std::cout << std::endl;
    
    std::cout << "2. Asset Package Workflow\n";
    Example_AssetPackageWorkflow();
    std::cout << std::endl;
    
    std::cout << "3. Async File Loading\n";
    Example_AsyncFileLoading();
    std::cout << std::endl;
    
    std::cout << "4. Memory Management\n";
    Example_MemoryManagement();
    std::cout << std::endl;
}

// To integrate with your application:
// 1. Call RunResourceStreamingExamples() from main or a test function
// 2. Create GameScene instance and use it in your game loop
// 3. Customize resource types for your asset formats
// 4. Adjust memory budgets based on platform capabilities
