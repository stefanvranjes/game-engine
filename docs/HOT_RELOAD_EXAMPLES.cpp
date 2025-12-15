// Hot-Reload System - Practical Examples
// This file demonstrates common usage patterns

#include "AssetHotReloadManager.h"
#include "Application.h"
#include "FileWatcher.h"
#include "TextureManager.h"

// ============================================================================
// EXAMPLE 1: Basic Initialization (in Application::Init)
// ============================================================================

void Application::InitializeHotReload() {
    // Create the manager
    m_HotReloadManager = std::make_unique<AssetHotReloadManager>();
    
    // Initialize with renderer and texture manager references
    m_HotReloadManager->Initialize(
        m_Renderer.get(),
        m_Renderer->GetTextureManager()
    );
    
    // Enable hot-reload (can also be toggled via ImGui)
    m_HotReloadManager->SetEnabled(true);
    
    // Watch the main shader directory
    m_HotReloadManager->WatchShaderDirectory("shaders/");
    
    // Watch texture directories
    m_HotReloadManager->WatchTextureDirectory("assets/");
    m_HotReloadManager->WatchTextureDirectory("assets/textures/");
    
    std::cout << "Hot-reload system initialized" << std::endl;
}

// ============================================================================
// EXAMPLE 2: Update Loop Integration (in Application::Update)
// ============================================================================

void Application::UpdateHotReload(float deltaTime) {
    // Call this once per frame to check for file changes
    if (m_HotReloadManager) {
        m_HotReloadManager->Update();
    }
}

// ============================================================================
// EXAMPLE 3: Register Callbacks for Asset Changes
// ============================================================================

void Application::SetupReloadCallbacks() {
    // Get notified when any asset is reloaded
    m_HotReloadManager->SetOnAssetReloaded([this](const std::string& type, const std::string& path) {
        if (type == "shader") {
            std::cout << "✓ Shader reloaded: " << path << std::endl;
            // Optional: Trigger UI refresh
            OnShaderReloaded(path);
        }
        else if (type == "texture") {
            std::cout << "✓ Texture reloaded: " << path << std::endl;
            // Optional: Update material references
            OnTextureReloaded(path);
        }
    });
}

void Application::OnShaderReloaded(const std::string& path) {
    // Custom logic when shader is reloaded
    // Update any dependent systems here
}

void Application::OnTextureReloaded(const std::string& path) {
    // Custom logic when texture is reloaded
    // Update materials or refresh preview
}

// ============================================================================
// EXAMPLE 4: Watch Custom Directories
// ============================================================================

void Application::WatchCustomDirectories() {
    if (!m_HotReloadManager) return;
    
    // Watch additional shader directories
    m_HotReloadManager->WatchShaderDirectory("my_custom_shaders/");
    m_HotReloadManager->WatchShaderDirectory("assets/shaders/");
    
    // Watch additional texture directories
    m_HotReloadManager->WatchTextureDirectory("content/textures/");
    m_HotReloadManager->WatchTextureDirectory("user_content/assets/");
}

// ============================================================================
// EXAMPLE 5: Monitor Hot-Reload Status
// ============================================================================

void Application::PrintHotReloadStats() {
    if (!m_HotReloadManager) return;
    
    std::cout << "=== Hot-Reload Status ===" << std::endl;
    std::cout << "Enabled: " << (m_HotReloadManager->IsEnabled() ? "Yes" : "No") << std::endl;
    std::cout << "Watched Files: " << m_HotReloadManager->GetWatchedFileCount() << std::endl;
    std::cout << "Total Reloads: " << m_HotReloadManager->GetReloadCount() << std::endl;
}

// ============================================================================
// EXAMPLE 6: File Watcher - Low Level API
// ============================================================================

void FileWatcherExample() {
    FileWatcher watcher;
    
    // Example 1: Watch a single file
    auto handle1 = watcher.WatchFile("shaders/main.frag", [](const std::string& path) {
        std::cout << "File changed: " << path << std::endl;
    });
    
    // Example 2: Watch a directory for specific file type
    auto handle2 = watcher.WatchDirectory("assets/textures/", ".png", [](const std::string& path) {
        std::cout << "PNG texture changed: " << path << std::endl;
    });
    
    // Example 3: In game loop
    for (int i = 0; i < 1000; ++i) {
        watcher.Update(100);  // Check for changes every 100ms
        // Game logic here
    }
    
    // Example 4: Stop watching
    watcher.Unwatch(handle1);
    watcher.Unwatch(handle2);
    watcher.Clear();  // Clear all watches
}

// ============================================================================
// EXAMPLE 7: TextureManager Hot-Reload - Low Level API
// ============================================================================

void TextureManagerExample(TextureManager* texMgr) {
    // Enable hot-reload on the texture manager
    texMgr->SetHotReloadEnabled(true);
    
    // Watch a texture directory
    texMgr->WatchTextureDirectory("assets/textures/");
    
    // Register callback for texture reloads
    texMgr->SetOnTextureReloaded([](const std::string& path) {
        std::cout << "Texture reloaded via TextureManager: " << path << std::endl;
    });
    
    // In game loop, call Update to process reloads
    // (Already called by TextureManager::Update)
}

// ============================================================================
// EXAMPLE 8: ImGui Panel for Hot-Reload Control
// ============================================================================

void RenderHotReloadPanel() {
    ImGui::Begin("Asset Hot-Reload Control");
    
    if (ImGui::Button("Enable All", ImVec2(100, 0))) {
        m_HotReloadManager->SetEnabled(true);
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Disable All", ImVec2(100, 0))) {
        m_HotReloadManager->SetEnabled(false);
    }
    
    ImGui::Separator();
    
    // Display statistics
    ImGui::Text("Watched: %zu files", m_HotReloadManager->GetWatchedFileCount());
    ImGui::Text("Reloaded: %u times", m_HotReloadManager->GetReloadCount());
    
    ImGui::Separator();
    
    // Reset counter
    if (ImGui::Button("Reset Counter", ImVec2(100, 0))) {
        m_HotReloadManager->ResetReloadCount();
    }
    
    ImGui::End();
}

// ============================================================================
// EXAMPLE 9: Practical Shader Development Workflow
// ============================================================================

/*
Workflow:

1. Open shader file:
   shaders/pbr_lighting.frag

2. Edit the shader:
   uniform vec3 u_AlbedoColor;
   
   void main() {
       vec3 finalColor = u_AlbedoColor;
       // Add more calculations...
       gl_FragColor = vec4(finalColor, 1.0);
   }

3. Save the file
   → File watcher detects change
   → Shader automatically recompiles
   → Changes appear instantly in viewport
   → No engine restart needed!

4. If there's an error:
   → Console shows: "Shader compilation failed"
   → Previous shader remains active
   → Fix the error and save again

5. Iterate quickly:
   → Change values
   → Save
   → See result
   → Repeat
*/

// ============================================================================
// EXAMPLE 10: Practical Texture Development Workflow
// ============================================================================

/*
Workflow:

1. Open texture in external editor:
   → Photoshop / GIMP / Paint.NET
   → File: assets/textures/material_pbr.png

2. Edit the texture:
   → Adjust colors, add details
   → Change metallic/roughness values

3. Save the file
   → File watcher detects change
   → TextureManager auto-reloads
   → New texture appears in viewport
   → All materials using it update instantly

4. Iterate quickly:
   → Make adjustments
   → Save in editor
   → Check result in engine
   → No export/import needed!

5. Compare different versions:
   → Keep backup copies
   → Swap between versions by renaming
   → Hot-reload picks up each change
*/

// ============================================================================
// EXAMPLE 11: Error Recovery Pattern
// ============================================================================

class ShaderDevelopmentSession {
public:
    ShaderDevelopmentSession(AssetHotReloadManager* mgr) : m_Manager(mgr) {
        // Track reload history
        m_Manager->SetOnAssetReloaded([this](const std::string& type, const std::string& path) {
            if (type == "shader") {
                m_LastReloadPath = path;
                m_ReloadCount++;
                std::cout << "Reload #" << m_ReloadCount << ": " << path << std::endl;
            }
        });
    }
    
    void PrintStats() const {
        std::cout << "Session Stats:" << std::endl;
        std::cout << "  Total reloads: " << m_ReloadCount << std::endl;
        std::cout << "  Last reload: " << m_LastReloadPath << std::endl;
    }
    
private:
    AssetHotReloadManager* m_Manager;
    std::string m_LastReloadPath;
    int m_ReloadCount = 0;
};

// ============================================================================
// EXAMPLE 12: Multi-File Editing
// ============================================================================

/*
You can edit multiple files simultaneously:

1. Edit shader:
   → shaders/lighting.frag
   → Save

2. Edit material texture:
   → assets/textures/surface.png
   → Save

3. Both updates happen instantly:
   → Shader recompiles
   → Texture reloads
   → Combined effect visible immediately

4. No coordination needed:
   → Each file is monitored independently
   → Changes propagate instantly
   → Perfect for rapid prototyping
*/

// ============================================================================
// EXAMPLE 13: Performance Optimization
// ============================================================================

void OptimizeHotReload() {
    if (!m_HotReloadManager) return;
    
    // For slower systems, disable hot-reload
    m_HotReloadManager->SetEnabled(false);
    
    // Or selectively watch only critical directories
    m_HotReloadManager->Clear();
    m_HotReloadManager->WatchShaderDirectory("shaders/");  // Only shaders
    // Don't watch textures to reduce polling overhead
}

// ============================================================================
// EXAMPLE 14: Integration with Editor Tools
// ============================================================================

class ShaderEditorPanel {
public:
    ShaderEditorPanel(AssetHotReloadManager* mgr) : m_Manager(mgr) {}
    
    void Render() {
        ImGui::Begin("Shader Editor");
        
        // Show hot-reload status
        bool enabled = m_Manager->IsEnabled();
        if (ImGui::Checkbox("Hot Reload", &enabled)) {
            m_Manager->SetEnabled(enabled);
        }
        
        ImGui::Separator();
        
        // Show watched shaders
        ImGui::Text("Watched Files: %zu", m_Manager->GetWatchedFileCount());
        ImGui::Text("Reloads: %u", m_Manager->GetReloadCount());
        
        ImGui::Separator();
        
        // Quick edit hints
        ImGui::TextWrapped("Tip: Edit shaders and save to see changes instantly!");
        
        ImGui::End();
    }
    
private:
    AssetHotReloadManager* m_Manager;
};

// ============================================================================
// EXAMPLE 15: Production vs Development Mode
// ============================================================================

void SetupHotReload(bool isDevelopmentMode) {
    m_HotReloadManager = std::make_unique<AssetHotReloadManager>();
    m_HotReloadManager->Initialize(m_Renderer.get(), m_Renderer->GetTextureManager());
    
    if (isDevelopmentMode) {
        // Development: Enable hot-reload with full monitoring
        m_HotReloadManager->SetEnabled(true);
        m_HotReloadManager->WatchShaderDirectory("shaders/");
        m_HotReloadManager->WatchTextureDirectory("assets/");
        std::cout << "Hot-reload ENABLED for development" << std::endl;
    }
    else {
        // Production: Disable hot-reload for safety
        m_HotReloadManager->SetEnabled(false);
        std::cout << "Hot-reload DISABLED for production" << std::endl;
    }
}

// ============================================================================
// QUICK REFERENCE
// ============================================================================

/*
Key Methods:

FileWatcher:
  - WatchFile(path, callback)
  - WatchDirectory(directory, extension, callback)
  - Update(pollIntervalMs)
  - Unwatch(handle)
  - Clear()

TextureManager:
  - SetHotReloadEnabled(enabled)
  - WatchTextureDirectory(directory)
  - SetOnTextureReloaded(callback)

AssetHotReloadManager:
  - Initialize(renderer, textureManager)
  - SetEnabled(enabled)
  - WatchShaderDirectory(directory)
  - WatchTextureDirectory(directory)
  - Update()
  - SetOnAssetReloaded(callback)
  - GetWatchedFileCount()
  - GetReloadCount()
  - ResetReloadCount()
  - IsEnabled()
  - Clear()

Supported Extensions:
  Shaders: .glsl, .vert, .frag, .geom, .comp, .tese, .tesc
  Textures: .png, .jpg, .jpeg, .bmp, .tga, .hdr, .exr
*/
