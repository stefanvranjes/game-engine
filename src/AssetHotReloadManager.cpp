#include "AssetHotReloadManager.h"
#include "Renderer.h"
#include "TextureManager.h"
#include "Shader.h"
#include <iostream>
#include <algorithm>

AssetHotReloadManager::AssetHotReloadManager() {
    m_FileWatcher = std::make_unique<FileWatcher>();
}

AssetHotReloadManager::~AssetHotReloadManager() {
    Clear();
}

void AssetHotReloadManager::Initialize(Renderer* renderer, TextureManager* textureManager) {
    m_Renderer = renderer;
    m_TextureManager = textureManager;
}

void AssetHotReloadManager::SetEnabled(bool enabled) {
    m_Enabled = enabled;
    
    if (enabled) {
        std::cout << "AssetHotReloadManager: Hot-reload ENABLED" << std::endl;
        if (m_TextureManager) {
            m_TextureManager->SetHotReloadEnabled(true);
        }
    } else {
        std::cout << "AssetHotReloadManager: Hot-reload DISABLED" << std::endl;
        if (m_TextureManager) {
            m_TextureManager->SetHotReloadEnabled(false);
        }
        if (m_FileWatcher) {
            m_FileWatcher->Clear();
        }
    }
}

void AssetHotReloadManager::WatchShaderDirectory(const std::string& shaderDirectory) {
    if (!m_Enabled || !m_FileWatcher) return;
    
    // Watch for shader file extensions
    static const std::vector<std::string> extensions = {
        ".glsl", ".vert", ".frag", ".geom", ".comp", ".tese", ".tesc"
    };
    
    for (const auto& ext : extensions) {
        m_FileWatcher->WatchDirectory(shaderDirectory, ext, [this](const std::string& path) {
            OnShaderChanged(path);
        });
    }
    
    std::cout << "AssetHotReloadManager: Watching shader directory: " << shaderDirectory << std::endl;
}

void AssetHotReloadManager::WatchTextureDirectory(const std::string& textureDirectory) {
    if (!m_TextureManager || !m_Enabled) return;
    
    m_TextureManager->WatchTextureDirectory(textureDirectory);
}

void AssetHotReloadManager::Update() {
    if (!m_Enabled || !m_FileWatcher) return;
    
    m_FileWatcher->Update(1000);

}

size_t AssetHotReloadManager::GetWatchedFileCount() const {
    if (!m_FileWatcher) return 0;
    return m_FileWatcher->GetWatchedFiles().size();
}

void AssetHotReloadManager::Clear() {
    if (m_FileWatcher) {
        m_FileWatcher->Clear();
    }
}

void AssetHotReloadManager::OnShaderChanged(const std::string& path) {
    if (!m_Renderer) return;
    
    std::cout << "AssetHotReloadManager: Shader changed, reloading all shaders: " << path << std::endl;
    
    // The Renderer::UpdateShaders() method checks each shader's timestamp
    // and reloads if needed
    m_Renderer->UpdateShaders();
    
    m_ReloadCount++;
    
    if (m_OnAssetReloaded) {
        m_OnAssetReloaded("shader", path);
    }
}

void AssetHotReloadManager::OnTextureChanged(const std::string& path) {
    std::cout << "AssetHotReloadManager: Texture changed: " << path << std::endl;
    
    m_ReloadCount++;
    
    if (m_OnAssetReloaded) {
        m_OnAssetReloaded("texture", path);
    }
}
