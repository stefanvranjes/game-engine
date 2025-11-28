#include "TextureManager.h"
#include <iostream>

TextureManager::TextureManager() {
}

TextureManager::~TextureManager() {
    Clear();
}

std::shared_ptr<Texture> TextureManager::LoadTexture(const std::string& path) {
    // Check if texture is already loaded
    auto it = m_Textures.find(path);
    if (it != m_Textures.end()) {
        return it->second;
    }

    // Load new texture
    std::shared_ptr<Texture> texture = std::make_shared<Texture>();
    if (texture->LoadFromFile(path)) {
        m_Textures[path] = texture;
        std::cout << "Texture loaded: " << path << std::endl;
        return texture;
    }

    std::cerr << "Failed to load texture: " << path << std::endl;
    return nullptr;
}

std::shared_ptr<Texture> TextureManager::GetTexture(const std::string& path) {
    auto it = m_Textures.find(path);
    if (it != m_Textures.end()) {
        return it->second;
    }
    return nullptr;
}

void TextureManager::Clear() {
    m_Textures.clear();
}
