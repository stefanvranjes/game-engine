#pragma once

#include "Texture.h"
#include <map>
#include <string>
#include <memory>

class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    // Load texture from file, or return existing if already loaded
    std::shared_ptr<Texture> LoadTexture(const std::string& path);

    // Get existing texture, returns nullptr if not found
    std::shared_ptr<Texture> GetTexture(const std::string& path);

    // Clear all textures
    void Clear();

private:
    std::map<std::string, std::shared_ptr<Texture>> m_Textures;
};
