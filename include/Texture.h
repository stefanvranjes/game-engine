#pragma once

#include <string>

class Texture {
public:
    Texture();
    ~Texture();

    // Load image from file and create OpenGL texture
    bool LoadFromFile(const std::string& path);
    bool LoadHDR(const std::string& path);
    bool LoadFromData(const unsigned char* data, int width, int height, int channels, bool sRGB = false);

    // Bind texture to specified texture unit (default 0)
    void Bind(unsigned int unit = 0) const;

    // Get OpenGL texture ID
    unsigned int GetID() const { return m_TextureID; }

private:
    unsigned int m_TextureID;
    int m_Width;
    int m_Height;
    int m_Channels;
};
