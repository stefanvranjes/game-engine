#include "Texture.h"
#include "GLExtensions.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <GLFW/glfw3.h>
#include <iostream>

Texture::Texture() : m_TextureID(0), m_Width(0), m_Height(0), m_Channels(0) {}

Texture::~Texture() {
    if (m_TextureID != 0) {
        glDeleteTextures(1, &m_TextureID);
    }
}

bool Texture::LoadFromFile(const std::string& path) {
    // Load image data using stb_image
    unsigned char* data = stbi_load(path.c_str(), &m_Width, &m_Height, &m_Channels, 0);
    bool usedStbImage = (data != nullptr);
    
    // If loading fails, create a procedural checkerboard texture
    if (!data) {
        std::cerr << "Failed to load texture: " << path << ", using procedural texture" << std::endl;
        m_Width = 64;
        m_Height = 64;
        m_Channels = 3;
        data = new unsigned char[m_Width * m_Height * m_Channels];
        
        // Create red/blue checkerboard pattern
        for (int y = 0; y < m_Height; ++y) {
            for (int x = 0; x < m_Width; ++x) {
                int index = (y * m_Width + x) * m_Channels;
                bool isRed = ((x / 8) + (y / 8)) % 2 == 0;
                data[index + 0] = isRed ? 255 : 0;   // R
                data[index + 1] = 0;                  // G
                data[index + 2] = isRed ? 0 : 255;   // B
            }
        }
    }

    // Determine format
    GLenum format = GL_RGB;
    if (m_Channels == 1) format = GL_RED;
    else if (m_Channels == 3) format = GL_RGB;
    else if (m_Channels == 4) format = GL_RGBA;

    glGenTextures(1, &m_TextureID);
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, format, m_Width, m_Height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Free data based on how it was allocated
    if (usedStbImage) {
        stbi_image_free(data);
    } else {
        delete[] data;
    }
    
    return true;
}

bool Texture::LoadFromData(const unsigned char* data, int width, int height, int channels, bool sRGB) {
    m_Width = width;
    m_Height = height;
    m_Channels = channels;

    GLenum format = GL_RGB;
    GLenum internalFormat = sRGB ? GL_SRGB : GL_RGB;

    if (m_Channels == 1) {
        format = GL_RED;
        internalFormat = GL_RED;
    } else if (m_Channels == 3) {
        format = GL_RGB;
        internalFormat = sRGB ? GL_SRGB : GL_RGB;
    } else if (m_Channels == 4) {
        format = GL_RGBA;
        internalFormat = sRGB ? GL_SRGB_ALPHA : GL_RGBA;
    }

    if (m_TextureID != 0) {
        glDeleteTextures(1, &m_TextureID);
    }

    glGenTextures(1, &m_TextureID);
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_Width, m_Height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

bool Texture::LoadHDR(const std::string& path) {
    stbi_set_flip_vertically_on_load(true);
    float* data = stbi_loadf(path.c_str(), &m_Width, &m_Height, &m_Channels, 0);
    
    if (data) {
        glGenTextures(1, &m_TextureID);
        glBindTexture(GL_TEXTURE_2D, m_TextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_Width, m_Height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }
    
    std::cerr << "Failed to load HDR texture: " << path << std::endl;
    return false;
}

void Texture::Bind(unsigned int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
}
