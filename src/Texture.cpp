#include "Texture.h"
#include "GLExtensions.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <GLFW/glfw3.h>
#include <iostream>

Texture::Texture() 
    : m_TextureID(0)
    , m_Width(0)
    , m_Height(0)
    , m_Channels(0)
    , m_State(TextureState::Unloaded)
    , m_LocalBuffer(nullptr)
    , m_LastUsedTime(0.0)
    , m_IsHDR(false)
    , m_LocalBufferHDR(nullptr)
{
    Touch();
}

Texture::~Texture() {
    Unload();
    if (m_LocalBuffer) {
        stbi_image_free(m_LocalBuffer);
    }
    if (m_LocalBufferHDR) {
        stbi_image_free(m_LocalBufferHDR);
    }
}

void Texture::Touch() {
    m_LastUsedTime = glfwGetTime();
}

void Texture::Unload() {
    if (m_TextureID != 0) {
        glDeleteTextures(1, &m_TextureID);
        m_TextureID = 0;
    }
    m_State = TextureState::Unloaded;
    m_Width = 0;
    m_Height = 0;
    m_Channels = 0;
}

void Texture::LoadDataAsync(const std::string& path) {
    if (m_State != TextureState::Unloaded) return;
    
    m_State = TextureState::Loading;
    
    // Check if HDR
    if (path.find(".hdr") != std::string::npos) {
        m_IsHDR = true;
        stbi_set_flip_vertically_on_load(true);
        m_LocalBufferHDR = stbi_loadf(path.c_str(), &m_Width, &m_Height, &m_Channels, 0);
        if (!m_LocalBufferHDR) {
            std::cerr << "Failed to load HDR texture async: " << path << std::endl;
            m_State = TextureState::Error;
        }
    } else {
        m_IsHDR = false;
        // Standard image
        m_LocalBuffer = stbi_load(path.c_str(), &m_Width, &m_Height, &m_Channels, 0);
        if (!m_LocalBuffer) {
            std::cerr << "Failed to load texture async: " << path << std::endl;
            m_State = TextureState::Error;
        }
    }
}

bool Texture::UploadToGPU() {
    if (m_State != TextureState::Loading) return false;
    
    // If loading failed on thread
    if ((!m_IsHDR && !m_LocalBuffer) || (m_IsHDR && !m_LocalBufferHDR)) {
        m_State = TextureState::Error;
        return false;
    }
    
    if (m_IsHDR) {
        glGenTextures(1, &m_TextureID);
        glBindTexture(GL_TEXTURE_2D, m_TextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_Width, m_Height, 0, GL_RGB, GL_FLOAT, m_LocalBufferHDR);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        stbi_image_free(m_LocalBufferHDR);
        m_LocalBufferHDR = nullptr;
    } else {
        // Determine format
        GLenum format = GL_RGB;
        if (m_Channels == 1) format = GL_RED;
        else if (m_Channels == 3) format = GL_RGB;
        else if (m_Channels == 4) format = GL_RGBA;

        glGenTextures(1, &m_TextureID);
        glBindTexture(GL_TEXTURE_2D, m_TextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, m_Width, m_Height, 0, format, GL_UNSIGNED_BYTE, m_LocalBuffer);
        glGenerateMipmap(GL_TEXTURE_2D);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        stbi_image_free(m_LocalBuffer);
        m_LocalBuffer = nullptr;
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
    m_State = TextureState::Loaded;
    Touch();
    return true;
}

bool Texture::LoadFromFile(const std::string& path) {
    // Legacy blocking load - implemented via async methods for consistency
    LoadDataAsync(path);
    if (m_State == TextureState::Error) return false;
    return UploadToGPU();
}

bool Texture::LoadFromData(const unsigned char* data, int width, int height, int channels, bool sRGB) {
    Unload(); // Clear existing
    
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

    glGenTextures(1, &m_TextureID);
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_Width, m_Height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    m_State = TextureState::Loaded;
    Touch();
    return true;
}

bool Texture::LoadHDR(const std::string& path) {
    // Legacy blocking load
    LoadDataAsync(path);
    if (m_State == TextureState::Error) return false;
    return UploadToGPU();
}

void Texture::Bind(unsigned int unit) {
    Touch(); // Mark as used
    glActiveTexture(GL_TEXTURE0 + unit);
    
    if (m_State == TextureState::Loaded && m_TextureID != 0) {
        glBindTexture(GL_TEXTURE_2D, m_TextureID);
    } else {
        // Bind 0 or a placeholder if not loaded
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

size_t Texture::GetMemorySize() const {
    if (m_State != TextureState::Loaded) return 0;
    // Estimate: width * height * channels * bytes_per_channel * mips(approx 1.33)
    size_t baseSize = m_Width * m_Height * m_Channels * (m_IsHDR ? 4 : 1);
    return (size_t)(baseSize * 1.33);
}
