#include "Texture.h"
#include "GLExtensions.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>

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
    , m_AnisotropyLevel(1.0f)
    , m_DownscaleLevel(0)
    , m_GenerateMipmaps(true)
{
    Touch();
}

Texture::Texture(unsigned int id, int width, int height, int channels)
    : m_TextureID(id)
    , m_Width(width)
    , m_Height(height)
    , m_Channels(channels)
    , m_State(TextureState::Loaded) // Assume loaded since we have an ID
    , m_LocalBuffer(nullptr)
    , m_LastUsedTime(0.0)
    , m_IsHDR(false)
    , m_LocalBufferHDR(nullptr)
    , m_AnisotropyLevel(1.0f)
    , m_DownscaleLevel(0)
    , m_GenerateMipmaps(true)
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

void Texture::SetAnisotropy(float level) {
    m_AnisotropyLevel = level;
}

void Texture::SetDownscaleLevel(int level) {
    m_DownscaleLevel = level;
}

void Texture::GenerateMipmaps(bool enable) {
    m_GenerateMipmaps = enable;
}


// Helper for software downscaling (Box Filter)
unsigned char* DownscaleImage(unsigned char* data, int& width, int& height, int channels, int levels) {
    if (levels <= 0 || !data) return data;

    unsigned char* currentData = data;
    int currentW = width;
    int currentH = height;

    for (int l = 0; l < levels; ++l) {
        if (currentW <= 1 || currentH <= 1) break;

        int newW = currentW / 2;
        int newH = currentH / 2;
        unsigned char* newData = (unsigned char*)malloc(newW * newH * channels);

        for (int y = 0; y < newH; ++y) {
            for (int x = 0; x < newW; ++x) {
                for (int c = 0; c < channels; ++c) {
                    // Average 2x2 block
                    int sum = 0;
                    sum += currentData[((y * 2) * currentW + (x * 2)) * channels + c];
                    sum += currentData[((y * 2) * currentW + (x * 2) + 1) * channels + c];
                    sum += currentData[((y * 2 + 1) * currentW + (x * 2)) * channels + c];
                    sum += currentData[((y * 2 + 1) * currentW + (x * 2) + 1) * channels + c];
                    newData[(y * newW + x) * channels + c] = (unsigned char)(sum / 4);
                }
            }
        }

        // Free old data if it wasn't the original buffer passed in (handled by caller logic usually, 
        // but here we need to be careful. The first iteration 'currentData' is 'data' which is stbi_malloc'd.
        // Subsequent iterations 'currentData' is malloc'd.
        // We should probably free 'currentData' if it's not the initial 'data', OR just rely on the fact 
        // that we will replace m_LocalBuffer with the result.
        // Actually, let's assume we free the input 'currentData' and return 'newData'.
        // BUT the first input is m_LocalBuffer which is stbi_malloc. We can use stbi_image_free on it?
        // Or just standard free? stbi uses malloc usually.
        
        if (l == 0) {
            stbi_image_free(currentData); // Free original stbi buffer
        } else {
            free(currentData); // Free intermediate buffer
        }

        currentData = newData;
        currentW = newW;
        currentH = newH;
    }

    width = currentW;
    height = currentH;
    return currentData;
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
        // Apply software downscaling if requested
        if (m_DownscaleLevel > 0 && m_LocalBuffer) {
            m_LocalBuffer = DownscaleImage(m_LocalBuffer, m_Width, m_Height, m_Channels, m_DownscaleLevel);
        }

        // Determine format
        GLenum format = GL_RGB;
        if (m_Channels == 1) format = GL_RED;
        else if (m_Channels == 3) format = GL_RGB;
        else if (m_Channels == 4) format = GL_RGBA;

        glGenTextures(1, &m_TextureID);
        glBindTexture(GL_TEXTURE_2D, m_TextureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, m_Width, m_Height, 0, format, GL_UNSIGNED_BYTE, m_LocalBuffer);
        
        if (m_GenerateMipmaps) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        if (m_GenerateMipmaps) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        } else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        
        // Apply Anisotropy (disabled - GLEW extension constants not available)
        // TODO: Define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT and GL_TEXTURE_MAX_ANISOTROPY_EXT in GLExtensions.h
        /*
        if (GLEW_EXT_texture_filter_anisotropic && m_AnisotropyLevel > 1.0f) {
            float maxAnisotropy;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(m_AnisotropyLevel, maxAnisotropy));
        }
        */
        
        // If we downscaled, m_LocalBuffer is now a malloc'd buffer, not stbi.
        // DownscaleImage handles the freeing of the original stbi buffer.
        // So we just need to free m_LocalBuffer using free() if it was downscaled, or stbi_free if not?
        // DownscaleImage frees the input and returns new malloc'd data.
        // So if m_DownscaleLevel > 0, m_LocalBuffer is malloc'd.
        // If m_DownscaleLevel == 0, m_LocalBuffer is stbi_malloc'd.
        // This is messy. Let's make DownscaleImage always return a buffer we can free with stbi_image_free?
        // No, stbi_image_free is just free() usually but we shouldn't rely on it.
        // Better: Check flag.
        
        if (m_DownscaleLevel > 0) {
            free(m_LocalBuffer);
        } else {
            stbi_image_free(m_LocalBuffer);
        }
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
