#pragma once

#include <string>

enum class TextureState {
    Unloaded,
    Loading,
    Loaded,
    Error
};

class Texture {
public:
    Texture();
    ~Texture();

    // Load image from file (Blocking - Legacy)
    bool LoadFromFile(const std::string& path);
    
    // Async Loading Interface
    void LoadDataAsync(const std::string& path); // Called on worker thread
    bool UploadToGPU();                          // Called on main thread
    void Unload();                               // Free GPU resources
    
    bool LoadHDR(const std::string& path);
    bool LoadFromData(const unsigned char* data, int width, int height, int channels, bool sRGB = false);

    // Bind texture to specified texture unit (default 0)
    void Bind(unsigned int unit = 0);

    // Get OpenGL texture ID
    unsigned int GetID() const { return m_TextureID; }
    
    // State & LRU
    TextureState GetState() const { return m_State; }
    void Touch(); // Update last used time
    double GetLastUsedTime() const { return m_LastUsedTime; }
    size_t GetMemorySize() const; // Estimate GPU memory usage

private:
    unsigned int m_TextureID;
    int m_Width;
    int m_Height;
    int m_Channels;
    
    // Async loading state
    TextureState m_State;
    unsigned char* m_LocalBuffer; // Temporary buffer for async loading
    double m_LastUsedTime;
    bool m_IsHDR;
    float* m_LocalBufferHDR; // Temporary buffer for HDR async loading
};
