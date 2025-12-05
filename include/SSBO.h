#pragma once

#include "GLExtensions.h"
#include <vector>

// Shader Storage Buffer Object wrapper class
class SSBO {
public:
    SSBO();
    ~SSBO();

    // Create SSBO with specified size and usage
    bool Create(size_t sizeBytes, GLenum usage = GL_DYNAMIC_DRAW);
    
    // Bind to a specific binding point
    void Bind(unsigned int bindingPoint) const;
    void Unbind() const;
    
    // Upload data to GPU
    void Upload(const void* data, size_t sizeBytes, size_t offset = 0);
    
    // Download data from GPU
    void Download(void* data, size_t sizeBytes, size_t offset = 0) const;
    
    // Map buffer for direct access
    void* Map(GLenum access = GL_READ_WRITE);
    void Unmap();
    
    // Get buffer properties
    unsigned int GetID() const { return m_BufferID; }
    size_t GetSize() const { return m_SizeBytes; }
    bool IsValid() const { return m_BufferID != 0; }
    
    // Resize buffer (destroys existing data)
    bool Resize(size_t newSizeBytes);
    
    // Clear buffer to zero
    void Clear();

private:
    unsigned int m_BufferID;
    size_t m_SizeBytes;
    GLenum m_Usage;
    bool m_IsMapped;
};
