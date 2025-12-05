#include "SSBO.h"
#include <iostream>
#include <cstring>

SSBO::SSBO() : m_BufferID(0), m_SizeBytes(0), m_Usage(GL_DYNAMIC_DRAW), m_IsMapped(false) {
}

SSBO::~SSBO() {
    if (m_BufferID != 0) {
        if (m_IsMapped) {
            Unmap();
        }
        glDeleteBuffers(1, &m_BufferID);
    }
}

bool SSBO::Create(size_t sizeBytes, GLenum usage) {
    if (!glBindBufferBase) {
        std::cerr << "SSBO not supported (OpenGL 4.3+ required)" << std::endl;
        return false;
    }
    
    if (m_BufferID != 0) {
        glDeleteBuffers(1, &m_BufferID);
    }
    
    glGenBuffers(1, &m_BufferID);
    if (m_BufferID == 0) {
        std::cerr << "Failed to create SSBO" << std::endl;
        return false;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeBytes, nullptr, usage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    m_SizeBytes = sizeBytes;
    m_Usage = usage;
    m_IsMapped = false;
    
    return true;
}

void SSBO::Bind(unsigned int bindingPoint) const {
    if (m_BufferID == 0) {
        std::cerr << "Cannot bind invalid SSBO" << std::endl;
        return;
    }
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, m_BufferID);
}

void SSBO::Unbind() const {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SSBO::Upload(const void* data, size_t sizeBytes, size_t offset) {
    if (m_BufferID == 0) {
        std::cerr << "Cannot upload to invalid SSBO" << std::endl;
        return;
    }
    
    if (offset + sizeBytes > m_SizeBytes) {
        std::cerr << "Upload size exceeds buffer size" << std::endl;
        return;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, sizeBytes, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void SSBO::Download(void* data, size_t sizeBytes, size_t offset) const {
    if (m_BufferID == 0) {
        std::cerr << "Cannot download from invalid SSBO" << std::endl;
        return;
    }
    
    if (offset + sizeBytes > m_SizeBytes) {
        std::cerr << "Download size exceeds buffer size" << std::endl;
        return;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    
    // Map buffer for reading
    void* bufferData = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (bufferData) {
        memcpy(data, static_cast<char*>(bufferData) + offset, sizeBytes);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    } else {
        std::cerr << "Failed to map SSBO for download" << std::endl;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void* SSBO::Map(GLenum access) {
    if (m_BufferID == 0) {
        std::cerr << "Cannot map invalid SSBO" << std::endl;
        return nullptr;
    }
    
    if (m_IsMapped) {
        std::cerr << "SSBO already mapped" << std::endl;
        return nullptr;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    void* ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
    
    if (ptr) {
        m_IsMapped = true;
    } else {
        std::cerr << "Failed to map SSBO" << std::endl;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    
    return ptr;
}

void SSBO::Unmap() {
    if (!m_IsMapped) {
        return;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    m_IsMapped = false;
}

bool SSBO::Resize(size_t newSizeBytes) {
    if (m_BufferID == 0) {
        return Create(newSizeBytes, m_Usage);
    }
    
    if (m_IsMapped) {
        Unmap();
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BufferID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, newSizeBytes, nullptr, m_Usage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    m_SizeBytes = newSizeBytes;
    return true;
}

void SSBO::Clear() {
    if (m_BufferID == 0) {
        return;
    }
    
    std::vector<char> zeros(m_SizeBytes, 0);
    Upload(zeros.data(), m_SizeBytes, 0);
}
