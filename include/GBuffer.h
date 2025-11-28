#pragma once

#include <memory>

class GBuffer {
public:
    GBuffer();
    ~GBuffer();

    bool Init(unsigned int width, unsigned int height);
    void BindForWriting();
    void BindForReading();
    void Shutdown();

    unsigned int GetPositionTexture() const { return m_PositionTexture; }
    unsigned int GetNormalTexture() const { return m_NormalTexture; }
    unsigned int GetAlbedoSpecTexture() const { return m_AlbedoSpecTexture; }
    
    unsigned int GetWidth() const { return m_Width; }
    unsigned int GetHeight() const { return m_Height; }

private:
    unsigned int m_FBO;
    unsigned int m_PositionTexture;
    unsigned int m_NormalTexture;
    unsigned int m_AlbedoSpecTexture;
    unsigned int m_DepthTexture;
    
    unsigned int m_Width;
    unsigned int m_Height;
};
