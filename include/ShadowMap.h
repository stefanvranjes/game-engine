#pragma once

class ShadowMap {
public:
    ShadowMap();
    ~ShadowMap();

    bool Init(unsigned int width, unsigned int height);
    void BindForWriting();
    void BindForReading(unsigned int textureUnit);
    
    unsigned int GetWidth() const { return m_Width; }
    unsigned int GetHeight() const { return m_Height; }

private:
    unsigned int m_FBO;
    unsigned int m_DepthMap;
    unsigned int m_Width;
    unsigned int m_Height;
};
