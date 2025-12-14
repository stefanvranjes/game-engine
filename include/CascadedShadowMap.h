#pragma once

#include <vector>
#include "Math/Mat4.h"

class CascadedShadowMap {
public:
    CascadedShadowMap();
    ~CascadedShadowMap();

    bool Init(unsigned int width, unsigned int height);
    void BindForWriting(unsigned int cascadeIndex);
    void BindForReading(unsigned int textureUnit);

    unsigned int GetWidth() const { return m_Width; }
    unsigned int GetHeight() const { return m_Height; }
    unsigned int GetCascadeCount() const { return 3; }
    unsigned int GetShadowMap() const { return m_DepthMapArray; }

private:
    unsigned int m_FBO;
    unsigned int m_DepthMapArray;
    unsigned int m_Width;
    unsigned int m_Height;
};
