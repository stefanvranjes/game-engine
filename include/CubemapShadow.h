#pragma once
#include <memory>
#include <vector>
#include "GLExtensions.h"
#include "Math/Mat4.h"
#include "Math/Vec3.h"

class Shader;

class CubemapShadow {
public:
    CubemapShadow();
    ~CubemapShadow();

    bool Init(unsigned int width, unsigned int height);
    
    void BindForWriting();
    void BindForReading(unsigned int textureUnit);
    
    unsigned int GetWidth() const { return m_Width; }
    unsigned int GetHeight() const { return m_Height; }
    
    // Calculate view matrices for the 6 faces of the cubemap
    void CalculateViewMatrices(const Vec3& lightPos, std::vector<Mat4>& outTransforms, float& outFarPlane);

private:
    unsigned int m_Width, m_Height;
    unsigned int m_FBO;
    unsigned int m_DepthCubemap;
    float m_FarPlane;
};
