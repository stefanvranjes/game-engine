#pragma once

#include "Math/Vec3.h"
#include "Shader.h"
#include <memory>
#include <vector>

class Renderer; // Forward declaration

class LightProbe {
public:
    LightProbe(const Vec3& position, float radius);
    ~LightProbe();

    bool Init(unsigned int resolution = 512);
    // Getters
    unsigned int GetEnvironmentMap() const { return m_EnvironmentMap; }
    unsigned int GetIrradianceMap() const { return m_IrradianceMap; }
    unsigned int GetPrefilterMap() const { return m_PrefilterMap; }
    Vec3 GetPosition() const { return m_Position; }
    float GetRadius() const { return m_Radius; }

private:
    Vec3 m_Position;
    float m_Radius;
    unsigned int m_Resolution;
    
    unsigned int m_EnvironmentMap;
    unsigned int m_IrradianceMap;
    unsigned int m_PrefilterMap;
};
