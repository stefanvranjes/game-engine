#pragma once

#include "Math/Vec3.h"
#include <memory>

class Renderer; // Forward declaration

class ReflectionProbe {
public:
    ReflectionProbe(const Vec3& position, float radius, unsigned int resolution = 256);
    ~ReflectionProbe();

    bool Init();
    void Capture(Renderer* renderer); // Render scene to cubemap
    
    // Getters
    unsigned int GetCubemap() const { return m_Cubemap; }
    Vec3 GetPosition() const { return m_Position; }
    float GetRadius() const { return m_Radius; }
    bool NeedsUpdate() const { return m_NeedsUpdate; }
    
    // Setters
    void SetNeedsUpdate(bool update) { m_NeedsUpdate = update; }

private:
    Vec3 m_Position;
    float m_Radius;
    unsigned int m_Resolution;
    
    unsigned int m_Cubemap;
    unsigned int m_FBO;
    unsigned int m_RBO;
    
    bool m_NeedsUpdate;
};
