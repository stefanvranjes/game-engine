#pragma once

#include "Math/Mat4.h"
#include <memory>
#include <string>

class Texture;

class Decal {
public:
    Decal();
    ~Decal();

    // Decals are typically attached to a GameObject, so we rely on the GameObject's transform.
    // However, to keep it standalone or simple, we might just store logic parameters here.
    
    void SetAlbedoTexture(std::shared_ptr<Texture> texture) { m_AlbedoTexture = texture; }
    std::shared_ptr<Texture> GetAlbedoTexture() const { return m_AlbedoTexture; }

    void SetNormalTexture(std::shared_ptr<Texture> texture) { m_NormalTexture = texture; }
    std::shared_ptr<Texture> GetNormalTexture() const { return m_NormalTexture; }

    void SetNormalBlending(float factor) { m_NormalBlending = factor; }
    float GetNormalBlending() const { return m_NormalBlending; }

    void SetPriority(int priority) { m_Priority = priority; }
    int GetPriority() const { return m_Priority; }

private:
    std::shared_ptr<Texture> m_AlbedoTexture;
    std::shared_ptr<Texture> m_NormalTexture;
    float m_NormalBlending; // 0 for no normal mod, 1 for full replacement/blending
    int m_Priority;         // Render order
};
