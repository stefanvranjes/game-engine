#pragma once

#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include "Math/Mat4.h"
#include "GLExtensions.h"
#include <vector>
#include <memory>
#include <random>

class Shader;
class Texture;
class Terrain;

/**
 * @brief Instance data for a single grass blade
 */
struct GrassInstance {
    Vec3 position;       // World position
    float rotation;      // Y-axis rotation in radians
    float scale;         // Height scale
    float colorVar;      // Color variation [0-1]
    float windPhase;     // Phase offset for wind animation
};

/**
 * @brief GPU-instanced vegetation rendering system
 * 
 * Renders thousands of grass blades efficiently using instancing.
 * Supports wind animation and terrain-based placement.
 */
class Vegetation {
public:
    Vegetation();
    ~Vegetation();

    /**
     * @brief Initialize the vegetation system
     * @return true if successful
     */
    bool Init();

    /**
     * @brief Generate grass instances from terrain
     * @param terrain Terrain to spawn grass on
     * @param splatChannel Which splatmap channel controls grass (0-3)
     * @param minWeight Minimum splatmap weight to spawn grass
     */
    void GenerateFromTerrain(Terrain* terrain, int splatChannel = 0, float minWeight = 0.3f);

    /**
     * @brief Manually add grass instances in a region
     * @param center Center position
     * @param radius Spawn radius
     * @param count Number of instances
     * @param baseHeight Y offset
     */
    void AddGrassRegion(const Vec3& center, float radius, int count, float baseHeight = 0.0f);

    /**
     * @brief Update instance buffer for rendering
     * @param cameraPos Camera position for distance culling
     */
    void UpdateInstances(const Vec3& cameraPos);

    /**
     * @brief Render all visible grass instances
     * @param shader Grass shader
     * @param view View matrix
     * @param projection Projection matrix
     * @param time Current time for animation
     */
    void Render(Shader* shader, const Mat4& view, const Mat4& projection, float time);

    /**
     * @brief Clear all instances
     */
    void Clear();

    // ========== Configuration ==========

    void SetDensity(float density) { m_Density = density; }
    float GetDensity() const { return m_Density; }

    void SetMaxDistance(float dist) { m_MaxDistance = dist; }
    float GetMaxDistance() const { return m_MaxDistance; }

    void SetFadeStart(float start) { m_FadeStart = start; }
    float GetFadeStart() const { return m_FadeStart; }

    void SetWindStrength(float strength) { m_WindStrength = strength; }
    float GetWindStrength() const { return m_WindStrength; }

    void SetWindSpeed(float speed) { m_WindSpeed = speed; }
    float GetWindSpeed() const { return m_WindSpeed; }

    void SetWindDirection(const Vec2& dir) { m_WindDirection = dir; }
    Vec2 GetWindDirection() const { return m_WindDirection; }

    void SetGrassTexture(std::shared_ptr<Texture> tex) { m_GrassTexture = tex; }
    void SetNoiseTexture(std::shared_ptr<Texture> tex) { m_NoiseTexture = tex; }

    void SetBladeWidth(float width) { m_BladeWidth = width; }
    void SetBladeHeight(float height) { m_BladeHeight = height; }

    void SetColorBase(const Vec3& color) { m_ColorBase = color; }
    void SetColorTip(const Vec3& color) { m_ColorTip = color; }

    int GetInstanceCount() const { return static_cast<int>(m_Instances.size()); }
    int GetVisibleCount() const { return m_VisibleCount; }

private:
    void CreateGrassMesh();
    void UpdateInstanceBuffer();

    // Instance data
    std::vector<GrassInstance> m_Instances;
    std::vector<GrassInstance> m_VisibleInstances;
    int m_VisibleCount = 0;

    // GPU resources
    GLuint m_VAO = 0;
    GLuint m_VBO = 0;           // Grass blade mesh
    GLuint m_InstanceVBO = 0;   // Instance data buffer
    int m_VertexCount = 0;

    // Textures
    std::shared_ptr<Texture> m_GrassTexture;
    std::shared_ptr<Texture> m_NoiseTexture;

    // Settings
    float m_Density = 5.0f;         // Blades per unit area
    float m_MaxDistance = 80.0f;    // Max render distance
    float m_FadeStart = 60.0f;      // Distance to start fading
    float m_WindStrength = 1.0f;
    float m_WindSpeed = 1.5f;
    Vec2 m_WindDirection = Vec2(1.0f, 0.5f);

    float m_BladeWidth = 0.05f;
    float m_BladeHeight = 0.4f;

    Vec3 m_ColorBase = Vec3(0.1f, 0.3f, 0.05f);
    Vec3 m_ColorTip = Vec3(0.3f, 0.5f, 0.1f);

    // Random generation
    std::mt19937 m_RNG;

    bool m_Initialized = false;
    bool m_BufferDirty = true;
};
