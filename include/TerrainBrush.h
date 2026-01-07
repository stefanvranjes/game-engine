#pragma once

#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include <functional>

class Terrain;

/**
 * @brief Brush operation modes for terrain editing
 */
enum class BrushMode {
    Raise,      // Increase height
    Lower,      // Decrease height
    Flatten,    // Set to target height
    Smooth,     // Average neighboring heights
    Paint,      // Modify splatmap
    SetHeight   // Set exact height value
};

/**
 * @brief Falloff shape for brush edge
 */
enum class BrushFalloff {
    Linear,     // Linear falloff
    Smooth,     // Smooth (cosine-based)
    Sphere,     // Spherical falloff
    Tip         // Sharp tip (quadratic)
};

/**
 * @brief Terrain brush for height/texture painting
 * 
 * Provides various brush operations for terrain sculpting and texture painting.
 */
class TerrainBrush {
public:
    TerrainBrush();
    ~TerrainBrush() = default;

    /**
     * @brief Apply brush at world position
     * @param terrain Target terrain to modify
     * @param worldX World X coordinate of brush center
     * @param worldZ World Z coordinate of brush center
     * @param deltaTime Time step for continuous application
     */
    void Apply(Terrain* terrain, float worldX, float worldZ, float deltaTime);

    /**
     * @brief Apply brush with custom per-vertex function
     */
    void ApplyCustom(Terrain* terrain, float worldX, float worldZ, float deltaTime,
                     std::function<float(float currentHeight, float brushInfluence)> func);

    // ========== Brush Settings ==========

    void SetMode(BrushMode mode) { m_Mode = mode; }
    BrushMode GetMode() const { return m_Mode; }

    void SetRadius(float radius) { m_Radius = radius; }
    float GetRadius() const { return m_Radius; }

    void SetStrength(float strength) { m_Strength = strength; }
    float GetStrength() const { return m_Strength; }

    void SetFalloff(float falloff) { m_Falloff = falloff; }
    float GetFalloff() const { return m_Falloff; }

    void SetFalloffType(BrushFalloff type) { m_FalloffType = type; }
    BrushFalloff GetFalloffType() const { return m_FalloffType; }

    void SetTargetHeight(float height) { m_TargetHeight = height; }
    float GetTargetHeight() const { return m_TargetHeight; }

    void SetPaintChannel(int channel) { m_PaintChannel = channel; }
    int GetPaintChannel() const { return m_PaintChannel; }

    void SetPaintWeight(float weight) { m_PaintWeight = weight; }
    float GetPaintWeight() const { return m_PaintWeight; }

    // ========== Preview ==========

    /**
     * @brief Get brush preview data for visualization
     * @param centerX World X center
     * @param centerZ World Z center
     * @param outPoints Output vector for preview circle points
     */
    void GetPreviewCircle(float centerX, float centerZ, std::vector<Vec3>& outPoints, int segments = 32);

private:
    /**
     * @brief Calculate brush influence at distance
     */
    float CalculateFalloff(float distance) const;

    BrushMode m_Mode = BrushMode::Raise;
    BrushFalloff m_FalloffType = BrushFalloff::Smooth;
    
    float m_Radius = 5.0f;
    float m_Strength = 0.5f;
    float m_Falloff = 0.5f;        // 0 = hard edge, 1 = soft edge
    float m_TargetHeight = 0.0f;   // For flatten mode
    
    int m_PaintChannel = 0;        // Splatmap channel (0-3)
    float m_PaintWeight = 1.0f;    // Paint amount
};
