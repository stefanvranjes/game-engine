#pragma once

#include "Math/Vec3.h"
#include "Math/Mat4.h"
#include "PartialTearSystem.h"
#include <vector>
#include <memory>

class Shader;

/**
 * @brief Renders crack lines on soft bodies with damage-based visual effects
 * 
 * Visualizes cracks with configurable appearance including damage-based
 * opacity, color interpolation, width variation, and optional glow effects.
 */
class CrackRenderer {
public:
    /**
     * @brief Rendering configuration
     */
    struct RenderSettings {
        Vec3 crackColor;           // Base crack color (default: dark red)
        float baseWidth;           // Base line width (default: 0.02)
        float maxWidth;            // Max width at 100% damage (default: 0.05)
        bool useGlow;              // Add glow effect
        float glowIntensity;       // Glow brightness (0.0 - 1.0)
        bool useDamageColor;       // Color based on damage
        Vec3 lowDamageColor;       // Color at 0% damage (default: yellow)
        Vec3 highDamageColor;      // Color at 100% damage (default: red)
        float minOpacity;          // Minimum opacity at 0% damage
        float maxOpacity;          // Maximum opacity at 100% damage
        
        // Animation settings
        bool enablePulsing;        // Enable pulsing effect
        float pulseSpeed;          // Pulse frequency in Hz (default: 2.0)
        float pulseAmplitude;      // Pulse intensity 0.0-1.0 (default: 0.3)
        
        bool enableFlickering;     // Enable flickering effect
        float flickerSpeed;        // Flicker frequency in Hz (default: 10.0)
        float flickerAmplitude;    // Flicker intensity 0.0-1.0 (default: 0.2)
        
        bool enableGrowth;         // Animate crack growth/fade-in
        float growthDuration;      // Growth animation duration in seconds (default: 0.5)
        
        RenderSettings()
            : crackColor(0.2f, 0.0f, 0.0f)
            , baseWidth(0.02f)
            , maxWidth(0.05f)
            , useGlow(false)
            , glowIntensity(0.3f)
            , useDamageColor(true)
            , lowDamageColor(0.5f, 0.5f, 0.0f)  // Yellow
            , highDamageColor(1.0f, 0.0f, 0.0f) // Red
            , minOpacity(0.3f)
            , maxOpacity(1.0f)
            , enablePulsing(false)
            , pulseSpeed(2.0f)
            , pulseAmplitude(0.3f)
            , enableFlickering(false)
            , flickerSpeed(10.0f)
            , flickerAmplitude(0.2f)
            , enableGrowth(true)
            , growthDuration(0.5f)
            , damageAffectsSpeed(true)
            , minPulseSpeed(1.0f)
            , maxPulseSpeed(4.0f)
        {}
    };

    CrackRenderer();
    ~CrackRenderer();

    /**
     * @brief Update animation state
     * 
     * @param deltaTime Time since last update
     */
    void Update(float deltaTime);

    /**
     * @brief Render all cracks
     * 
     * @param cracks Active cracks to render
     * @param vertices Vertex positions
     * @param tetrahedra Tetrahedral indices
     * @param shader Shader to use for rendering
     * @param modelMatrix Model transformation matrix
     * @param currentTime Current simulation time for animations
     */
    void RenderCracks(
        const std::vector<PartialTearSystem::Crack>& cracks,
        const Vec3* vertices,
        const int* tetrahedra,
        Shader* shader,
        const Mat4& modelMatrix,
        float currentTime = 0.0f
    );

    /**
     * @brief Generate geometry for a single crack
     * 
     * @param crack Crack to generate geometry for
     * @param vertices Vertex positions
     * @param tetrahedra Tetrahedral indices
     * @param outVertices Output vertex positions
     * @param outColors Output vertex colors (with alpha in w component)
     * @param currentTime Current simulation time for animations
     */
    void GenerateCrackGeometry(
        const PartialTearSystem::Crack& crack,
        const Vec3* vertices,
        const int* tetrahedra,
        std::vector<Vec3>& outVertices,
        std::vector<Vec3>& outColors,
        float currentTime = 0.0f
    );

    /**
     * @brief Set rendering settings
     */
    void SetRenderSettings(const RenderSettings& settings);

    /**
     * @brief Get current rendering settings
     */
    RenderSettings GetRenderSettings() const { return m_Settings; }

    /**
     * @brief Initialize OpenGL resources
     */
    void Initialize();

    /**
     * @brief Clean up OpenGL resources
     */
    void Cleanup();

private:
    RenderSettings m_Settings;
    
    // OpenGL resources
    unsigned int m_VAO;
    unsigned int m_VBO_Positions;
    unsigned int m_VBO_Colors;
    bool m_Initialized;
    
    // Cached geometry
    std::vector<Vec3> m_CachedVertices;
    std::vector<Vec3> m_CachedColors;
    
    // Animation state
    float m_AnimationTime;
    
    /**
     * @brief Calculate color for crack based on damage
     */
    Vec3 CalculateCrackColor(float damage) const;
    
    /**
     * @brief Calculate opacity for crack based on damage
     */
    float CalculateCrackOpacity(float damage) const;
    
    /**
     * @brief Calculate width for crack based on damage
     */
    float CalculateCrackWidth(float damage) const;
    
    /**
     * @brief Calculate animated intensity multiplier
     */
    float CalculateAnimatedIntensity(
        const PartialTearSystem::Crack& crack,
        float currentTime
    ) const;
    
    /**
     * @brief Calculate growth fade-in factor
     */
    float CalculateGrowthFactor(
        const PartialTearSystem::Crack& crack,
        float currentTime
    ) const;
    
    /**
     * @brief Update OpenGL buffers with current geometry
     */
    void UpdateBuffers();
};
