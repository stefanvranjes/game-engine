#pragma once

#include "FluidSimulation.h"
#include "Shader.h"
#include "Camera.h"
#include <memory>

class FluidSimulation;

/**
 * @brief Fluid renderer with screen-space surface reconstruction
 * 
 * Renders fluid particles using:
 * - Particle-based rendering (simple billboards)
 * - Screen-space fluid rendering (advanced surface reconstruction)
 */
class FluidRenderer {
public:
    enum class RenderMode {
        Particles,      // Simple particle billboards
        ScreenSpace     // Screen-space surface reconstruction
    };
    
    FluidRenderer();
    ~FluidRenderer();
    
    /**
     * @brief Initialize the renderer
     * @param screenWidth Screen width
     * @param screenHeight Screen height
     */
    void Initialize(int screenWidth, int screenHeight);
    
    /**
     * @brief Set scene depth texture for soft particles
     */
    void SetSceneDepthTexture(unsigned int texture) { m_SceneDepthTexture = texture; }
    
    /**
     * @brief Shutdown and cleanup
     */
    void Shutdown();
    
    /**
     * @brief Render fluid simulation
     * @param simulation Fluid simulation to render
     * @param camera Camera for rendering
     */
    void Render(const FluidSimulation* simulation, Camera* camera);
    
    /**
     * @brief Handle screen resize
     */
    void OnResize(int width, int height);
    
    /**
     * @brief Set render mode
     */
    void SetRenderMode(RenderMode mode) { m_RenderMode = mode; }
    RenderMode GetRenderMode() const { return m_RenderMode; }
    
    /**
     * @brief Set rendering parameters
     */
    void SetParticleSize(float size) { m_ParticleSize = size; }
    void SetSmoothingRadius(float radius) { m_SmoothingRadius = radius; }
    void SetThicknessScale(float scale) { m_ThicknessScale = scale; }
    void SetRefractiveIndex(float index) { m_RefractiveIndex = index; }
    void SetAbsorptionColor(const Vec3& color) { m_AbsorptionColor = color; }
    void SetFresnelPower(float power) { m_FresnelPower = power; }
    
    /**
     * @brief Get rendering parameters
     */
    float GetParticleSize() const { return m_ParticleSize; }
    float GetSmoothingRadius() const { return m_SmoothingRadius; }
    float GetThicknessScale() const { return m_ThicknessScale; }
    
    /**
     * @brief Foam rendering
     */
    void SetRenderFoam(bool render) { m_RenderFoam = render; }
    bool GetRenderFoam() const { return m_RenderFoam; }
    void SetFoamSize(float size) { m_FoamSize = size; }
    float GetFoamSize() const { return m_FoamSize; }
    void SetUseFoamAnimation(bool use) { m_UseFoamAnimation = use; }
    bool GetUseFoamAnimation() const { return m_UseFoamAnimation; }

private:
    // Render mode
    RenderMode m_RenderMode;
    
    // Screen dimensions
    int m_ScreenWidth;
    int m_ScreenHeight;
    
    // Rendering parameters
    float m_ParticleSize;
    float m_SmoothingRadius;
    float m_ThicknessScale;
    float m_RefractiveIndex;
    Vec3 m_AbsorptionColor;
    float m_FresnelPower;
    bool m_RenderFoam;
    float m_FoamSize;
    bool m_UseFoamAnimation;
    
    // Shaders
    std::unique_ptr<Shader> m_ParticleShader;
    std::unique_ptr<Shader> m_DepthShader;
    std::unique_ptr<Shader> m_SmoothShader;
    std::unique_ptr<Shader> m_NormalShader;
    std::unique_ptr<Shader> m_ThicknessShader;
    std::unique_ptr<Shader> m_ShadingShader;
    std::unique_ptr<Shader> m_FoamShader;
    
    // Framebuffers and textures for screen-space rendering
    unsigned int m_DepthFBO;
    unsigned int m_DepthTexture;
    unsigned int m_SceneDepthTexture; // External scene depth
    unsigned int m_SmoothedDepthFBO;
    unsigned int m_SmoothedDepthTexture;
    unsigned int m_NormalFBO;
    unsigned int m_NormalTexture;
    unsigned int m_ThicknessFBO;
    unsigned int m_ThicknessTexture;
    
    // Particle rendering
    unsigned int m_ParticleVAO;
    unsigned int m_ParticleVBO;
    unsigned int m_ParticleInstanceVBO;
    
    // Foam rendering
    unsigned int m_FoamVAO;
    unsigned int m_FoamVBO;
    unsigned int m_FoamInstanceVBO;
    unsigned int m_FoamTexture;
    bool m_FoamTextureLoaded;
    unsigned int m_FoamAnimationTexture;
    bool m_FoamAnimationTextureLoaded;
    unsigned int m_FoamNormalTexture;
    bool m_FoamNormalTextureLoaded;
    
    // Fullscreen quad for post-processing
    unsigned int m_QuadVAO;
    unsigned int m_QuadVBO;
    
    // Rendering methods
    void RenderParticles(const FluidSimulation* simulation, Camera* camera);
    void RenderScreenSpace(const FluidSimulation* simulation, Camera* camera);
    
    void RenderDepthPass(const FluidSimulation* simulation, Camera* camera);
    void SmoothDepthPass();
    void ComputeNormalsPass(Camera* camera);
    void RenderThicknessPass(const FluidSimulation* simulation, Camera* camera);
    void FinalShadingPass(Camera* camera);
    
    void SetupParticleMesh();
    void SetupFoamMesh();
    void SetupFullscreenQuad();
    void SetupFramebuffers();
    void CleanupFramebuffers();
    void LoadFoamTexture();
    void LoadFoamAnimationTexture();
    void LoadFoamNormalTexture();
    
    void UpdateInstanceData(const FluidSimulation* simulation);
    void RenderFoamParticles(const FluidSimulation* simulation, Camera* camera);
};
