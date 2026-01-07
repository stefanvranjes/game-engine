#pragma once

#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include <memory>

class Texture;

class Water {
public:
    Water();
    ~Water();

    // Water Properties
    Vec3 m_DeepColor;
    Vec3 m_ShallowColor;
    
    float m_Tiling;
    float m_WaveSpeed;
    float m_WaveStrength;
    float m_WaveFoamThreshold;
    float m_Clarity; // Transparency/Depth visibility

    // Textures
    std::shared_ptr<Texture> m_NormalMap;
    std::shared_ptr<Texture> m_FoamMap; // Optional specific foam texture
    std::shared_ptr<Texture> m_FlowMap; // Optional flow map

    // Methods
    void SetNormalMap(std::shared_ptr<Texture> texture) { m_NormalMap = texture; }
    std::shared_ptr<Texture> GetNormalMap() const { return m_NormalMap; }

    // FFT Ocean Simulation
    bool InitFFT();
    void UpdateFFT(float time);
    void BindFFTResources(int startUnit);
    
    // FFT Parameters
    int m_FFTResolution = 256; // N
    float m_OceanSize = 100.0f; // L
    Vec2 m_WindDirection = Vec2(1.0f, 1.0f);
    float m_WindSpeed = 30.0f;
    float m_Amplitude = 1.0f; // A
    float m_Choppiness = 1.0f;

    // FFT Internal Resources (Raw OpenGL IDs)
    unsigned int m_H0k = 0;           // Initial Spectrum
    unsigned int m_Hkt = 0;           // Time-evolved Spectrum (Height, X, Z)
    unsigned int m_TwiddleFactors = 0; // Precomputed butterflies
    unsigned int m_Displacement = 0;   // Final Displacement Map
    unsigned int m_Normal = 0;         // Final Normal Map
    unsigned int m_PingPong = 0;       // Ping Pong for FFT
    
    // Planar Reflection Settings
    bool m_UsePlanarReflection = true;    // Enable planar reflections
    float m_ReflectionDistortion = 0.02f; // Wave distortion strength
    float m_PlaneY = 0.0f;                // Height of water surface for reflection
    
    // Subsurface Scattering & Absorption
    Vec3 m_AbsorptionColor = Vec3(0.45f, 0.09f, 0.04f); // Red/green absorb more (blue water)
    float m_AbsorptionScale = 0.25f;      // How quickly light is absorbed
    Vec3 m_ScatterColor = Vec3(0.0f, 0.4f, 0.3f); // Subsurface scatter tint
    float m_ScatterStrength = 1.5f;       // SSS intensity
    
    // Foam
    float m_FoamIntensity = 1.0f;         // Overall foam brightness
    float m_FoamThreshold = 0.4f;         // Wave height threshold for foam
    float m_FoamFalloff = 3.0f;           // How quickly foam fades
    Vec3 m_FoamColor = Vec3(1.0f, 1.0f, 1.0f);
    float m_ShorelineFoamWidth = 2.0f;    // Depth range for shoreline foam
    
    // Specular / PBR
    float m_Roughness = 0.08f;            // Water surface roughness
    float m_SpecularIntensity = 1.0f;     // Specular highlight strength
    
    // Gerstner Waves (procedural detail)
    bool m_UseGerstnerWaves = true;
    float m_GerstnerAmplitude = 0.15f;    // Height of Gerstner waves
    float m_GerstnerFrequency = 0.8f;     // Wave frequency
    float m_GerstnerSteepness = 0.5f;     // Wave sharpness (0-1)
    
    // Caustics
    bool m_UseCaustics = true;            // Enable underwater caustics
    float m_CausticsIntensity = 0.5f;     // Brightness of caustics
    float m_CausticsScale = 0.1f;         // World-space scale of caustics pattern
    float m_CausticsSpeed = 0.5f;         // Animation speed
    float m_CausticsDepth = 20.0f;        // Max depth for caustics effect
    
    // Spray Particles
    bool m_UseSprayParticles = true;      // Enable spray/foam particles
    float m_SpraySpawnThreshold = 0.4f;   // Min wave height to spawn particles
    float m_SprayIntensity = 50.0f;       // Particles per second at max spawn
    float m_SprayVelocityScale = 5.0f;    // Upward spray velocity multiplier
    float m_SprayLifetime = 1.5f;         // Particle lifetime in seconds
    float m_SpraySizeMin = 0.05f;         // Minimum particle size
    float m_SpraySizeMax = 0.15f;         // Maximum particle size
    float m_SprayGravity = -9.8f;         // Gravity applied to spray
    
    // Underwater Effects
    bool m_EnableUnderwaterEffects = true;
    Vec3 m_UnderwaterTint = Vec3(0.1f, 0.4f, 0.5f);  // Blue-green tint
    float m_UnderwaterFogDensity = 0.04f;            // Fog density
    float m_UnderwaterFogStart = 0.0f;               // Fog start distance
    float m_UnderwaterFogEnd = 50.0f;                // Fog end distance
    float m_UnderwaterDistortion = 0.008f;           // Screen distortion strength
    float m_UnderwaterDistortionSpeed = 2.0f;        // Distortion animation speed
    float m_UnderwaterVignette = 0.3f;               // Edge darkening
    
private:
    void CreateFFTTextures();
    void ComputeTwiddleFactors();
    
    std::shared_ptr<class Shader> m_InitialSpectrumShader;
    std::shared_ptr<class Shader> m_TimeUpdateShader;
    std::shared_ptr<class Shader> m_ButterflyShader;
    std::shared_ptr<class Shader> m_InversionShader;
    std::shared_ptr<class Shader> m_NormalMapShader;
    
    bool m_FFTInitialized = false;
};
