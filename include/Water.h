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
    
private:
    void CreateFFTTextures();
    void ComputeTwiddleFactors();
    
    std::shared_ptr<class Shader> m_InitialSpectrumShader;
    std::shared_ptr<class Shader> m_TimeUpdateShader;
    std::shared_ptr<class Shader> m_ButterflyShader;
    std::shared_ptr<class Shader> m_InversionShader;
    std::shared_ptr<class Shader> m_NormalMapShader;
    
    bool m_FFTInitialized = false;
