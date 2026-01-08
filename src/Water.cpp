#include "Water.h"
#include "Texture.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

Water::Water() 
    : m_DeepColor(0.0f, 0.1f, 0.4f)     // Dark Blue
    , m_ShallowColor(0.0f, 0.5f, 0.8f)  // Lighter Blue
    , m_Tiling(4.0f)
    , m_WaveSpeed(0.1f)
    , m_WaveStrength(0.02f)
    , m_WaveFoamThreshold(0.5f)
    , m_Clarity(0.5f)
{
}

Water::~Water() {
    if (m_H0k) glDeleteTextures(1, &m_H0k);
    if (m_Hkt) glDeleteTextures(1, &m_Hkt);
    if (m_TwiddleFactors) glDeleteTextures(1, &m_TwiddleFactors);
    if (m_Displacement) glDeleteTextures(1, &m_Displacement);
    if (m_Normal) glDeleteTextures(1, &m_Normal);
    if (m_PingPong) glDeleteTextures(1, &m_PingPong);
}

const float PI = 3.14159265359f;

bool Water::InitFFT() {
    if (m_FFTInitialized) return true;

    // Load Compute Shaders
    m_InitialSpectrumShader = std::make_shared<Shader>();
    if (!m_InitialSpectrumShader->LoadComputeShader("shaders/ocean_initial_spectrum.comp")) {
        std::cerr << "Failed to load ocean_initial_spectrum.comp" << std::endl;
        return false;
    }

    m_TimeUpdateShader = std::make_shared<Shader>();
    if (!m_TimeUpdateShader->LoadComputeShader("shaders/ocean_time_update.comp")) {
        std::cerr << "Failed to load ocean_time_update.comp" << std::endl;
        return false;
    }

    m_ButterflyShader = std::make_shared<Shader>();
    if (!m_ButterflyShader->LoadComputeShader("shaders/ocean_butterfly.comp")) {
        std::cerr << "Failed to load ocean_butterfly.comp" << std::endl;
        return false;
    }

    m_InversionShader = std::make_shared<Shader>();
    if (!m_InversionShader->LoadComputeShader("shaders/ocean_inversion.comp")) {
        std::cerr << "Failed to load ocean_inversion.comp" << std::endl;
        return false;
    }
    
    // Create Textures
    CreateFFTTextures();
    
    // Precompute Twiddle Factors
    ComputeTwiddleFactors();
    
    // Run Initial Spectrum Shader Once
    m_InitialSpectrumShader->Use();
    m_InitialSpectrumShader->SetInt("u_N", m_FFTResolution);
    m_InitialSpectrumShader->SetInt("u_L", (int)m_OceanSize);
    m_InitialSpectrumShader->SetVec2("u_WindDirection", m_WindDirection.x, m_WindDirection.y);
    m_InitialSpectrumShader->SetFloat("u_WindSpeed", m_WindSpeed);
    m_InitialSpectrumShader->SetFloat("u_Amplitude", m_Amplitude);
    
    glBindImageTexture(0, m_H0k, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    m_InitialSpectrumShader->Dispatch(m_FFTResolution / 16, m_FFTResolution / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    m_FFTInitialized = true;
    return true;
}

void Water::CreateFFTTextures() {
    // H0k (Initial Spectrum) - RGBA32F
    glGenTextures(1, &m_H0k);
    glBindTexture(GL_TEXTURE_2D, m_H0k);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_FFTResolution, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Hkt (Time-evolved Spectrum) - RGBA32F
    glGenTextures(1, &m_Hkt);
    glBindTexture(GL_TEXTURE_2D, m_Hkt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_FFTResolution, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Twiddle Factors - RGBA32F (Log2(N) width, N height)
    int log2N = (int)std::log2(m_FFTResolution);
    glGenTextures(1, &m_TwiddleFactors);
    glBindTexture(GL_TEXTURE_2D, m_TwiddleFactors);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, log2N, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Displacement Map - RGBA32F
    glGenTextures(1, &m_Displacement);
    glBindTexture(GL_TEXTURE_2D, m_Displacement);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_FFTResolution, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Normal Map - RG16F or RGBA16F
    glGenTextures(1, &m_Normal);
    glBindTexture(GL_TEXTURE_2D, m_Normal);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, m_FFTResolution, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // PingPong - RGBA32F
    glGenTextures(1, &m_PingPong);
    glBindTexture(GL_TEXTURE_2D, m_PingPong);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_FFTResolution, m_FFTResolution);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

// Bit reversal helper
unsigned int BitReverse(unsigned int x, int N_bits) {
    unsigned int n = 0;
    for (int i = 0; i < N_bits; ++i) {
        if ((x >> i) & 1)
            n |= (1 << (N_bits - 1 - i));
    }
    return n;
}

// Complex multiplication helper
struct Complex {
    float r, i;
};

void Water::ComputeTwiddleFactors() {
    int log2N = (int)std::log2(m_FFTResolution);
    std::vector<float> data(log2N * m_FFTResolution * 4);
    
    for (int y = 0; y < m_FFTResolution; ++y) {
        // Reverse bit index for row y
        unsigned int reversedY = BitReverse(y, log2N);
        
        for (int x = 0; x < log2N; ++x) {
            int butterflySpan = (int)std::pow(2, x); // 1, 2, 4...
            
            int butterflyWing;
            if (x == 0) butterflyWing = 1;
            else butterflyWing = 0; // Logic is simpler:
            
            // Standard Cooley-Tukey butterfly logic for texture storage
            // This texture stores indices and weights for the shader to read.
            // Following common GPU FFT implementation (e.g. Tessendorf/nVidia)
            
            bool topWing = (y % (butterflySpan * 2)) < butterflySpan;
            
            // First stage x=0 (butterflySpan=1)
            // Storing:
            // R: Real part of weight
            // G: Imag part of weight
            // B: Index of first input
            // A: Index of second input
            
            float k = (y * (m_FFTResolution / (float)std::pow(2, x+1))); // k for twist
             // Actually, the k depends on the stage.
             // Let's use a known clean algo reference for "Precomputed Butterfly Texture"
             
             // Updated logic based on "Realtime Water Rendering - GPU Gems / Tessendorf"
             // x is stage (0 to log2N-1)
             // y is row index (0 to N-1)
             
            float p = 2.0f * PI * (float)y / (float)m_FFTResolution; // Not quite
            // Let's simplify: shader does the heavy lifting if we provide the indices and k.
            
            int stage = x;
            int k_val = (y * (m_FFTResolution / (int)std::pow(2, stage + 1))) % m_FFTResolution;
            float twiddle_r = cos(2.0f * PI * k_val / m_FFTResolution);
            float twiddle_i = sin(2.0f * PI * k_val / m_FFTResolution);
            
            int span = (int)std::pow(2, stage);
            int step = span * 2;
            int box_idx = y / step;
            int local_idx = y % step;
            
            int index1, index2;
            if (local_idx < span) {
                index1 = box_idx * step + local_idx;
                index2 = box_idx * step + local_idx + span;
            } else {
                index1 = box_idx * step + local_idx - span;
                index2 = box_idx * step + local_idx;
                // In second wing, we multiply by weight?
                // Actually the weight is W_N^k.
            }

            int index = (y * log2N + x) * 4;
            data[index + 0] = twiddle_r;
            data[index + 1] = twiddle_i;
            data[index + 2] = (float)index1; 
            data[index + 3] = (float)index2;
        }
    }
    
    glBindTexture(GL_TEXTURE_2D, m_TwiddleFactors);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, log2N, m_FFTResolution, GL_RGBA, GL_FLOAT, data.data());
}

void Water::UpdateFFT(float time) {
    if (!m_FFTInitialized) return;

    // 1. Update Spectrum (Hkt)
    m_TimeUpdateShader->Use();
    m_TimeUpdateShader->SetFloat("u_Time", time);
    m_TimeUpdateShader->SetInt("u_N", m_FFTResolution);
    m_TimeUpdateShader->SetInt("u_L", (int)m_OceanSize);
    
    glBindImageTexture(0, m_H0k, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, m_Hkt, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    m_TimeUpdateShader->Dispatch(m_FFTResolution / 16, m_FFTResolution / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    // 2. Horizontal FFT
    m_ButterflyShader->Use();
    int log2N = (int)std::log2(m_FFTResolution);
    int pingpong = 0; 
    
    // For Horizontal: direction = 0
    m_ButterflyShader->SetInt("u_Direction", 0); 
    
    for (int stage = 0; stage < log2N; ++stage) {
        m_ButterflyShader->SetInt("u_Stage", stage);
        
        // Input: Hkt (or PingPong), Output: PingPong (or Hkt)
        // Bind Twiddle
        glBindImageTexture(0, m_TwiddleFactors, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        
        if (stage == 0) {
            glBindImageTexture(1, m_Hkt, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
            glBindImageTexture(2, m_PingPong, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        } else {
            if (pingpong == 0) {
                glBindImageTexture(1, m_PingPong, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
                glBindImageTexture(2, m_Hkt, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
            } else {
                glBindImageTexture(1, m_Hkt, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
                glBindImageTexture(2, m_PingPong, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
            }
        }
        
        m_ButterflyShader->Dispatch(m_FFTResolution / 16, m_FFTResolution / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        if (stage > 0) pingpong = !pingpong; // Toggle after first stage
        else pingpong = 1; // Since stage 0 writes to PingPong (1), next should read from 1
    }
    
    // 3. Vertical FFT
    m_ButterflyShader->SetInt("u_Direction", 1);
    for (int stage = 0; stage < log2N; ++stage) {
        m_ButterflyShader->SetInt("u_Stage", stage);
        
        // Input from previous pass
        if (pingpong == 0) { // Read Hkt, Write PingPong
             glBindImageTexture(1, m_Hkt, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
             glBindImageTexture(2, m_PingPong, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        } else { // Read PingPong, Write Hkt
             glBindImageTexture(1, m_PingPong, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
             glBindImageTexture(2, m_Hkt, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        }
        
        m_ButterflyShader->Dispatch(m_FFTResolution / 16, m_FFTResolution / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        pingpong = !pingpong;
    }
    
    // 4. Inversion / Permutation
    // Final result is in the buffer written to in the last stage.
    // If pingpong is 0, result in Hkt. If 1, result in PingPong.
    // But wait, my pingpong logic is simple.
    // Inversion shader should take final complex buffer and write Component Displacement
    
    unsigned int finalBuffer = (pingpong == 0) ? m_Hkt : m_PingPong;
    
    m_InversionShader->Use();
    m_InversionShader->SetInt("u_N", m_FFTResolution);
    m_InversionShader->SetFloat("u_Choppiness", m_Choppiness);
    
    glBindImageTexture(0, finalBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, m_Displacement, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(2, m_Normal, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F); // Assuming RGBA for Normal+J
    
    m_InversionShader->Dispatch(m_FFTResolution / 16, m_FFTResolution / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Water::BindFFTResources(int startUnit) {
    if (!m_FFTInitialized) return;
    
    glActiveTexture(GL_TEXTURE0 + startUnit);
    glBindTexture(GL_TEXTURE_2D, m_Displacement);
    
    glActiveTexture(GL_TEXTURE0 + startUnit + 1);
    glBindTexture(GL_TEXTURE_2D, m_Normal);
}
