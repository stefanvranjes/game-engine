#pragma once

#include <cmath>

// Simple 3D Perlin-like noise using sine waves
// For production use, consider implementing proper Perlin or Simplex noise
class NoiseGenerator {
public:
    // Generate 3D noise value in range [-1, 1]
    static float Noise3D(float x, float y, float z) {
        // Use multiple octaves of sine waves for pseudo-random noise
        float noise = 0.0f;
        
        // First octave
        noise += sin(x * 2.0f + y * 1.3f) * cos(y * 2.0f + z * 1.7f) * sin(z * 2.0f + x * 1.1f);
        
        // Second octave (higher frequency, lower amplitude)
        noise += sin(x * 4.0f + z * 2.1f) * cos(y * 4.0f + x * 2.3f) * 0.5f;
        
        // Third octave
        noise += sin(x * 8.0f + y * 3.7f) * cos(z * 8.0f + y * 3.1f) * 0.25f;
        
        // Normalize to [-1, 1] range
        return noise / 1.75f;
    }
    
    // Generate smooth 3D noise with interpolation
    static float SmoothNoise3D(float x, float y, float z) {
        float noise = Noise3D(x, y, z);
        
        // Apply smoothing function
        return noise * noise * noise * (noise * (noise * 6.0f - 15.0f) + 10.0f);
    }
};
