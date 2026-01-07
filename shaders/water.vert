#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

uniform mat4 u_Projection;
uniform mat4 u_View;
uniform mat4 u_Model;

// FFT Displacement
uniform sampler2D u_DisplacementMap;
uniform float u_L; // Ocean patch size

// Gerstner Wave Parameters
uniform float u_Time;
uniform int u_UseGerstner;
uniform float u_GerstnerAmplitude;
uniform float u_GerstnerFrequency;
uniform float u_GerstnerSteepness;

out vec3 FragPos;
out vec2 TexCoords;
out vec3 Normal;
out vec4 ClipSpace;
out float WaveHeight; // For foam calculation

// ============================================================================
// Gerstner Wave Function
// ============================================================================

// Single Gerstner wave component
vec3 gerstnerWave(vec2 position, vec2 direction, float amplitude, float frequency, 
                   float steepness, float phase, out vec3 tangent, out vec3 binormal) {
    float k = 2.0 * 3.14159 * frequency;
    float c = sqrt(9.8 / k); // Phase speed (deep water)
    vec2 d = normalize(direction);
    
    float f = k * (dot(d, position) - c * phase);
    float a = amplitude;
    
    // Gerstner displacement
    float cosF = cos(f);
    float sinF = sin(f);
    
    vec3 displacement;
    displacement.x = steepness * a * d.x * cosF;
    displacement.z = steepness * a * d.y * cosF;
    displacement.y = a * sinF;
    
    // Tangent (partial derivative w.r.t. x)
    tangent = vec3(
        1.0 - steepness * d.x * d.x * k * a * sinF,
        d.x * k * a * cosF,
        -steepness * d.x * d.y * k * a * sinF
    );
    
    // Binormal (partial derivative w.r.t. z)
    binormal = vec3(
        -steepness * d.x * d.y * k * a * sinF,
        d.y * k * a * cosF,
        1.0 - steepness * d.y * d.y * k * a * sinF
    );
    
    return displacement;
}

void main()
{
    // Transform to world space
    vec4 worldPos = u_Model * vec4(aPos, 1.0);
    
    // UV for FFT sampling based on world position
    float L = u_L;
    vec2 dispUV = worldPos.xz / L;
    
    // Sample FFT displacement
    vec3 fftDisplacement = texture(u_DisplacementMap, dispUV).xyz;
    
    // Track total wave height for foam
    float totalHeight = fftDisplacement.y;
    
    // Apply FFT displacement
    worldPos.xyz += fftDisplacement;
    
    // ----- GERSTNER WAVES -----
    vec3 gerstnerDisp = vec3(0.0);
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 binormal = vec3(0.0, 0.0, 1.0);
    
    if (u_UseGerstner == 1) {
        vec3 t1, b1, t2, b2, t3, b3, t4, b4;
        
        // Wave 1: Primary direction
        gerstnerDisp += gerstnerWave(
            worldPos.xz, 
            vec2(1.0, 0.3),           // Direction
            u_GerstnerAmplitude,       // Amplitude
            u_GerstnerFrequency,       // Frequency
            u_GerstnerSteepness,       // Steepness
            u_Time,                    // Phase
            t1, b1
        );
        
        // Wave 2: Secondary direction (different freq)
        gerstnerDisp += gerstnerWave(
            worldPos.xz,
            vec2(0.5, 0.8),
            u_GerstnerAmplitude * 0.5,
            u_GerstnerFrequency * 1.3,
            u_GerstnerSteepness * 0.8,
            u_Time * 1.1,
            t2, b2
        );
        
        // Wave 3: Cross waves
        gerstnerDisp += gerstnerWave(
            worldPos.xz,
            vec2(-0.4, 0.6),
            u_GerstnerAmplitude * 0.3,
            u_GerstnerFrequency * 2.1,
            u_GerstnerSteepness * 0.6,
            u_Time * 0.9,
            t3, b3
        );
        
        // Wave 4: Small ripples
        gerstnerDisp += gerstnerWave(
            worldPos.xz,
            vec2(0.7, -0.5),
            u_GerstnerAmplitude * 0.15,
            u_GerstnerFrequency * 3.5,
            u_GerstnerSteepness * 0.4,
            u_Time * 1.3,
            t4, b4
        );
        
        // Accumulate tangent space
        tangent = normalize(t1 + t2 + t3 + t4);
        binormal = normalize(b1 + b2 + b3 + b4);
        
        worldPos.xyz += gerstnerDisp;
        totalHeight += gerstnerDisp.y;
    }
    
    // Calculate normal from tangent and binormal
    vec3 calculatedNormal;
    if (u_UseGerstner == 1) {
        calculatedNormal = normalize(cross(binormal, tangent));
    } else {
        // Use model normal if no Gerstner
        calculatedNormal = mat3(transpose(inverse(u_Model))) * aNormal;
    }
    
    // Output
    ClipSpace = u_Projection * u_View * worldPos;
    gl_Position = ClipSpace;
    
    FragPos = worldPos.xyz;
    TexCoords = dispUV;
    Normal = calculatedNormal;
    WaveHeight = totalHeight;
}
