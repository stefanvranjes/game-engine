// probe_sampling.glsl
// Shared GLSL functions for light probe sampling
// Include this file in shaders that need probe-based GI

#ifndef PROBE_SAMPLING_GLSL
#define PROBE_SAMPLING_GLSL

// Spherical Harmonics evaluation (L0 + L1 bands)
vec3 EvaluateSH_L1(float shCoeffs[9], vec3 normal) {
    // L0 band (constant)
    vec3 result = vec3(shCoeffs[0], shCoeffs[3], shCoeffs[6]) * 0.282095;
    
    // L1 band (linear)
    vec3 l1_y = vec3(shCoeffs[1], shCoeffs[4], shCoeffs[7]);
    vec3 l1_z = vec3(shCoeffs[2], shCoeffs[5], shCoeffs[8]);
    vec3 l1_x = vec3(shCoeffs[0], shCoeffs[3], shCoeffs[6]);  // Note: indices adjusted for storage
    
    result += l1_y * (0.488603 * normal.y);
    result += l1_z * (0.488603 * normal.z);
    result += l1_x * (0.488603 * normal.x);
    
    return max(result, vec3(0.0));
}

// Sample probe grid with trilinear interpolation
vec3 SampleProbeGrid(vec3 worldPos, vec3 normal, 
                     samplerBuffer probeData, 
                     vec3 gridMin, vec3 gridMax, ivec3 gridRes) {
    // Convert world position to grid coordinates
    vec3 gridPos = (worldPos - gridMin) / (gridMax - gridMin);
    gridPos = clamp(gridPos, vec3(0.0), vec3(1.0));
    
    vec3 gridCoord = gridPos * vec3(gridRes - 1);
    ivec3 baseCoord = ivec3(floor(gridCoord));
    vec3 frac = fract(gridCoord);
    
    // Sample 8 surrounding probes
    vec3 irradiance = vec3(0.0);
    float totalWeight = 0.0;
    
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ivec3 coord = baseCoord + ivec3(x, y, z);
                coord = clamp(coord, ivec3(0), gridRes - 1);
                
                int probeIndex = coord.x + coord.y * gridRes.x + coord.z * gridRes.x * gridRes.y;
                
                // Fetch SH coefficients from buffer
                // Each probe stores: position (vec3) + 27 SH coeffs + flags + radius
                // Total: 32 floats per probe
                int baseOffset = probeIndex * 32;
                
                // Skip position (3 floats)
                float shCoeffs[9];
                for (int i = 0; i < 9; i++) {
                    shCoeffs[i] = texelFetch(probeData, baseOffset + 3 + i).r;
                }
                
                vec3 probeIrradiance = EvaluateSH_L1(shCoeffs, normal);
                
                // Trilinear weight
                vec3 weight3D = mix(vec3(1.0) - frac, frac, vec3(x, y, z));
                float weight = weight3D.x * weight3D.y * weight3D.z;
                
                irradiance += probeIrradiance * weight;
                totalWeight += weight;
            }
        }
    }
    
    if (totalWeight > 0.0) {
        irradiance /= totalWeight;
    }
    
    return irradiance;
}

// Simplified version using SSBO instead of samplerBuffer
struct ProbeData {
    vec3 position;
    float shCoeffs[27];
    uint flags;
    float radius;
};

vec3 SampleProbeGridSSBO(vec3 worldPos, vec3 normal,
                         vec3 gridMin, vec3 gridMax, ivec3 gridRes) {
    // This version assumes probes are in an SSBO bound to binding point 0
    // The actual SSBO binding is done in C++ code
    
    // Convert to grid coordinates
    vec3 gridPos = (worldPos - gridMin) / (gridMax - gridMin);
    gridPos = clamp(gridPos, vec3(0.0), vec3(1.0));
    
    vec3 gridCoord = gridPos * vec3(gridRes - 1);
    ivec3 baseCoord = ivec3(floor(gridCoord));
    vec3 frac = fract(gridCoord);
    
    vec3 irradiance = vec3(0.0);
    
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ivec3 coord = baseCoord + ivec3(x, y, z);
                coord = clamp(coord, ivec3(0), gridRes - 1);
                
                int probeIndex = coord.x + coord.y * gridRes.x + coord.z * gridRes.x * gridRes.y;
                
                // Evaluate SH for this probe
                // Note: In actual implementation, fetch from SSBO
                // For now, this is a placeholder
                float shCoeffs[9];
                // shCoeffs = fetch from SSBO at probeIndex
                
                vec3 probeIrradiance = EvaluateSH_L1(shCoeffs, normal);
                
                // Trilinear weight
                vec3 weight3D = mix(vec3(1.0) - frac, frac, vec3(x, y, z));
                float weight = weight3D.x * weight3D.y * weight3D.z;
                
                irradiance += probeIrradiance * weight;
            }
        }
    }
    
    return irradiance;
}

#endif // PROBE_SAMPLING_GLSL
