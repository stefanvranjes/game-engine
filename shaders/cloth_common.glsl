// cloth_common.glsl
// Shared utility functions for cloth rendering

#ifndef CLOTH_COMMON_GLSL
#define CLOTH_COMMON_GLSL

/**
 * Calculate cloth normal from position derivatives
 * Useful when normals are not pre-computed
 */
vec3 calculateClothNormal(vec3 worldPos) {
    vec3 dPdx = dFdx(worldPos);
    vec3 dPdy = dFdy(worldPos);
    return normalize(cross(dPdx, dPdy));
}

/**
 * Apply wind deformation to cloth vertex
 * @param pos World position
 * @param normal Surface normal
 * @param windVel Wind velocity vector
 * @param time Current time for animation
 * @param flexibility How much the cloth responds to wind (0-1)
 * @return Deformed position
 */
vec3 applyWindDeformation(vec3 pos, vec3 normal, vec3 windVel, float time, float flexibility) {
    vec3 windDir = normalize(windVel);
    float windStrength = length(windVel);
    
    if (windStrength < 0.001) {
        return pos;
    }
    
    // Wind influence based on normal alignment
    float windAlignment = max(0.0, dot(normal, windDir));
    
    // Multi-frequency wave motion
    float wave1 = sin(pos.x * 0.5 + time * 2.0) * 0.5 + 0.5;
    float wave2 = sin(pos.z * 0.3 + time * 1.5) * 0.5 + 0.5;
    float wave3 = sin((pos.x + pos.z) * 0.4 + time * 2.5) * 0.5 + 0.5;
    
    float waveFactor = wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2;
    float windInfluence = windAlignment * waveFactor * windStrength * flexibility;
    
    return pos + windDir * windInfluence;
}

/**
 * Calculate translucency (subsurface scattering approximation)
 * @param normal Surface normal
 * @param lightDir Light direction
 * @param viewDir View direction
 * @param translucency Translucency amount (0-1)
 * @return Translucency factor
 */
float calculateClothTranslucency(vec3 normal, vec3 lightDir, vec3 viewDir, float translucency) {
    // Backlight (light hitting the back of the surface)
    float backlight = max(0.0, -dot(normal, lightDir));
    
    // View-dependent translucency
    float VdotL = max(0.0, dot(viewDir, -lightDir));
    
    return backlight * VdotL * translucency;
}

/**
 * Enhance cloth with procedural wrinkle detail
 * @param normal Base normal
 * @param uv Texture coordinates
 * @param scale Wrinkle frequency scale
 * @return Enhanced normal with wrinkle detail
 */
vec3 enhanceClothWrinkles(vec3 normal, vec2 uv, float scale) {
    // Generate wrinkle pattern using trigonometric functions
    float wrinkle1 = sin(uv.x * 50.0 * scale) * cos(uv.y * 50.0 * scale);
    float wrinkle2 = sin(uv.x * 80.0 * scale + 1.5) * cos(uv.y * 80.0 * scale + 2.3);
    
    float wrinkleIntensity = (wrinkle1 * 0.6 + wrinkle2 * 0.4) * 0.05;
    
    // Perturb normal
    vec3 perturbedNormal = normal;
    perturbedNormal.xy += vec2(wrinkleIntensity);
    
    return normalize(perturbedNormal);
}

/**
 * Calculate cloth-specific ambient occlusion
 * Enhances AO in folds and creases
 */
float calculateClothAO(vec3 normal, vec3 worldPos, float baseAO) {
    // Use world position variation to simulate fold darkening
    float foldFactor = sin(worldPos.x * 10.0) * sin(worldPos.z * 10.0);
    foldFactor = foldFactor * 0.5 + 0.5; // Remap to 0-1
    
    // Combine with base AO
    return baseAO * (0.7 + foldFactor * 0.3);
}

#endif // CLOTH_COMMON_GLSL
