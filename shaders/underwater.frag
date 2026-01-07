#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

// Scene texture from previous pass
uniform sampler2D u_SceneTexture;
uniform sampler2D u_DepthTexture;

// Underwater settings
uniform int u_IsUnderwater;
uniform vec3 u_UnderwaterTint;
uniform float u_FogDensity;
uniform float u_FogStart;
uniform float u_FogEnd;
uniform float u_Distortion;
uniform float u_DistortionSpeed;
uniform float u_Vignette;
uniform float u_Time;

// Camera
uniform float u_NearPlane;
uniform float u_FarPlane;

// Constants
const float PI = 3.14159265359;

// ============================================================================
// Helper Functions
// ============================================================================

// Linearize depth from depth buffer
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * u_NearPlane * u_FarPlane) / (u_FarPlane + u_NearPlane - z * (u_FarPlane - u_NearPlane));
}

// Noise function for distortion
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// ============================================================================
// Main
// ============================================================================

void main()
{
    if (u_IsUnderwater == 0) {
        // Not underwater - pass through
        FragColor = texture(u_SceneTexture, TexCoord);
        return;
    }
    
    // ----- DISTORTION -----
    // Animated wave distortion based on noise
    float time = u_Time * u_DistortionSpeed;
    
    vec2 distortedUV = TexCoord;
    
    // Create wave-like distortion pattern
    float waveX = sin(TexCoord.y * 20.0 + time) * u_Distortion;
    float waveY = cos(TexCoord.x * 15.0 + time * 0.8) * u_Distortion;
    
    // Add noise-based distortion for organic feel
    float n1 = noise(TexCoord * 10.0 + time * 0.5);
    float n2 = noise(TexCoord * 8.0 - time * 0.3);
    
    distortedUV.x += waveX + (n1 - 0.5) * u_Distortion * 0.5;
    distortedUV.y += waveY + (n2 - 0.5) * u_Distortion * 0.5;
    
    // Clamp to valid range
    distortedUV = clamp(distortedUV, 0.001, 0.999);
    
    // Sample scene with distortion
    vec3 sceneColor = texture(u_SceneTexture, distortedUV).rgb;
    
    // ----- DEPTH-BASED FOG -----
    float depth = texture(u_DepthTexture, distortedUV).r;
    float linearDepth = linearizeDepth(depth);
    
    // Exponential fog
    float fogFactor = 1.0 - exp(-linearDepth * u_FogDensity);
    
    // Also apply distance-based fog
    float distanceFog = smoothstep(u_FogStart, u_FogEnd, linearDepth);
    fogFactor = max(fogFactor, distanceFog);
    fogFactor = clamp(fogFactor, 0.0, 0.95); // Never fully opaque
    
    // ----- COLOR TINTING -----
    // Apply underwater tint based on fog factor
    vec3 tintedColor = mix(sceneColor, sceneColor * u_UnderwaterTint, 0.5);
    
    // Blend with fog color (deeper = more fog color)
    vec3 fogColor = u_UnderwaterTint * 0.3; // Dark water fog
    vec3 foggedColor = mix(tintedColor, fogColor, fogFactor);
    
    // ----- VIGNETTE -----
    // Darken edges of screen
    vec2 vignetteUV = TexCoord * 2.0 - 1.0;
    float vignetteFactor = 1.0 - length(vignetteUV) * u_Vignette;
    vignetteFactor = clamp(vignetteFactor, 0.3, 1.0);
    
    foggedColor *= vignetteFactor;
    
    // ----- LIGHT ABSORPTION -----
    // Water absorbs red first, then green, leaving blue
    // This adds extra absorption based on depth
    float absorption = 1.0 - fogFactor * 0.5;
    foggedColor.r *= absorption;
    foggedColor.g *= absorption * 1.1;
    foggedColor.b *= absorption * 1.3;
    
    // ----- CAUSTICS OVERLAY -----
    // Simple animated caustics pattern on screen
    float causticTime = time * 0.3;
    vec2 causticUV = distortedUV * 3.0;
    float caustic1 = sin(causticUV.x * 10.0 + causticTime) * sin(causticUV.y * 8.0 - causticTime);
    float caustic2 = sin(causticUV.x * 8.0 - causticTime * 0.7) * sin(causticUV.y * 12.0 + causticTime * 0.9);
    float caustics = (caustic1 + caustic2 + 2.0) * 0.25;
    caustics = pow(caustics, 3.0) * 0.15;
    
    // Add subtle caustics to bright areas
    float brightness = dot(foggedColor, vec3(0.299, 0.587, 0.114));
    foggedColor += vec3(caustics) * (1.0 - fogFactor) * smoothstep(0.3, 0.8, brightness);
    
    FragColor = vec4(foggedColor, 1.0);
}
