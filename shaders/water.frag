#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoords;
in vec3 Normal;
in vec4 ClipSpace;
in float WaveHeight; // From vertex shader (Jacobian or height)

// Camera & Scene
uniform vec3 u_ViewPos;
uniform vec3 u_LightDir;
uniform vec3 u_LightColor;
uniform float u_Time;

// Water Colors
uniform vec3 u_DeepColor;
uniform vec3 u_ShallowColor;
uniform float u_Clarity;

// Absorption & Scattering
uniform vec3 u_AbsorptionColor;
uniform float u_AbsorptionScale;
uniform vec3 u_ScatterColor;
uniform float u_ScatterStrength;

// Foam
uniform float u_FoamIntensity;
uniform float u_FoamThreshold;
uniform float u_FoamFalloff;
uniform vec3 u_FoamColor;
uniform float u_ShorelineFoamWidth;

// Specular / PBR
uniform float u_Roughness;
uniform float u_SpecularIntensity;

// Textures
uniform samplerCube skybox;
uniform sampler2D u_NormalMap;
uniform sampler2D u_DerivativesMap;   // FFT Normal
uniform sampler2D u_RefractionMap;
uniform sampler2D u_FoamTexture;
uniform sampler2D u_DepthTexture;     // Scene depth for shoreline effects

// Planar Reflection
uniform sampler2D u_PlanarReflection;
uniform int u_UsePlanarReflection;
uniform float u_ReflectionDistortion;

// Screen dimensions for depth calculation
uniform float u_NearPlane;
uniform float u_FarPlane;

// Constants
const float PI = 3.14159265359;
const float WATER_IOR = 1.333;
const float AIR_IOR = 1.0;

// ============================================================================
// Helper Functions
// ============================================================================

// Schlick Fresnel approximation
float fresnelSchlick(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX Normal Distribution Function
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / max(denom, 0.0001);
}

// GGX Geometry function (Schlick-GGX)
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / max(denom, 0.0001);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// Linearize depth from depth buffer
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC
    return (2.0 * u_NearPlane * u_FarPlane) / (u_FarPlane + u_NearPlane - z * (u_FarPlane - u_NearPlane));
}

// ============================================================================
// Main
// ============================================================================

void main()
{
    // Screen coordinates
    vec2 ndc = (ClipSpace.xy / ClipSpace.w) * 0.5 + 0.5;
    
    // Sample FFT normal map
    vec3 fftNormal = texture(u_DerivativesMap, TexCoords).rgb;
    if (length(fftNormal) < 0.1) {
        fftNormal = vec3(0.0, 1.0, 0.0); // Fallback flat normal
    }
    vec3 normal = normalize(fftNormal);
    
    // View direction
    vec3 viewDir = normalize(u_ViewPos - FragPos);
    float NdotV = max(dot(normal, viewDir), 0.0);
    
    // ----- DEPTH CALCULATION -----
    float sceneDepth = texture(u_DepthTexture, ndc).r;
    float linearSceneDepth = linearizeDepth(sceneDepth);
    float waterSurfaceDepth = linearizeDepth(gl_FragCoord.z);
    float waterDepth = max(linearSceneDepth - waterSurfaceDepth, 0.0);
    
    // ----- REFRACTION -----
    vec2 refractionDistortion = normal.xz * 0.05 * min(waterDepth, 1.0);
    vec2 refractionUV = clamp(ndc + refractionDistortion, 0.001, 0.999);
    vec3 refractedColor = texture(u_RefractionMap, refractionUV).rgb;
    
    // ----- ABSORPTION (Beer-Lambert Law) -----
    // Light is absorbed exponentially as it travels through water
    vec3 absorption = exp(-u_AbsorptionColor * waterDepth * u_AbsorptionScale);
    refractedColor *= absorption;
    
    // ----- DEPTH-BASED COLOR -----
    float depthFactor = 1.0 - exp(-waterDepth * u_Clarity * 0.5);
    vec3 waterColor = mix(u_ShallowColor, u_DeepColor, depthFactor);
    refractedColor = mix(refractedColor, refractedColor * waterColor, depthFactor);
    
    // ----- SUBSURFACE SCATTERING -----
    // Light scattering through water at shallow viewing angles
    vec3 lightDir = normalize(u_LightDir);
    float scatterDot = pow(max(dot(viewDir, -lightDir), 0.0), 4.0);
    float heightFactor = max(WaveHeight * 2.0, 0.0); // Higher waves scatter more
    vec3 subsurfaceScatter = u_ScatterColor * scatterDot * u_ScatterStrength * (1.0 - NdotV);
    subsurfaceScatter *= (1.0 + heightFactor);
    
    // ----- REFLECTION -----
    vec3 reflection;
    if (u_UsePlanarReflection == 1) {
        vec2 reflectUV = ndc;
        reflectUV.y = 1.0 - reflectUV.y;
        reflectUV += normal.xz * u_ReflectionDistortion;
        reflectUV = clamp(reflectUV, 0.001, 0.999);
        reflection = texture(u_PlanarReflection, reflectUV).rgb;
    } else {
        vec3 reflectDir = reflect(-viewDir, normal);
        reflection = texture(skybox, reflectDir).rgb;
    }
    
    // ----- FRESNEL -----
    float f0 = pow((AIR_IOR - WATER_IOR) / (AIR_IOR + WATER_IOR), 2.0); // ~0.02
    float fresnel = fresnelSchlick(NdotV, f0);
    
    // ----- SPECULAR (GGX BRDF) -----
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float NDF = distributionGGX(normal, halfwayDir, u_Roughness);
    float G = geometrySmith(normal, viewDir, lightDir, u_Roughness);
    float F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), f0);
    
    float NdotL = max(dot(normal, lightDir), 0.0);
    float specularFactor = (NDF * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3 specular = specularFactor * u_LightColor * u_SpecularIntensity * NdotL;
    
    // ----- FOAM -----
    vec3 foamContribution = vec3(0.0);
    
    // Wave crest foam (based on wave height / Jacobian)
    float waveFoam = smoothstep(u_FoamThreshold, u_FoamThreshold + 0.3, WaveHeight);
    
    // Shoreline foam (based on water depth)
    float shoreFoam = 1.0 - smoothstep(0.0, u_ShorelineFoamWidth, waterDepth);
    shoreFoam *= shoreFoam; // Quadratic falloff
    
    // Sample foam texture
    vec2 foamUV = TexCoords * 4.0 + u_Time * 0.02;
    float foamPattern = texture(u_FoamTexture, foamUV).r;
    foamPattern *= texture(u_FoamTexture, foamUV * 0.5 + vec2(0.5)).r; // Layer for complexity
    
    float totalFoam = max(waveFoam, shoreFoam) * foamPattern * u_FoamIntensity;
    totalFoam = pow(totalFoam, u_FoamFalloff);
    foamContribution = u_FoamColor * totalFoam;
    
    // ----- FINAL COMPOSITION -----
    // Blend refraction and reflection based on Fresnel
    vec3 waterSurface = mix(refractedColor, reflection, fresnel);
    
    // Add subsurface scattering
    waterSurface += subsurfaceScatter;
    
    // Add specular highlights
    waterSurface += specular;
    
    // Add foam on top
    waterSurface = mix(waterSurface, foamContribution + waterSurface * 0.3, totalFoam);
    
    // HDR output
    FragColor = vec4(waterSurface, 1.0);
}
