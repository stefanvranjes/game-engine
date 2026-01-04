#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gEmissive;
uniform sampler2DArray shadowMap;

struct Light {
    int type; // 0 = Directional, 1 = Point, 2 = Spot
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    
    float constant;
    float linear;
    float quadratic;
    
    float cutOff;
    float outerCutOff;
    
    float range;
    float shadowSoftness;
    int castsShadows;
    float lightSize;
};

#define MAX_LIGHTS 32
uniform Light u_Lights[MAX_LIGHTS];
uniform int u_LightCount;
uniform vec3 u_ViewPos;
uniform mat4 cascadeLightSpaceMatrices[3];
uniform float cascadePlaneDistances[3];
uniform mat4 view; // Need view matrix for depth calculation
uniform bool u_ShowCascades; // toggle cascade visualization
uniform float u_ShadowFadeStart; // Distance where shadows start fading
uniform float u_ShadowFadeEnd; // Distance where shadows completely disappear

uniform samplerCube pointShadowMaps[4]; // Support up to 4 point lights with shadows
uniform sampler2D spotShadowMaps[4]; // Support up to 4 spot lights with shadows
uniform mat4 spotLightSpaceMatrices[4];

// IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

// SSAO
uniform sampler2D ssaoTexture;
uniform int ssaoEnabled;

// SSR
uniform sampler2D ssrTexture;
uniform int ssrEnabled;

// Light Probe
uniform samplerCube probeIrradianceMap;
uniform samplerCube probePrefilterMap;
uniform int u_HasLightProbe;
uniform vec3 u_ProbePos;
uniform float u_ProbeRadius;

// Reflection Probes
uniform int u_ReflectionProbeCount;
uniform vec3 u_ReflectionProbePositions[4];
uniform float u_ReflectionProbeRadii[4];
uniform samplerCube u_ReflectionProbeCubemaps[4];

// Volumetric Fog
uniform sampler2D volumetricFogTexture;
uniform int volumetricFogEnabled;

// Global Illumination
uniform int u_GIEnabled;
uniform int u_GITechnique;  // 0=None, 1=VCT, 2=LPV, 3=SSGI, 4=Hybrid, 5=Probes, 6=ProbesVCT
uniform float u_GIIntensity;
uniform sampler2D giTexture;  // Pre-computed GI from GI pass
uniform sampler3D voxelAlbedo;  // For VCT
uniform sampler3D voxelNormal;  // For VCT
uniform sampler3D lpvTextureR;  // For LPV
uniform sampler3D lpvTextureG;  // For LPV
uniform sampler3D lpvTextureB;  // For LPV
uniform vec3 u_VoxelGridMin;
uniform vec3 u_VoxelGridMax;
uniform vec3 u_LPVGridMin;
uniform vec3 u_LPVGridMax;

// Probe-based GI
uniform int u_UseProbes;
uniform samplerBuffer u_ProbeData;  // Probe SSBO as texture buffer
uniform vec3 u_ProbeGridMin;
uniform vec3 u_ProbeGridMax;
uniform ivec3 u_ProbeGridResolution;
uniform float u_ProbeBlendWeight;

// Array of offset direction for sampling
vec3 gridSamplingDisk[20] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

float PointShadowCalculation(vec3 fragPos, vec3 lightPos, float far_plane, samplerCube shadowMap, float softness)
{
    vec3 fragToLight = fragPos - lightPos;
    float currentDepth = length(fragToLight);
    
    float shadow = 0.0;
    float bias = 0.15;
    int samples = 20;
    float viewDistance = length(u_ViewPos - fragPos);
    float diskRadius = (1.0 + (viewDistance / far_plane)) / 25.0 * softness;
    
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(shadowMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= far_plane;   // undo mapping [0;1]
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);
        
    return shadow;
}


// PCSS Helper Functions
#define BLOCKER_SEARCH_NUM_SAMPLES 16
#define PCF_NUM_SAMPLES 16

vec2 poissonDisk[16] = vec2[](
   vec2(-0.94201624, -0.39906216),
   vec2(0.94558609, -0.76890725),
   vec2(-0.094184101, -0.92938870),
   vec2(0.34495938, 0.29387760),
   vec2(-0.91588581, 0.45771432),
   vec2(-0.81544232, -0.87912464),
   vec2(-0.38277543, 0.27676845),
   vec2(0.97484398, 0.75648379),
   vec2(0.44323325, -0.97511554),
   vec2(0.53742981, -0.47373420),
   vec2(-0.26496911, -0.41893023),
   vec2(0.79197514, 0.19090188),
   vec2(-0.24188840, 0.99706507),
   vec2(-0.81409955, 0.91437590),
   vec2(0.19984126, 0.78641367),
   vec2(0.14383161, -0.14100790)
);

float FindBlockerDistance(sampler2DArray shadowMap, vec3 projCoords, int layer, float bias, float lightSize)
{
    float searchWidth = lightSize / projCoords.z; // Adjust search width based on depth
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    
    float blockerSum = 0.0;
    int numBlockers = 0;
    
    for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++)
    {
        vec2 offset = poissonDisk[i] * searchWidth * texelSize;
        float shadowMapDepth = texture(shadowMap, vec3(projCoords.xy + offset, layer)).r;
        
        if (shadowMapDepth < projCoords.z - bias)
        {
            blockerSum += shadowMapDepth;
            numBlockers++;
        }
    }
    
    if (numBlockers == 0)
        return -1.0; // No blockers found
        
    return blockerSum / float(numBlockers);
}

float PCF_Filter(sampler2DArray shadowMap, vec3 projCoords, int layer, float bias, float filterRadius)
{
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    
    for (int i = 0; i < PCF_NUM_SAMPLES; i++)
    {
        vec2 offset = poissonDisk[i] * filterRadius * texelSize;
        float pcfDepth = texture(shadowMap, vec3(projCoords.xy + offset, layer)).r;
        shadow += (projCoords.z - bias) > pcfDepth ? 1.0 : 0.0;
    }
    
    return shadow / float(PCF_NUM_SAMPLES);
}

float PCSS(sampler2DArray shadowMap, vec3 projCoords, int layer, float bias, float lightSize)
{
    // Step 1: Blocker search
    float avgBlockerDistance = FindBlockerDistance(shadowMap, projCoords, layer, bias, lightSize);
    
    if (avgBlockerDistance < 0.0)
        return 0.0; // No blockers, fully lit
    
    // Step 2: Penumbra size estimation
    float penumbraWidth = (projCoords.z - avgBlockerDistance) / avgBlockerDistance;
    float filterRadius = penumbraWidth * lightSize;
    
    // Clamp filter radius to reasonable values
    filterRadius = clamp(filterRadius, 1.0, 10.0);
    
    // Step 3: PCF filtering with variable kernel
    return PCF_Filter(shadowMap, projCoords, layer, bias, filterRadius);
}

float ShadowCalculation(vec3 fragPos, vec3 normal, vec3 lightDir, float lightSize)
{
    // Select cascade layer
    vec4 fragPosViewSpace = view * vec4(fragPos, 1.0);
    float depthValue = abs(fragPosViewSpace.z);

    int layer = -1;
    for (int i = 0; i < 3; ++i)
    {
        if (depthValue < cascadePlaneDistances[i])
        {
            layer = i;
            break;
        }
    }
    if (layer == -1)
    {
        layer = 2;
    }

    vec4 fragPosLightSpace = cascadeLightSpaceMatrices[layer] * vec4(fragPos, 1.0);
    
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Calculate bias to prevent shadow acne
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    if (layer == 2) bias *= 0.5;
    else bias *= 1.0 / (float(layer) + 1.0);
    
    // Use PCSS for soft shadows
    float shadow = PCSS(shadowMap, projCoords, layer, bias, lightSize);
    
    // Keep the shadow at 0.0 when outside the far_plane region
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

// PCSS for Spot Lights
float FindBlockerDistanceSpot(sampler2D shadowMap, vec2 projCoords, float currentDepth, float bias, float lightSize)
{
    float searchWidth = lightSize / currentDepth;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    float blockerSum = 0.0;
    int numBlockers = 0;
    
    for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++)
    {
        vec2 offset = poissonDisk[i] * searchWidth * texelSize;
        float shadowMapDepth = texture(shadowMap, projCoords + offset).r;
        
        if (shadowMapDepth < currentDepth - bias)
        {
            blockerSum += shadowMapDepth;
            numBlockers++;
        }
    }
    
    if (numBlockers == 0)
        return -1.0;
        
    return blockerSum / float(numBlockers);
}

float PCF_FilterSpot(sampler2D shadowMap, vec2 projCoords, float currentDepth, float bias, float filterRadius)
{
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    float shadow = 0.0;
    
    for (int i = 0; i < PCF_NUM_SAMPLES; i++)
    {
        vec2 offset = poissonDisk[i] * filterRadius * texelSize;
        float pcfDepth = texture(shadowMap, projCoords + offset).r;
        shadow += (currentDepth - bias) > pcfDepth ? 1.0 : 0.0;
    }
    
    return shadow / float(PCF_NUM_SAMPLES);
}

float SpotShadowCalculation(vec3 fragPos, vec3 normal, vec3 lightDir, mat4 lightSpaceMatrix, sampler2D shadowMap, float lightSize)
{
    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
    
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Calculate bias to prevent shadow acne
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // Use PCSS for soft shadows
    float avgBlockerDistance = FindBlockerDistanceSpot(shadowMap, projCoords.xy, currentDepth, bias, lightSize);
    
    float shadow = 0.0;
    if (avgBlockerDistance < 0.0)
    {
        shadow = 0.0; // No blockers, fully lit
    }
    else
    {
        // Penumbra size estimation
        float penumbraWidth = (currentDepth - avgBlockerDistance) / avgBlockerDistance;
        float filterRadius = penumbraWidth * lightSize;
        filterRadius = clamp(filterRadius, 1.0, 10.0);
        
        // PCF filtering with variable kernel
        shadow = PCF_FilterSpot(shadowMap, projCoords.xy, currentDepth, bias, filterRadius);
    }
    
    // Keep the shadow at 0.0 when outside the far_plane region
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
    
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;
    
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Sample LPV for indirect lighting
vec3 SampleLPV(vec3 worldPos, vec3 normal)
{
    // Convert world position to LPV grid coordinates
    vec3 gridPos = (worldPos - u_LPVGridMin) / (u_LPVGridMax - u_LPVGridMin);
    
    if (any(lessThan(gridPos, vec3(0.0))) || any(greaterThan(gridPos, vec3(1.0)))) {
        return vec3(0.0);
    }
    
    // Sample spherical harmonic coefficients
    vec4 shR = texture(lpvTextureR, gridPos);
    vec4 shG = texture(lpvTextureG, gridPos);
    vec4 shB = texture(lpvTextureB, gridPos);
    
    // Evaluate SH with normal direction (simplified)
    // In full implementation, use proper SH evaluation
    vec3 indirectLight = vec3(shR.r, shG.r, shB.r);
    
    return indirectLight;
}

// Sample probe grid for indirect lighting
vec3 SampleProbeGrid(vec3 worldPos, vec3 normal)
{
    // Convert world position to grid coordinates
    vec3 gridPos = (worldPos - u_ProbeGridMin) / (u_ProbeGridMax - u_ProbeGridMin);
    gridPos = clamp(gridPos, vec3(0.0), vec3(1.0));
    
    vec3 gridCoord = gridPos * vec3(u_ProbeGridResolution - 1);
    ivec3 baseCoord = ivec3(floor(gridCoord));
    vec3 frac = fract(gridCoord);
    
    // Sample 8 surrounding probes with trilinear interpolation
    vec3 irradiance = vec3(0.0);
    
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ivec3 coord = baseCoord + ivec3(x, y, z);
                coord = clamp(coord, ivec3(0), u_ProbeGridResolution - 1);
                
                int probeIndex = coord.x + coord.y * u_ProbeGridResolution.x + 
                                coord.z * u_ProbeGridResolution.x * u_ProbeGridResolution.y;
                
                // Fetch SH coefficients (simplified - L0 + L1 bands)
                // Each probe: 32 floats (3 pos + 27 SH + flags + radius)
                int baseOffset = probeIndex * 32 + 3;  // Skip position
                
                // Evaluate SH (L0 + L1 only for performance)
                vec3 sh_l0 = vec3(
                    texelFetch(u_ProbeData, baseOffset + 0).r,
                    texelFetch(u_ProbeData, baseOffset + 9).r,
                    texelFetch(u_ProbeData, baseOffset + 18).r
                ) * 0.282095;
                
                vec3 sh_l1_y = vec3(
                    texelFetch(u_ProbeData, baseOffset + 1).r,
                    texelFetch(u_ProbeData, baseOffset + 10).r,
                    texelFetch(u_ProbeData, baseOffset + 19).r
                ) * 0.488603 * normal.y;
                
                vec3 sh_l1_z = vec3(
                    texelFetch(u_ProbeData, baseOffset + 2).r,
                    texelFetch(u_ProbeData, baseOffset + 11).r,
                    texelFetch(u_ProbeData, baseOffset + 20).r
                ) * 0.488603 * normal.z;
                
                vec3 sh_l1_x = vec3(
                    texelFetch(u_ProbeData, baseOffset + 3).r,
                    texelFetch(u_ProbeData, baseOffset + 12).r,
                    texelFetch(u_ProbeData, baseOffset + 21).r
                ) * 0.488603 * normal.x;
                
                vec3 probeIrradiance = sh_l0 + sh_l1_y + sh_l1_z + sh_l1_x;
                probeIrradiance = max(probeIrradiance, vec3(0.0));
                
                // Trilinear weight
                vec3 weight3D = mix(vec3(1.0) - frac, frac, vec3(x, y, z));
                float weight = weight3D.x * weight3D.y * weight3D.z;
                
                irradiance += probeIrradiance * weight;
            }
        }
    }
    
    return irradiance;
}

void main()
{
    // Retrieve data from G-Buffer
    vec4 gPosData = texture(gPosition, TexCoord);
    vec3 FragPos = gPosData.rgb;
    float AO = gPosData.a;
    
    vec4 gNormData = texture(gNormal, TexCoord);
    vec3 Normal = gNormData.rgb;
    float Roughness = gNormData.a;
    
    vec4 gAlbedoData = texture(gAlbedoSpec, TexCoord);
    vec3 Albedo = gAlbedoData.rgb;
    float Metallic = gAlbedoData.a;

    vec3 Emissive = texture(gEmissive, TexCoord).rgb;
    
    vec3 N = normalize(Normal);
    vec3 V = normalize(u_ViewPos - FragPos);
    
    // Apply SSAO
    if (ssaoEnabled == 1) {
        float ssao = texture(ssaoTexture, TexCoord).r;
        AO *= ssao;
    }
    
    // F0: Surface reflection at zero incidence
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, Albedo, Metallic);
    
    vec3 Lo = vec3(0.0);
    
    // Calculate shadow fade factor based on distance
    float fragDistance = length(u_ViewPos - FragPos);
    float shadowFadeFactor = 1.0 - smoothstep(u_ShadowFadeStart, u_ShadowFadeEnd, fragDistance);
    
    int pointShadowIndex = 0;
    
    for(int i = 0; i < u_LightCount && i < MAX_LIGHTS; ++i) {
        // Calculate per-light radiance
        vec3 L = vec3(0.0);
        vec3 H = vec3(0.0);
        float attenuation = 1.0;
        vec3 radiance = vec3(0.0);
        float shadow = 0.0;
        
        if (u_Lights[i].type == 0) { // Directional
             L = normalize(-u_Lights[i].direction);
             attenuation = 1.0; // Directional light has no attenuation
             
             if(i == 0 && u_Lights[i].castsShadows == 1) {
                 shadow = ShadowCalculation(FragPos, N, L, u_Lights[i].lightSize);
                 shadow *= shadowFadeFactor;
             }
        } else if (u_Lights[i].type == 1) { // Point
             L = normalize(u_Lights[i].position - FragPos);
             float distance = length(u_Lights[i].position - FragPos);
             attenuation = 1.0 / (distance * distance); // Inverse square law
             
             // Range cutoff (optional for PBR but good for performance)
             if (u_Lights[i].range > 0.0) {
                float cutoff = u_Lights[i].range;
                float rangeFactor = max(min(1.0 - pow(distance / cutoff, 4.0), 1.0), 0.0);
                attenuation *= rangeFactor;
             }
             
             if (u_Lights[i].castsShadows == 1 && pointShadowIndex < 4) {
                 if (pointShadowIndex == 0) shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[0], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 1) shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[1], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 2) shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[2], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 3) shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[3], u_Lights[i].shadowSoftness);
                 shadow *= shadowFadeFactor;
                 pointShadowIndex++;
             }
        } else if (u_Lights[i].type == 2) { // Spot
             L = normalize(u_Lights[i].position - FragPos);
             float distance = length(u_Lights[i].position - FragPos);
             attenuation = 1.0 / (distance * distance);
             
             float theta = dot(L, normalize(-u_Lights[i].direction)); 
             float epsilon = u_Lights[i].cutOff - u_Lights[i].outerCutOff;
             float intensity = clamp((theta - u_Lights[i].outerCutOff) / epsilon, 0.0, 1.0);
             attenuation *= intensity;
             
             // Shadow logic for spot (simplified for brevity, assume similar to point)
             // ... (Spot shadow logic omitted for brevity in this PBR block, can be re-added if needed)
        }
        
        radiance = u_Lights[i].color * u_Lights[i].intensity * attenuation;
        
        // Cook-Torrance BRDF
        H = normalize(V + L);
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        
        float NDF = DistributionGGX(N, H, Roughness);
        float G   = GeometrySmith(N, V, L, Roughness);
        vec3 F    = FresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        vec3 specular     = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - Metallic;
        
        Lo += (kD * Albedo / 3.14159265359 + specular) * radiance * NdotL * (1.0 - shadow);
    }
    
    
    // IBL Ambient Lighting
    vec3 R = reflect(-V, N);
    
    // Diffuse IBL
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * Albedo;
    
    // Specular IBL
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, Roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), Roughness)).rg;
    vec3 specular = prefilteredColor * (F0 * brdf.x + brdf.y);
    
    // Apply SSR if enabled
    if (ssrEnabled == 1) {
        vec4 ssrSample = texture(ssrTexture, TexCoord);
        vec3 ssrColor = ssrSample.rgb;
        float ssrStrength = ssrSample.a;
        
        // Blend SSR with IBL based on roughness and SSR confidence
        // SSR is more prominent on smooth surfaces
        float ssrBlend = ssrStrength * (1.0 - Roughness * 0.8);
        
        // Apply BRDF to SSR color (same as IBL)
        vec3 ssrSpecular = ssrColor * (F0 * brdf.x + brdf.y);
        
        // Blend SSR with IBL specular
        specular = mix(specular, ssrSpecular, ssrBlend);
    }
    
    // Combine IBL
    vec3 kS = FresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - Metallic;
    
    // Local Light Probe Blending
    if (u_HasLightProbe == 1) {
        float dist = length(FragPos - u_ProbePos);
        float blend = 1.0 - smoothstep(u_ProbeRadius * 0.8, u_ProbeRadius, dist); // 1.0 at center, 0.0 at radius
        
        if (blend > 0.0) {
            // Sample Probe
            vec3 probeIrradiance = texture(probeIrradianceMap, N).rgb;
            vec3 probeDiffuse = probeIrradiance * Albedo;
            
            vec3 probePrefilteredColor = textureLod(probePrefilterMap, R, Roughness * MAX_REFLECTION_LOD).rgb;
            vec3 probeSpecular = probePrefilteredColor * (F0 * brdf.x + brdf.y);
            
            // Blend
            diffuse = mix(diffuse, probeDiffuse, blend);
            specular = mix(specular, probeSpecular, blend);
        }
    }
    
    // Reflection Probe Blending
    if (u_ReflectionProbeCount > 0) {
        // Find closest probe within range
        float closestDist = 1000000.0;
        int closestProbe = -1;
        
        for (int i = 0; i < u_ReflectionProbeCount && i < 4; ++i) {
            float dist = length(FragPos - u_ReflectionProbePositions[i]);
            if (dist < u_ReflectionProbeRadii[i] && dist < closestDist) {
                closestDist = dist;
                closestProbe = i;
            }
        }
        
        // Blend with closest probe
        if (closestProbe >= 0) {
            float blend = 1.0 - smoothstep(u_ReflectionProbeRadii[closestProbe] * 0.7, u_ReflectionProbeRadii[closestProbe], closestDist);
            
            if (blend > 0.0) {
                // Sample reflection probe cubemap (using if-else instead of dynamic indexing)
                vec3 reflectionColor = vec3(0.0);
                if (closestProbe == 0) {
                    reflectionColor = textureLod(u_ReflectionProbeCubemaps[0], R, Roughness * MAX_REFLECTION_LOD).rgb;
                } else if (closestProbe == 1) {
                    reflectionColor = textureLod(u_ReflectionProbeCubemaps[1], R, Roughness * MAX_REFLECTION_LOD).rgb;
                } else if (closestProbe == 2) {
                    reflectionColor = textureLod(u_ReflectionProbeCubemaps[2], R, Roughness * MAX_REFLECTION_LOD).rgb;
                } else if (closestProbe == 3) {
                    reflectionColor = textureLod(u_ReflectionProbeCubemaps[3], R, Roughness * MAX_REFLECTION_LOD).rgb;
                }
                
                vec3 reflectionSpecular = reflectionColor * (F0 * brdf.x + brdf.y);
                
                // Blend with global IBL specular
                specular = mix(specular, reflectionSpecular, blend);
            }
        }
    }
    
    // === Global Illumination ===
    vec3 indirectDiffuse = vec3(0.0);
    
    if (u_GIEnabled == 1) {
        if (u_GITechnique == 1) {  // VCT
            // Sample pre-computed GI texture from cone tracing pass
            indirectDiffuse = texture(giTexture, TexCoord).rgb;
        } else if (u_GITechnique == 2) {  // LPV
            indirectDiffuse = SampleLPV(FragPos, N);
        } else if (u_GITechnique == 3) {  // SSGI
            indirectDiffuse = texture(giTexture, TexCoord).rgb;
        } else if (u_GITechnique == 4) {  // Hybrid (VCT + SSGI)
            vec3 vctGI = texture(giTexture, TexCoord).rgb;
            // SSGI is blended in the GI pass itself
            indirectDiffuse = vctGI;
        } else if (u_GITechnique == 5) {  // Probes only
            if (u_UseProbes == 1) {
                indirectDiffuse = SampleProbeGrid(FragPos, N);
            }
        } else if (u_GITechnique == 6) {  // Probes + VCT hybrid
            vec3 probeGI = vec3(0.0);
            if (u_UseProbes == 1) {
                probeGI = SampleProbeGrid(FragPos, N);
            }
            vec3 vctGI = texture(giTexture, TexCoord).rgb;
            indirectDiffuse = mix(probeGI, vctGI, u_ProbeBlendWeight);
        }
        
        // Apply GI intensity
        indirectDiffuse *= u_GIIntensity;
    }
    
    vec3 ambient = (kD * diffuse + specular) * AO + indirectDiffuse;
    
    // Fallback ambient if IBL is not loaded (prevents black screen)
    float iblStrength = max(max(irradiance.r, irradiance.g), irradiance.b);
    if (iblStrength < 0.001) {
        ambient = vec3(0.03) * Albedo * AO; // Simple ambient fallback
    }
    
    vec3 color = ambient + Lo + Emissive; // Add emissive component
    
    // Apply Volumetric Fog
    if (volumetricFogEnabled == 1) {
        vec4 fogData = texture(volumetricFogTexture, TexCoord);
        vec3 fogColor = fogData.rgb;
        float transmittance = fogData.a;
        color = color * transmittance + fogColor;
    }

    FragColor = vec4(color, 1.0);
}
