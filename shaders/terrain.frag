#version 330 core

// G-Buffer outputs for deferred rendering
layout (location = 0) out vec4 gPosition;   // RGB: Position, A: AO
layout (location = 1) out vec4 gNormal;     // RGB: Normal, A: Roughness
layout (location = 2) out vec4 gAlbedoSpec; // RGB: Albedo, A: Metallic
layout (location = 3) out vec4 gEmissive;   // RGB: Emissive

in vec3 FragPos;
in vec2 TexCoord;
in vec3 Normal;
in vec4 ClipPos;

// Splatmap and layer textures
uniform sampler2D u_Splatmap;

uniform sampler2D u_Layer0_Albedo;
uniform sampler2D u_Layer0_Normal;
uniform float u_Layer0_Tiling;

uniform sampler2D u_Layer1_Albedo;
uniform sampler2D u_Layer1_Normal;
uniform float u_Layer1_Tiling;

uniform sampler2D u_Layer2_Albedo;
uniform sampler2D u_Layer2_Normal;
uniform float u_Layer2_Tiling;

uniform sampler2D u_Layer3_Albedo;
uniform sampler2D u_Layer3_Normal;
uniform float u_Layer3_Tiling;

// Material properties per layer
uniform float u_Roughness0 = 0.8;
uniform float u_Roughness1 = 0.9;
uniform float u_Roughness2 = 0.95;
uniform float u_Roughness3 = 0.85;

// Triplanar settings
uniform int u_UseTriplanar = 1;
uniform float u_TriplanarSharpness = 4.0;

// ============================================================================
// Helper Functions
// ============================================================================

// Triplanar texture sampling
vec4 SampleTriplanar(sampler2D tex, vec3 worldPos, vec3 normal, float tiling) {
    vec3 blending = pow(abs(normal), vec3(u_TriplanarSharpness));
    blending = blending / (blending.x + blending.y + blending.z);
    
    vec4 xAxis = texture(tex, worldPos.yz * tiling);
    vec4 yAxis = texture(tex, worldPos.xz * tiling);
    vec4 zAxis = texture(tex, worldPos.xy * tiling);
    
    return xAxis * blending.x + yAxis * blending.y + zAxis * blending.z;
}

// Sample normal map with triplanar
vec3 SampleNormalTriplanar(sampler2D normalMap, vec3 worldPos, vec3 normal, float tiling) {
    vec3 blending = pow(abs(normal), vec3(u_TriplanarSharpness));
    blending = blending / (blending.x + blending.y + blending.z);
    
    vec3 nX = texture(normalMap, worldPos.yz * tiling).rgb * 2.0 - 1.0;
    vec3 nY = texture(normalMap, worldPos.xz * tiling).rgb * 2.0 - 1.0;
    vec3 nZ = texture(normalMap, worldPos.xy * tiling).rgb * 2.0 - 1.0;
    
    // Blend normals
    vec3 result = normalize(nX * blending.x + nY * blending.y + nZ * blending.z);
    return result;
}

// Blend two normals
vec3 BlendNormals(vec3 n1, vec3 n2) {
    return normalize(vec3(n1.xy + n2.xy, n1.z * n2.z));
}

// ============================================================================
// Main
// ============================================================================

void main()
{
    // Sample splatmap weights
    vec4 splat = texture(u_Splatmap, TexCoord);
    float w0 = splat.r;
    float w1 = splat.g;
    float w2 = splat.b;
    float w3 = splat.a;
    
    // Normalize weights
    float totalWeight = w0 + w1 + w2 + w3;
    if (totalWeight > 0.001) {
        w0 /= totalWeight;
        w1 /= totalWeight;
        w2 /= totalWeight;
        w3 /= totalWeight;
    } else {
        w0 = 1.0;
    }
    
    vec3 worldNormal = normalize(Normal);
    
    // Sample albedo for each layer
    vec3 albedo0, albedo1, albedo2, albedo3;
    vec3 normal0, normal1, normal2, normal3;
    
    if (u_UseTriplanar == 1) {
        // Use triplanar for steep slopes
        float slope = 1.0 - worldNormal.y;
        float triplanarBlend = smoothstep(0.3, 0.7, slope);
        
        // Layer 0
        vec3 planarAlbedo0 = texture(u_Layer0_Albedo, TexCoord * u_Layer0_Tiling).rgb;
        vec3 triAlbedo0 = SampleTriplanar(u_Layer0_Albedo, FragPos, worldNormal, u_Layer0_Tiling * 0.1).rgb;
        albedo0 = mix(planarAlbedo0, triAlbedo0, triplanarBlend);
        
        vec3 planarNormal0 = texture(u_Layer0_Normal, TexCoord * u_Layer0_Tiling).rgb * 2.0 - 1.0;
        vec3 triNormal0 = SampleNormalTriplanar(u_Layer0_Normal, FragPos, worldNormal, u_Layer0_Tiling * 0.1);
        normal0 = mix(planarNormal0, triNormal0, triplanarBlend);
        
        // Layer 1
        vec3 planarAlbedo1 = texture(u_Layer1_Albedo, TexCoord * u_Layer1_Tiling).rgb;
        vec3 triAlbedo1 = SampleTriplanar(u_Layer1_Albedo, FragPos, worldNormal, u_Layer1_Tiling * 0.1).rgb;
        albedo1 = mix(planarAlbedo1, triAlbedo1, triplanarBlend);
        
        vec3 planarNormal1 = texture(u_Layer1_Normal, TexCoord * u_Layer1_Tiling).rgb * 2.0 - 1.0;
        vec3 triNormal1 = SampleNormalTriplanar(u_Layer1_Normal, FragPos, worldNormal, u_Layer1_Tiling * 0.1);
        normal1 = mix(planarNormal1, triNormal1, triplanarBlend);
        
        // Layer 2 (rock - always triplanar)
        albedo2 = SampleTriplanar(u_Layer2_Albedo, FragPos, worldNormal, u_Layer2_Tiling * 0.1).rgb;
        normal2 = SampleNormalTriplanar(u_Layer2_Normal, FragPos, worldNormal, u_Layer2_Tiling * 0.1);
        
        // Layer 3
        vec3 planarAlbedo3 = texture(u_Layer3_Albedo, TexCoord * u_Layer3_Tiling).rgb;
        vec3 triAlbedo3 = SampleTriplanar(u_Layer3_Albedo, FragPos, worldNormal, u_Layer3_Tiling * 0.1).rgb;
        albedo3 = mix(planarAlbedo3, triAlbedo3, triplanarBlend);
        
        vec3 planarNormal3 = texture(u_Layer3_Normal, TexCoord * u_Layer3_Tiling).rgb * 2.0 - 1.0;
        vec3 triNormal3 = SampleNormalTriplanar(u_Layer3_Normal, FragPos, worldNormal, u_Layer3_Tiling * 0.1);
        normal3 = mix(planarNormal3, triNormal3, triplanarBlend);
    } else {
        // Simple planar sampling
        albedo0 = texture(u_Layer0_Albedo, TexCoord * u_Layer0_Tiling).rgb;
        albedo1 = texture(u_Layer1_Albedo, TexCoord * u_Layer1_Tiling).rgb;
        albedo2 = texture(u_Layer2_Albedo, TexCoord * u_Layer2_Tiling).rgb;
        albedo3 = texture(u_Layer3_Albedo, TexCoord * u_Layer3_Tiling).rgb;
        
        normal0 = texture(u_Layer0_Normal, TexCoord * u_Layer0_Tiling).rgb * 2.0 - 1.0;
        normal1 = texture(u_Layer1_Normal, TexCoord * u_Layer1_Tiling).rgb * 2.0 - 1.0;
        normal2 = texture(u_Layer2_Normal, TexCoord * u_Layer2_Tiling).rgb * 2.0 - 1.0;
        normal3 = texture(u_Layer3_Normal, TexCoord * u_Layer3_Tiling).rgb * 2.0 - 1.0;
    }
    
    // Blend layers based on splatmap
    vec3 finalAlbedo = albedo0 * w0 + albedo1 * w1 + albedo2 * w2 + albedo3 * w3;
    vec3 finalNormalTS = normal0 * w0 + normal1 * w1 + normal2 * w2 + normal3 * w3;
    float finalRoughness = u_Roughness0 * w0 + u_Roughness1 * w1 + u_Roughness2 * w2 + u_Roughness3 * w3;
    
    // Transform tangent-space normal to world space (simplified - terrain is mostly flat)
    // For proper implementation, would need TBN matrix from terrain
    vec3 finalNormal = normalize(worldNormal + finalNormalTS * 0.3);
    
    // Output to G-Buffer
    gPosition = vec4(FragPos, 1.0);                        // Position, AO=1.0
    gNormal = vec4(finalNormal, finalRoughness);           // Normal, Roughness
    gAlbedoSpec = vec4(finalAlbedo, 0.0);                  // Albedo, Metallic=0
    gEmissive = vec4(0.0, 0.0, 0.0, 0.0);                  // No emissive
}
