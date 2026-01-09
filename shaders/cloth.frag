#version 330 core

// G-Buffer Outputs (Deferred Rendering)
layout (location = 0) out vec4 gPosition;   // RGB: Position, A: AO
layout (location = 1) out vec4 gNormal;     // RGB: Normal, A: Roughness
layout (location = 2) out vec4 gAlbedoSpec; // RGB: Albedo, A: Metallic
layout (location = 3) out vec3 gEmissive;   // RGB: Emissive
layout (location = 4) out vec2 gVelocity;   // RG: Motion Vector

// Input from Vertex Shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 ViewPos;
in vec4 CurrentPos;
in vec4 PreviousPos;
in mat3 TBN;

// Material Structure
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float roughness;
    float metallic;
    sampler2D texture;
    sampler2D specularMap;
    sampler2D normalMap;
    sampler2D roughnessMap;
    sampler2D metallicMap;
    sampler2D aoMap;
    sampler2D ormMap;  // Combined Occlusion-Roughness-Metallic
    sampler2D heightMap;
    float heightScale;
    sampler2D emissiveMap;
    vec3 emissiveColor;
};

uniform Material material;
uniform int u_HasTexture;
uniform int u_HasSpecularMap;
uniform int u_HasNormalMap;
uniform int u_HasRoughnessMap;
uniform int u_HasMetallicMap;
uniform int u_HasAOMap;
uniform int u_HasORMMap;
uniform int u_HasHeightMap;
uniform int u_HasEmissiveMap;

// Cloth-Specific Uniforms
uniform bool u_TwoSidedRendering;      // Enable two-sided rendering
uniform bool u_EnableSubsurface;       // Enable subsurface scattering approximation
uniform float u_Translucency;          // Translucency amount (0-1)
uniform float u_WrinkleScale;          // Wrinkle detail scale
uniform bool u_EnableWrinkleDetail;    // Enable procedural wrinkle enhancement

// Subsurface scattering approximation for thin cloth
float calculateTranslucency(vec3 normal, vec3 viewDir, vec3 lightDir) {
    if (!u_EnableSubsurface) {
        return 0.0;
    }
    
    // Simple translucency: light passing through thin material
    float NdotL = dot(normal, lightDir);
    float backlight = max(0.0, -NdotL);
    
    // View-dependent translucency
    float VdotL = max(0.0, dot(viewDir, -lightDir));
    
    return backlight * VdotL * u_Translucency;
}

// Procedural wrinkle detail (adds high-frequency normal variation)
vec3 addWrinkleDetail(vec3 normal, vec2 uv) {
    if (!u_EnableWrinkleDetail) {
        return normal;
    }
    
    // Generate procedural wrinkle pattern using noise-like functions
    float wrinkle1 = sin(uv.x * 50.0 * u_WrinkleScale) * cos(uv.y * 50.0 * u_WrinkleScale);
    float wrinkle2 = sin(uv.x * 80.0 * u_WrinkleScale + 1.5) * cos(uv.y * 80.0 * u_WrinkleScale + 2.3);
    
    // Combine wrinkles
    float wrinkleIntensity = (wrinkle1 * 0.6 + wrinkle2 * 0.4) * 0.05;
    
    // Perturb normal
    vec3 wrinkleNormal = normal;
    wrinkleNormal.xy += vec2(wrinkleIntensity);
    
    return normalize(wrinkleNormal);
}

void main()
{
    // Two-sided rendering: flip normal for back faces
    vec3 normal = Normal;
    if (u_TwoSidedRendering && !gl_FrontFacing) {
        normal = -normal;
    }
    
    // Sample material properties
    float ao = 1.0;
    float roughness = material.roughness;
    float metallic = material.metallic;
    
    if (u_HasORMMap == 1) {
        // Use combined ORM texture (R=AO, G=Roughness, B=Metallic)
        vec3 orm = texture(material.ormMap, TexCoord).rgb;
        ao = orm.r;
        roughness = orm.g;
        metallic = orm.b;
    } else {
        // Use separate textures
        if (u_HasAOMap == 1) {
            ao = texture(material.aoMap, TexCoord).r;
        }
        
        if (u_HasRoughnessMap == 1) {
            roughness = texture(material.roughnessMap, TexCoord).r;
        }
        
        if (u_HasMetallicMap == 1) {
            metallic = texture(material.metallicMap, TexCoord).r;
        }
    }
    
    // Cloth typically has low metallic values
    // Override if metallic is too high for cloth material
    metallic = min(metallic, 0.1);
    
    // Store fragment position and AO
    gPosition = vec4(FragPos, ao);
    
    // Apply normal mapping if available
    vec3 finalNormal = normal;
    
    if (u_HasNormalMap == 1) {
        // Sample normal map
        vec3 normalMapSample = texture(material.normalMap, TexCoord).rgb;
        normalMapSample = normalMapSample * 2.0 - 1.0;
        
        // Transform to world space using TBN matrix
        finalNormal = normalize(TBN * normalMapSample);
        
        // Flip for backfaces if two-sided
        if (u_TwoSidedRendering && !gl_FrontFacing) {
            finalNormal = -finalNormal;
        }
    }
    
    // Add procedural wrinkle detail
    finalNormal = addWrinkleDetail(finalNormal, TexCoord);
    
    // Store normal and roughness
    // Cloth typically has higher roughness than other materials
    roughness = max(roughness, 0.3); // Minimum roughness for cloth
    gNormal = vec4(finalNormal, roughness);
    
    // Sample albedo (diffuse color)
    vec3 albedo = material.diffuse;
    if (u_HasTexture == 1) {
        vec4 texColor = texture(material.texture, TexCoord);
        albedo = texColor.rgb;
        
        // Optional: Use alpha for translucency
        // This could be used for semi-transparent cloth
    }
    
    gAlbedoSpec.rgb = albedo;
    gAlbedoSpec.a = metallic;
    
    // Emissive (for glowing cloth materials)
    vec3 emissive = material.emissiveColor;
    if (u_HasEmissiveMap == 1) {
        emissive = texture(material.emissiveMap, TexCoord).rgb;
        if (length(material.emissiveColor) > 0.01) {
            emissive *= material.emissiveColor;
        }
    }
    gEmissive = emissive;
    
    // Calculate motion vector (screen-space velocity) for TAA
    vec2 currentNDC = CurrentPos.xy / CurrentPos.w;
    vec2 previousNDC = PreviousPos.xy / PreviousPos.w;
    gVelocity = (currentNDC - previousNDC) * 0.5;
    
    // Note: Subsurface scattering translucency is stored in the normal alpha channel
    // or can be added as a separate render target if needed
    // For now, it's handled in the lighting pass by checking material properties
}
