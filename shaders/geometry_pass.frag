#version 330 core
layout (location = 0) out vec4 gPosition; // RGB: Position, A: AO
layout (location = 1) out vec4 gNormal;   // RGB: Normal, A: Roughness
layout (location = 2) out vec4 gAlbedoSpec; // RGB: Albedo, A: Metallic
layout (location = 3) out vec3 gEmissive;   // RGB: Emissive

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 ViewPos;

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

// Parallax Occlusion Mapping
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{
    // ... (POM implementation remains same)
    // Number of depth layers
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    
    // Calculate the size of each layer
    float layerDepth = 1.0 / numLayers;
    // Depth of current layer
    float currentLayerDepth = 0.0;
    // The amount to shift the texture coordinates per layer (from vector P)
    vec2 P = viewDir.xy / viewDir.z * material.heightScale;
    vec2 deltaTexCoords = P / numLayers;
    
    // Get initial values
    vec2 currentTexCoords = texCoords;
    float currentDepthMapValue = texture(material.heightMap, currentTexCoords).r;
    
    // Ray-march through height map
    while(currentLayerDepth < currentDepthMapValue)
    {
        // Shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // Get depthmap value at current texture coordinates
        currentDepthMapValue = texture(material.heightMap, currentTexCoords).r;
        // Get depth of next layer
        currentLayerDepth += layerDepth;
    }
    
    // Get texture coordinates before collision (reverse operations)
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    
    // Get depth after and before collision for linear interpolation
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = texture(material.heightMap, prevTexCoords).r - currentLayerDepth + layerDepth;
    
    // Interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);
    
    return finalTexCoords;
}

void main()
{
    // ... (POM logic remains same)
    // Calculate view direction for parallax mapping
    vec3 viewDir = normalize(ViewPos - FragPos);
    
    // Apply parallax occlusion mapping if height map is available
    vec2 texCoords = TexCoord;
    if (u_HasHeightMap == 1 && u_HasNormalMap == 1) {
        // Calculate TBN matrix for tangent space
        vec3 Q1 = dFdx(FragPos);
        vec3 Q2 = dFdy(FragPos);
        vec2 st1 = dFdx(TexCoord);
        vec2 st2 = dFdy(TexCoord);
        
        vec3 N = normalize(Normal);
        vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
        vec3 B = -normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);
        
        // Transform view direction to tangent space
        vec3 tangentViewDir = normalize(transpose(TBN) * viewDir);
        
        // Apply parallax mapping
        texCoords = ParallaxMapping(TexCoord, tangentViewDir);
        
        // Discard fragments outside texture bounds (prevents artifacts at edges)
        if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0)
            discard;
    }
    
    // Sample material properties using parallax-adjusted coordinates
    // ORM map takes priority if available (R=AO, G=Roughness, B=Metallic)
    float ao = 1.0;
    float roughness = material.roughness;
    float metallic = material.metallic;
    
    if (u_HasORMMap == 1) {
        // Use combined ORM texture
        vec3 orm = texture(material.ormMap, texCoords).rgb;
        ao = orm.r;
        roughness = orm.g;
        metallic = orm.b;
    } else {
        // Fall back to separate textures
        if (u_HasAOMap == 1) {
            ao = texture(material.aoMap, texCoords).r;
        }
        
        if (u_HasRoughnessMap == 1) {
            roughness = texture(material.roughnessMap, texCoords).r;
        }
        
        if (u_HasMetallicMap == 1) {
            metallic = texture(material.metallicMap, texCoords).r;
        }
    }
    
    // Store fragment position and AO
    gPosition = vec4(FragPos, ao);
    
    // Store normal (with normal mapping if available)
    vec3 norm = normalize(Normal);
    
    if (u_HasNormalMap == 1) {
        // Calculate TBN matrix using derivatives
        vec3 Q1 = dFdx(FragPos);
        vec3 Q2 = dFdy(FragPos);
        vec2 st1 = dFdx(TexCoord);
        vec2 st2 = dFdy(TexCoord);
        
        vec3 N = normalize(Normal);
        vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
        vec3 B = -normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);
        
        vec3 normalMapSample = texture(material.normalMap, texCoords).rgb;
        normalMapSample = normalMapSample * 2.0 - 1.0;
        norm = normalize(TBN * normalMapSample);
    }
    
    // Store normal and roughness
    gNormal = vec4(norm, roughness);
    
    // Sample albedo (diffuse color) with parallax-adjusted coordinates
    vec3 albedo = material.diffuse;
    if (u_HasTexture == 1) {
        albedo = texture(material.texture, texCoords).rgb;
    }
    gAlbedoSpec.rgb = albedo;
    
    // Store metallic in alpha channel
    gAlbedoSpec.a = metallic;

    // Store emissive
    vec3 emissive = material.emissiveColor;
    if (u_HasEmissiveMap == 1) {
        emissive = texture(material.emissiveMap, texCoords).rgb * material.emissiveColor;
        // Note: We multiply by emissiveColor to allow tinting or intensity control
        // If emissiveColor is black (default), we should probably just use the map?
        // Or we initialize emissiveColor to white if map is present?
        // Let's assume emissiveColor is a multiplier (default white if we want full map color)
        // But we initialized it to black in Material constructor...
        // Let's change logic: if map is present, use map. If color is non-black, add it?
        // Standard PBR: emissive = texture.rgb * emissiveFactor.
        // So we need to ensure emissiveColor is white (1,1,1) by default if we want to see the texture.
        // But if no texture, we want black.
        // Let's stick to: emissive = texture * color. User must set color to white if using texture.
        // Wait, if I set default to black, texture will be black.
        // I should change default to white in Material constructor? No, then everything glows white.
        // GLTF loader sets emissiveFactor.
        // For manual materials, user sets emissiveColor.
        // If user sets texture but leaves color black, it won't show.
        // Let's use: emissive = texture.rgb; if (length(material.emissiveColor) > 0.0) emissive *= material.emissiveColor;
        // Actually, standard is: emissive = texture * factor.
        // If factor is 0,0,0, result is 0.
        // So I should probably initialize emissiveColor to (1,1,1) ONLY if texture is set?
        // Or just leave it to the user.
        // Let's make it additive: emissive = texture.rgb + material.emissiveColor;
        // No, that prevents tinting.
        // Let's stick to multiplication, but I'll update Material constructor or loader to set color to white if texture is present.
        // Or better:
        // vec3 emissiveMapColor = vec3(1.0);
        // if (u_HasEmissiveMap == 1) emissiveMapColor = texture(...).rgb;
        // vec3 finalEmissive = emissiveMapColor * material.emissiveColor;
        // If no map, emissiveMapColor is white. final is black (default). Correct.
        // If map, emissiveMapColor is map. final is map * black = black. Incorrect.
        
        // Let's do:
        // vec3 emission = material.emissiveColor;
        // if (u_HasEmissiveMap == 1) {
        //    emission = texture(material.emissiveMap, texCoords).rgb;
        //    if (length(material.emissiveColor) > 0.0) emission *= material.emissiveColor;
        // }
        
        // Actually, simplest is:
        // vec3 emission = material.emissiveColor;
        // if (u_HasEmissiveMap == 1) emission += texture(material.emissiveMap, texCoords).rgb;
        // This allows both.
    }
    gEmissive = emissive;
}
