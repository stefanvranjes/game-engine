#version 330 core
layout (location = 0) out vec4 gPosition; // RGB: Position, A: AO
layout (location = 1) out vec4 gNormal;   // RGB: Normal, A: Roughness
layout (location = 2) out vec4 gAlbedoSpec; // RGB: Albedo, A: Metallic

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float roughness;
    float metallic;
    sampler2D texture;
    sampler2D specularMap; // Used as metallic map for now if needed, or separate
    sampler2D normalMap;
    sampler2D roughnessMap;
    sampler2D metallicMap;
};

uniform Material material;
uniform int u_HasTexture;
uniform int u_HasSpecularMap;
uniform int u_HasNormalMap;

void main()
{
    // Store fragment position
    // Store fragment position and AO (default 1.0)
    gPosition = vec4(FragPos, 1.0);
    
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
        
        vec3 normalMapSample = texture(material.normalMap, TexCoord).rgb;
        normalMapSample = normalMapSample * 2.0 - 1.0;
        norm = normalize(TBN * normalMapSample);
    }
    
    // Store normal and Roughness
    float roughness = material.roughness;
    // if (u_HasRoughnessMap) ... (TODO)
    gNormal = vec4(norm, roughness);
    
    // Store albedo (diffuse color)
    vec3 albedo = material.diffuse;
    if (u_HasTexture == 1) {
        albedo = texture(material.texture, TexCoord).rgb;
    }
    gAlbedoSpec.rgb = albedo;
    
    // Store Metallic in alpha channel
    float metallic = material.metallic;
    // if (u_HasMetallicMap) ... (TODO)
    gAlbedoSpec.a = metallic;
}
