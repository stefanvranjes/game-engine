#version 330 core
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    sampler2D texture;
    sampler2D specularMap;
    sampler2D normalMap;
};

uniform Material material;
uniform int u_HasTexture;
uniform int u_HasSpecularMap;
uniform int u_HasNormalMap;

void main()
{
    // Store fragment position
    gPosition = FragPos;
    
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
    
    gNormal = norm;
    
    // Store albedo (diffuse color)
    vec3 albedo = material.diffuse;
    if (u_HasTexture == 1) {
        albedo = texture(material.texture, TexCoord).rgb;
    }
    gAlbedoSpec.rgb = albedo;
    
    // Store specular intensity in alpha channel
    float specular = material.specular.r; // Use red channel as intensity
    if (u_HasSpecularMap == 1) {
        specular = texture(material.specularMap, TexCoord).r;
    }
    gAlbedoSpec.a = specular;
}
