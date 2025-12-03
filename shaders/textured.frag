#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
in vec4 FragPosLightSpace;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    sampler2D texture;
    sampler2D specularMap;
    sampler2D normalMap;
    float opacity;
};

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

#define MAX_LIGHTS 4
uniform Material material;
uniform int u_HasTexture;
uniform int u_HasSpecularMap;
uniform int u_HasNormalMap;

uniform Light u_Lights[MAX_LIGHTS];
uniform int u_LightCount;

uniform vec3 u_ViewPos;
uniform sampler2D shadowMap;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // Get closest depth value from light's perspective
    // float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Calculate bias to prevent shadow acne
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // Keep the shadow at 0.0 when outside the far_plane region of the light's frustum
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

void main()
{
    vec3 norm = normalize(Normal);
    
    // Calculate TBN matrix using derivatives (for normal mapping)
    if (u_HasNormalMap == 1) {
        vec3 Q1 = dFdx(FragPos);
        vec3 Q2 = dFdy(FragPos);
        vec2 st1 = dFdx(TexCoord);
        vec2 st2 = dFdy(TexCoord);
        
        vec3 N = normalize(Normal);
        vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
        vec3 B = -normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);
        
        // Sample normal map and transform to [-1, 1] range
        vec3 normalMapSample = texture(material.normalMap, TexCoord).rgb;
        normalMapSample = normalMapSample * 2.0 - 1.0;
        
        // Transform normal from tangent space to world space
        norm = normalize(TBN * normalMapSample);
    }
    
    vec3 viewDir = normalize(u_ViewPos - FragPos);
    
    vec3 result = vec3(0.0);
    
    for(int i = 0; i < u_LightCount && i < MAX_LIGHTS; ++i) {
        // Ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * u_Lights[i].color * u_Lights[i].intensity * material.ambient;
        
        // Diffuse 
        vec3 lightDir = normalize(u_Lights[i].position - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * u_Lights[i].color * u_Lights[i].intensity * material.diffuse;
        
        // Specular
        float specularStrength = 0.5;
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        vec3 specular = specularStrength * spec * u_Lights[i].color * u_Lights[i].intensity * material.specular;
        
        if (u_HasSpecularMap == 1) {
            specular *= texture(material.specularMap, TexCoord).r;
        }  
        
        // Calculate shadow (only for first light)
        float shadow = 0.0;
        if(i == 0) {
            shadow = ShadowCalculation(FragPosLightSpace, norm, lightDir);
        }
        
        // Combine with shadow
        result += ambient + (1.0 - shadow) * (diffuse + specular);
    }
    
    if (u_HasTexture == 1) {
        vec4 texColor = texture(material.texture, TexCoord);
        FragColor = vec4(texColor.rgb * result, texColor.a * material.opacity);
    } else {
        FragColor = vec4(result, material.opacity);
    }
}
