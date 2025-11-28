#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D shadowMap;

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
};

#define MAX_LIGHTS 32
uniform Light u_Lights[MAX_LIGHTS];
uniform int u_LightCount;
uniform vec3 u_ViewPos;
uniform mat4 u_LightSpaceMatrix;

uniform samplerCube pointShadowMaps[4]; // Support up to 4 point lights with shadows

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

float ShadowCalculation(vec3 fragPos, vec3 normal, vec3 lightDir, float softness)
{
    vec4 fragPosLightSpace = u_LightSpaceMatrix * vec4(fragPos, 1.0);
    
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Calculate bias to prevent shadow acne
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    // Adjust kernel size based on softness
    int kernelSize = int(softness);
    if (kernelSize < 1) kernelSize = 1;
    
    for(int x = -kernelSize; x <= kernelSize; ++x)
    {
        for(int y = -kernelSize; y <= kernelSize; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= float((2 * kernelSize + 1) * (2 * kernelSize + 1));
    
    // Keep the shadow at 0.0 when outside the far_plane region
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

vec3 CalcDirLight(Light light, vec3 normal, vec3 viewDir, vec3 albedo, float specular, float shadow)
{
    vec3 lightDir = normalize(-light.direction);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    // Combine
    vec3 ambient = 0.1 * light.color * light.intensity * albedo;
    vec3 diffuse = diff * light.color * light.intensity * albedo;
    vec3 specularColor = 0.5 * spec * light.color * light.intensity * specular;
    
    return ambient + (1.0 - shadow) * (diffuse + specularColor);
}

vec3 CalcPointLight(Light light, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo, float specular, float shadow)
{
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
    
    // Range cutoff
    if (light.range > 0.0) {
        attenuation *= smoothstep(light.range, light.range * 0.75, distance);
    }
    
    // Combine
    vec3 ambient = 0.1 * light.color * light.intensity * albedo;
    vec3 diffuse = diff * light.color * light.intensity * albedo;
    vec3 specularColor = 0.5 * spec * light.color * light.intensity * specular;
    
    ambient *= attenuation;
    diffuse *= attenuation;
    specularColor *= attenuation;
    
    return ambient + (1.0 - shadow) * (diffuse + specularColor);
}

vec3 CalcSpotLight(Light light, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 albedo, float specular, float shadow)
{
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
    
    // Spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // Combine
    vec3 ambient = 0.1 * light.color * light.intensity * albedo;
    vec3 diffuse = diff * light.color * light.intensity * albedo;
    vec3 specularColor = 0.5 * spec * light.color * light.intensity * specular;
    
    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specularColor *= attenuation * intensity;
    
    return ambient + (1.0 - shadow) * (diffuse + specularColor);
}

void main()
{
    // Retrieve data from G-Buffer
    vec3 FragPos = texture(gPosition, TexCoord).rgb;
    vec3 Normal = texture(gNormal, TexCoord).rgb;
    vec3 Albedo = texture(gAlbedoSpec, TexCoord).rgb;
    float Specular = texture(gAlbedoSpec, TexCoord).a;
    
    vec3 viewDir = normalize(u_ViewPos - FragPos);
    vec3 result = vec3(0.0);
    
    int pointShadowIndex = 0;
    
    for(int i = 0; i < u_LightCount && i < MAX_LIGHTS; ++i) {
        float shadow = 0.0;
        
        if (u_Lights[i].type == 0) { // Directional
            // Only calculate directional shadow for first light (or if we had multiple shadow maps)
            if(i == 0) {
                 vec3 lightDir = normalize(-u_Lights[i].direction);
                 shadow = ShadowCalculation(FragPos, Normal, lightDir, u_Lights[i].shadowSoftness);
            }
            result += CalcDirLight(u_Lights[i], Normal, viewDir, Albedo, Specular, shadow);
        } else if (u_Lights[i].type == 1) { // Point
            // Calculate point shadow if enabled and we have slots
            if (pointShadowIndex < 4) { // Assuming first few point lights cast shadows
                 // Use static indexing to avoid GLSL 330 error
                 if (pointShadowIndex == 0)
                    shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[0], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 1)
                    shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[1], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 2)
                    shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[2], u_Lights[i].shadowSoftness);
                 else if (pointShadowIndex == 3)
                    shadow = PointShadowCalculation(FragPos, u_Lights[i].position, 25.0, pointShadowMaps[3], u_Lights[i].shadowSoftness);
                 
                 pointShadowIndex++;
            }
            result += CalcPointLight(u_Lights[i], FragPos, Normal, viewDir, Albedo, Specular, shadow);
        } else if (u_Lights[i].type == 2) { // Spot
             // Spot lights could use the same shadow map technique as directional lights
             result += CalcSpotLight(u_Lights[i], FragPos, Normal, viewDir, Albedo, Specular, shadow);
        }
    }
    
    FragColor = vec4(result, 1.0);
}
