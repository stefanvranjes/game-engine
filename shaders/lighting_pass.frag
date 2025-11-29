#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
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

uniform samplerCube pointShadowMaps[4]; // Support up to 4 point lights with shadows
uniform sampler2D spotShadowMaps[4]; // Support up to 4 spot lights with shadows
uniform mat4 spotLightSpaceMatrices[4];

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
    
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    
    // Adjust kernel size based on softness
    int kernelSize = int(softness);
    if (kernelSize < 1) kernelSize = 1;
    
    for(int x = -kernelSize; x <= kernelSize; ++x)
    {
        for(int y = -kernelSize; y <= kernelSize; ++y)
        {
            float pcfDepth = texture(shadowMap, vec3(projCoords.xy + vec2(x, y) * texelSize, layer)).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= float((2 * kernelSize + 1) * (2 * kernelSize + 1));
    
    // Keep the shadow at 0.0 when outside the far_plane region
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
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
            if(i == 0 && u_Lights[i].castsShadows == 1) {
                 vec3 lightDir = normalize(-u_Lights[i].direction);
                 shadow = ShadowCalculation(FragPos, Normal, lightDir, u_Lights[i].lightSize);
            }
            result += CalcDirLight(u_Lights[i], Normal, viewDir, Albedo, Specular, shadow);
        } else if (u_Lights[i].type == 1) { // Point
            // Calculate point shadow if enabled and we have slots
            if (u_Lights[i].castsShadows == 1 && pointShadowIndex < 4) { // Assuming first few point lights cast shadows
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
             // Calculate spot shadow if enabled and we have slots
             int spotShadowIndex = 0;
             for (int j = 0; j < i; ++j) {
                 if (u_Lights[j].type == 2 && u_Lights[j].castsShadows == 1) {
                     spotShadowIndex++;
                 }
             }
             
             if (u_Lights[i].castsShadows == 1 && spotShadowIndex < 4) {
                 vec3 lightDir = normalize(u_Lights[i].position - FragPos);
                 // Use static indexing to avoid GLSL 330 error
                 if (spotShadowIndex == 0)
                    shadow = SpotShadowCalculation(FragPos, Normal, lightDir, spotLightSpaceMatrices[0], spotShadowMaps[0], u_Lights[i].lightSize);
                 else if (spotShadowIndex == 1)
                    shadow = SpotShadowCalculation(FragPos, Normal, lightDir, spotLightSpaceMatrices[1], spotShadowMaps[1], u_Lights[i].lightSize);
                 else if (spotShadowIndex == 2)
                    shadow = SpotShadowCalculation(FragPos, Normal, lightDir, spotLightSpaceMatrices[2], spotShadowMaps[2], u_Lights[i].lightSize);
                 else if (spotShadowIndex == 3)
                    shadow = SpotShadowCalculation(FragPos, Normal, lightDir, spotLightSpaceMatrices[3], spotShadowMaps[3], u_Lights[i].lightSize);
             }
             
             result += CalcSpotLight(u_Lights[i], FragPos, Normal, viewDir, Albedo, Specular, shadow);
        }
    }
    
    FragColor = vec4(result, 1.0);
}
