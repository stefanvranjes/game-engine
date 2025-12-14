#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gDepth;       // Depth Texture
uniform sampler2DArray shadowMap; // CSM Array

uniform mat4 inverseView;
uniform mat4 inverseProjection;
uniform vec3 viewPos;

// Light Info (Directional Only for now)
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform float lightIntensity;

// Shadow Info
uniform mat4 cascadeLightSpaceMatrices[3];
uniform float cascadePlaneDistances[3];

// Fog Info
uniform float u_Density;
uniform float u_Anisotropy; // g factor (-1 to 1)
uniform int u_MaxSteps;
uniform float u_StepSize;

// Utils to reconstruct world position
vec3 ReconstructWorldPos(float depth, vec2 texCoord) {
    float z = depth * 2.0 - 1.0;
    vec4 clipSpacePosition = vec4(texCoord * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = inverseProjection * clipSpacePosition;
    viewSpacePosition /= viewSpacePosition.w;
    vec4 worldSpacePosition = inverseView * viewSpacePosition;
    return worldSpacePosition.xyz;
}

// Henyey-Greenstein Phase Function
float ComputeScattering(float lightDotView) {
    float g = u_Anisotropy;
    float g2 = g * g;
    float result = 1.0 - g2;
    result /= (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * lightDotView, 1.5));
    return result;
}

// Shadow Calculation (Simplified from lighting_pass but for volume)
float CalculateShadow(vec3 worldPos) {
    vec4 fragPosViewSpace = inverse(inverseView) * vec4(worldPos, 1.0);
    float depthValue = abs(fragPosViewSpace.z);
    
    int layer = -1;
    for (int i = 0; i < 3; ++i) {
        if (depthValue < cascadePlaneDistances[i]) {
            layer = i;
            break;
        }
    }
    if (layer == -1) layer = 2;
    
    vec4 lightSpacePos = cascadeLightSpaceMatrices[layer] * vec4(worldPos, 1.0);
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    if (projCoords.z > 1.0) return 0.0;
    
    float shadowMapDepth = texture(shadowMap, vec3(projCoords.xy, layer)).r;
    float currentDepth = projCoords.z;
    
    // Simple shadow, no PCF for performance in volume
    float bias = 0.005;
    float shadow = currentDepth - bias > shadowMapDepth ? 1.0 : 0.0;
    
    return shadow;
}

void main() {
    float depth = texture(gDepth, TexCoord).r;
    vec3 worldPos = ReconstructWorldPos(depth, TexCoord);
    
    vec3 startPos = viewPos;
    vec3 rayDir = normalize(worldPos - startPos);
    float rayLength = length(worldPos - startPos);
    
    // Limit ray length to far plane or reasonable distance to prevent skybox over-fogging
    // (Optional: if depth == 1.0, clamp to max distance)
    float maxDist = cascadePlaneDistances[2]; 
    if (depth >= 1.0) {
        rayLength = min(rayLength, maxDist);
        worldPos = startPos + rayDir * rayLength; // Clamp worldPos for correct stepping
    }
    
    vec3 currentPos = startPos;
    float stepLen = u_StepSize;
    
    // Jitter start position for noise (dithering) to reduce banding
    // float noise = texture(noiseTex, TexCoord * noiseScale).r; // Not implemented yet
    // currentPos += rayDir * stepLen * noise;
    
    vec3 accumulatedLight = vec3(0.0);
    float accumulatedTransmittance = 1.0;
    
    int steps = int(min(rayLength / stepLen, float(u_MaxSteps)));
    
    // Optimization: Don't march if too far
    // if (rayLength > maxDist) ...
    
    for (int i = 0; i < steps; ++i) {
        if (accumulatedTransmittance < 0.01) break;
        
        float shadow = CalculateShadow(currentPos);
        
        if (shadow < 1.0) { // If not in shadow (1.0 means fully shadowed in our CalculateShadow func above? Wait check logic)
            // Logic above: returns 1.0 if shadowed. So we want shadow < 1.0 (lit)
            
            float density = u_Density;
            
            // Optional: Height fog
            // float heightFactor = exp(-heightFalloff * currentPos.y);
            // density *= heightFactor;
            
            float scattering = ComputeScattering(dot(rayDir, -lightDir));
            
            // Light contribution
            vec3 lightContribution = lightColor * lightIntensity * scattering * density * stepLen;
            
            // Accumulate
            accumulatedLight += lightContribution * accumulatedTransmittance; // * (1.0 - shadow) handled by if check
        }
        
        // Attenuate transmittance
        accumulatedTransmittance *= exp(-u_Density * stepLen);
        
        currentPos += rayDir * stepLen;
    }
    
    // Output: RGB = Fog Color, A = Transmittance (how much background shows through)
    FragColor = vec4(accumulatedLight, accumulatedTransmittance);
}
