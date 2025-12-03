#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;

uniform mat4 view;
uniform mat4 projection;

uniform int maxSteps;
uniform float stepSize;
uniform float thickness;
uniform float maxDistance;
uniform float fadeStart;
uniform float fadeEnd;
uniform vec2 screenSize;

// Convert world position to screen space
vec2 WorldToScreen(vec3 worldPos) {
    vec4 clipSpace = projection * view * vec4(worldPos, 1.0);
    vec3 ndc = clipSpace.xyz / clipSpace.w;
    return ndc.xy * 0.5 + 0.5;
}

// Get view space position from depth
vec3 GetViewPosition(vec2 uv) {
    vec3 worldPos = texture(gPosition, uv).rgb;
    vec4 viewPos = view * vec4(worldPos, 1.0);
    return viewPos.xyz;
}

// Ray march through depth buffer
bool RayMarch(vec3 rayOrigin, vec3 rayDir, out vec2 hitUV, out float hitConfidence) {
    vec3 rayPos = rayOrigin;
    vec2 screenPos;
    
    float stepLength = stepSize;
    
    for (int i = 0; i < maxSteps; i++) {
        rayPos += rayDir * stepLength;
        
        // Check if ray is too far
        if (length(rayPos - rayOrigin) > maxDistance) {
            return false;
        }
        
        // Convert to screen space
        vec4 clipSpace = projection * vec4(rayPos, 1.0);
        vec3 ndc = clipSpace.xyz / clipSpace.w;
        screenPos = ndc.xy * 0.5 + 0.5;
        
        // Check if outside screen bounds
        if (screenPos.x < 0.0 || screenPos.x > 1.0 || 
            screenPos.y < 0.0 || screenPos.y > 1.0) {
            return false;
        }
        
        // Sample depth buffer
        vec3 sampleViewPos = GetViewPosition(screenPos);
        
        // Check for intersection
        float depthDiff = rayPos.z - sampleViewPos.z;
        
        if (depthDiff > 0.0 && depthDiff < thickness) {
            // Binary search for more accurate hit
            vec3 searchStart = rayPos - rayDir * stepLength;
            vec3 searchEnd = rayPos;
            
            for (int j = 0; j < 4; j++) {
                vec3 searchMid = (searchStart + searchEnd) * 0.5;
                
                vec4 searchClip = projection * vec4(searchMid, 1.0);
                vec3 searchNDC = searchClip.xyz / searchClip.w;
                vec2 searchUV = searchNDC.xy * 0.5 + 0.5;
                
                vec3 searchSample = GetViewPosition(searchUV);
                float searchDiff = searchMid.z - searchSample.z;
                
                if (searchDiff > 0.0) {
                    searchEnd = searchMid;
                } else {
                    searchStart = searchMid;
                }
            }
            
            // Use refined position
            vec4 finalClip = projection * vec4(searchEnd, 1.0);
            vec3 finalNDC = finalClip.xyz / finalClip.w;
            hitUV = finalNDC.xy * 0.5 + 0.5;
            
            // Calculate confidence based on hit quality
            hitConfidence = 1.0 - (depthDiff / thickness);
            
            return true;
        }
    }
    
    return false;
}

void main() {
    // Sample G-Buffer
    vec3 worldPos = texture(gPosition, TexCoord).rgb;
    vec3 normal = normalize(texture(gNormal, TexCoord).rgb);
    vec4 albedoSpec = texture(gAlbedoSpec, TexCoord);
    float roughness = texture(gNormal, TexCoord).a;
    float metallic = albedoSpec.a;
    
    // Early exit for non-reflective surfaces
    if (metallic < 0.01 && roughness > 0.9) {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    
    // Convert to view space
    vec3 viewPos = GetViewPosition(TexCoord);
    vec3 viewNormal = mat3(view) * normal;
    vec3 viewDir = normalize(viewPos);
    
    // Calculate reflection ray
    vec3 reflectDir = normalize(reflect(viewDir, viewNormal));
    
    // Ray march
    vec2 hitUV;
    float hitConfidence;
    
    if (RayMarch(viewPos, reflectDir, hitUV, hitConfidence)) {
        // Sample color at hit point
        vec3 reflectionColor = texture(gAlbedoSpec, hitUV).rgb;
        
        // Calculate fade factors
        
        // 1. Screen edge fade
        vec2 edgeDist = abs(hitUV - 0.5) * 2.0;
        float edgeFade = 1.0 - smoothstep(fadeStart, fadeEnd, max(edgeDist.x, edgeDist.y));
        
        // 2. Distance fade
        float rayLength = length(GetViewPosition(hitUV) - viewPos);
        float distanceFade = 1.0 - smoothstep(maxDistance * 0.5, maxDistance, rayLength);
        
        // 3. Roughness fade (less reflections on rough surfaces)
        float roughnessFade = 1.0 - roughness;
        
        // 4. Fresnel-like fade (more reflections at grazing angles)
        float fresnelFade = pow(1.0 - max(dot(-viewDir, viewNormal), 0.0), 2.0);
        fresnelFade = mix(0.3, 1.0, fresnelFade);
        
        // Combine all fade factors
        float finalStrength = hitConfidence * edgeFade * distanceFade * roughnessFade * fresnelFade;
        
        // Boost strength for metallic surfaces
        finalStrength *= mix(0.5, 1.0, metallic);
        
        FragColor = vec4(reflectionColor, finalStrength);
    } else {
        // No hit
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
