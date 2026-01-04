#version 430 core

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D depthTexture;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_InvProjection;
uniform vec3 u_ViewPos;
uniform vec2 u_ScreenSize;
uniform float u_GIIntensity;

const int NUM_SAMPLES = 16;
const float MAX_RAY_DISTANCE = 10.0;
const float STEP_SIZE = 0.1;
const float THICKNESS = 0.5;

// Random number generation
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Generate random direction in hemisphere
vec3 randomHemisphereDirection(vec3 normal, vec2 seed) {
    float u = rand(seed);
    float v = rand(seed + vec2(1.0, 0.0));
    
    float theta = acos(sqrt(1.0 - u));
    float phi = 2.0 * 3.14159265359 * v;
    
    vec3 tangentSpaceDir = vec3(
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    );
    
    // Transform to world space
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    
    return normalize(tangent * tangentSpaceDir.x + bitangent * tangentSpaceDir.y + normal * tangentSpaceDir.z);
}

// Ray march in screen space
vec3 screenSpaceRayMarch(vec3 origin, vec3 direction, out bool hit) {
    hit = false;
    vec3 rayPos = origin;
    
    for (int i = 0; i < 64; i++) {
        rayPos += direction * STEP_SIZE;
        
        // Check if ray is out of bounds
        if (length(rayPos - origin) > MAX_RAY_DISTANCE) {
            break;
        }
        
        // Project to screen space
        vec4 projPos = u_Projection * u_View * vec4(rayPos, 1.0);
        projPos.xyz /= projPos.w;
        projPos.xy = projPos.xy * 0.5 + 0.5;
        
        // Check if in screen bounds
        if (projPos.x < 0.0 || projPos.x > 1.0 || projPos.y < 0.0 || projPos.y > 1.0) {
            break;
        }
        
        // Sample depth at this screen position
        float sceneDepth = texture(gPosition, projPos.xy).z;
        float rayDepth = rayPos.z;
        
        // Check for intersection
        if (rayDepth > sceneDepth && rayDepth < sceneDepth + THICKNESS) {
            hit = true;
            return texture(gAlbedoSpec, projPos.xy).rgb;
        }
    }
    
    return vec3(0.0);
}

void main()
{
    // Sample G-Buffer
    vec3 fragPos = texture(gPosition, TexCoord).rgb;
    vec3 normal = texture(gNormal, TexCoord).rgb;
    vec4 albedoSpec = texture(gAlbedoSpec, TexCoord);
    vec3 albedo = albedoSpec.rgb;
    
    // Check if valid fragment
    if (length(normal) < 0.1) {
        FragColor = vec4(0.0);
        return;
    }
    
    normal = normalize(normal);
    
    // Accumulate indirect lighting from multiple samples
    vec3 indirectLight = vec3(0.0);
    int hitCount = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        vec2 seed = TexCoord + vec2(float(i) * 0.1, 0.0);
        vec3 rayDir = randomHemisphereDirection(normal, seed);
        
        bool hit;
        vec3 sampleColor = screenSpaceRayMarch(fragPos, rayDir, hit);
        
        if (hit) {
            float NdotL = max(dot(normal, rayDir), 0.0);
            indirectLight += sampleColor * NdotL;
            hitCount++;
        }
    }
    
    // Average samples
    if (hitCount > 0) {
        indirectLight /= float(hitCount);
    }
    
    // Apply intensity and albedo
    vec3 giColor = indirectLight * albedo * u_GIIntensity;
    
    FragColor = vec4(giColor, 1.0);
}
