#version 430 core

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler3D voxelAlbedo;
uniform sampler3D voxelNormal;
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;

uniform vec3 u_ViewPos;
uniform vec3 u_GridMin;
uniform vec3 u_GridMax;
uniform float u_VoxelSize;
uniform int u_NumCones;
uniform float u_GIIntensity;

const float PI = 3.14159265359;
const float MAX_DISTANCE = 100.0;

// Cone directions for diffuse cone tracing (5 cones)
const vec3 coneDirections[5] = vec3[](
    vec3(0.0, 1.0, 0.0),           // Up
    vec3(0.0, 0.5, 0.866025),      // Forward-up
    vec3(0.823639, 0.5, 0.267617), // Right-forward-up
    vec3(0.509037, 0.5, -0.7006629), // Right-back-up
    vec3(-0.823639, 0.5, -0.267617)  // Left-back-up
);

const float coneWeights[5] = float[](
    0.25,  // Up cone
    0.15,  // Forward-up
    0.15,  // Right-forward-up
    0.15,  // Right-back-up
    0.15   // Left-back-up
);

// Sample voxel grid with trilinear interpolation
vec4 sampleVoxel(sampler3D voxelTex, vec3 worldPos, float lod)
{
    vec3 voxelPos = (worldPos - u_GridMin) / (u_GridMax - u_GridMin);
    
    // Check bounds
    if (any(lessThan(voxelPos, vec3(0.0))) || any(greaterThan(voxelPos, vec3(1.0)))) {
        return vec4(0.0);
    }
    
    return textureLod(voxelTex, voxelPos, lod);
}

// Trace a cone through the voxel grid
vec3 traceCone(vec3 origin, vec3 direction, float aperture)
{
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float dist = u_VoxelSize; // Start one voxel away
    
    const int maxSteps = 128;
    const float minDiameter = u_VoxelSize;
    
    for (int i = 0; i < maxSteps && alpha < 0.95 && dist < MAX_DISTANCE; i++)
    {
        vec3 samplePos = origin + direction * dist;
        
        // Calculate cone diameter at current distance
        float diameter = max(minDiameter, 2.0 * aperture * dist);
        
        // Calculate mip level based on cone diameter
        float mipLevel = log2(diameter / u_VoxelSize);
        
        // Sample voxel grid
        vec4 voxelSample = sampleVoxel(voxelAlbedo, samplePos, mipLevel);
        
        // Accumulate color with front-to-back blending
        float sampleAlpha = voxelSample.a;
        color += (1.0 - alpha) * voxelSample.rgb * sampleAlpha;
        alpha += (1.0 - alpha) * sampleAlpha;
        
        // Step forward (larger steps for larger cones)
        dist += diameter * 0.5;
    }
    
    return color;
}

// Compute indirect diffuse lighting using cone tracing
vec3 computeIndirectDiffuse(vec3 position, vec3 normal)
{
    vec3 indirectDiffuse = vec3(0.0);
    
    // Create tangent space basis
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    // Trace multiple cones
    float aperture = tan(PI / 6.0); // 30 degree cone
    
    int numCones = min(u_NumCones, 5);
    for (int i = 0; i < numCones; i++)
    {
        vec3 coneDir = TBN * coneDirections[i];
        vec3 coneColor = traceCone(position + normal * u_VoxelSize * 2.0, coneDir, aperture);
        indirectDiffuse += coneColor * coneWeights[i];
    }
    
    return indirectDiffuse * u_GIIntensity;
}

// Compute indirect specular using a single cone
vec3 computeIndirectSpecular(vec3 position, vec3 normal, vec3 viewDir, float roughness)
{
    vec3 reflectDir = reflect(-viewDir, normal);
    
    // Cone aperture based on roughness
    float aperture = tan(roughness * PI * 0.5);
    
    vec3 specularColor = traceCone(position + normal * u_VoxelSize * 2.0, reflectDir, aperture);
    
    return specularColor * u_GIIntensity * (1.0 - roughness);
}

void main()
{
    // Sample G-Buffer
    vec3 fragPos = texture(gPosition, TexCoord).rgb;
    vec3 normal = texture(gNormal, TexCoord).rgb;
    vec4 albedoSpec = texture(gAlbedoSpec, TexCoord);
    vec3 albedo = albedoSpec.rgb;
    float roughness = texture(gNormal, TexCoord).a;
    
    // Check if this is a valid fragment
    if (length(normal) < 0.1) {
        FragColor = vec4(0.0);
        return;
    }
    
    normal = normalize(normal);
    vec3 viewDir = normalize(u_ViewPos - fragPos);
    
    // Compute indirect lighting
    vec3 indirectDiffuse = computeIndirectDiffuse(fragPos, normal);
    vec3 indirectSpecular = computeIndirectSpecular(fragPos, normal, viewDir, roughness);
    
    // Combine diffuse and specular
    vec3 giColor = indirectDiffuse * albedo + indirectSpecular;
    
    FragColor = vec4(giColor, 1.0);
}
