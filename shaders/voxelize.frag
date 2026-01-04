#version 430 core

layout(rgba8, binding = 0) uniform image3D voxelAlbedo;
layout(rgba8, binding = 1) uniform image3D voxelNormal;

in GS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    flat int axis;
} fs_in;

uniform vec3 u_GridMin;
uniform vec3 u_GridMax;
uniform int u_Resolution;
uniform float u_VoxelSize;

// Material properties
uniform int u_HasTexture;
uniform sampler2D u_AlbedoTexture;
uniform vec3 u_AlbedoColor;

void main()
{
    // Convert world position to voxel grid coordinates
    vec3 voxelPos = (fs_in.worldPos - u_GridMin) / (u_GridMax - u_GridMin);
    voxelPos = clamp(voxelPos, vec3(0.0), vec3(1.0));
    
    ivec3 voxelCoord = ivec3(voxelPos * float(u_Resolution));
    
    // Clamp to valid range
    voxelCoord = clamp(voxelCoord, ivec3(0), ivec3(u_Resolution - 1));
    
    // Sample albedo
    vec3 albedo = u_AlbedoColor;
    if (u_HasTexture == 1) {
        albedo = texture(u_AlbedoTexture, fs_in.texCoord).rgb;
    }
    
    // Normalize normal
    vec3 normal = normalize(fs_in.normal);
    
    // Encode normal from [-1, 1] to [0, 1]
    vec3 encodedNormal = normal * 0.5 + 0.5;
    
    // Write to voxel textures using atomic operations to handle overlapping fragments
    // Store albedo (RGB) and ambient occlusion (A, default 1.0)
    vec4 albedoData = vec4(albedo, 1.0);
    imageStore(voxelAlbedo, voxelCoord, albedoData);
    
    // Store normal (RGB) and emissive (A, default 0.0)
    vec4 normalData = vec4(encodedNormal, 0.0);
    imageStore(voxelNormal, voxelCoord, normalData);
    
    // Note: For proper handling of overlapping fragments, we should use imageAtomicMax
    // or similar atomic operations, but for simplicity we use imageStore here.
    // In production, implement proper atomic averaging or max operations.
}
