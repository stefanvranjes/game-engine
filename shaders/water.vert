#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform sampler2D u_DisplacementMap;
uniform float u_L;

out vec3 FragPos;
out vec2 TexConfigs;
out vec3 Normal;
out vec4 ClipSpace;

void main()
{
    // Local coords for displacement sampling
    vec4 worldPos = model * vec4(aPos, 1.0);
    
    // Use uniform Ocean Size (L)
    float L = u_L;
    vec2 dispUV = worldPos.xz / L;
    
    vec3 displacement = texture(u_DisplacementMap, dispUV).xyz;
    
    // Apply displacement
    // FFT Displacement is (X, Y, Z). simple implementation used Y mainly.
    // But inversion shader wrote vec3(0, h, 0).
    
    worldPos.xyz += displacement;

    ClipSpace = projection * view * worldPos;
    gl_Position = ClipSpace;
    
    FragPos = worldPos.xyz;
    TexConfigs = dispUV; // Pass UV for normal sampling
    
    // Basic normal from vertex, updated in Frag by FFT normal map
    Normal = mat3(transpose(inverse(model))) * aNormal; 
}
