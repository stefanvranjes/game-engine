#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
} vs_out;

uniform mat4 u_Model;

void main()
{
    vs_out.worldPos = vec3(u_Model * vec4(aPos, 1.0));
    vs_out.normal = mat3(transpose(inverse(u_Model))) * aNormal;
    vs_out.texCoord = aTexCoord;
    
    // Position is computed in geometry shader
    gl_Position = vec4(vs_out.worldPos, 1.0);
}
