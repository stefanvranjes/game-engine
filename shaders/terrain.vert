#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform float u_HeightScale;
uniform vec3 u_TerrainPos;

out vec3 FragPos;
out vec2 TexCoord;
out vec3 Normal;
out vec4 ClipPos;

void main()
{
    vec4 worldPos = u_Model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    TexCoord = aTexCoord;
    Normal = aNormal;
    
    ClipPos = u_Projection * u_View * worldPos;
    gl_Position = ClipPos;
}
