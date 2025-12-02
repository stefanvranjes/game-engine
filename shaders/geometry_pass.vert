#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 ViewPos;

uniform mat4 u_Model;
uniform mat4 u_MVP;
uniform vec3 u_ViewPos;

void main()
{
    vec4 worldPos = u_Model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(u_Model))) * aNormal;
    TexCoord = aTexCoord;
    ViewPos = u_ViewPos;
    
    gl_Position = u_MVP * vec4(aPos, 1.0);
}
