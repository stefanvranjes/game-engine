#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;
out vec4 FragPosLightSpace;

uniform mat4 u_MVP;
uniform mat4 u_Model;
uniform mat4 u_LightSpaceMatrix;

void main()
{
    gl_Position = u_MVP * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    Normal = mat3(transpose(inverse(u_Model))) * aNormal;
    FragPos = vec3(u_Model * vec4(aPos, 1.0));
    FragPosLightSpace = u_LightSpaceMatrix * vec4(FragPos, 1.0);
}
