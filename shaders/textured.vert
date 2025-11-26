#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 u_MVP;
uniform mat4 u_Model; // Need model matrix for world space calculations

void main(){
    gl_Position = u_MVP * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    
    // Calculate world space position
    FragPos = vec3(u_Model * vec4(aPos, 1.0));
    
    // Transform normal to world space (using normal matrix to handle non-uniform scaling)
    // Note: In production, pass normal matrix as uniform to avoid inverse() in shader
    Normal = mat3(transpose(inverse(u_Model))) * aNormal;
}
