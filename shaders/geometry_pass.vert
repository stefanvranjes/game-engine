#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

layout (location = 3) in mat4 aInstanceMatrix;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 ViewPos;
out vec4 CurrentPos;
out vec4 PreviousPos;

uniform mat4 u_Model;
uniform mat4 u_MVP;
uniform mat4 u_PrevMVP;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform vec3 u_ViewPos;
uniform bool u_Instanced;

void main()
{
    mat4 modelMatrix;
    if (u_Instanced) {
        modelMatrix = aInstanceMatrix;
    } else {
        modelMatrix = u_Model;
    }

    vec4 worldPos = modelMatrix * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(modelMatrix))) * aNormal;
    TexCoord = aTexCoord;
    ViewPos = u_ViewPos;
    
    // Calculate current and previous clip-space positions for motion vectors
    // Note: For instanced, we need to calculate MVP manually or pass View/Proj
    if (u_Instanced) {
        CurrentPos = u_Projection * u_View * worldPos;
        // Approximation for previous pos in instanced mode (assuming no motion for now)
        // Ideally we'd need previous instance matrix too
        PreviousPos = CurrentPos; 
    } else {
        CurrentPos = u_MVP * vec4(aPos, 1.0);
        PreviousPos = u_PrevMVP * vec4(aPos, 1.0);
    }
    
    gl_Position = CurrentPos;
}
