#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in ivec4 aBoneIDs;
layout (location = 4) in vec4 aBoneWeights;

layout (location = 5) in mat4 aInstanceMatrix;

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

// Skeletal animation
const int MAX_BONES = 100;
uniform bool u_Skinned;
uniform mat4 u_BoneMatrices[MAX_BONES];

// Clip plane for planar reflections
uniform vec4 u_ClipPlane;
uniform int u_UseClipPlane;

void main()
{
    vec4 localPos = vec4(aPos, 1.0);
    vec3 localNormal = aNormal;
    
    // Apply skeletal animation if enabled
    if (u_Skinned) {
        mat4 boneTransform = mat4(0.0);
        for (int i = 0; i < 4; ++i) {
            if (aBoneWeights[i] > 0.0) {
                boneTransform += u_BoneMatrices[aBoneIDs[i]] * aBoneWeights[i];
            }
        }
        localPos = boneTransform * localPos;
        localNormal = mat3(boneTransform) * aNormal;
    }
    
    mat4 modelMatrix;
    if (u_Instanced) {
        modelMatrix = aInstanceMatrix;
    } else {
        modelMatrix = u_Model;
    }

    vec4 worldPos = modelMatrix * localPos;
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(modelMatrix))) * localNormal;
    TexCoord = aTexCoord;
    ViewPos = u_ViewPos;
    
    // Clip plane for planar reflections
    if (u_UseClipPlane == 1) {
        gl_ClipDistance[0] = dot(worldPos, u_ClipPlane);
    } else {
        gl_ClipDistance[0] = 1.0; // Always pass when not using clip plane
    }
    
    // Calculate current and previous clip-space positions for motion vectors
    // Note: For instanced, we need to calculate MVP manually or pass View/Proj
    if (u_Instanced) {
        CurrentPos = u_Projection * u_View * worldPos;
        // Approximation for previous pos in instanced mode (assuming no motion for now)
        // Ideally we'd need previous instance matrix too
        PreviousPos = CurrentPos; 
    } else {
        CurrentPos = u_MVP * localPos;
        PreviousPos = u_PrevMVP * localPos;
    }
    
    gl_Position = CurrentPos;
}
