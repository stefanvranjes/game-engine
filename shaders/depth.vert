#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 3) in ivec4 aBoneIDs;
layout (location = 4) in vec4 aBoneWeights;

uniform mat4 u_LightSpaceMatrix;
uniform mat4 u_Model;

// Skeletal animation
const int MAX_BONES = 100;
uniform bool u_Skinned;
uniform mat4 u_BoneMatrices[MAX_BONES];

void main()
{
    vec4 localPos = vec4(aPos, 1.0);
    
    // Apply skeletal animation if enabled
    if (u_Skinned) {
        mat4 boneTransform = mat4(0.0);
        for (int i = 0; i < 4; ++i) {
            if (aBoneWeights[i] > 0.0) {
                boneTransform += u_BoneMatrices[aBoneIDs[i]] * aBoneWeights[i];
            }
        }
        localPos = boneTransform * localPos;
    }
    
    gl_Position = u_LightSpaceMatrix * u_Model * localPos;
}
