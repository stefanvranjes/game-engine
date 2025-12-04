#version 330 core

layout(location = 0) in vec3 a_Position;  // Quad vertex position
layout(location = 1) in vec2 a_TexCoord;  // Quad texture coordinates

// Instance data
layout(location = 2) in vec3 a_ParticlePos;   // Particle world position
layout(location = 3) in vec4 a_ParticleColor;  // Particle color
layout(location = 4) in float a_ParticleSize;  // Particle size

out vec2 v_TexCoord;
out vec4 v_Color;
out vec4 v_FragPos;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform float u_AtlasRows;
uniform float u_AtlasCols;

layout(location = 5) in float a_LifeRatio;

void main() {
    v_TexCoord = a_TexCoord;
    v_Color = a_ParticleColor;
    
    // Billboard: extract camera right and up vectors from view matrix
    vec3 cameraRight = vec3(u_View[0][0], u_View[1][0], u_View[2][0]);
    vec3 cameraUp = vec3(u_View[0][1], u_View[1][1], u_View[2][1]);
    
    // Create billboard quad facing camera
    vec3 worldPos = a_ParticlePos 
                  + cameraRight * a_Position.x * a_ParticleSize
                  + cameraUp * a_Position.y * a_ParticleSize;
    
    vec4 clipPos = u_Projection * u_View * vec4(worldPos, 1.0);
    v_FragPos = clipPos;
    gl_Position = clipPos;

    // Atlas Animation Logic
    float totalFrames = u_AtlasRows * u_AtlasCols;
    if (totalFrames > 1.0) {
        float currentFrame = floor(a_LifeRatio * totalFrames);
        currentFrame = clamp(currentFrame, 0.0, totalFrames - 1.0);
        
        float row = floor(currentFrame / u_AtlasCols);
        float col = mod(currentFrame, u_AtlasCols);
        
        // Invert row because texture coordinates usually start from bottom-left
        // But atlas usually starts top-left. Let's assume standard top-left start for atlas logic
        // If texture V is 0 at bottom, then row 0 (top) corresponds to V=1.
        // Let's stick to standard UV: (0,0) bottom-left.
        // If atlas is generated top-to-bottom, row 0 is at V=1.
        // row = 0 -> V range [1 - 1/rows, 1]
        // row = 1 -> V range [1 - 2/rows, 1 - 1/rows]
        
        // Calculate UV offset and scale
        vec2 uvScale = vec2(1.0 / u_AtlasCols, 1.0 / u_AtlasRows);
        
        // For standard top-down atlas with OpenGL bottom-up texture coords:
        // col 0 -> U=0
        // row 0 -> V=1-1/rows
        
        vec2 uvOffset = vec2(
            col / u_AtlasCols, 
            1.0 - (row + 1.0) / u_AtlasRows // +1 because we want bottom of the cell
        );
        
        // Adjust TexCoord to atlas sub-region
        // a_TexCoord is 0..1 for the quad
        v_TexCoord = a_TexCoord * uvScale + uvOffset;
    } else {
        v_TexCoord = a_TexCoord;
    }
}
