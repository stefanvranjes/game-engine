#version 330 core

layout(location = 0) in vec3 a_Position;  // Quad vertex position

// Instance data
layout(location = 1) in vec3 a_ParticlePos;   // Particle world position
layout(location = 2) in vec4 a_ParticleColor;  // Particle color
layout(location = 3) in float a_ParticleSize;  // Particle size
layout(location = 4) in float a_TextureIndex;  // Texture atlas index
layout(location = 5) in float a_AnimationTime; // Animation time (0-1)
layout(location = 6) in vec3 a_Velocity;       // Particle velocity for flow distortion

out vec2 v_TexCoord;
out vec2 v_TexCoordNext; // Next frame UVs
out float v_BlendFactor; // Blend factor (0-1)
out vec4 v_Color;
flat out int v_TextureIndex;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform bool u_UseAnimation;
uniform int u_AnimationFrames;  // Number of frames in animation (e.g., 16 for 4x4)

void main() {
    v_Color = a_ParticleColor;
    v_TextureIndex = int(a_TextureIndex);
    
    // Billboard: create quad facing camera
    vec3 cameraRight = vec3(u_View[0][0], u_View[1][0], u_View[2][0]);
    vec3 cameraUp = vec3(u_View[0][1], u_View[1][1], u_View[2][1]);
    
    vec3 worldPos = a_ParticlePos 
                  + cameraRight * a_Position.x * a_ParticleSize
                  + cameraUp * a_Position.y * a_ParticleSize;
    
    gl_Position = u_Projection * u_View * vec4(worldPos, 1.0);
    
    // Flow Distortion: Stretch/Shear UVs based on view-space velocity
    vec2 flowDistortion = vec2(0.0);
    if (length(a_Velocity) > 0.01) {
       // Project velocity to view space to match billboard orientation
       vec3 viewVel = mat3(u_View) * a_Velocity;
       // Distort UVs along the velocity direction in view space (X/Y)
       // This stretches the texture along the direction of motion
       flowDistortion = viewVel.xy * 0.5; 
    }

    // Calculate texture coordinates
    if (u_UseAnimation) {
        // Flipbook animation (e.g., 4x4 grid = 16 frames)
        int framesPerRow = int(sqrt(float(u_AnimationFrames)));
        float exactFrame = a_AnimationTime * float(u_AnimationFrames);
        int currentFrame = int(exactFrame) % u_AnimationFrames;
        int nextFrame = (currentFrame + 1) % u_AnimationFrames;
        
        // Calculate interpolation factor
        v_BlendFactor = fract(exactFrame);
        
        // --- Current Frame ---
        int row = currentFrame / framesPerRow;
        int col = currentFrame % framesPerRow;
        
        vec2 quadUV = (a_Position.xy + 1.0) * 0.5;
        // Apply flow distortion
        vec2 distortedUV = quadUV - flowDistortion * quadUV.y; 
        
        float cellSize = 1.0 / float(framesPerRow);
        vec2 cellUV = distortedUV * cellSize;
        vec2 cellOffset = vec2(float(col) * cellSize, float(row) * cellSize);
        
        v_TexCoord = cellUV + cellOffset;
        
        // --- Next Frame ---
        row = nextFrame / framesPerRow;
        col = nextFrame % framesPerRow;
        
        vec2 cellOffsetNext = vec2(float(col) * cellSize, float(row) * cellSize);
        v_TexCoordNext = cellUV + cellOffsetNext;
        
    } else {
        // Static texture atlas (2x2 for different foam types)
        int row = v_TextureIndex / 2;
        int col = v_TextureIndex % 2;
        
        vec2 quadUV = (a_Position.xy + 1.0) * 0.5;
         // Apply flow distortion
        quadUV -= flowDistortion * (quadUV.y);

        vec2 cellUV = quadUV * 0.5;
        vec2 cellOffset = vec2(float(col) * 0.5, float(row) * 0.5);
        
        v_TexCoord = cellUV + cellOffset;
        v_TexCoordNext = v_TexCoord; // No blending for static
        v_BlendFactor = 0.0;
    }
}
