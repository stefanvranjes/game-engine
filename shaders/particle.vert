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
}
