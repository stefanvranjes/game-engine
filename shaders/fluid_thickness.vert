#version 330 core

layout(location = 0) in vec3 a_Position;  // Quad vertex position

// Instance data
layout(location = 1) in vec3 a_ParticlePos;   // Particle world position
layout(location = 2) in float a_ParticleSize;  // Particle size

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform float u_ParticleSizeScale;

void main() {
    // Billboard: create quad facing camera
    vec3 cameraRight = vec3(u_View[0][0], u_View[1][0], u_View[2][0]);
    vec3 cameraUp = vec3(u_View[0][1], u_View[1][1], u_View[2][1]);
    
    vec3 worldPos = a_ParticlePos 
                  + cameraRight * a_Position.x * a_ParticleSize * u_ParticleSizeScale
                  + cameraUp * a_Position.y * a_ParticleSize * u_ParticleSizeScale;
    
    gl_Position = u_Projection * u_View * vec4(worldPos, 1.0);
}
