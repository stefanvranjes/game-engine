#version 430 core

// Particle structure matching SSBO layout
struct Particle {
    vec3 position;
    float life;
    vec3 velocity;
    float size;
    vec4 color;
    float rotation;
    float angularVelocity;
    float mass;
    float atlasIndex;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// Uniforms
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform float u_AtlasRows;
uniform float u_AtlasCols;

// Outputs to fragment shader
out vec2 v_TexCoord;
out vec4 v_Color;

void main() {
    // Calculate particle index and vertex index within the quad
    // We draw 6 vertices per particle (2 triangles)
    uint particleIndex = gl_VertexID / 6;
    uint vertexIndex = gl_VertexID % 6;
    
    Particle p = particles[particleIndex];
    
    // Quad vertices (local space)
    vec2 quadVertices[6] = vec2[](
        vec2(-0.5,  0.5), // TL
        vec2(-0.5, -0.5), // BL
        vec2( 0.5, -0.5), // BR
        vec2(-0.5,  0.5), // TL
        vec2( 0.5, -0.5), // BR
        vec2( 0.5,  0.5)  // TR
    );
    
    vec2 quadTexCoords[6] = vec2[](
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0)
    );
    
    vec2 localPos = quadVertices[vertexIndex];
    v_TexCoord = quadTexCoords[vertexIndex];
    v_Color = p.color;
    
    // Apply rotation if needed (optional, can be added later)
    
    // Billboard calculation
    vec3 cameraRight = vec3(u_View[0][0], u_View[1][0], u_View[2][0]);
    vec3 cameraUp = vec3(u_View[0][1], u_View[1][1], u_View[2][1]);
    
    vec3 worldPos = p.position + 
                    (cameraRight * localPos.x * p.size) + 
                    (cameraUp * localPos.y * p.size);
                    
    gl_Position = u_Projection * u_View * vec4(worldPos, 1.0);
}
