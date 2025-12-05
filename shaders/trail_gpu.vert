#version 430 core

// Trail point structure
struct TrailPoint {
    vec4 position; // w = width
};

layout(std430, binding = 2) buffer TrailBuffer {
    TrailPoint trailPoints[];
};

// Uniforms
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform int u_TrailLength;
uniform float u_TrailWidth;
uniform vec4 u_TrailColor;

out vec4 v_Color;
out vec2 v_TexCoord;

void main() {
    // Determine which segment and particle this vertex belongs to
    // We draw 6 vertices per segment
    uint totalVerticesPerSegment = 6;
    uint segmentIndex = gl_VertexID / totalVerticesPerSegment;
    uint vertexIndex = gl_VertexID % totalVerticesPerSegment;
    
    // Which particle?
    uint segmentsPerParticle = u_TrailLength - 1;
    uint particleIndex = segmentIndex / segmentsPerParticle;
    uint pointIndex = segmentIndex % segmentsPerParticle;
    
    // Fetch trail points
    uint trailBase = particleIndex * u_TrailLength;
    TrailPoint p1 = trailPoints[trailBase + pointIndex];
    TrailPoint p2 = trailPoints[trailBase + pointIndex + 1];
    
    vec3 pos1 = p1.position.xyz;
    vec3 pos2 = p2.position.xyz;
    float width1 = p1.position.w; // or use uniform u_TrailWidth? Let's use stored width from particle size
    float width2 = p2.position.w;
    
    // Calculate ribbon quad
    vec3 forward = normalize(pos2 - pos1);
    vec3 cameraPos = vec3(inverse(u_View)[3]); // Expensive? Pass as uniform?
    vec3 toCamera = normalize(cameraPos - pos1);
    vec3 right = normalize(cross(forward, toCamera));
    
    // Fallback if degenerate
    if (length(forward) < 0.001) {
        forward = vec3(0, 0, 1);
        right = vec3(1, 0, 0);
    }
    
    float halfWidth = 0.5 * u_TrailWidth; // Global multiplier
    // Modulate by particle size if desired:
    halfWidth *= width1; // Assuming stored size is valid
    
    // Quad vertices relative to the segment line
    vec3 vPos;
    vec2 vUV;
    
    // 0: P1 - Right, 1: P2 - Right, 2: P2 + Right
    // 3: P1 - Right, 4: P2 + Right, 5: P1 + Right
    
    if (vertexIndex == 0 || vertexIndex == 3) {
        vPos = pos1 - right * halfWidth;
        vUV = vec2(0.0, 0.0);
    } else if (vertexIndex == 1) {
        vPos = pos2 - right * halfWidth;
        vUV = vec2(1.0, 0.0);
    } else if (vertexIndex == 2 || vertexIndex == 4) {
        vPos = pos2 + right * halfWidth;
        vUV = vec2(1.0, 1.0);
    } else { // 5
        vPos = pos1 + right * halfWidth;
        vUV = vec2(0.0, 1.0);
    }
    
    // Fade alpha based on point index (age)
    float alpha = 1.0 - (float(pointIndex) / float(segmentsPerParticle));
    v_Color = u_TrailColor * vec4(1, 1, 1, alpha);
    v_TexCoord = vUV;
    
    gl_Position = u_Projection * u_View * vec4(vPos, 1.0);
}
