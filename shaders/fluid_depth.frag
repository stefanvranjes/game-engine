#version 330 core

in vec3 v_ViewPos;
in float v_PointSize;

out float FragDepth;

uniform float u_ParticleRadius;

void main() {
    // Calculate distance from center of point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // -1 to 1
    float distSq = dot(coord, coord);
    
    // Discard fragments outside sphere
    if (distSq > 1.0) {
        discard;
    }
    
    // Calculate sphere depth offset
    float z = sqrt(1.0 - distSq);
    vec3 normal = vec3(coord, z);
    
    // Adjust depth based on sphere geometry
    float depthOffset = z * u_ParticleRadius;
    vec3 fragPos = v_ViewPos + normal * u_ParticleRadius;
    
    // Output eye-space depth
    FragDepth = fragPos.z;
    gl_FragDepth = gl_FragCoord.z;
}
