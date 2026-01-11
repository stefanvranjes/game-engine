#version 330 core

out float FragThickness;

uniform float u_ThicknessScale;

void main() {
    // Calculate distance from center of point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // -1 to 1
    float distSq = dot(coord, coord);
    
    // Discard fragments outside sphere
    if (distSq > 1.0) {
        discard;
    }
    
    // Thickness based on sphere depth
    float thickness = sqrt(1.0 - distSq);
    
    FragThickness = thickness * u_ThicknessScale;
}
