#version 330 core
in vec2 TexCoord;
in vec4 Color;

out vec4 FragColor;

uniform sampler2D trailTexture;
uniform bool useTexture;

void main() {
    vec4 texColor = useTexture ? texture(trailTexture, TexCoord) : vec4(1.0);
    FragColor = Color * texColor;
    
    // Premultiply alpha for correct blending
    FragColor.rgb *= FragColor.a;
}
