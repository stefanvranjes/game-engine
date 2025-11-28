#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D hdrBuffer;
uniform float threshold;

void main()
{
    vec3 color = texture(hdrBuffer, TexCoord).rgb;
    
    // Calculate brightness using luminance formula
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Extract only bright areas above threshold
    if (brightness > threshold) {
        FragColor = vec4(color, 1.0);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
