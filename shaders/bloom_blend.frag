#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D hdrBuffer;
uniform sampler2D bloomBlur;
uniform float bloomIntensity;
uniform float exposure;
uniform float gamma;
uniform int toneMappingMode;

// Reinhard tone mapping
vec3 ReinhardToneMapping(vec3 color) {
    return color / (color + vec3(1.0));
}

// ACES filmic tone mapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 hdrColor = texture(hdrBuffer, TexCoord).rgb;
    vec3 bloomColor = texture(bloomBlur, TexCoord).rgb;
    
    // Add bloom to HDR color
    hdrColor += bloomColor * bloomIntensity;
    
    // Apply exposure
    hdrColor *= exposure;
    
    // Tone mapping
    vec3 mapped;
    if (toneMappingMode == 0) {
        mapped = ReinhardToneMapping(hdrColor);
    } else {
        mapped = ACESFilm(hdrColor);
    }
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    
    FragColor = vec4(mapped, 1.0);
}
