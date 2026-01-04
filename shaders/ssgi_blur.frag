#version 430 core

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D ssgiInput;
uniform vec2 u_ScreenSize;
uniform int u_BlurRadius;

// Bilateral blur to preserve edges
void main()
{
    vec2 texelSize = 1.0 / u_ScreenSize;
    vec3 result = vec3(0.0);
    float totalWeight = 0.0;
    
    vec3 centerColor = texture(ssgiInput, TexCoord).rgb;
    
    for (int x = -u_BlurRadius; x <= u_BlurRadius; x++) {
        for (int y = -u_BlurRadius; y <= u_BlurRadius; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 sampleColor = texture(ssgiInput, TexCoord + offset).rgb;
            
            // Bilateral weight based on color similarity
            float colorDiff = length(sampleColor - centerColor);
            float weight = exp(-colorDiff * 10.0);
            
            result += sampleColor * weight;
            totalWeight += weight;
        }
    }
    
    result /= totalWeight;
    FragColor = vec4(result, 1.0);
}
