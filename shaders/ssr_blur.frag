#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D ssrTexture;
uniform sampler2D gPosition;
uniform vec2 screenSize;

const int BLUR_RADIUS = 2;
const float DEPTH_THRESHOLD = 0.5;

void main() {
    vec2 texelSize = 1.0 / screenSize;
    vec3 centerPos = texture(gPosition, TexCoord).rgb;
    
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Bilateral blur
    for (int x = -BLUR_RADIUS; x <= BLUR_RADIUS; x++) {
        for (int y = -BLUR_RADIUS; y <= BLUR_RADIUS; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec2 sampleUV = TexCoord + offset;
            
            // Sample SSR and position
            vec4 sampleSSR = texture(ssrTexture, sampleUV);
            vec3 samplePos = texture(gPosition, sampleUV).rgb;
            
            // Calculate spatial weight (Gaussian-like)
            float spatialWeight = exp(-float(x*x + y*y) / (2.0 * float(BLUR_RADIUS * BLUR_RADIUS)));
            
            // Calculate depth weight (preserve edges)
            float depthDiff = length(samplePos - centerPos);
            float depthWeight = exp(-depthDiff / DEPTH_THRESHOLD);
            
            // Combine weights
            float weight = spatialWeight * depthWeight;
            
            result += sampleSSR * weight;
            totalWeight += weight;
        }
    }
    
    if (totalWeight > 0.0) {
        result /= totalWeight;
    }
    
    FragColor = result;
}
