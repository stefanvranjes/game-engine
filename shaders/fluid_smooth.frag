#version 330 core

in vec2 v_TexCoord;

out float FragDepth;

uniform sampler2D u_DepthTexture;
uniform vec2 u_TexelSize;
uniform float u_FilterRadius;
uniform float u_DepthThreshold;

void main() {
    float centerDepth = texture(u_DepthTexture, v_TexCoord).r;
    
    // If no depth, skip smoothing
    if (centerDepth == 0.0) {
        FragDepth = 0.0;
        return;
    }
    
    // Bilateral filter: smooth while preserving edges
    float weightSum = 0.0;
    float depthSum = 0.0;
    
    int kernelSize = int(u_FilterRadius);
    for (int x = -kernelSize; x <= kernelSize; ++x) {
        for (int y = -kernelSize; y <= kernelSize; ++y) {
            vec2 offset = vec2(float(x), float(y)) * u_TexelSize;
            vec2 sampleCoord = v_TexCoord + offset;
            
            float sampleDepth = texture(u_DepthTexture, sampleCoord).r;
            
            if (sampleDepth == 0.0) continue;
            
            // Spatial weight (Gaussian)
            float spatialDist = length(vec2(x, y));
            float spatialWeight = exp(-spatialDist * spatialDist / (2.0 * u_FilterRadius * u_FilterRadius));
            
            // Range weight (depth difference)
            float depthDiff = abs(sampleDepth - centerDepth);
            float rangeWeight = exp(-depthDiff * depthDiff / (2.0 * u_DepthThreshold * u_DepthThreshold));
            
            float weight = spatialWeight * rangeWeight;
            
            weightSum += weight;
            depthSum += sampleDepth * weight;
        }
    }
    
    if (weightSum > 0.0) {
        FragDepth = depthSum / weightSum;
    } else {
        FragDepth = centerDepth;
    }
}
