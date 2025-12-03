#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D currentFrame;
uniform sampler2D historyFrame;
uniform sampler2D velocityTexture;

uniform float blendFactor;
uniform vec2 screenSize;
uniform int frameIndex;

// Sample with catmull-rom filtering for better quality
vec3 SampleCatmullRom(sampler2D tex, vec2 uv) {
    vec2 texelSize = 1.0 / screenSize;
    vec2 position = uv * screenSize;
    
    vec2 centerPosition = floor(position - 0.5) + 0.5;
    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;
    
    vec2 s0 = w0 + w1;
    vec2 s1 = w2 + w3;
    
    vec2 f0 = w1 / s0;
    vec2 f1 = w3 / s1;
    
    vec2 t0 = centerPosition - 1.0 + f0;
    vec2 t1 = centerPosition + 1.0 + f1;
    
    return (texture(tex, t0 * texelSize).rgb * s0.x +
            texture(tex, vec2(t1.x, t0.y) * texelSize).rgb * s1.x) * s0.y +
           (texture(tex, vec2(t0.x, t1.y) * texelSize).rgb * s0.x +
            texture(tex, t1 * texelSize).rgb * s1.x) * s1.y;
}

// RGB to YCoCg color space for better clamping
vec3 RGBToYCoCg(vec3 rgb) {
    float Y  =  0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
    float Co =  0.5  * rgb.r - 0.5 * rgb.b;
    float Cg = -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
    return vec3(Y, Co, Cg);
}

vec3 YCoCgToRGB(vec3 ycocg) {
    float tmp = ycocg.x - ycocg.z;
    float r = tmp + ycocg.y;
    float g = ycocg.x + ycocg.z;
    float b = tmp - ycocg.y;
    return vec3(r, g, b);
}

// Clip history to neighborhood AABB
vec3 ClipAABB(vec3 aabbMin, vec3 aabbMax, vec3 prevSample) {
    vec3 center = 0.5 * (aabbMax + aabbMin);
    vec3 extents = 0.5 * (aabbMax - aabbMin);
    
    vec3 offset = prevSample - center;
    vec3 ts = abs(extents) / (abs(offset) + 0.0001);
    float t = min(min(ts.x, ts.y), ts.z);
    
    if (t < 1.0) {
        return center + offset * t;
    }
    return prevSample;
}

void main() {
    vec2 texelSize = 1.0 / screenSize;
    
    // Sample current frame
    vec3 currentColor = texture(currentFrame, TexCoord).rgb;
    
    // Sample velocity
    vec2 velocity = texture(velocityTexture, TexCoord).rg;
    
    // Calculate previous frame UV
    vec2 prevUV = TexCoord - velocity;
    
    // Check if previous UV is valid
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        // Off-screen, use current frame only
        FragColor = vec4(currentColor, 1.0);
        return;
    }
    
    // Sample history with high-quality filtering
    vec3 historyColor = SampleCatmullRom(historyFrame, prevUV);
    
    // Compute neighborhood statistics for clamping (3x3 neighborhood)
    vec3 neighborhoodMin = vec3(1000.0);
    vec3 neighborhoodMax = vec3(-1000.0);
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 neighbor = texture(currentFrame, TexCoord + offset).rgb;
            
            // Convert to YCoCg for better color space
            neighbor = RGBToYCoCg(neighbor);
            
            neighborhoodMin = min(neighborhoodMin, neighbor);
            neighborhoodMax = max(neighborhoodMax, neighbor);
            
            m1 += neighbor;
            m2 += neighbor * neighbor;
        }
    }
    
    // Variance clipping (more aggressive than simple min/max)
    m1 /= 9.0;
    m2 /= 9.0;
    
    vec3 sigma = sqrt(max(m2 - m1 * m1, 0.0));
    vec3 boxMin = m1 - 1.0 * sigma;
    vec3 boxMax = m1 + 1.0 * sigma;
    
    // Expand box slightly to reduce flickering
    boxMin = min(boxMin, neighborhoodMin);
    boxMax = max(boxMax, neighborhoodMax);
    
    // Clip history to neighborhood
    vec3 historyYCoCg = RGBToYCoCg(historyColor);
    historyYCoCg = ClipAABB(boxMin, boxMax, historyYCoCg);
    historyColor = YCoCgToRGB(historyYCoCg);
    
    // Blend current and history
    // Higher blendFactor = more history (smoother but more ghosting)
    // Lower blendFactor = more current (less ghosting but more aliasing)
    vec3 result = mix(currentColor, historyColor, blendFactor);
    
    FragColor = vec4(result, 1.0);
}
