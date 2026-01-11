#version 330 core

in vec3 fragColor;
in float fragAlpha;

out vec4 FragColor;

uniform float glowIntensity;
uniform bool useGlow;

// Animation uniforms
uniform float time;
uniform float pulseSpeed;
uniform float pulseAmplitude;
uniform bool enablePulsing;
uniform float flickerSpeed;
uniform float flickerAmplitude;
uniform bool enableFlickering;

void main() {
    vec3 color = fragColor;
    float alpha = fragAlpha;
    
    // Pulsing effect - synchronized across all cracks
    if (enablePulsing) {
        float pulse = sin(time * pulseSpeed * 6.28318) * 0.5 + 0.5;
        alpha *= 1.0 + pulse * pulseAmplitude;
        
        // Add glow pulsing
        if (useGlow) {
            color += vec3(pulse * glowIntensity * alpha);
        }
    }
    
    // Flickering effect - per-fragment variation
    if (enableFlickering) {
        // Use fragment position as seed for variation
        float seed = gl_FragCoord.x * 0.01 + gl_FragCoord.y * 0.02;
        float flicker = sin(time * flickerSpeed * 6.28318 + seed);
        flicker = flicker * 0.5 + 0.5;
        flicker = pow(flicker, 3.0);  // Make it more sporadic
        alpha *= 1.0 + flicker * flickerAmplitude;
    }
    
    // Base glow effect
    if (useGlow && !enablePulsing) {
        color += vec3(glowIntensity * alpha);
    }
    
    FragColor = vec4(color, alpha);
}
