#version 330 core

in vec2 v_TexCoord;
in vec4 v_Color;
in vec4 v_FragPos; // Fragment position in clip space

out vec4 FragColor;

uniform sampler2D u_Texture;
uniform sampler2D u_DepthTexture;
uniform bool u_HasTexture;
uniform bool u_SoftParticles;
uniform vec2 u_ScreenSize;
uniform float u_Softness; // Fade distance

void main() {
    vec4 texColor = vec4(1.0);
    
    if (u_HasTexture) {
        texColor = texture(u_Texture, v_TexCoord);
    } else {
        // Default: circular gradient for soft particles
        vec2 center = v_TexCoord - vec2(0.5);
        float dist = length(center);
        texColor.a = 1.0 - smoothstep(0.0, 0.5, dist);
    }
    
    FragColor = v_Color * texColor;
    
    // Soft particles: fade based on depth difference
    if (u_SoftParticles) {
        // Calculate screen-space UV coordinates
        vec2 screenUV = gl_FragCoord.xy / u_ScreenSize;
        
        // Sample scene depth
        float sceneDepth = texture(u_DepthTexture, screenUV).r;
        
        // Get particle depth (convert from clip space)
        float particleDepth = gl_FragCoord.z;
        
        // Calculate depth difference
        float depthDiff = sceneDepth - particleDepth;
        
        // Fade alpha based on proximity to geometry
        float fade = saturate(depthDiff / u_Softness);
        FragColor.a *= fade;
    }
    
    // Discard fully transparent pixels
    if (FragColor.a < 0.01) {
        discard;
    }
}

// GLSL doesn't have saturate, define it
float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}
