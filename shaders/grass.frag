#version 330 core

out vec4 FragColor;

in vec2 TexCoord;
in vec3 WorldPos;
in float ColorVariation;
in float DistanceFade;
in float HeightGradient;

uniform sampler2D u_GrassTexture;
uniform sampler2D u_NoiseTexture;

uniform vec3 u_ColorBase;
uniform vec3 u_ColorTip;
uniform vec3 u_LightDir;
uniform vec3 u_LightColor;
uniform float u_AmbientStrength;

// Subsurface scattering approximation
uniform float u_SSSStrength = 0.3;
uniform vec3 u_SSSColor = vec3(0.5, 0.8, 0.2);

void main()
{
    // Sample grass texture
    vec4 texColor = texture(u_GrassTexture, TexCoord);
    
    // Alpha cutout
    if (texColor.a < 0.5) {
        discard;
    }
    
    // Height-based color gradient
    vec3 baseColor = mix(u_ColorBase, u_ColorTip, HeightGradient);
    
    // Add color variation per instance
    vec3 variedColor = baseColor * (0.85 + ColorVariation * 0.3);
    
    // Combine with texture
    vec3 grassColor = variedColor * texColor.rgb;
    
    // ----- LIGHTING -----
    // Simple diffuse + ambient
    vec3 normal = vec3(0.0, 1.0, 0.0); // Simplified up-facing normal
    vec3 lightDir = normalize(u_LightDir);
    
    float NdotL = max(dot(normal, -lightDir), 0.0);
    float ambient = u_AmbientStrength;
    
    // Subsurface scattering approximation for back-lighting
    float backLight = max(dot(normal, lightDir), 0.0);
    vec3 sss = u_SSSColor * backLight * u_SSSStrength * HeightGradient;
    
    // Final lighting
    vec3 diffuse = grassColor * (NdotL + ambient) * u_LightColor;
    vec3 finalColor = diffuse + sss;
    
    // Apply distance fade
    float alpha = min(texColor.a, DistanceFade);
    
    // Boost saturation slightly for vivid grass
    float luminance = dot(finalColor, vec3(0.299, 0.587, 0.114));
    finalColor = mix(vec3(luminance), finalColor, 1.2);
    
    FragColor = vec4(finalColor, alpha);
}
