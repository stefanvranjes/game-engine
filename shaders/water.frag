#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexConfigs;
in vec3 Normal;
in vec4 ClipSpace;

uniform vec3 viewPos;
uniform vec3 u_DeepColor;
uniform vec3 u_ShallowColor;
uniform float u_Clarity;
uniform float u_WaveFoamThreshold;

uniform samplerCube skybox;
uniform sampler2D u_NormalMap; // Detail normal
uniform sampler2D u_DerivativesMap; // FFT Normal
uniform sampler2D u_RefractionMap;

// Helper for Fresnel
float calculateFresnel(vec3 normal, vec3 viewDir, float f0) {
    float cosTheta = clamp(dot(normal, viewDir), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    // Screen coords for refraction
    vec2 ndc = (ClipSpace.xy / ClipSpace.w) * 0.5 + 0.5;
    
    // Sample FFT Normal
    // Texture contains normal in RGB?
    // In InitFFT: glTexStorage2D(..., GL_RGBA16F, ...)
    // In Inversion: imageStore(u_Normal, id, vec4(0,1,0,1)); // Placeholder
    // So currently it is flat.
    
    vec3 fftNormal = texture(u_DerivativesMap, TexConfigs).rgb;
    // If we computed slope, we'd reconstruct normal here.
    // For now assuming texture has valid normal.
    // Or if Inversion shader calculated slopes (slopeX, slopeZ), we do cross product.
    // Let's assume 'u_DerivativesMap' holds the Normal vector in World Space directly.
    
    vec3 normal = normalize(fftNormal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Refraction Distortion
    vec2 distortion = (normal.xz * 0.05); // Scale distortion
    vec3 refraction = texture(u_RefractionMap, ndc + distortion).rgb;
    
    // Reflection (Approximate with Cubemap)
    vec3 reflectDir = reflect(-viewDir, normal);
    vec3 reflection = texture(skybox, reflectDir).rgb;
    
    // Fresnel
    float fresnel = calculateFresnel(normal, viewDir, 0.02); // Water F0 ~ 0.02
    
    // Color Mixing
    vec3 waterColor = mix(u_ShallowColor, u_DeepColor, 0.5); // Simple depth approx
    
    // Specular
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3)); // Sun dir (hardcoded for now)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 128.0);
    vec3 specular = vec3(1.0) * spec; // White sun
    
    vec3 finalColor = mix(refraction * waterColor, reflection, fresnel) + specular;
    
    FragColor = vec4(finalColor, 1.0);
}
