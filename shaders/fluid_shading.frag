#version 330 core

in vec2 v_TexCoord;

out vec4 FragColor;

uniform sampler2D u_DepthTexture;
uniform sampler2D u_NormalTexture;
uniform sampler2D u_ThicknessTexture;
uniform sampler2D u_SceneTexture;
uniform sampler2D u_SceneDepthTexture;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform mat4 u_InvProjection;
uniform vec3 u_CameraPos;
uniform vec3 u_LightDir;

uniform vec3 u_FluidColor;
uniform float u_RefractiveIndex;
uniform vec3 u_AbsorptionColor;
uniform float u_FresnelPower;
uniform float u_SpecularPower;

vec3 ReconstructViewPos(vec2 texCoord, float depth) {
    vec2 ndc = texCoord * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, depth, 1.0);
    vec4 viewPos = u_InvProjection * clipPos;
    return viewPos.xyz / viewPos.w;
}

float FresnelSchlick(float cosTheta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, u_FresnelPower);
}

void main() {
    float depth = texture(u_DepthTexture, v_TexCoord).r;
    
    // If no fluid at this pixel, show scene
    if (depth == 0.0) {
        FragColor = texture(u_SceneTexture, v_TexCoord);
        return;
    }
    
    // Get fluid surface properties
    vec3 normal = texture(u_NormalTexture, v_TexCoord).rgb;
    float thickness = texture(u_ThicknessTexture, v_TexCoord).r;
    vec3 viewPos = ReconstructViewPos(v_TexCoord, depth);
    
    // View direction
    vec3 viewDir = normalize(-viewPos);
    
    // Fresnel effect
    float fresnel = FresnelSchlick(max(dot(normal, viewDir), 0.0), 1.0, u_RefractiveIndex);
    
    // Refraction
    vec3 refractDir = refract(-viewDir, normal, 1.0 / u_RefractiveIndex);
    vec2 refractOffset = refractDir.xy * thickness * 0.1;
    vec2 refractCoord = clamp(v_TexCoord + refractOffset, 0.0, 1.0);
    vec3 refractColor = texture(u_SceneTexture, refractCoord).rgb;
    
    // Absorption based on thickness
    vec3 absorption = exp(-u_AbsorptionColor * thickness);
    refractColor *= absorption;
    
    // Reflection (simple environment reflection)
    vec3 reflectDir = reflect(-viewDir, normal);
    vec3 reflectColor = u_FluidColor * 0.5;  // Simplified, could sample environment map
    
    // Specular highlight
    vec3 halfDir = normalize(u_LightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), u_SpecularPower);
    vec3 specular = vec3(spec);
    
    // Combine refraction and reflection using Fresnel
    vec3 color = mix(refractColor, reflectColor, fresnel);
    color += specular * 0.5;
    
    // Add base fluid color tint
    color = mix(color, u_FluidColor, 0.2);
    
    FragColor = vec4(color, 1.0);
}
