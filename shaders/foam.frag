#version 330 core

in vec2 v_TexCoord;
in vec4 v_Color;
flat in int v_TextureIndex;

out vec4 FragColor;

in vec2 v_TexCoordNext; // Next frame UVs
in float v_BlendFactor; // Blend factor (0-1)

uniform sampler2D u_FoamTexture;
uniform bool u_HasTexture;
uniform bool u_UseAnimation;
uniform sampler2D u_FoamNormalMap;
uniform bool u_HasNormalMap;

// Soft Particles
uniform bool u_UseSoftParticles;
uniform sampler2D u_SceneDepth;
uniform vec2 u_ScreenSize;
uniform vec2 u_CameraNearFar; // x = Start/Near, y = End/Far

uniform vec3 u_LightDir;     
uniform vec3 u_LightColor;
uniform vec3 u_AmbientColor;
uniform mat4 u_View; 

float LinearizeDepth(float depth) {
    float n = u_CameraNearFar.x;
    float f = u_CameraNearFar.y;
    return (2.0 * n) / (f + n - depth * (f - n)); // Determine exact projection? Assuming standard perspective
    // Alternative standard linearization: (2.0 * n * f) / (f + n - (depth * 2.0 - 1.0) * (f - n)); 
    // If depth is 0..1
}

void main() {
    vec4 texColor = vec4(1.0);
    vec3 normal = vec3(0.0, 0.0, 1.0); // Default normal (facing camera)
    
    if (u_HasTexture) {
        vec4 color1 = texture(u_FoamTexture, v_TexCoord);
        
        if (u_UseAnimation) {
            vec4 color2 = texture(u_FoamTexture, v_TexCoordNext);
            texColor = mix(color1, color2, v_BlendFactor);
            
            // Sample normal map if available
            if (u_HasNormalMap) {
                vec3 n1 = texture(u_FoamNormalMap, v_TexCoord).rgb;
                vec3 n2 = texture(u_FoamNormalMap, v_TexCoordNext).rgb;
                vec3 n = mix(n1, n2, v_BlendFactor);
                normal = normalize(n * 2.0 - 1.0);
            }
        } else {
            texColor = color1;
        }
    } else {
        // Fallback: circular gradient
        vec2 center = (v_TexCoord - vec2(0.25, 0.25)) * 2.0;
        float dist = length(center);
        texColor.a = 1.0 - smoothstep(0.0, 1.0, dist);
        texColor.rgb = vec3(1.0);
        
        // Approximate sphere normal for fallback
        if (dist < 1.0) {
            float z = sqrt(1.0 - dist*dist);
            normal = vec3(center.x, center.y, z); // Tangent space roughly
        }
    }
    
    // Discard fully transparent pixels
    if (texColor.a < 0.01) {
        discard;
    }

    // --- Lighting Calculation ---
    // Normals are in Tangent Space (which for a billboard align with View Space)
    // T = (1,0,0), B = (0,1,0), N = (0,0,1)
    // So Tangent Space Normal == View Space Normal
    
    // Transform light direction to View Space
    // u_LightDir is World Space, so we multiply by View Matrix
    vec3 lightDirView = mat3(u_View) * normalize(u_LightDir);
    
    // Diffuse
    float diff = max(dot(normal, lightDirView), 0.0);
    vec3 diffuse = diff * u_LightColor;
    
    // Specular (Blinn-Phong)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // In view space, viewer is always at 0,0,0 looking down -Z? No, view dir is +Z relative to surface?
    // Actually, in View Space, camera is at origin (0,0,0). Surface is at some negative Z.
    // View Vector is normalize(CameraPos - FragPos).
    // For a billboard, View Vector is effectively (0,0,1) in Tangent Space.
    vec3 halfwayDir = normalize(lightDirView + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = vec3(0.5) * spec; // Moderate specular
    
    vec3 lighting = u_AmbientColor + diffuse + specular;
    
    vec3 lighting = u_AmbientColor + diffuse + specular;
    
    // Apply particle color and alpha
    FragColor = v_Color * texColor;
    FragColor.rgb *= lighting; 
    
    // Soft Particles (Depth Fading)
    if (u_UseSoftParticles) {
        float depth = texture(u_SceneDepth, gl_FragCoord.xy / u_ScreenSize).r;
        
        // Linearize depths for comparison (assuming 0..1 depth range)
        // Standard projection unprojection
        float zNear = u_CameraNearFar.x;
        float zFar = u_CameraNearFar.y;
        float z_scene = (2.0 * zNear * zFar) / (zFar + zNear - (depth * 2.0 - 1.0) * (zFar - zNear));
        float z_particle = (2.0 * zNear * zFar) / (zFar + zNear - (gl_FragCoord.z * 2.0 - 1.0) * (zFar - zNear));
        
        float fadeDistance = 0.5; // World units
        float diff = z_scene - z_particle;
        float fade = clamp(diff / fadeDistance, 0.0, 1.0);
        
        FragColor.a *= fade;
    } 
}
