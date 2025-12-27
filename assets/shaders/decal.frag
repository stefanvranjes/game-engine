#version 330 core
out vec4 gAlbedoSpec; // We only write to Albedo/Spec for now. What about Normal?
// We need to write to multiple targets if we want to update Normal buffer too.
// GBuffer Layout:
// 0: Position (Not updated by decal usually, or we don't care?)
// 1: Normal
// 2: Albedo + Spec
// 3: Emissive
// 4: Velocity
// 5: Depth (Read-Only)

// Wait, we can't read/write same attachment. 
// Standard Deferred Decal:
// WE READ DEPTH. WE WRITE ALBEDO/NORMAL.
// Depth is attached as Read-Only Depth Stencil or Texture.
// But we render into G-Buffer FBO. 
// If we want to depth test against scene geometry, we use hardware depth test (GL_LEQUAL) and disable Depth Write.
// But here we might want to project onto existing geometry, so we need to READ depth to reconstruct position.

layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

uniform sampler2D gDepth;
uniform sampler2D u_DecalAlbedo;
uniform sampler2D u_DecalNormal;

uniform mat4 u_InvModel;
uniform mat4 u_InvView;
uniform mat4 u_InvProjection;
uniform float u_NormalBlending;

uniform vec2 u_ScreenSize;

vec3 ReconstructWorldPos(float depth, vec2 texCoord) {
    float z = depth * 2.0 - 1.0;
    vec4 clipSpacePosition = vec4(texCoord * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = u_InvProjection * clipSpacePosition;
    viewSpacePosition /= viewSpacePosition.w;
    vec4 worldSpacePosition = u_InvView * viewSpacePosition;
    return worldSpacePosition.xyz;
}

void main() {
    vec2 texCoord = gl_FragCoord.xy / u_ScreenSize;
    float depth = texture(gDepth, texCoord).r;
    
    vec3 worldPos = ReconstructWorldPos(depth, texCoord);
    
    // Transform to local space of the decal cube
    vec4 localPos = u_InvModel * vec4(worldPos, 1.0);
    
    // Check if inside unit cube (-0.5 to 0.5)
    // We add small bias to avoid z-fighting at borders? No, clip logic.
    if (abs(localPos.x) > 0.5 || abs(localPos.y) > 0.5 || abs(localPos.z) > 0.5) {
        discard;
    }
    
    // Project UVs (XZ plane usually for top-down decal, but depends on local orientation)
    // Decal Local Space: X=Right, Y=Up, Z=Forward.
    // If we want to project along +Y (Downwards), UV = XZ.
    // Map -0.5..0.5 to 0..1
    vec2 uv = localPos.xz + 0.5;
    
    vec4 albedoSample = texture(u_DecalAlbedo, uv);
    if (albedoSample.a < 0.1) discard; // Alpha clip
    
    // Output
    gAlbedoSpec.rgb = albedoSample.rgb;
    gAlbedoSpec.a = albedoSample.a; // or Specular? GBuffer uses A for Specular.
    // If decal has specular map, use it. For now assume constant.
    gAlbedoSpec.a = 0.5; 
    
    // Normal Mapping (Optional)
    // vec3 normalSample = texture(u_DecalNormal, uv).rgb;
    // ... normal blending logic ...
    // gNormal = ...
}
