#version 330 core

in vec2 v_TexCoord;

out vec3 FragNormal;

uniform sampler2D u_DepthTexture;
uniform vec2 u_TexelSize;
uniform mat4 u_Projection;

vec3 ReconstructViewPos(vec2 texCoord, float depth) {
    // Convert texture coordinates to NDC
    vec2 ndc = texCoord * 2.0 - 1.0;
    
    // Reconstruct view-space position from depth
    vec4 clipPos = vec4(ndc, depth, 1.0);
    vec4 viewPos = inverse(u_Projection) * clipPos;
    return viewPos.xyz / viewPos.w;
}

void main() {
    float centerDepth = texture(u_DepthTexture, v_TexCoord).r;
    
    if (centerDepth == 0.0) {
        FragNormal = vec3(0.0, 0.0, 1.0);
        return;
    }
    
    // Sample neighboring depths
    float depthRight = texture(u_DepthTexture, v_TexCoord + vec2(u_TexelSize.x, 0.0)).r;
    float depthLeft = texture(u_DepthTexture, v_TexCoord - vec2(u_TexelSize.x, 0.0)).r;
    float depthUp = texture(u_DepthTexture, v_TexCoord + vec2(0.0, u_TexelSize.y)).r;
    float depthDown = texture(u_DepthTexture, v_TexCoord - vec2(0.0, u_TexelSize.y)).r;
    
    // Use center depth if neighbors are missing
    if (depthRight == 0.0) depthRight = centerDepth;
    if (depthLeft == 0.0) depthLeft = centerDepth;
    if (depthUp == 0.0) depthUp = centerDepth;
    if (depthDown == 0.0) depthDown = centerDepth;
    
    // Reconstruct positions
    vec3 posCenter = ReconstructViewPos(v_TexCoord, centerDepth);
    vec3 posRight = ReconstructViewPos(v_TexCoord + vec2(u_TexelSize.x, 0.0), depthRight);
    vec3 posLeft = ReconstructViewPos(v_TexCoord - vec2(u_TexelSize.x, 0.0), depthLeft);
    vec3 posUp = ReconstructViewPos(v_TexCoord + vec2(0.0, u_TexelSize.y), depthUp);
    vec3 posDown = ReconstructViewPos(v_TexCoord - vec2(0.0, u_TexelSize.y), depthDown);
    
    // Compute gradients
    vec3 dx = posRight - posLeft;
    vec3 dy = posUp - posDown;
    
    // Compute normal via cross product
    vec3 normal = normalize(cross(dx, dy));
    
    // Ensure normal points toward camera
    if (normal.z < 0.0) {
        normal = -normal;
    }
    
    FragNormal = normal;
}
