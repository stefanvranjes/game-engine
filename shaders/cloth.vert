#version 330 core

// Vertex Attributes
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

// Output to Fragment Shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 ViewPos;
out vec4 CurrentPos;
out vec4 PreviousPos;
out mat3 TBN;

// Transformation Matrices
uniform mat4 u_Model;
uniform mat4 u_MVP;
uniform mat4 u_PrevMVP;
uniform mat4 u_View;
uniform mat4 u_Projection;
uniform vec3 u_ViewPos;

// Cloth-Specific Uniforms
uniform bool u_UsePrecomputedNormals;  // Use normals from mesh data vs. computed
uniform vec3 u_WindVelocity;           // Wind direction and strength
uniform float u_WindStrength;          // Wind intensity multiplier
uniform float u_Time;                  // Time for wind animation
uniform float u_ClothFlexibility;      // How much cloth responds to wind (0-1)

// Clip plane for planar reflections
uniform vec4 u_ClipPlane;
uniform int u_UseClipPlane;

// Wind deformation function
vec3 applyWindDeformation(vec3 worldPos, vec3 normal) {
    if (u_WindStrength < 0.001) {
        return worldPos;
    }
    
    // Calculate wind influence based on normal alignment with wind direction
    vec3 windDir = normalize(u_WindVelocity);
    float windAlignment = max(0.0, dot(normal, windDir));
    
    // Create wave-like motion using sine waves
    float wave1 = sin(worldPos.x * 0.5 + u_Time * 2.0) * 0.5 + 0.5;
    float wave2 = sin(worldPos.z * 0.3 + u_Time * 1.5) * 0.5 + 0.5;
    float wave3 = sin((worldPos.x + worldPos.z) * 0.4 + u_Time * 2.5) * 0.5 + 0.5;
    
    // Combine waves for more natural motion
    float waveFactor = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2);
    
    // Apply wind displacement
    float windInfluence = windAlignment * waveFactor * u_WindStrength * u_ClothFlexibility;
    vec3 displacement = windDir * windInfluence;
    
    return worldPos + displacement;
}

void main()
{
    // Transform to world space
    vec4 worldPos = u_Model * vec4(aPos, 1.0);
    
    // Calculate normal in world space
    vec3 worldNormal;
    if (u_UsePrecomputedNormals) {
        // Use normals from ClothMeshSynchronizer (already calculated from PhysX)
        worldNormal = normalize(mat3(transpose(inverse(u_Model))) * aNormal);
    } else {
        // Compute normals from position derivatives (fallback)
        vec3 dPdx = dFdx(worldPos.xyz);
        vec3 dPdy = dFdy(worldPos.xyz);
        worldNormal = normalize(cross(dPdx, dPdy));
    }
    
    // Apply wind deformation
    worldPos.xyz = applyWindDeformation(worldPos.xyz, worldNormal);
    
    // Recalculate normal after wind deformation
    // This is important for proper lighting on deformed cloth
    vec3 T, B;
    if (length(aTangent) > 0.01) {
        // Use provided tangent/bitangent if available
        T = normalize(mat3(u_Model) * aTangent);
        B = normalize(mat3(u_Model) * aBitangent);
    } else {
        // Compute tangent space from derivatives
        vec3 dPdx = dFdx(worldPos.xyz);
        vec3 dPdy = dFdy(worldPos.xyz);
        vec2 dUVdx = dFdx(aTexCoord);
        vec2 dUVdy = dFdy(aTexCoord);
        
        // Calculate tangent and bitangent
        float r = 1.0 / (dUVdx.x * dUVdy.y - dUVdx.y * dUVdy.x);
        T = normalize((dPdx * dUVdy.y - dPdy * dUVdx.y) * r);
        B = normalize((dPdy * dUVdx.x - dPdx * dUVdy.x) * r);
    }
    
    // Construct TBN matrix for normal mapping
    TBN = mat3(T, B, worldNormal);
    
    // Output to fragment shader
    FragPos = worldPos.xyz;
    Normal = worldNormal;
    TexCoord = aTexCoord;
    ViewPos = u_ViewPos;
    
    // Clip plane for planar reflections
    if (u_UseClipPlane == 1) {
        gl_ClipDistance[0] = dot(worldPos, u_ClipPlane);
    } else {
        gl_ClipDistance[0] = 1.0;
    }
    
    // Calculate current and previous clip-space positions for motion vectors
    // Use local position for MVP multiplication (before wind deformation)
    vec4 localPos = vec4(aPos, 1.0);
    CurrentPos = u_MVP * localPos;
    PreviousPos = u_PrevMVP * localPos;
    
    // Final position
    gl_Position = u_Projection * u_View * worldPos;
}
