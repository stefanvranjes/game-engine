#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;


out vec2 TexCoord;
out vec3 WorldPos;
out vec3 Normal;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform vec3 u_CenterPos; // World position of the billboard center
uniform vec2 u_Size;      // Width, Height

void main()
{
    TexCoord = aTexCoords;
    
    // Billboard Logic:
    vec3 CameraRight = vec3(u_View[0][0], u_View[1][0], u_View[2][0]);
    vec3 CameraUp    = vec3(u_View[0][1], u_View[1][1], u_View[2][1]);
    
    // Normal always points back to camera (approx)
    // Or simpler: Camera Forward inverted. 
    // Camera Forward is Row 2 of View Matrix (if un-transposed) -> Col 2 if transposed?
    // In Column-Major View Matrix: Col 2 is Forward (or -Forward).
    vec3 CameraForward = vec3(u_View[0][2], u_View[1][2], u_View[2][2]);
    Normal = CameraForward; // Points towards camera (View Z is positive in eye space usually? Wait. View is usually -Z forward. Camera looks down -Z. So Forward vector is -Z. Normal should point to +Z in View space = towards camera.
    // Actually, View Matrix Row 2 is Z axis. If right-handed, Z is BACK. Camera Back vector.
    // So Normal = Camera Back works (points to camera).
    
    vec3 worldPos = u_CenterPos 
                  + CameraRight * aPos.x * u_Size.x 
                  + CameraUp * aPos.y * u_Size.y;
    
    WorldPos = worldPos;
    gl_Position = u_Projection * u_View * vec4(worldPos, 1.0);
}
