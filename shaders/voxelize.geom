#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
} gs_in[];

out GS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texCoord;
    flat int axis;
} gs_out;

uniform int u_Resolution;

void main()
{
    // ... dominant axis calculation ...

    // Emit triangle vertices with chosen projection
    for (int i = 0; i < 3; i++) {
        gs_out.worldPos = gs_in[i].worldPos;
        gs_out.normal = gs_in[i].normal;
        gs_out.texCoord = gs_in[i].texCoord;
        gs_out.axis = axis;
        
        gl_Position = projection * vec4(gs_in[i].worldPos, 1.0);
        
        // Conservative rasterization: expand triangle slightly
        vec2 screenPos = gl_Position.xy / gl_Position.w;
        vec2 pixelSize = vec2(2.0) / vec2(float(u_Resolution));
        
        // Expand outward from triangle center
        vec2 center = (gl_in[0].gl_Position.xy + gl_in[1].gl_Position.xy + gl_in[2].gl_Position.xy) / 3.0;
        vec2 dir = normalize(screenPos - center);
        
        // Check for NaN/Inf in dir if screenPos == center
        if (length(screenPos - center) > 1e-6) {
             screenPos += dir * pixelSize * 0.7071f; // Expand by half diagonal
        }
        
        gl_Position.xy = screenPos * gl_Position.w;
        
        EmitVertex();
    }
    
    EndPrimitive();
}
