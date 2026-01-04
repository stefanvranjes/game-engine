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

uniform mat4 u_ProjectionX;
uniform mat4 u_ProjectionY;
uniform mat4 u_ProjectionZ;

void main()
{
    // Calculate triangle normal to determine dominant axis
    vec3 edge1 = gs_in[1].worldPos - gs_in[0].worldPos;
    vec3 edge2 = gs_in[2].worldPos - gs_in[0].worldPos;
    vec3 faceNormal = abs(cross(edge1, edge2));
    
    // Choose projection axis based on dominant normal component
    int axis = 2; // Z-axis by default
    mat4 projection = u_ProjectionZ;
    
    if (faceNormal.x > faceNormal.y && faceNormal.x > faceNormal.z) {
        axis = 0; // X-axis
        projection = u_ProjectionX;
    } else if (faceNormal.y > faceNormal.z) {
        axis = 1; // Y-axis
        projection = u_ProjectionY;
    }
    
    // Emit triangle vertices with chosen projection
    for (int i = 0; i < 3; i++) {
        gs_out.worldPos = gs_in[i].worldPos;
        gs_out.normal = gs_in[i].normal;
        gs_out.texCoord = gs_in[i].texCoord;
        gs_out.axis = axis;
        
        gl_Position = projection * vec4(gs_in[i].worldPos, 1.0);
        
        // Conservative rasterization: expand triangle slightly
        vec2 screenPos = gl_Position.xy / gl_Position.w;
        vec2 pixelSize = vec2(2.0) / vec2(textureSize(usampler3D(0), 0).xy);
        
        // Expand outward from triangle center
        vec2 center = (gl_in[0].gl_Position.xy + gl_in[1].gl_Position.xy + gl_in[2].gl_Position.xy) / 3.0;
        vec2 dir = normalize(screenPos - center);
        screenPos += dir * pixelSize * 0.5;
        
        gl_Position.xy = screenPos * gl_Position.w;
        
        EmitVertex();
    }
    
    EndPrimitive();
}
