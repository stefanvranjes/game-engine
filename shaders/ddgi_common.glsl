#version 460 core

// Constants
const float PI = 3.14159265359;

// --- Octahedral Encoding ---

float signNotZero(float v) {
    return (v >= 0.0) ? 1.0 : -1.0;
}

vec2 signNotZero(vec2 v) {
    return vec2(signNotZero(v.x), signNotZero(v.y));
}

// Encode normalized direction to [-1, 1] octahedron UV
vec2 OctEncode(vec3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0) {
        result = (1.0 - abs(result.yx)) * signNotZero(result);
    }
    return result;
}

// Decode [-1, 1] octahedron UV to normalized direction
vec3 OctDecode(vec2 o) {
    vec3 v = vec3(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
    if (v.z < 0.0) {
        v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}

// Map probe index to 3D Grid Coordinate
ivec3 GetProbeGridCoord(int probeIndex, ivec3 gridDim) {
    int z = probeIndex / (gridDim.x * gridDim.y);
    int y = (probeIndex % (gridDim.x * gridDim.y)) / gridDim.x;
    int x = probeIndex % gridDim.x;
    return ivec3(x, y, z);
}

// Compute texture coordinate for a probe's texel
// index: Probe index (0 to TotalProbes-1)
// texel: Texel offset within probe (0 to 7)
// res: Resolution of probe (e.g. 8)
// border: Border size (e.g. 1)
ivec2 GetProbeTexelCoord(int probeIndex, ivec2 texelOffset, int gridSizeX, int probeRes, int border) {
    // Atlas layout assumption:
    // Width = AtlasCols * (Res + 2*Border)
    // We need to match C++ layout
    
    int totalRes = probeRes + 2 * border;
    
    // Assume simple grid layout in Atlas
    // X wraps at gridDim.x * gridDim.y?
    // In C++ we did: atlasCols = ceil(sqrt(TotalProbes))
    // Let's rely on uniforms for atlas dimensions if possible, or recalculate.
    // For now, simpler layout:
    int atlasWidthProbes = int(ceil(sqrt(float(gridSizeX)))); // Recompute similar to C++? 
    // No, passing uniforms is safer.
    // Placeholder: assume caller passes base Coord.
    return ivec2(0);
}
