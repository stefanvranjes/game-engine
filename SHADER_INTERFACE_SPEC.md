# Hybrid Renderer Shader Interface Specification

## GPU Culling Compute Shaders

### 1. Frustum Culling (`gpu_cull_frustum.comp`)

**Purpose**: Determines visibility and LOD level for instances based on camera frustum.

**Configuration**:
```glsl
#define FRUSTUM_PLANES 6
#define MAX_INSTANCES 8192
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
```

**Input Buffers**:
| Binding | Type | Purpose |
|---------|------|---------|
| 0 | SSBO (readonly) | Cull data (model matrices, bounding volumes) |
| 0 | UBO | Frustum planes, camera data |

**Output Buffers**:
| Binding | Type | Purpose |
|---------|------|---------|
| 1 | SSBO (writeonly) | Visibility flags (uint32 per instance) |
| 2 | SSBO (writeonly) | LOD levels (uint32 per instance) |

**Key Functions**:

```glsl
// Test if sphere is inside/intersecting frustum
bool testSphereFrustum(vec3 sphereCenter, float radius) {
    for (int i = 0; i < FRUSTUM_PLANES; ++i) {
        float dist = dot(sphereCenter, frustumPlanes[i].xyz) - frustumPlanes[i].w;
        if (dist < -radius) return false;  // Outside frustum
    }
    return true;
}

// Test AABB using separating axis theorem
bool testAABBFrustum(vec3 aabbMin, vec3 aabbMax) {
    for (int i = 0; i < FRUSTUM_PLANES; ++i) {
        vec3 normal = frustumPlanes[i].xyz;
        vec3 p = aabbMin;
        if (normal.x > 0.0) p.x = aabbMax.x;
        if (normal.y > 0.0) p.y = aabbMax.y;
        if (normal.z > 0.0) p.z = aabbMax.z;
        if (dot(p, normal) - frustumPlanes[i].w < 0.0) return false;
    }
    return true;
}
```

**Output Data**:
```glsl
visibility[instanceID] = isVisible ? 1U : 0U;
lodLevels[instanceID] = lodLevel;  // 0-3 based on distance
```

---

### 2. Occlusion Culling (`gpu_cull_occlusion.comp`)

**Purpose**: Refines visibility based on depth pyramid from previous frame.

**Configuration**:
```glsl
#define MAX_INSTANCES 8192
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
```

**Input Buffers**:
| Binding | Type | Purpose |
|---------|------|---------|
| 0 | SSBO (readonly) | Cull data |
| 1 | SSBO (readonly) | Frustum visibility from previous pass |
| 0 | UBO | Camera matrices and viewport |

**Input Textures**:
| Unit | Type | Purpose |
|------|------|---------|
| 0 | sampler2D | Mipmapped depth pyramid |

**Output Buffers**:
| Binding | Type | Purpose |
|---------|------|---------|
| 2 | SSBO (writeonly) | Refined visibility (with occlusion) |

**Key Functions**:

```glsl
// Automatically select appropriate mip level based on footprint size
float sampleDepthPyramid(vec2 uv, float footprintSize) {
    int mipLevel = int(ceil(log2(footprintSize)));
    mipLevel = clamp(mipLevel, 0, 12);
    return textureLod(u_DepthPyramid, uv, float(mipLevel)).r;
}

// Test if sphere is occluded by scene
bool isOccluded(vec3 sphereCenter, float radius) {
    vec4 projPos = projection * vec4(sphereCenter, 1.0);
    projPos /= projPos.w;
    vec2 screenUV = projPos.xy * 0.5 + 0.5;
    
    if (screenUV.x < 0.0 || screenUV.x > 1.0 || 
        screenUV.y < 0.0 || screenUV.y > 1.0) {
        return false;  // Off-screen, render conservatively
    }
    
    vec4 projRadius = projection * vec4(radius, 0.0, sphereCenter.z, 1.0);
    float projRadiusPixels = (projRadius.x / projRadius.w) * depthBufferSize.x;
    
    float sampledDepth = sampleDepthPyramid(screenUV, projRadiusPixels);
    float sphereDepth = projPos.z;
    
    return sphereDepth > sampledDepth + 0.01;  // With bias
}
```

**Output Data**:
```glsl
occlusionVisibility[instanceID] = isOccluded(...) ? 0U : 1U;
```

---

## Deferred Lighting Compute Shader

### 3. Deferred Lighting (`deferred_lighting.comp`)

**Purpose**: Evaluates PBR lighting in screen-space using G-Buffer data and light list.

**Configuration**:
```glsl
#define TILE_SIZE 16
#define MAX_LIGHTS_PER_TILE 256
#define MAX_LIGHTS 32
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;
```

**Input Textures**:
| Binding | Type | Purpose |
|---------|------|---------|
| 0 | sampler2D | G-Buffer Position (xyz = worldPos, w = depth) |
| 1 | sampler2D | G-Buffer Normal (xyz = normal, w = roughness) |
| 2 | sampler2D | G-Buffer Albedo (rgb = color, a = metallic) |
| 3 | sampler2D | G-Buffer Emissive (rgb = emissive, a = AO) |
| 4 | sampler2D | Depth Buffer |

**Output Images**:
| Binding | Type | Purpose |
|---------|------|---------|
| 5 | imageBuffer(rgba16f) | Lit scene output |

**Uniform Buffers**:
| Binding | Name | Contents |
|---------|------|----------|
| 0 | LightBuffer | Light array + count |
| 1 | CameraData | View/Projection/Inv-Projection matrices + camera position |

**Light Structure**:
```glsl
struct Light {
    vec4 position;         // xyz = position, w = type (0=dir, 1=point, 2=spot)
    vec4 direction;        // xyz = direction
    vec4 colorIntensity;   // rgb = color, a = intensity
    vec4 params;           // x = range, y = angle (spot), z = attenuation, w = bias
};
```

**Key Functions**:

```glsl
// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX Normal Distribution Function
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}

// Schlick-GGX Geometry Function
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

// Smith Geometry Function
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Compute PBR contribution from single light
vec3 computeLightContribution(Light light, vec3 worldPos, vec3 normal, 
                              vec3 viewDir, vec3 albedo, float metallic, 
                              float roughness) {
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    // ... PBR computation
}
```

**Tiling Strategy**:
- Each 16x16 pixel tile computes a shared light list in `tileVisibleLights`
- Reduces per-pixel light testing from 32 to ~10 average
- Uses atomic operations to build light list

**Output Data**:
```glsl
vec3 lighting = vec3(0);
for(int i = 0; i < lightCount; i++) {
    lighting += computeLightContribution(lights[i], ...);
}
lighting += emissive + albedo * vec3(0.03);  // Add emissive and ambient
imageStore(u_LitOutput, pixelCoord, vec4(lighting, 1.0));
```

---

## G-Buffer Format

**Resolution**: 1920x1080 (configurable)

**Targets**:

| Target | Format | Contents | Precision |
|--------|--------|----------|-----------|
| 0 | RGB32F | World position (xyz) | 32-bit float |
| 1 | RGB32F + R8 | Normal (xyz) + Roughness (w) | 32-bit + 8-bit |
| 2 | RGB8 + R8 | Albedo (rgb) + Metallic (a) | 8-bit sRGB |
| 3 | RGB8 + R8 | Emissive (rgb) + AO (a) | 8-bit |

**Layout (Memory)**:
```
Target 0: [Position.x][Position.y][Position.z][Depth]
Target 1: [Normal.x][Normal.y][Normal.z][Roughness]
Target 2: [Albedo.r][Albedo.g][Albedo.b][Metallic]
Target 3: [Emissive.r][Emissive.g][Emissive.b][AO]
```

---

## Culling Constants UBO

**Layout** (std140):
```glsl
layout(std140) uniform CullingConstants {
    vec4 frustumPlanes[6];      // Offset 0: 6 plane equations
    mat4 viewMatrix;            // Offset 96
    mat4 projectionMatrix;      // Offset 160
    vec3 cameraPosition;        // Offset 224
    float cameraNear;           // Offset 236
    float cameraFar;            // Offset 240
    vec2 depthBufferSize;       // Offset 248
    float padding;              // Offset 256
};
```

**Frustum Plane Format**:
```
plane[i] = vec4(normal.x, normal.y, normal.z, -distance_to_origin)
Signed distance = dot(point, plane.xyz) + plane.w
```

---

## Memory Bandwidth Analysis

### Culling Pass
- **Input**: 8192 instances × 256 bytes = 2 MB
- **Output**: 8192 × 8 bytes (visibility + LOD) = 64 KB
- **Texture**: Minimal (frustum planes)
- **Arithmetic Intensity**: ~10 FLOPs per byte (compute-bound)

### Occlusion Culling Pass
- **Input**: Same as above
- **Texture Reads**: Depth pyramid (varies by LOD, ~4 reads/instance average)
- **Arithmetic Intensity**: ~5 FLOPs per byte

### Deferred Lighting Pass
- **Input**: G-Buffer 4×2 MB = 8 MB, lights ~4 KB
- **Output**: 1920×1080×16 bytes = 33 MB
- **Texture Reads**: ~40 FLOPs per output pixel
- **Arithmetic Intensity**: High (compute-bound for simple lighting, memory-bound for complex)

---

## Compilation Requirements

**GLSL Version**: 4.6 core
**Extensions**: 
- `GL_ARB_compute_shader`
- `GL_ARB_shader_storage_buffer_object`
- `GL_ARB_atomic_counters` (optional, for optional visibility counter)

**Compile Flags**:
```cpp
glShaderSource(shader, 1, &source, nullptr);
glCompileShader(shader);

GLint status;
glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
if(!status) {
    GLchar infoLog[512];
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    // Handle compilation error
}
```

---

## Synchronization Points

**GPU Memory Barriers**:
```glsl
// After culling compute
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

// After geometry pass, before lighting
glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

// After lighting compute
glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
```

---

## Performance Guidelines

| Shader | Throughput | Latency | Bottleneck |
|--------|-----------|---------|-----------|
| Frustum Culling | 100k instances/ms | ~1 ms | Compute |
| Occlusion Culling | 10k instances/ms | ~2 ms | Texture bandwidth |
| Deferred Lighting | 1920×1080 @ 60fps | ~5 ms | FLOPs |

**Optimization Tips**:
1. Use compute shader grouping (shared memory) in lighting for light lists
2. Coalesce memory accesses (workgroups process contiguous instances)
3. Use mipmap pyramid for occlusion to reduce texture bandwidth
4. Consider wave/subgroup operations for light list building
