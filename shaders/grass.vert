#version 330 core

// Grass blade mesh
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

// Per-instance data
layout (location = 2) in vec3 aInstancePos;
layout (location = 3) in float aRotation;
layout (location = 4) in float aScale;
layout (location = 5) in float aColorVar;
layout (location = 6) in float aWindPhase;

uniform mat4 u_View;
uniform mat4 u_Projection;
uniform float u_Time;
uniform float u_WindStrength;
uniform float u_WindSpeed;
uniform vec2 u_WindDirection;
uniform float u_BladeHeight;
uniform float u_FadeStart;
uniform float u_FadeEnd;

uniform vec3 u_ViewPos;

out vec2 TexCoord;
out vec3 WorldPos;
out float ColorVariation;
out float DistanceFade;
out float HeightGradient;

// Simple noise function
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main()
{
    // Height gradient for wind and color
    float heightT = aPos.y / u_BladeHeight;
    HeightGradient = heightT;
    
    // Rotation matrix around Y axis
    float c = cos(aRotation);
    float s = sin(aRotation);
    mat3 rotY = mat3(
        c, 0.0, -s,
        0.0, 1.0, 0.0,
        s, 0.0, c
    );
    
    // Apply rotation and scale to local position
    vec3 localPos = rotY * (aPos * vec3(1.0, aScale, 1.0));
    
    // ----- WIND -----
    // Sample noise based on world position for wind variation
    vec2 windSamplePos = aInstancePos.xz * 0.1 + u_WindDirection * u_Time * u_WindSpeed;
    float windNoise = noise(windSamplePos) * 2.0 - 1.0;
    
    // Wind strength increases with height (top bends more)
    float windEffect = heightT * heightT;
    
    // Add phase offset for variation
    float phase = u_Time * u_WindSpeed + aWindPhase;
    float windWave = sin(phase + windNoise * 3.14159) * 0.5 + 0.5;
    
    // Calculate wind displacement
    vec2 windDir = normalize(u_WindDirection);
    float windAmount = windEffect * u_WindStrength * (windWave * 0.7 + windNoise * 0.3);
    
    localPos.x += windDir.x * windAmount * 0.3;
    localPos.z += windDir.y * windAmount * 0.3;
    
    // Bend the blade (lean in wind direction)
    localPos.y -= windAmount * windAmount * 0.1;
    
    // World position
    vec3 worldPos = aInstancePos + localPos;
    WorldPos = worldPos;
    
    // Distance fade
    float dist = length(worldPos.xz - u_ViewPos.xz);
    DistanceFade = 1.0 - smoothstep(u_FadeStart, u_FadeEnd, dist);
    
    TexCoord = aTexCoord;
    ColorVariation = aColorVar;
    
    gl_Position = u_Projection * u_View * vec4(worldPos, 1.0);
}
