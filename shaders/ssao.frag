#version 330 core
out float FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];

// Parameters (passed from C++)
uniform mat4 projection;
uniform mat4 view;
uniform float radius;
uniform float bias;
uniform vec2 noiseScale;

void main()
{
    // 1. Get position and normal in View Space
    vec3 fragPosWorld = texture(gPosition, TexCoord).xyz;
    vec3 fragPosView = (view * vec4(fragPosWorld, 1.0)).xyz;
    
    vec3 normalWorld = texture(gNormal, TexCoord).rgb;
    vec3 normalView = normalize(mat3(view) * normalWorld);
    
    // 2. Create TBN matrix (Tangent, Bitangent, Normal)
    vec3 randomVec = texture(texNoise, TexCoord * noiseScale).xyz;
    
    vec3 tangent = normalize(randomVec - normalView * dot(randomVec, normalView));
    vec3 bitangent = cross(normalView, tangent);
    mat3 TBN = mat3(tangent, bitangent, normalView);
    
    // 3. Iterate over samples
    float occlusion = 0.0;
    for(int i = 0; i < 64; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * samples[i]; // From tangent to view-space
        samplePos = fragPosView + samplePos * radius; 
        
        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
        
        // get sample depth
        vec3 samplePosWorld = texture(gPosition, offset.xy).xyz;
        vec3 samplePosView = (view * vec4(samplePosWorld, 1.0)).xyz;
        float sampleDepth = samplePosView.z; // get depth value of kernel sample
        
        // range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPosView.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;           
    }
    
    occlusion = 1.0 - (occlusion / 64.0);
    FragColor = occlusion;
}
