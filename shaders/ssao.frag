#version 330 core
out float FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform vec3 samples[64];
uniform mat4 projection;
uniform mat4 view;
uniform float radius;
uniform float bias;
uniform vec2 noiseScale;

void main()
{
    // Get input for SSAO algorithm
    vec3 fragPos = texture(gPosition, TexCoord).xyz;
    vec3 normal = normalize(texture(gNormal, TexCoord).rgb);
    vec3 randomVec = normalize(texture(texNoise, TexCoord * noiseScale).xyz);
    
    // Transform to view space
    vec3 fragPosView = (view * vec4(fragPos, 1.0)).xyz;
    vec3 normalView = normalize((view * vec4(normal, 0.0)).xyz);
    
    // Create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 tangent = normalize(randomVec - normalView * dot(randomVec, normalView));
    vec3 bitangent = cross(normalView, tangent);
    mat3 TBN = mat3(tangent, bitangent, normalView);
    
    // Iterate over the sample kernel and calculate occlusion factor
    float occlusion = 0.0;
    for(int i = 0; i < 64; ++i)
    {
        // Get sample position
        vec3 samplePos = TBN * samples[i]; // From tangent to view-space
        samplePos = fragPosView + samplePos * radius; 
        
        // Project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
        
        // Get sample depth
        vec3 sampleFragPos = texture(gPosition, offset.xy).xyz;
        vec3 sampleFragPosView = (view * vec4(sampleFragPos, 1.0)).xyz;
        float sampleDepth = sampleFragPosView.z;
        
        // Range check & accumulate
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPosView.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }
    
    occlusion = 1.0 - (occlusion / 64.0);
    
    FragColor = occlusion;
}
