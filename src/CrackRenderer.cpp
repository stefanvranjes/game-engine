#include "CrackRenderer.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <algorithm>
#include <cmath>

CrackRenderer::CrackRenderer()
    : m_VAO(0)
    , m_VBO_Positions(0)
    , m_VBO_Colors(0)
    , m_Initialized(false)
    , m_AnimationTime(0.0f)
{
}

CrackRenderer::~CrackRenderer() {
    Cleanup();
}

void CrackRenderer::Initialize() {
    if (m_Initialized) {
        return;
    }

    // Generate OpenGL resources
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO_Positions);
    glGenBuffers(1, &m_VBO_Colors);

    // Setup VAO
    glBindVertexArray(m_VAO);

    // Position attribute
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO_Positions);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO_Colors);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    m_Initialized = true;
}

void CrackRenderer::Cleanup() {
    if (m_VAO) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
    if (m_VBO_Positions) {
        glDeleteBuffers(1, &m_VBO_Positions);
        m_VBO_Positions = 0;
    }
    if (m_VBO_Colors) {
        glDeleteBuffers(1, &m_VBO_Colors);
        m_VBO_Colors = 0;
    }
    m_Initialized = false;
}

void CrackRenderer::Update(float deltaTime) {
    m_AnimationTime += deltaTime;
}

void CrackRenderer::RenderCracks(
    const std::vector<PartialTearSystem::Crack>& cracks,
    const Vec3* vertices,
    const int* tetrahedra,
    Shader* shader,
    const Mat4& modelMatrix,
    float currentTime)
{
    if (cracks.empty() || !vertices || !tetrahedra || !shader) {
        return;
    }

    if (!m_Initialized) {
        Initialize();
    }

    // Clear cached geometry
    m_CachedVertices.clear();
    m_CachedColors.clear();

    // Ensure pulse state tracking matches crack count
    if (m_LastPulseState.size() != cracks.size()) {
        m_LastPulseState.resize(cracks.size(), false);
    }
    
    // Ensure particle timing tracking matches crack count
    if (m_LastParticleTime.size() != cracks.size()) {
        m_LastParticleTime.resize(cracks.size(), 0.0f);
    }

    // Generate geometry for all cracks and check for pulse peaks
    for (size_t i = 0; i < cracks.size(); ++i) {
        const auto& crack = cracks[i];
        GenerateCrackGeometry(crack, vertices, tetrahedra, m_CachedVertices, m_CachedColors, currentTime);
        
        // Calculate crack position and normal for effects
        const int edgeIndices[6][2] = {
            {0, 1}, {0, 2}, {0, 3},
            {1, 2}, {1, 3}, {2, 3}
        };
        const int* tet = &tetrahedra[crack.tetrahedronIndex * 4];
        int edgeIdx = crack.edgeIndex;
        
        if (edgeIdx >= 0 && edgeIdx < 6) {
            int v0 = tet[edgeIndices[edgeIdx][0]];
            int v1 = tet[edgeIndices[edgeIdx][1]];
            Vec3 p0 = vertices[v0];
            Vec3 p1 = vertices[v1];
            Vec3 crackPos = (p0 + p1) * 0.5f;  // Midpoint
            Vec3 crackDir = p1 - p0;
            float length = std::sqrt(crackDir.x * crackDir.x + crackDir.y * crackDir.y + crackDir.z * crackDir.z);
            if (length > 0.001f) {
                crackDir.x /= length;
                crackDir.y /= length;
                crackDir.z /= length;
            }
            // Normal perpendicular to crack
            Vec3 crackNormal(-crackDir.y, crackDir.x, 0.0f);
            
            // Particle effects
            if (m_Settings.enableParticles && m_ParticleCallback) {
                // Continuous particle spawning
                float timeSinceLastParticle = currentTime - m_LastParticleTime[i];
                float spawnInterval = 1.0f / m_Settings.particleSpawnRate;
                
                if (timeSinceLastParticle >= spawnInterval) {
                    // Spawn continuous particles
                    Vec3 velocity = crackNormal * (0.5f + crack.damage * 0.5f);  // Faster with more damage
                    m_ParticleCallback(crackPos, crackNormal, velocity, crack.damage, 1);
                    m_LastParticleTime[i] = currentTime;
                }
            }
        
            // Sound synchronization - detect pulse peaks
            if (m_Settings.enableSoundSync && m_Settings.enablePulsing && m_SoundCallback) {
                const float PI = 3.14159265359f;
                
                // Calculate pulse speed (with damage modulation if enabled)
                float effectivePulseSpeed = m_Settings.pulseSpeed;
                if (m_Settings.damageAffectsSpeed) {
                    float damageT = std::clamp(crack.damage, 0.0f, 1.0f);
                    effectivePulseSpeed = m_Settings.minPulseSpeed + 
                                         damageT * (m_Settings.maxPulseSpeed - m_Settings.minPulseSpeed);
                }
                
                // Calculate current pulse value (0.0 to 1.0)
                float pulse = std::sin(currentTime * effectivePulseSpeed * 2.0f * PI) * 0.5f + 0.5f;
                
                // Detect rising edge crossing threshold
                bool aboveThreshold = pulse >= m_Settings.soundSyncThreshold;
                if (aboveThreshold && !m_LastPulseState[i]) {
                    // Pulse peak detected - trigger sound
                    float intensity = m_Settings.pulseAmplitude * pulse;
                    m_SoundCallback(static_cast<int>(i), crack.damage, intensity);
                    
                    // Spawn particle burst on pulse if enabled
                    if (m_Settings.enableParticles && m_Settings.particlesOnPulse && m_ParticleCallback) {
                        Vec3 burstVelocity = crackNormal * (1.0f + crack.damage * 2.0f);  // Stronger burst
                        m_ParticleCallback(crackPos, crackNormal, burstVelocity, crack.damage, m_Settings.particleBurstCount);
                    }
                }
                m_LastPulseState[i] = aboveThreshold;
            }
        }
    }

    if (m_CachedVertices.empty()) {
        return;
    }

    // Update buffers
    UpdateBuffers();

    // Render
    shader->Use();
    shader->SetMat4("model", modelMatrix);
    shader->SetFloat("glowIntensity", m_Settings.glowIntensity);
    shader->SetBool("useGlow", m_Settings.useGlow);
    
    // Animation uniforms
    shader->SetFloat("time", m_AnimationTime);
    shader->SetFloat("pulseSpeed", m_Settings.pulseSpeed);
    shader->SetFloat("pulseAmplitude", m_Settings.pulseAmplitude);
    shader->SetBool("enablePulsing", m_Settings.enablePulsing);
    shader->SetFloat("flickerSpeed", m_Settings.flickerSpeed);
    shader->SetFloat("flickerAmplitude", m_Settings.flickerAmplitude);
    shader->SetBool("enableFlickering", m_Settings.enableFlickering);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Disable depth writing for cracks (render on top)
    glDepthMask(GL_FALSE);

    // Set line width
    glLineWidth(3.0f);

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_CachedVertices.size()));
    glBindVertexArray(0);

    // Restore state
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

void CrackRenderer::GenerateCrackGeometry(
    const PartialTearSystem::Crack& crack,
    const Vec3* vertices,
    const int* tetrahedra,
    std::vector<Vec3>& outVertices,
    std::vector<Vec3>& outColors,
    float currentTime)
{
    // Edge indices for a tetrahedron
    const int edgeIndices[6][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };

    // Get tetrahedron vertices
    const int* tet = &tetrahedra[crack.tetrahedronIndex * 4];
    int edgeIdx = crack.edgeIndex;

    if (edgeIdx < 0 || edgeIdx >= 6) {
        return;  // Invalid edge
    }

    // Get edge endpoints
    int v0 = tet[edgeIndices[edgeIdx][0]];
    int v1 = tet[edgeIndices[edgeIdx][1]];

    Vec3 p0 = vertices[v0];
    Vec3 p1 = vertices[v1];

    // Calculate crack color and opacity based on damage
    Vec3 color = CalculateCrackColor(crack.damage);
    float opacity = CalculateCrackOpacity(crack.damage);
    
    // Apply color pulsing if enabled
    if (m_Settings.enableColorPulsing && m_Settings.enablePulsing) {
        const float PI = 3.14159265359f;
        
        // Calculate pulse speed (with damage modulation if enabled)
        float effectivePulseSpeed = m_Settings.pulseSpeed;
        if (m_Settings.damageAffectsSpeed) {
            float damageT = std::clamp(crack.damage, 0.0f, 1.0f);
            effectivePulseSpeed = m_Settings.minPulseSpeed + 
                                 damageT * (m_Settings.maxPulseSpeed - m_Settings.minPulseSpeed);
        }
        
        // Calculate pulse value (0.0 to 1.0)
        float pulse = std::sin(currentTime * effectivePulseSpeed * 2.0f * PI) * 0.5f + 0.5f;
        
        // Interpolate between min and max pulse colors
        Vec3 pulseColor;
        pulseColor.x = m_Settings.pulseColorMin.x + pulse * (m_Settings.pulseColorMax.x - m_Settings.pulseColorMin.x);
        pulseColor.y = m_Settings.pulseColorMin.y + pulse * (m_Settings.pulseColorMax.y - m_Settings.pulseColorMin.y);
        pulseColor.z = m_Settings.pulseColorMin.z + pulse * (m_Settings.pulseColorMax.z - m_Settings.pulseColorMin.z);
        
        // Blend pulse color with base color
        color = pulseColor;
    }
    
    // Apply animation effects
    float animIntensity = CalculateAnimatedIntensity(crack, currentTime);
    opacity *= animIntensity;
    
    // Apply growth fade-in
    float growthFactor = CalculateGrowthFactor(crack, currentTime);
    opacity *= growthFactor;
    
    // Apply crack propagation (length growth)
    float propagationFactor = CalculatePropagationFactor(crack, currentTime);
    
    // Interpolate crack endpoints based on propagation
    Vec3 crackStart = p0;
    Vec3 crackEnd = p0 + (p1 - p0) * propagationFactor;

    // Store color with opacity in RGB (will use red channel as alpha in shader)
    Vec3 colorWithAlpha = color * opacity;

    // Add line segment (may be partial if still propagating)
    outVertices.push_back(crackStart);
    outVertices.push_back(crackEnd);
    outColors.push_back(colorWithAlpha);
    outColors.push_back(colorWithAlpha);
}

void CrackRenderer::SetRenderSettings(const RenderSettings& settings) {
    m_Settings = settings;
}

void CrackRenderer::SetSoundCallback(std::function<void(int, float, float)> callback) {
    m_SoundCallback = callback;
}

void CrackRenderer::SetParticleCallback(std::function<void(Vec3, Vec3, Vec3, float, int)> callback) {
    m_ParticleCallback = callback;
}

Vec3 CrackRenderer::CalculateCrackColor(float damage) const {
    if (!m_Settings.useDamageColor) {
        return m_Settings.crackColor;
    }

    // Interpolate between low and high damage colors
    float t = std::clamp(damage, 0.0f, 1.0f);
    Vec3 color;
    color.x = m_Settings.lowDamageColor.x + t * (m_Settings.highDamageColor.x - m_Settings.lowDamageColor.x);
    color.y = m_Settings.lowDamageColor.y + t * (m_Settings.highDamageColor.y - m_Settings.lowDamageColor.y);
    color.z = m_Settings.lowDamageColor.z + t * (m_Settings.highDamageColor.z - m_Settings.lowDamageColor.z);
    return color;
}

float CrackRenderer::CalculateCrackOpacity(float damage) const {
    float t = std::clamp(damage, 0.0f, 1.0f);
    return m_Settings.minOpacity + t * (m_Settings.maxOpacity - m_Settings.minOpacity);
}

float CrackRenderer::CalculateCrackWidth(float damage) const {
    float t = std::clamp(damage, 0.0f, 1.0f);
    return m_Settings.baseWidth + t * (m_Settings.maxWidth - m_Settings.baseWidth);
}

float CrackRenderer::CalculateAnimatedIntensity(
    const PartialTearSystem::Crack& crack,
    float currentTime) const
{
    float intensity = 1.0f;
    
    // Pulsing effect - smooth sine wave
    if (m_Settings.enablePulsing) {
        const float PI = 3.14159265359f;
        
        // Calculate pulse speed based on damage if enabled
        float effectivePulseSpeed = m_Settings.pulseSpeed;
        if (m_Settings.damageAffectsSpeed) {
            // Interpolate between min and max speed based on damage
            float damageT = std::clamp(crack.damage, 0.0f, 1.0f);
            effectivePulseSpeed = m_Settings.minPulseSpeed + 
                                 damageT * (m_Settings.maxPulseSpeed - m_Settings.minPulseSpeed);
        }
        
        float pulse = std::sin(currentTime * effectivePulseSpeed * 2.0f * PI) * 0.5f + 0.5f;
        intensity *= 1.0f + pulse * m_Settings.pulseAmplitude;
    }
    
    // Flickering effect - random variations per crack
    if (m_Settings.enableFlickering) {
        const float PI = 3.14159265359f;
        // Use crack index as seed for variation
        float crackSeed = static_cast<float>(crack.tetrahedronIndex) * 0.1f;
        float flicker = std::sin(currentTime * m_Settings.flickerSpeed * 2.0f * PI + crackSeed);
        flicker = flicker * 0.5f + 0.5f;
        // Make it more sporadic with power function
        flicker = std::pow(flicker, 3.0f);
        intensity *= 1.0f + flicker * m_Settings.flickerAmplitude;
    }
    
    return intensity;
}

float CrackRenderer::CalculateGrowthFactor(
    const PartialTearSystem::Crack& crack,
    float currentTime) const
{
    if (!m_Settings.enableGrowth) {
        return 1.0f;
    }
    
    // Calculate age of crack
    float age = currentTime - crack.creationTime;
    
    // Fade in over growth duration
    float growth = std::clamp(age / m_Settings.growthDuration, 0.0f, 1.0f);
    
    // Smooth step for nicer fade-in
    growth = growth * growth * (3.0f - 2.0f * growth);
    
    return growth;
}

float CrackRenderer::CalculatePropagationFactor(
    const PartialTearSystem::Crack& crack,
    float currentTime) const
{
    if (!m_Settings.enablePropagation) {
        return 1.0f;  // Full length immediately
    }
    
    // Calculate age of crack
    float age = currentTime - crack.creationTime;
    
    // Propagate over propagation duration
    float propagation = std::clamp(age / m_Settings.propagationDuration, 0.0f, 1.0f);
    
    // Apply easing if enabled
    if (m_Settings.propagationEasing) {
        // Ease-out cubic for fast start, slow finish
        propagation = 1.0f - std::pow(1.0f - propagation, 3.0f);
    }
    
    return propagation;
}

void CrackRenderer::UpdateBuffers() {
    if (!m_Initialized || m_CachedVertices.empty()) {
        return;
    }

    // Update position buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO_Positions);
    glBufferData(GL_ARRAY_BUFFER,
                 m_CachedVertices.size() * sizeof(Vec3),
                 m_CachedVertices.data(),
                 GL_DYNAMIC_DRAW);

    // Update color buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO_Colors);
    glBufferData(GL_ARRAY_BUFFER,
                 m_CachedColors.size() * sizeof(Vec3),
                 m_CachedColors.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
