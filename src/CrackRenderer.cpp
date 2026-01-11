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

    // Generate geometry for all cracks
    for (const auto& crack : cracks) {
        GenerateCrackGeometry(crack, vertices, tetrahedra, m_CachedVertices, m_CachedColors, currentTime);
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
    
    // Apply animation effects
    float animIntensity = CalculateAnimatedIntensity(crack, currentTime);
    opacity *= animIntensity;
    
    // Apply growth fade-in
    float growthFactor = CalculateGrowthFactor(crack, currentTime);
    opacity *= growthFactor;

    // Store color with opacity in RGB (will use red channel as alpha in shader)
    Vec3 colorWithAlpha = color * opacity;

    // Add line segment
    outVertices.push_back(p0);
    outVertices.push_back(p1);
    outColors.push_back(colorWithAlpha);
    outColors.push_back(colorWithAlpha);
}

void CrackRenderer::SetRenderSettings(const RenderSettings& settings) {
    m_Settings = settings;
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
