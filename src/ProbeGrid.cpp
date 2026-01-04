#include "ProbeGrid.h"
#include "GameObject.h"
#include "Mesh.h"
#include "Camera.h"
#include "Shader.h"
#include <glad/glad.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

ProbeGrid::ProbeGrid(const glm::vec3& min, const glm::vec3& max, const glm::ivec3& resolution)
    : m_GridMin(min)
    , m_GridMax(max)
    , m_Resolution(resolution)
    , m_ProbeSSBO(0)
    , m_DebugVAO(0)
    , m_DebugVBO(0)
{
    m_CellSize = (m_GridMax - m_GridMin) / glm::vec3(m_Resolution);
    m_IrradianceTextures[0] = 0;
    m_IrradianceTextures[1] = 0;
}

ProbeGrid::~ProbeGrid()
{
    Shutdown();
}

bool ProbeGrid::Initialize()
{
    std::cout << "[ProbeGrid] Initializing probe grid (" << m_Resolution.x << "x" 
              << m_Resolution.y << "x" << m_Resolution.z << ")..." << std::endl;

    // Create SSBO for probe data
    glGenBuffers(1, &m_ProbeSSBO);

    // Create debug visualization resources
    glGenVertexArrays(1, &m_DebugVAO);
    glGenBuffers(1, &m_DebugVBO);

    // Create 3D Textures
    CreateIrradianceTextures();

    std::cout << "[ProbeGrid] Probe grid initialized successfully!" << std::endl;
    return true;
}

void ProbeGrid::Shutdown()
{
    if (m_ProbeSSBO) glDeleteBuffers(1, &m_ProbeSSBO);
    if (m_DebugVAO) glDeleteVertexArrays(1, &m_DebugVAO);
    if (m_DebugVBO) glDeleteBuffers(1, &m_DebugVBO);
    
    if (m_IrradianceTextures[0]) glDeleteTextures(2, m_IrradianceTextures);
    m_IrradianceTextures[0] = 0;
}

void ProbeGrid::GenerateProbes()
{
    m_Probes.clear();

    for (int z = 0; z < m_Resolution.z; z++) {
        for (int y = 0; y < m_Resolution.y; y++) {
            for (int x = 0; x < m_Resolution.x; x++) {
                glm::vec3 position = m_GridMin + glm::vec3(x, y, z) * m_CellSize + m_CellSize * 0.5f;

                // Validate probe position
                if (IsValidProbePosition(position)) {
                    LightProbeData probe;
                    probe.position = position;
                    probe.flags = 0;
                    probe.radius = glm::length(m_CellSize) * 0.5f;
                    m_Probes.push_back(probe);
                }
            }
        }
    }

    std::cout << "[ProbeGrid] Generated " << m_Probes.size() << " probes" << std::endl;
    
    // Update textures if they exist
    if (m_IrradianceTextures[0]) UpdateIrradianceTextures();
}

void ProbeGrid::GenerateAdaptiveProbes(float varianceThreshold, int maxIterations)
{
    // Start with coarse grid
    GenerateProbes();

    // Adaptive subdivision based on lighting variance
    // NOTE: Adaptive probes break 3D texture mapping unless using octree structure.
    // For Irradiance Volume mode, we might want to warn or skip.
    // We proceed but texture update will only capture base grid.
    
    for (int iteration = 0; iteration < maxIterations; iteration++) {
        std::vector<LightProbeData> newProbes;

        for (size_t i = 0; i < m_Probes.size(); i++) {
            float variance = ComputeLightingVariance(static_cast<int>(i));

            if (variance > varianceThreshold) {
                AddSubdivisionProbes(static_cast<int>(i), newProbes);
            }
        }

        if (newProbes.empty()) break;

        m_Probes.insert(m_Probes.end(), newProbes.begin(), newProbes.end());
        std::cout << "[ProbeGrid] Iteration " << iteration << ": Added " 
                  << newProbes.size() << " probes (total: " << m_Probes.size() << ")" << std::endl;
    }
}

void ProbeGrid::AddProbe(const glm::vec3& position)
{
    LightProbeData probe;
    probe.position = position;
    probe.flags = 0;
    probe.radius = glm::length(m_CellSize) * 0.5f;
    m_Probes.push_back(probe);
}

void ProbeGrid::RemoveProbe(int index)
{
    if (index >= 0 && index < static_cast<int>(m_Probes.size())) {
        m_Probes.erase(m_Probes.begin() + index);
    }
}

void ProbeGrid::ClearProbes()
{
    m_Probes.clear();
}

glm::vec3 ProbeGrid::SampleIrradiance(const glm::vec3& position, const glm::vec3& normal) const
{
    if (m_Probes.empty()) return glm::vec3(0.0f);

    int indices[8];
    float weights[8];
    GetInterpolationWeights(position, indices, weights);

    glm::vec3 irradiance(0.0f);

    for (int i = 0; i < 8; i++) {
        if (indices[i] >= 0 && weights[i] > 0.0f) {
            const LightProbeData& probe = m_Probes[indices[i]];
            glm::vec3 probeIrradiance = EvaluateSH(probe.shCoefficients, normal);
            irradiance += probeIrradiance * weights[i];
        }
    }

    return irradiance;
}

void ProbeGrid::GetInterpolationWeights(const glm::vec3& position, int indices[8], float weights[8]) const
{
    // Initialize
    for (int i = 0; i < 8; i++) {
        indices[i] = -1;
        weights[i] = 0.0f;
    }

    // Convert to grid coordinates
    glm::vec3 gridPos = (position - m_GridMin) / (m_GridMax - m_GridMin);
    gridPos = glm::clamp(gridPos, glm::vec3(0.0f), glm::vec3(1.0f));

    glm::vec3 gridCoord = gridPos * glm::vec3(m_Resolution - glm::ivec3(1));
    glm::ivec3 baseCoord = glm::ivec3(glm::floor(gridCoord));
    glm::vec3 frac = glm::fract(gridCoord);

    // Sample 8 surrounding probes
    int idx = 0;
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                glm::ivec3 coord = baseCoord + glm::ivec3(x, y, z);
                coord = glm::clamp(coord, glm::ivec3(0), m_Resolution - glm::ivec3(1));

                int probeIndex = GetProbeIndex(coord);
                if (probeIndex >= 0) {
                    indices[idx] = probeIndex;

                    // Trilinear weight
                    glm::vec3 weight3D = glm::mix(glm::vec3(1.0f) - frac, frac, glm::vec3(x, y, z));
                    weights[idx] = weight3D.x * weight3D.y * weight3D.z;
                }
                idx++;
            }
        }
    }
}

void ProbeGrid::SetGridBounds(const glm::vec3& min, const glm::vec3& max)
{
    m_GridMin = min;
    m_GridMax = max;
    m_CellSize = (m_GridMax - m_GridMin) / glm::vec3(m_Resolution);
    
    // Recreate textures if resolution changed? (Resolution is const currently, bounds change is fine)
}

bool ProbeGrid::SaveToFile(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ProbeGrid] Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write header
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write grid properties
    file.write(reinterpret_cast<const char*>(&m_GridMin), sizeof(m_GridMin));
    file.write(reinterpret_cast<const char*>(&m_GridMax), sizeof(m_GridMax));
    file.write(reinterpret_cast<const char*>(&m_Resolution), sizeof(m_Resolution));

    // Write probe count
    uint32_t probeCount = static_cast<uint32_t>(m_Probes.size());
    file.write(reinterpret_cast<const char*>(&probeCount), sizeof(probeCount));

    // Write probe data
    file.write(reinterpret_cast<const char*>(m_Probes.data()), 
               probeCount * sizeof(LightProbeData));

    file.close();
    std::cout << "[ProbeGrid] Saved " << probeCount << " probes to " << filename << std::endl;
    return true;
}

bool ProbeGrid::LoadFromFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ProbeGrid] Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // Read header
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != 1) {
        std::cerr << "[ProbeGrid] Unsupported file version: " << version << std::endl;
        return false;
    }

    // Read grid properties
    file.read(reinterpret_cast<char*>(&m_GridMin), sizeof(m_GridMin));
    file.read(reinterpret_cast<char*>(&m_GridMax), sizeof(m_GridMax));
    file.read(reinterpret_cast<char*>(&m_Resolution), sizeof(m_Resolution));

    m_CellSize = (m_GridMax - m_GridMin) / glm::vec3(m_Resolution);

    // Read probe count
    uint32_t probeCount;
    file.read(reinterpret_cast<char*>(&probeCount), sizeof(probeCount));

    // Read probe data
    m_Probes.resize(probeCount);
    file.read(reinterpret_cast<char*>(m_Probes.data()), 
              probeCount * sizeof(LightProbeData));

    file.close();
    std::cout << "[ProbeGrid] Loaded " << probeCount << " probes from " << filename << std::endl;
    
    // Update GPU
    UploadToGPU();
    UpdateIrradianceTextures();
    
    return true;
}

void ProbeGrid::UploadToGPU()
{
    if (m_Probes.empty()) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ProbeSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
                 m_Probes.size() * sizeof(LightProbeData),
                 m_Probes.data(), 
                 GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Also update 3D textures
    UpdateIrradianceTextures();

    std::cout << "[ProbeGrid] Uploaded " << m_Probes.size() << " probes to GPU" << std::endl;
}

// === Irradiance Volume Implementation ===

void ProbeGrid::CreateIrradianceTextures() {
    if (m_IrradianceTextures[0]) glDeleteTextures(2, m_IrradianceTextures);
    
    glGenTextures(2, m_IrradianceTextures);
    
    // Texture A: RGBA16F (L0.r, L0.g, L0.b, L1_x_Luma)
    glBindTexture(GL_TEXTURE_3D, m_IrradianceTextures[0]);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_Resolution.x, m_Resolution.y, m_Resolution.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Texture B: RG16F (L1_y_Luma, L1_z_Luma)
    glBindTexture(GL_TEXTURE_3D, m_IrradianceTextures[1]);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RG16F, m_Resolution.x, m_Resolution.y, m_Resolution.z, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_3D, 0);
}

void ProbeGrid::UpdateIrradianceTextures() {
    if (!m_IrradianceTextures[0] || m_Probes.empty()) return;
    
    int numProbes = m_Resolution.x * m_Resolution.y * m_Resolution.z;
    if (m_Probes.size() != numProbes) {
        return;
    }
    
    // Texture A: L0_RGB + L1_X_Luma
    std::vector<glm::vec4> textureA(numProbes);
    // Texture B: L1_Y_Luma + L1_Z_Luma
    std::vector<glm::vec2> textureB(numProbes);
    
    // Luminance weights (Rec. 709)
    const glm::vec3 lumaWeights(0.2126f, 0.7152f, 0.0722f);

    for (int i = 0; i < numProbes; ++i) {
        const float* sh = m_Probes[i].shCoefficients;
        
        // L0 (Constant) - Indices: R[0], G[9], B[18]
        float l0_r = sh[0];
        float l0_g = sh[9];
        float l0_b = sh[18];

        // L1 Bands - RGB
        // Y (Index 1): R[1], G[10], B[19]
        glm::vec3 l1_y_rgb(sh[1], sh[10], sh[19]);
        
        // Z (Index 2): R[2], G[11], B[20]
        glm::vec3 l1_z_rgb(sh[2], sh[11], sh[20]);
        
        // X (Index 3): R[3], G[12], B[21]
        glm::vec3 l1_x_rgb(sh[3], sh[12], sh[21]);
        
        // Compute Luminance for L1 bands
        float l1_y_luma = glm::dot(l1_y_rgb, lumaWeights);
        float l1_z_luma = glm::dot(l1_z_rgb, lumaWeights);
        float l1_x_luma = glm::dot(l1_x_rgb, lumaWeights);
        
        // Texture A: R=L0.r, G=L0.g, B=L0.b, A=L1_x_luma
        textureA[i] = glm::vec4(l0_r, l0_g, l0_b, l1_x_luma);
        
        // Texture B: R=L1_y_luma, G=L1_z_luma
        textureB[i] = glm::vec2(l1_y_luma, l1_z_luma);
    }
    
    glBindTexture(GL_TEXTURE_3D, m_IrradianceTextures[0]);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_Resolution.x, m_Resolution.y, m_Resolution.z, GL_RGBA, GL_FLOAT, textureA.data());
    
    glBindTexture(GL_TEXTURE_3D, m_IrradianceTextures[1]);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_Resolution.x, m_Resolution.y, m_Resolution.z, GL_RG, GL_FLOAT, textureB.data());
    
    glBindTexture(GL_TEXTURE_3D, 0);
}

void ProbeGrid::BindIrradianceTextures(int startUnit) {
    if (!m_IrradianceTextures[0]) return;
    for(int i=0; i<2; ++i) {
        glActiveTexture(GL_TEXTURE0 + startUnit + i);
        glBindTexture(GL_TEXTURE_3D, m_IrradianceTextures[i]);
    }
}

// ... Debug Render & Helpers ...

void ProbeGrid::RenderDebug(Camera* camera, Shader* shader)
{
    if (!shader || m_Probes.empty()) return;

    // Render probes as spheres or points
    shader->Use();
    shader->SetMat4("u_ViewProjection", camera->GetProjectionMatrix() * camera->GetViewMatrix());

    // Simple point rendering for now
    std::vector<glm::vec3> positions;
    for (const auto& probe : m_Probes) {
        positions.push_back(probe.position);
    }

    glBindVertexArray(m_DebugVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_DebugVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), 
                 positions.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glPointSize(5.0f);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));

    glBindVertexArray(0);
}

// Private helper methods

bool ProbeGrid::IsValidProbePosition(const glm::vec3& position) const
{
    // TODO: Check if position is inside geometry using raycasting
    // For now, accept all positions
    return true;
}

int ProbeGrid::GetProbeIndex(const glm::ivec3& gridCoord) const
{
    // Simple linear indexing
    int index = gridCoord.x + gridCoord.y * m_Resolution.x + 
                gridCoord.z * m_Resolution.x * m_Resolution.y;

    if (index >= 0 && index < static_cast<int>(m_Probes.size())) {
        return index;
    }
    return -1;
}

glm::ivec3 ProbeGrid::WorldToGrid(const glm::vec3& worldPos) const
{
    glm::vec3 gridPos = (worldPos - m_GridMin) / m_CellSize;
    return glm::ivec3(glm::floor(gridPos));
}

glm::vec3 ProbeGrid::GridToWorld(const glm::ivec3& gridCoord) const
{
    return m_GridMin + glm::vec3(gridCoord) * m_CellSize + m_CellSize * 0.5f;
}

float ProbeGrid::ComputeLightingVariance(int probeIndex) const
{
    // TODO: Compute variance with neighboring probes
    // For now, return 0 (no subdivision)
    return 0.0f;
}

void ProbeGrid::AddSubdivisionProbes(int probeIndex, std::vector<LightProbeData>& newProbes)
{
    // TODO: Add probes between this probe and its neighbors
    // For now, do nothing
}

glm::vec3 ProbeGrid::EvaluateSH(const float shCoeffs[9], const glm::vec3& normal) const
{
    // Evaluate spherical harmonics (L0 + L1 bands)
    // L0 band (constant)
    glm::vec3 result(shCoeffs[0], shCoeffs[9], shCoeffs[18]);
    result *= 0.282095f;

    // L1 band (linear)
    glm::vec3 l1_y(shCoeffs[1], shCoeffs[10], shCoeffs[19]);
    glm::vec3 l1_z(shCoeffs[2], shCoeffs[11], shCoeffs[20]);
    glm::vec3 l1_x(shCoeffs[3], shCoeffs[12], shCoeffs[21]);

    result += l1_y * (0.488603f * normal.y);
    result += l1_z * (0.488603f * normal.z);
    result += l1_x * (0.488603f * normal.x);

    return glm::max(result, glm::vec3(0.0f));
}
