#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

class GameObject;
class Light;

/**
 * @struct LightProbeData
 * @brief Data stored per light probe
 */
struct LightProbeData {
    glm::vec3 position;
    float shCoefficients[27];  // 9 SH bands × RGB (L0, L1, L2)
    uint32_t flags;
    float radius;
    
    LightProbeData() 
        : position(0.0f)
        , flags(0)
        , radius(5.0f)
    {
        for (int i = 0; i < 27; i++) {
            shCoefficients[i] = 0.0f;
        }
    }
};

/**
 * @class ProbeGrid
 * @brief Manages light probe placement, baking, and runtime sampling
 * 
 * Organizes probes in a 3D grid for efficient lookup and interpolation.
 * Supports both automatic grid generation and manual probe placement.
 */
class ProbeGrid {
public:
    ProbeGrid(const glm::vec3& min, const glm::vec3& max, const glm::ivec3& resolution);
    ~ProbeGrid();

    // Initialization
    bool Initialize();
    void Shutdown();

    // Probe placement
    void GenerateProbes();
    void GenerateAdaptiveProbes(float varianceThreshold = 0.1f, int maxIterations = 3);
    void AddProbe(const glm::vec3& position);
    void RemoveProbe(int index);
    void ClearProbes();

    // Probe access
    int GetProbeCount() const { return static_cast<int>(m_Probes.size()); }
    const LightProbeData& GetProbe(int index) const { return m_Probes[index]; }
    LightProbeData& GetProbe(int index) { return m_Probes[index]; }
    const std::vector<LightProbeData>& GetProbes() const { return m_Probes; }

    // Runtime sampling
    glm::vec3 SampleIrradiance(const glm::vec3& position, const glm::vec3& normal) const;
    void GetInterpolationWeights(const glm::vec3& position, int indices[8], float weights[8]) const;

    // Grid properties
    glm::vec3 GetGridMin() const { return m_GridMin; }
    glm::vec3 GetGridMax() const { return m_GridMax; }
    glm::ivec3 GetResolution() const { return m_Resolution; }
    void SetGridBounds(const glm::vec3& min, const glm::vec3& max);

    // Serialization
    bool SaveToFile(const std::string& filename) const;
    bool LoadFromFile(const std::string& filename);

    // GPU upload
    void UploadToGPU();
    unsigned int GetProbeSSBO() const { return m_ProbeSSBO; }

    // Debug visualization
    void RenderDebug(class Camera* camera, class Shader* shader);

private:
    // Helper methods
    bool IsValidProbePosition(const glm::vec3& position) const;
    int GetProbeIndex(const glm::ivec3& gridCoord) const;
    glm::ivec3 WorldToGrid(const glm::vec3& worldPos) const;
    glm::vec3 GridToWorld(const glm::ivec3& gridCoord) const;
    float ComputeLightingVariance(int probeIndex) const;
    void AddSubdivisionProbes(int probeIndex, std::vector<LightProbeData>& newProbes);

    // Spherical harmonics evaluation
    glm::vec3 EvaluateSH(const float shCoeffs[9], const glm::vec3& normal) const;

    // Grid properties
    glm::vec3 m_GridMin;
    glm::vec3 m_GridMax;
    glm::ivec3 m_Resolution;
    glm::vec3 m_CellSize;

    // Probe data
    std::vector<LightProbeData> m_Probes;

    // GPU resources
    unsigned int m_ProbeSSBO;  // Shader Storage Buffer Object

    // Debug visualization
    unsigned int m_DebugVAO;
    unsigned int m_DebugVBO;
};
