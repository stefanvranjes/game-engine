#pragma once

#include <memory>
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Camera.h"

class GameObject;

/**
 * @class VoxelGrid
 * @brief Manages 3D voxel grid for scene representation in GI
 * 
 * Voxelizes the scene geometry into a 3D texture for use in
 * Voxel Cone Tracing. Stores albedo, normal, and emissive data.
 */
class VoxelGrid {
public:
    VoxelGrid(int resolution = 128);
    ~VoxelGrid();

    // Initialization
    bool Initialize();
    void Shutdown();

    // Voxelization
    void Voxelize(const std::vector<GameObject*>& objects, Camera* camera);
    void Clear();
    void GenerateMipmaps();

    // Configuration
    void SetGridBounds(int cascadeIndex, const glm::vec3& min, const glm::vec3& max);
    void SetGridCenter(const glm::vec3& center);
    void SetResolution(int resolution);

    // Getters
    unsigned int GetVoxelAlbedoTexture(int cascadeIndex) const { return m_VoxelAlbedoTexture[cascadeIndex]; }
    unsigned int GetVoxelNormalTexture(int cascadeIndex) const { return m_VoxelNormalTexture[cascadeIndex]; }
    int GetResolution() const { return m_Resolution; }
    
    struct CascadeData {
        glm::vec3 min;
        glm::vec3 max;
        glm::vec3 center;
        float voxelSize;
        float extent; // Half-size
    };
    
    const CascadeData& GetCascade(int index) const { return m_Cascades[index]; }
    static const int MAX_CASCADES = 3;

    // Debug
    void RenderDebug(Camera* camera, Shader* debugShader);

private:
    void CreateVoxelTextures();
    void CreateVoxelizationResources();
    void UpdateCascadeBounds(Camera* camera);

    // Grid properties
    int m_Resolution;
    
    // Cascade data
    std::array<CascadeData, MAX_CASCADES> m_Cascades;

    // 3D Textures Arrays
    unsigned int m_VoxelAlbedoTexture[MAX_CASCADES];   
    unsigned int m_VoxelNormalTexture[MAX_CASCADES];

    // Voxelization shaders
    std::unique_ptr<Shader> m_VoxelizeShader;
    
    // Voxelization state
    unsigned int m_VoxelizationFBO;
    unsigned int m_DummyVAO;  // For voxelization without rasterization

    // Conservative rasterization support
    bool m_UseConservativeRasterization;
};
