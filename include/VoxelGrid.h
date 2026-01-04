#pragma once

#include <memory>
#include <vector>
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

    // Getters
    unsigned int GetVoxelAlbedoTexture() const { return m_VoxelAlbedoTexture; }
    unsigned int GetVoxelNormalTexture() const { return m_VoxelNormalTexture; }
    int GetResolution() const { return m_Resolution; }
    glm::vec3 GetGridCenter() const { return m_GridCenter; }
    glm::vec3 GetGridMin() const { return m_GridMin; }
    glm::vec3 GetGridMax() const { return m_GridMax; }
    float GetVoxelSize() const { return m_VoxelSize; }

    // Configuration
    void SetGridBounds(const glm::vec3& min, const glm::vec3& max);
    void SetGridCenter(const glm::vec3& center, float extent);
    void SetResolution(int resolution);

    // Debug
    void RenderDebug(Camera* camera, Shader* debugShader);

private:
    void CreateVoxelTextures();
    void CreateVoxelizationResources();
    void UpdateGridBounds(Camera* camera);

    // Grid properties
    int m_Resolution;
    glm::vec3 m_GridCenter;
    glm::vec3 m_GridMin;
    glm::vec3 m_GridMax;
    float m_VoxelSize;
    float m_GridExtent;  // Half-size of the grid

    // 3D Textures
    unsigned int m_VoxelAlbedoTexture;   // RGBA8: RGB=Albedo, A=Occlusion
    unsigned int m_VoxelNormalTexture;   // RGBA8: RGB=Normal, A=Emissive

    // Voxelization shaders
    std::unique_ptr<Shader> m_VoxelizeShader;
    
    // Voxelization state
    unsigned int m_VoxelizationFBO;
    unsigned int m_DummyVAO;  // For voxelization without rasterization

    // Conservative rasterization support
    bool m_UseConservativeRasterization;
};
