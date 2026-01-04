#include "VoxelGrid.h"
#include "GameObject.h"
#include "Mesh.h"
#include "Material.h"
#include <glad/glad.h>
#include <iostream>

VoxelGrid::VoxelGrid(int resolution)
    : m_Resolution(resolution)
    , m_GridCenter(0.0f)
    , m_GridMin(-50.0f)
    , m_GridMax(50.0f)
    , m_VoxelSize(0.0f)
    , m_GridExtent(50.0f)
    , m_VoxelAlbedoTexture(0)
    , m_VoxelNormalTexture(0)
    , m_VoxelizationFBO(0)
    , m_DummyVAO(0)
    , m_UseConservativeRasterization(false)
{
    m_VoxelSize = (m_GridMax.x - m_GridMin.x) / static_cast<float>(m_Resolution);
}

VoxelGrid::~VoxelGrid()
{
    Shutdown();
}

bool VoxelGrid::Initialize()
{
    std::cout << "[VoxelGrid] Initializing voxel grid (" << m_Resolution << "³)..." << std::endl;

    CreateVoxelTextures();
    CreateVoxelizationResources();

    // Load voxelization shader
    m_VoxelizeShader = std::make_unique<Shader>("shaders/voxelize.vert", 
                                                  "shaders/voxelize.frag",
                                                  "shaders/voxelize.geom");

    // Check for conservative rasterization support
    // GL_NV_conservative_raster or GL_INTEL_conservative_rasterization
    // For now, we'll use geometry shader-based conservative rasterization

    std::cout << "[VoxelGrid] Voxel grid initialized successfully!" << std::endl;
    return true;
}

void VoxelGrid::Shutdown()
{
    if (m_VoxelAlbedoTexture) glDeleteTextures(1, &m_VoxelAlbedoTexture);
    if (m_VoxelNormalTexture) glDeleteTextures(1, &m_VoxelNormalTexture);
    if (m_VoxelizationFBO) glDeleteFramebuffers(1, &m_VoxelizationFBO);
    if (m_DummyVAO) glDeleteVertexArrays(1, &m_DummyVAO);
}

void VoxelGrid::CreateVoxelTextures()
{
    // Create 3D texture for albedo + occlusion
    glGenTextures(1, &m_VoxelAlbedoTexture);
    glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, m_Resolution, m_Resolution, m_Resolution, 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    
    // Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_3D);

    // Create 3D texture for normal + emissive
    glGenTextures(1, &m_VoxelNormalTexture);
    glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, m_Resolution, m_Resolution, m_Resolution,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    
    glGenerateMipmap(GL_TEXTURE_3D);

    std::cout << "[VoxelGrid] Created 3D textures: " << m_Resolution << "³" << std::endl;
}

void VoxelGrid::CreateVoxelizationResources()
{
    // Create dummy VAO for voxelization
    glGenVertexArrays(1, &m_DummyVAO);

    // Create framebuffer (even though we won't use traditional rasterization)
    glGenFramebuffers(1, &m_VoxelizationFBO);
}

void VoxelGrid::Voxelize(const std::vector<GameObject*>& objects, Camera* camera)
{
    if (!m_VoxelizeShader) return;

    // Update grid bounds based on camera position
    UpdateGridBounds(camera);

    // Disable rasterization (we write directly to 3D texture via image store)
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    // Bind 3D textures as image units for writing
    glBindImageTexture(0, m_VoxelAlbedoTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glBindImageTexture(1, m_VoxelNormalTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);

    m_VoxelizeShader->Use();
    m_VoxelizeShader->SetVec3("u_GridMin", m_GridMin);
    m_VoxelizeShader->SetVec3("u_GridMax", m_GridMax);
    m_VoxelizeShader->SetInt("u_Resolution", m_Resolution);
    m_VoxelizeShader->SetFloat("u_VoxelSize", m_VoxelSize);

    // Orthographic projection matrices for voxelization (3 axes)
    glm::mat4 projX = glm::ortho(m_GridMin.z, m_GridMax.z, m_GridMin.y, m_GridMax.y, 
                                  m_GridMin.x, m_GridMax.x);
    glm::mat4 projY = glm::ortho(m_GridMin.x, m_GridMax.x, m_GridMin.z, m_GridMax.z,
                                  m_GridMin.y, m_GridMax.y);
    glm::mat4 projZ = glm::ortho(m_GridMin.x, m_GridMax.x, m_GridMin.y, m_GridMax.y,
                                  m_GridMin.z, m_GridMax.z);

    m_VoxelizeShader->SetMat4("u_ProjectionX", projX);
    m_VoxelizeShader->SetMat4("u_ProjectionY", projY);
    m_VoxelizeShader->SetMat4("u_ProjectionZ", projZ);

    // Render all objects
    for (GameObject* obj : objects) {
        if (!obj || !obj->IsActive()) continue;

        auto mesh = obj->GetComponent<Mesh>();
        auto material = obj->GetComponent<Material>();
        
        if (!mesh || !material) continue;

        // Set model matrix
        m_VoxelizeShader->SetMat4("u_Model", obj->GetTransform().GetWorldMatrix());

        // Bind material textures
        if (material->HasTexture()) {
            m_VoxelizeShader->SetInt("u_HasTexture", 1);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, material->GetTextureID());
        } else {
            m_VoxelizeShader->SetInt("u_HasTexture", 0);
            m_VoxelizeShader->SetVec3("u_AlbedoColor", material->GetDiffuse());
        }

        // Render mesh
        mesh->Draw();
    }

    // Re-enable rasterization
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Memory barrier to ensure writes are complete
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

void VoxelGrid::Clear()
{
    // Clear voxel textures to zero
    GLubyte clearColor[4] = {0, 0, 0, 0};
    
    glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture);
    glClearTexImage(m_VoxelAlbedoTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);
    
    glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture);
    glClearTexImage(m_VoxelNormalTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);
}

void VoxelGrid::GenerateMipmaps()
{
    glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture);
    glGenerateMipmap(GL_TEXTURE_3D);
    
    glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture);
    glGenerateMipmap(GL_TEXTURE_3D);
}

void VoxelGrid::SetGridBounds(const glm::vec3& min, const glm::vec3& max)
{
    m_GridMin = min;
    m_GridMax = max;
    m_GridCenter = (min + max) * 0.5f;
    m_VoxelSize = (max.x - min.x) / static_cast<float>(m_Resolution);
}

void VoxelGrid::SetGridCenter(const glm::vec3& center, float extent)
{
    m_GridCenter = center;
    m_GridExtent = extent;
    m_GridMin = center - glm::vec3(extent);
    m_GridMax = center + glm::vec3(extent);
    m_VoxelSize = (2.0f * extent) / static_cast<float>(m_Resolution);
}

void VoxelGrid::SetResolution(int resolution)
{
    if (resolution == m_Resolution) return;

    m_Resolution = resolution;
    m_VoxelSize = (m_GridMax.x - m_GridMin.x) / static_cast<float>(m_Resolution);

    // Recreate textures with new resolution
    if (m_VoxelAlbedoTexture) glDeleteTextures(1, &m_VoxelAlbedoTexture);
    if (m_VoxelNormalTexture) glDeleteTextures(1, &m_VoxelNormalTexture);

    CreateVoxelTextures();
}

void VoxelGrid::UpdateGridBounds(Camera* camera)
{
    // Center grid around camera position
    // This allows the voxel grid to follow the camera for large scenes
    if (camera) {
        SetGridCenter(camera->GetPosition(), m_GridExtent);
    }
}

void VoxelGrid::RenderDebug(Camera* camera, Shader* debugShader)
{
    if (!debugShader) return;

    // Render voxel grid as point cloud or wireframe cubes
    // This is a placeholder - full implementation would render visible voxels
    debugShader->Use();
    debugShader->SetMat4("u_ViewProjection", camera->GetProjectionMatrix() * camera->GetViewMatrix());
    debugShader->SetVec3("u_GridMin", m_GridMin);
    debugShader->SetVec3("u_GridMax", m_GridMax);
    debugShader->SetInt("u_Resolution", m_Resolution);

    // Bind voxel texture for sampling
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture);

    // Render points or instanced cubes
    // (Implementation depends on debug visualization approach)
}
