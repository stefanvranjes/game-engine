#include "VoxelGrid.h"
#include "GameObject.h"
#include "Mesh.h"
#include "Material.h"
#include <glad/glad.h>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

VoxelGrid::VoxelGrid(int resolution)
    : m_Resolution(resolution)
    , m_VoxelizationFBO(0)
    , m_DummyVAO(0)
    , m_UseConservativeRasterization(false)
{
    for(int i=0; i<MAX_CASCADES; ++i) {
        m_VoxelAlbedoTexture[i] = 0;
        m_VoxelNormalTexture[i] = 0;
        m_Cascades[i] = { glm::vec3(0), glm::vec3(0), glm::vec3(0), 0.0f, 0.0f };
    }
    
    // Initialize defaults: Cascade 0 base, others scaled by 4
    float baseExtent = 20.0f; 
    for(int i=0; i<MAX_CASCADES; ++i) {
        float extent = baseExtent * std::pow(4.0f, (float)i);
        m_Cascades[i].extent = extent;
        // Don't set min/max yet, wait for update
    }
}

VoxelGrid::~VoxelGrid()
{
    Shutdown();
}

bool VoxelGrid::Initialize()
{
    std::cout << "[VoxelGrid] Initializing voxel grid (" << m_Resolution << "³). Cascades: " << MAX_CASCADES << "..." << std::endl;

    CreateVoxelTextures();
    CreateVoxelizationResources();

    // Load voxelization shader
    m_VoxelizeShader = std::make_unique<Shader>();
    m_VoxelizeShader->LoadFromFiles("shaders/voxelize.vert", 
                                       "shaders/voxelize.frag",
                                       "shaders/voxelize.geom");

    std::cout << "[VoxelGrid] Voxel grid initialized successfully!" << std::endl;
    return true;
}

void VoxelGrid::Shutdown()
{
    for(int i=0; i<MAX_CASCADES; ++i) {
        if (m_VoxelAlbedoTexture[i]) glDeleteTextures(1, &m_VoxelAlbedoTexture[i]);
        if (m_VoxelNormalTexture[i]) glDeleteTextures(1, &m_VoxelNormalTexture[i]);
    }
    if (m_VoxelizationFBO) glDeleteFramebuffers(1, &m_VoxelizationFBO);
    if (m_DummyVAO) glDeleteVertexArrays(1, &m_DummyVAO);
}

void VoxelGrid::CreateVoxelTextures()
{
    for(int i=0; i<MAX_CASCADES; ++i) {
        // Create 3D texture for albedo + occlusion
        glGenTextures(1, &m_VoxelAlbedoTexture[i]);
        glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture[i]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, m_Resolution, m_Resolution, m_Resolution, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
        
        glGenerateMipmap(GL_TEXTURE_3D);

        // Create 3D texture for normal + emissive
        glGenTextures(1, &m_VoxelNormalTexture[i]);
        glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture[i]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, m_Resolution, m_Resolution, m_Resolution,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
        
        glGenerateMipmap(GL_TEXTURE_3D);
    }
    
    std::cout << "[VoxelGrid] Created 3D textures for " << MAX_CASCADES << " cascades" << std::endl;
}

void VoxelGrid::CreateVoxelizationResources()
{
    glGenVertexArrays(1, &m_DummyVAO);
    glGenFramebuffers(1, &m_VoxelizationFBO);
}

#include "Profiler.h"

void VoxelGrid::Voxelize(const std::vector<GameObject*>& objects, Camera* camera)
{
    PROFILE_SCOPE("VoxelGrid::Voxelize");
    
    if (!m_VoxelizeShader) return;

    // Update grid bounds based on camera position
    UpdateCascadeBounds(camera);

    // Disable rasterization (write via image store)
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    m_VoxelizeShader->Use();
    m_VoxelizeShader->SetInt("u_Resolution", m_Resolution);

    // Render loop for each cascade
    for(int i=0; i<MAX_CASCADES; ++i) {
        const auto& cascade = m_Cascades[i];

        // Bind textures for this cascade
        glBindImageTexture(0, m_VoxelAlbedoTexture[i], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glBindImageTexture(1, m_VoxelNormalTexture[i], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);

        m_VoxelizeShader->SetVec3("u_GridMin", cascade.min);
        m_VoxelizeShader->SetVec3("u_GridMax", cascade.max);
        m_VoxelizeShader->SetFloat("u_VoxelSize", cascade.voxelSize);

        // Orthographic matrices
        glm::mat4 projX = glm::ortho(cascade.min.z, cascade.max.z, cascade.min.y, cascade.max.y, 
                                      cascade.min.x, cascade.max.x);
        glm::mat4 projY = glm::ortho(cascade.min.x, cascade.max.x, cascade.min.z, cascade.max.z,
                                      cascade.min.y, cascade.max.y);
        glm::mat4 projZ = glm::ortho(cascade.min.x, cascade.max.x, cascade.min.y, cascade.max.y,
                                      cascade.min.z, cascade.max.z);

        m_VoxelizeShader->SetMat4("u_ProjectionX", &projX[0][0]);
        m_VoxelizeShader->SetMat4("u_ProjectionY", &projY[0][0]);
        m_VoxelizeShader->SetMat4("u_ProjectionZ", &projZ[0][0]);

        // Render objects
        // Optimization: Cull objects outside this cascade
        for (GameObject* obj : objects) {
            if (!obj || !obj->IsActive()) continue;

            // TODO: AABB check against cascade bounds would go here

            auto mesh = obj->GetMesh();
            auto material = obj->GetMaterial();
            
            if (!mesh || !material) continue;

            m_VoxelizeShader->SetMat4("u_Model", obj->GetTransform().GetModelMatrix().m);

            if (material->GetTexture()) {
                m_VoxelizeShader->SetInt("u_HasTexture", 1);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, material->GetTexture()->GetID());
            } else {
                m_VoxelizeShader->SetInt("u_HasTexture", 0);
                m_VoxelizeShader->SetVec3("u_AlbedoColor", material->GetDiffuse());
            }

            mesh->Draw();
        }
        
        // Memory barrier between cascades? Not strictly needed unless overlapping writes (which shouldn't happen per texture)
    }

    // Restore state
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

void VoxelGrid::Clear()
{
    GLubyte clearColor[4] = {0, 0, 0, 0};
    for(int i=0; i<MAX_CASCADES; ++i) {
        glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture[i]);
        glClearTexImage(m_VoxelAlbedoTexture[i], 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);
        
        glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture[i]);
        glClearTexImage(m_VoxelNormalTexture[i], 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);
    }
}

void VoxelGrid::GenerateMipmaps()
{
    for(int i=0; i<MAX_CASCADES; ++i) {
        glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture[i]);
        glGenerateMipmap(GL_TEXTURE_3D);
        
        glBindTexture(GL_TEXTURE_3D, m_VoxelNormalTexture[i]);
        glGenerateMipmap(GL_TEXTURE_3D);
    }
}

void VoxelGrid::SetGridBounds(int cascadeIndex, const glm::vec3& min, const glm::vec3& max)
{
    if(cascadeIndex < 0 || cascadeIndex >= MAX_CASCADES) return;
    m_Cascades[cascadeIndex].min = min;
    m_Cascades[cascadeIndex].max = max;
    m_Cascades[cascadeIndex].center = (min + max) * 0.5f;
    m_Cascades[cascadeIndex].voxelSize = (max.x - min.x) / static_cast<float>(m_Resolution);
}

void VoxelGrid::SetGridCenter(const glm::vec3& center)
{
    // Sets center for all cascades (extent remains fixed)
    for(int i=0; i<MAX_CASCADES; ++i) {
        float extent = m_Cascades[i].extent;
        m_Cascades[i].center = center;
        m_Cascades[i].min = center - glm::vec3(extent);
        m_Cascades[i].max = center + glm::vec3(extent);
        m_Cascades[i].voxelSize = (2.0f * extent) / static_cast<float>(m_Resolution);
    }
}

void VoxelGrid::SetResolution(int resolution)
{
    if (resolution == m_Resolution) return;
    m_Resolution = resolution;
    
    // Cleanup old
    Shutdown();
    // Recreate
    CreateVoxelTextures();
    CreateVoxelizationResources();
    
    // Recalc voxel sizes
    SetGridCenter(m_Cascades[0].center); // Helper to refresh bounds
}

void VoxelGrid::UpdateCascadeBounds(Camera* camera)
{
    if (!camera) return;

    Vec3 p = camera->GetPosition();
    glm::vec3 camPos(p.x, p.y, p.z);

    for(int i=0; i<MAX_CASCADES; ++i) {
        float extent = m_Cascades[i].extent;
        float voxelSize = (2.0f * extent) / static_cast<float>(m_Resolution);
        
        // Snap center to voxel grid
        glm::vec3 snappedCenter = glm::floor(camPos / voxelSize) * voxelSize;
        
        m_Cascades[i].center = snappedCenter;
        m_Cascades[i].min = snappedCenter - glm::vec3(extent);
        m_Cascades[i].max = snappedCenter + glm::vec3(extent);
        m_Cascades[i].voxelSize = voxelSize;
    }
}

void VoxelGrid::RenderDebug(Camera* camera, Shader* debugShader)
{
    if (!debugShader) return;

    debugShader->Use();
    debugShader->SetMat4("u_ViewProjection", (camera->GetProjectionMatrix() * camera->GetViewMatrix()).m);
    
    // Just render Cascade 0 for now or cycle?
    // Let's render Cascade 0
    // Or add uniforms to debug shader to select cascade?
    // Using Cascade 0 for debug
    
    debugShader->SetVec3("u_GridMin", m_Cascades[0].min);
    debugShader->SetVec3("u_GridMax", m_Cascades[0].max);
    debugShader->SetInt("u_Resolution", m_Resolution);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_VoxelAlbedoTexture[0]);

    // Draw
    // Placeholder - user didn't ask for full debug overhaul yet
}
