#include "GlobalIllumination.h"
#include "VoxelGrid.h"
#include "LightPropagationVolume.h"
#include "ProbeGrid.h"
#include "GameObject.h"
#include <glad/glad.h>
#include <iostream>

GlobalIllumination::GlobalIllumination()
    : m_Technique(Technique::VCT)
    , m_Quality(Quality::Medium)
    , m_Enabled(false)
    , m_Intensity(1.0f)
    , m_ShowVoxels(false)
    , m_ScreenWidth(0)
    , m_ScreenHeight(0)
    , m_GITexture(0)
    , m_GIFramebuffer(0)
    , m_SSGITexture(0)
    , m_SSGIBlurTexture(0)
    , m_CurrentTemporalIndex(0)
    , m_VoxelResolution(128)
    , m_NumDiffuseCones(5)
    , m_UseTemporalFiltering(true)
    , m_TemporalBlendFactor(0.9f)
    , m_DebugVAO(0)
    , m_DebugVBO(0)
    , m_ProbeBlendWeight(0.5f)
{
    m_TemporalTexture[0] = 0;
    m_TemporalTexture[1] = 0;
}

GlobalIllumination::~GlobalIllumination()
{
    Shutdown();
}

bool GlobalIllumination::Initialize(int screenWidth, int screenHeight)
{
    m_ScreenWidth = screenWidth;
    m_ScreenHeight = screenHeight;

    std::cout << "[GI] Initializing Global Illumination system..." << std::endl;

    // Create GI output texture
    CreateGITexture(screenWidth, screenHeight);

    // Initialize VoxelGrid
    m_VoxelGrid = std::make_unique<VoxelGrid>(m_VoxelResolution);
    if (!m_VoxelGrid->Initialize()) {
        std::cerr << "[GI] Failed to initialize VoxelGrid!" << std::endl;
        return false;
    }

    // Initialize LPV
    m_LPV = std::make_unique<LightPropagationVolume>(32);
    if (!m_LPV->Initialize()) {
        std::cerr << "[GI] Failed to initialize LPV!" << std::endl;
        return false;
    }

    // Initialize ProbeGrid
    glm::vec3 gridMin(-50.0f, 0.0f, -50.0f);
    glm::vec3 gridMax(50.0f, 50.0f, 50.0f);
    glm::ivec3 gridRes(8, 8, 8);
    m_ProbeGrid = std::make_unique<ProbeGrid>(gridMin, gridMax, gridRes);
    if (!m_ProbeGrid->Initialize()) {
        std::cerr << "[GI] Failed to initialize ProbeGrid!" << std::endl;
        return false;
    }

    // Create SSGI resources
    CreateSSGIResources(screenWidth, screenHeight);

    // Load shaders
    m_ConeTraceShader = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/cone_trace.frag");
    m_SSGIShader = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/ssgi.frag");
    m_SSGIBlurShader = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/ssgi_blur.frag");
    m_TemporalShader = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/gi_temporal.frag");
    m_VoxelDebugShader = std::make_unique<Shader>("shaders/voxel_debug.vert", "shaders/voxel_debug.frag");

    // Create debug visualization resources
    glGenVertexArrays(1, &m_DebugVAO);
    glGenBuffers(1, &m_DebugVBO);

    UpdateQualitySettings();

    std::cout << "[GI] Global Illumination initialized successfully!" << std::endl;
    return true;
}

void GlobalIllumination::Shutdown()
{
    if (m_GITexture) glDeleteTextures(1, &m_GITexture);
    if (m_GIFramebuffer) glDeleteFramebuffers(1, &m_GIFramebuffer);
    if (m_SSGITexture) glDeleteTextures(1, &m_SSGITexture);
    if (m_SSGIBlurTexture) glDeleteTextures(1, &m_SSGIBlurTexture);
    if (m_TemporalTexture[0]) glDeleteTextures(1, &m_TemporalTexture[0]);
    if (m_TemporalTexture[1]) glDeleteTextures(1, &m_TemporalTexture[1]);
    if (m_DebugVAO) glDeleteVertexArrays(1, &m_DebugVAO);
    if (m_DebugVBO) glDeleteBuffers(1, &m_DebugVBO);

    if (m_VoxelGrid) m_VoxelGrid->Shutdown();
    if (m_LPV) m_LPV->Shutdown();
}

void GlobalIllumination::Update(float deltaTime)
{
    // Update temporal accumulation index
    if (m_UseTemporalFiltering) {
        m_CurrentTemporalIndex = 1 - m_CurrentTemporalIndex;
    }
}

void GlobalIllumination::Render(Camera* camera, const std::vector<Light>& lights,
                                const std::vector<GameObject*>& objects)
{
    if (!m_Enabled) return;

    switch (m_Technique) {
        case Technique::VCT:
            RenderVCT(camera, lights, objects);
            break;
        case Technique::LPV:
            RenderLPV(camera, lights);
            break;
        case Technique::SSGI:
            RenderSSGI(camera);
            break;
        case Technique::Hybrid:
            RenderHybrid(camera, lights, objects);
            break;
        default:
            break;
    }
}

void GlobalIllumination::RenderVCT(Camera* camera, const std::vector<Light>& lights,
                                   const std::vector<GameObject*>& objects)
{
    // Step 1: Voxelize the scene
    m_VoxelGrid->Clear();
    m_VoxelGrid->Voxelize(objects, camera);
    m_VoxelGrid->GenerateMipmaps();

    // Step 2: Cone trace to compute indirect lighting
    glBindFramebuffer(GL_FRAMEBUFFER, m_GIFramebuffer);
    glViewport(0, 0, m_ScreenWidth, m_ScreenHeight);
    glClear(GL_COLOR_BUFFER_BIT);

    m_ConeTraceShader->Use();
    m_ConeTraceShader->SetInt("voxelAlbedo", 0);
    m_ConeTraceShader->SetInt("voxelNormal", 1);
    m_ConeTraceShader->SetInt("gPosition", 2);
    m_ConeTraceShader->SetInt("gNormal", 3);
    m_ConeTraceShader->SetInt("numCones", m_NumDiffuseCones);
    m_ConeTraceShader->SetFloat("voxelSize", m_VoxelGrid->GetVoxelSize());
    m_ConeTraceShader->SetVec3("gridMin", m_VoxelGrid->GetGridMin());
    m_ConeTraceShader->SetVec3("gridMax", m_VoxelGrid->GetGridMax());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, m_VoxelGrid->GetVoxelAlbedoTexture());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, m_VoxelGrid->GetVoxelNormalTexture());

    // Render fullscreen quad (assumes utility function exists)
    // RenderFullscreenQuad();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Step 3: Temporal filtering (if enabled)
    if (m_UseTemporalFiltering) {
        // Apply temporal accumulation
        // (Implementation would blend current frame with previous)
    }
}

void GlobalIllumination::RenderLPV(Camera* camera, const std::vector<Light>& lights)
{
    // Step 1: Inject light from RSM
    m_LPV->Clear();
    m_LPV->Inject(lights, camera);

    // Step 2: Propagate light through grid
    m_LPV->Propagate(4);

    // Step 3: Sample LPV in lighting pass (done in shader)
    // The LPV textures are bound in the lighting pass
}

void GlobalIllumination::RenderSSGI(Camera* camera)
{
    // Step 1: Compute SSGI via ray marching
    glBindFramebuffer(GL_FRAMEBUFFER, m_GIFramebuffer);
    glViewport(0, 0, m_ScreenWidth, m_ScreenHeight);

    m_SSGIShader->Use();
    // Bind G-Buffer textures
    // Perform screen-space ray marching

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Step 2: Bilateral blur for denoising
    m_SSGIBlurShader->Use();
    // Apply blur
}

void GlobalIllumination::RenderHybrid(Camera* camera, const std::vector<Light>& lights,
                                      const std::vector<GameObject*>& objects)
{
    // Combine VCT and SSGI
    RenderVCT(camera, lights, objects);
    RenderSSGI(camera);

    // Blend results (done in lighting pass shader)
}

void GlobalIllumination::SetQuality(Quality quality)
{
    m_Quality = quality;
    UpdateQualitySettings();

    // Recreate voxel grid with new resolution
    if (m_VoxelGrid) {
        m_VoxelGrid->SetResolution(m_VoxelResolution);
    }
}

void GlobalIllumination::UpdateQualitySettings()
{
    switch (m_Quality) {
        case Quality::Low:
            m_VoxelResolution = 64;
            m_NumDiffuseCones = 3;
            m_UseTemporalFiltering = false;
            m_TemporalBlendFactor = 0.0f;
            break;
        case Quality::Medium:
            m_VoxelResolution = 128;
            m_NumDiffuseCones = 5;
            m_UseTemporalFiltering = true;
            m_TemporalBlendFactor = 0.85f;
            break;
        case Quality::High:
            m_VoxelResolution = 256;
            m_NumDiffuseCones = 5;
            m_UseTemporalFiltering = true;
            m_TemporalBlendFactor = 0.9f;
            break;
        case Quality::Ultra:
            m_VoxelResolution = 512;
            m_NumDiffuseCones = 9;
            m_UseTemporalFiltering = true;
            m_TemporalBlendFactor = 0.95f;
            break;
    }
}

void GlobalIllumination::CreateGITexture(int width, int height)
{
    // Create RGBA16F texture for GI output
    glGenTextures(1, &m_GITexture);
    glBindTexture(GL_TEXTURE_2D, m_GITexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create framebuffer
    glGenFramebuffers(1, &m_GIFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_GIFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_GITexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[GI] GI framebuffer is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create temporal textures
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &m_TemporalTexture[i]);
        glBindTexture(GL_TEXTURE_2D, m_TemporalTexture[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
}

void GlobalIllumination::CreateSSGIResources(int width, int height)
{
    // Create SSGI texture
    glGenTextures(1, &m_SSGITexture);
    glBindTexture(GL_TEXTURE_2D, m_SSGITexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create blur texture
    glGenTextures(1, &m_SSGIBlurTexture);
    glBindTexture(GL_TEXTURE_2D, m_SSGIBlurTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void GlobalIllumination::RenderDebugVisualization(Camera* camera)
{
    if (!m_ShowVoxels || !m_VoxelGrid) return;

    m_VoxelGrid->RenderDebug(camera, m_VoxelDebugShader.get());
}
