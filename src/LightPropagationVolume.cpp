#include "LightPropagationVolume.h"
#include "Camera.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

LightPropagationVolume::LightPropagationVolume(int gridSize)
    : m_GridSize(gridSize)
    , m_GridMin(-50.0f)
    , m_GridMax(50.0f)
    , m_CellSize(0.0f)
    , m_CurrentBuffer(0)
    , m_RSMFramebuffer(0)
    , m_RSMPositionTexture(0)
    , m_RSMNormalTexture(0)
    , m_RSMFluxTexture(0)
    , m_RSMDepthTexture(0)
    , m_RSMResolution(512)
    , m_PropagationIterations(4)
{
    m_LPVTextureR[0] = m_LPVTextureR[1] = 0;
    m_LPVTextureG[0] = m_LPVTextureG[1] = 0;
    m_LPVTextureB[0] = m_LPVTextureB[1] = 0;

    m_CellSize = (m_GridMax.x - m_GridMin.x) / static_cast<float>(m_GridSize);
}

LightPropagationVolume::~LightPropagationVolume()
{
    Shutdown();
}

bool LightPropagationVolume::Initialize()
{
    std::cout << "[LPV] Initializing Light Propagation Volumes (" << m_GridSize << "³)..." << std::endl;

    CreateLPVTextures();
    CreateRSMResources();

    // Load shaders
    m_RSMShader = std::make_unique<Shader>("shaders/rsm.vert", "shaders/rsm.frag");
    m_InjectShader = std::make_unique<Shader>("shaders/lpv_inject.comp");
    m_PropagateShader = std::make_unique<Shader>("shaders/lpv_propagate.comp");

    std::cout << "[LPV] LPV initialized successfully!" << std::endl;
    return true;
}

void LightPropagationVolume::Shutdown()
{
    if (m_LPVTextureR[0]) glDeleteTextures(1, &m_LPVTextureR[0]);
    if (m_LPVTextureR[1]) glDeleteTextures(1, &m_LPVTextureR[1]);
    if (m_LPVTextureG[0]) glDeleteTextures(1, &m_LPVTextureG[0]);
    if (m_LPVTextureG[1]) glDeleteTextures(1, &m_LPVTextureG[1]);
    if (m_LPVTextureB[0]) glDeleteTextures(1, &m_LPVTextureB[0]);
    if (m_LPVTextureB[1]) glDeleteTextures(1, &m_LPVTextureB[1]);

    if (m_RSMFramebuffer) glDeleteFramebuffers(1, &m_RSMFramebuffer);
    if (m_RSMPositionTexture) glDeleteTextures(1, &m_RSMPositionTexture);
    if (m_RSMNormalTexture) glDeleteTextures(1, &m_RSMNormalTexture);
    if (m_RSMFluxTexture) glDeleteTextures(1, &m_RSMFluxTexture);
    if (m_RSMDepthTexture) glDeleteTextures(1, &m_RSMDepthTexture);
}

void LightPropagationVolume::CreateLPVTextures()
{
    // Create 3D textures for spherical harmonic coefficients (R, G, B channels)
    // Each texture stores SH coefficients for one color channel
    
    for (int i = 0; i < 2; i++) {
        // Red channel SH coefficients
        glGenTextures(1, &m_LPVTextureR[i]);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureR[i]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_GridSize, m_GridSize, m_GridSize,
                     0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        // Green channel SH coefficients
        glGenTextures(1, &m_LPVTextureG[i]);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureG[i]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_GridSize, m_GridSize, m_GridSize,
                     0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        // Blue channel SH coefficients
        glGenTextures(1, &m_LPVTextureB[i]);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureB[i]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, m_GridSize, m_GridSize, m_GridSize,
                     0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }

    std::cout << "[LPV] Created LPV 3D textures: " << m_GridSize << "³" << std::endl;
}

void LightPropagationVolume::CreateRSMResources()
{
    // Create Reflective Shadow Map textures
    glGenTextures(1, &m_RSMPositionTexture);
    glBindTexture(GL_TEXTURE_2D, m_RSMPositionTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_RSMResolution, m_RSMResolution, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &m_RSMNormalTexture);
    glBindTexture(GL_TEXTURE_2D, m_RSMNormalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_RSMResolution, m_RSMResolution, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &m_RSMFluxTexture);
    glBindTexture(GL_TEXTURE_2D, m_RSMFluxTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_RSMResolution, m_RSMResolution, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &m_RSMDepthTexture);
    glBindTexture(GL_TEXTURE_2D, m_RSMDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, m_RSMResolution, m_RSMResolution, 0, 
                 GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create framebuffer
    glGenFramebuffers(1, &m_RSMFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_RSMFramebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_RSMPositionTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_RSMNormalTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_RSMFluxTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_RSMDepthTexture, 0);

    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, drawBuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[LPV] RSM framebuffer is not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void LightPropagationVolume::Inject(const std::vector<Light>& lights, Camera* camera)
{
    if (lights.empty() || !m_InjectShader) return;

    // Generate RSM for the primary directional light
    for (const Light& light : lights) {
        if (light.type == LightType::Directional) {
            GenerateRSM(light, camera);
            break;  // Only use first directional light for now
        }
    }

    // Inject light from RSM into LPV grid
    m_InjectShader->Use();
    m_InjectShader->SetInt("rsmPosition", 0);
    m_InjectShader->SetInt("rsmNormal", 1);
    m_InjectShader->SetInt("rsmFlux", 2);
    m_InjectShader->SetVec3("gridMin", m_GridMin);
    m_InjectShader->SetVec3("gridMax", m_GridMax);
    m_InjectShader->SetInt("gridSize", m_GridSize);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_RSMPositionTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_RSMNormalTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_RSMFluxTexture);

    // Bind LPV textures as image units
    glBindImageTexture(0, m_LPVTextureR[m_CurrentBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(1, m_LPVTextureG[m_CurrentBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(2, m_LPVTextureB[m_CurrentBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    // Dispatch compute shader
    int numWorkGroups = (m_GridSize + 7) / 8;
    glDispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void LightPropagationVolume::Propagate(int iterations)
{
    if (!m_PropagateShader) return;

    m_PropagateShader->Use();
    m_PropagateShader->SetInt("gridSize", m_GridSize);

    for (int i = 0; i < iterations; i++) {
        int readBuffer = m_CurrentBuffer;
        int writeBuffer = 1 - m_CurrentBuffer;

        // Bind read textures
        m_PropagateShader->SetInt("lpvR", 0);
        m_PropagateShader->SetInt("lpvG", 1);
        m_PropagateShader->SetInt("lpvB", 2);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureR[readBuffer]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureG[readBuffer]);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureB[readBuffer]);

        // Bind write textures
        glBindImageTexture(3, m_LPVTextureR[writeBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glBindImageTexture(4, m_LPVTextureG[writeBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glBindImageTexture(5, m_LPVTextureB[writeBuffer], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        // Dispatch compute shader
        int numWorkGroups = (m_GridSize + 7) / 8;
        glDispatchCompute(numWorkGroups, numWorkGroups, numWorkGroups);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        SwapBuffers();
    }
}

void LightPropagationVolume::Clear()
{
    // Clear LPV textures to zero
    float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureR[i]);
        glClearTexImage(m_LPVTextureR[i], 0, GL_RGBA, GL_FLOAT, clearColor);
        
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureG[i]);
        glClearTexImage(m_LPVTextureG[i], 0, GL_RGBA, GL_FLOAT, clearColor);
        
        glBindTexture(GL_TEXTURE_3D, m_LPVTextureB[i]);
        glClearTexImage(m_LPVTextureB[i], 0, GL_RGBA, GL_FLOAT, clearColor);
    }
}

void LightPropagationVolume::SetGridBounds(const glm::vec3& min, const glm::vec3& max)
{
    m_GridMin = min;
    m_GridMax = max;
    m_CellSize = (max.x - min.x) / static_cast<float>(m_GridSize);
}

void LightPropagationVolume::GenerateRSM(const Light& light, Camera* camera)
{
    if (!m_RSMShader) return;

    // Render scene from light's perspective to generate RSM
    glBindFramebuffer(GL_FRAMEBUFFER, m_RSMFramebuffer);
    glViewport(0, 0, m_RSMResolution, m_RSMResolution);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Create light view-projection matrix
    glm::mat4 lightView = glm::lookAt(
        glm::vec3(light.position.x, light.position.y, light.position.z),
        glm::vec3(light.position.x, light.position.y, light.position.z) + 
            glm::vec3(light.direction.x, light.direction.y, light.direction.z),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    float orthoSize = 50.0f;
    glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 200.0f);

    glm::mat4 lightViewProjection = lightProjection * lightView;
    m_RSMShader->Use();
    m_RSMShader->SetMat4("lightViewProjection", &lightViewProjection[0][0]);
    m_RSMShader->SetVec3("lightColor", glm::vec3(light.color.x, light.color.y, light.color.z));
    m_RSMShader->SetFloat("lightIntensity", light.intensity);

    // Render scene geometry
    // (This would iterate through scene objects and render them)
    // For now, this is a placeholder

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void LightPropagationVolume::SwapBuffers()
{
    m_CurrentBuffer = 1 - m_CurrentBuffer;
}
