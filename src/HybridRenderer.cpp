#include "HybridRenderer.h"
#include "Camera.h"
#include "GameObject.h"
#include "PostProcessing.h"
#include "ParticleSystem.h"
#include "GlobalIllumination.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

HybridRenderer::HybridRenderer()
    : m_Camera(nullptr), m_RenderMode(RenderMode::HybridOptimized),
      m_LightingMode(LightingMode::Deferred), m_MainFramebuffer(0),
      m_TransparencyFramebuffer(0), m_LightDataUBO(0), m_LightGridSSBO(0),
      m_LightListSSBO(0), m_ShowCullingBounds(false), m_ShowGBuffer(false),
      m_ShowLightHeatmap(false), m_GPUCullingEnabled(true) {
}

HybridRenderer::~HybridRenderer() {
    Shutdown();
}

bool HybridRenderer::Initialize() {
    // Initialize G-Buffer
    m_GBuffer = std::make_unique<GBuffer>();
    if (!m_GBuffer->Init(1920, 1080)) {
        return false;
    }

    // Initialize GPU Culling System
    m_CullingSystem = std::make_unique<GPUCullingSystem>();
    if (!m_CullingSystem->Initialize()) {
        return false;
    }

    // Initialize Post-Processing
    m_PostProcessing = std::make_unique<PostProcessing>();
    
    // Initialize Particle System
    m_ParticleSystem = std::make_unique<ParticleSystem>();

    // Setup shaders
    SetupDefaultShaders();

    // Setup GPU buffers
    SetupGPUBuffers();

    // Create render pipeline
    RecreateDefaultPipeline();

    return true;
}

void HybridRenderer::Shutdown() {
    CleanupGPUBuffers();

    m_CullingSystem->Shutdown();
    m_GBuffer.reset();
    m_PostProcessing.reset();
    m_ParticleSystem.reset();

    if (m_MainFramebuffer) glDeleteFramebuffers(1, &m_MainFramebuffer);
    if (m_TransparencyFramebuffer) glDeleteFramebuffers(1, &m_TransparencyFramebuffer);
}

void HybridRenderer::Update(float deltaTime) {
    // Update particle system
    if (m_ParticleSystem) {
        m_ParticleSystem->Update(deltaTime);
    }
}

void HybridRenderer::Render() {
    if (!m_Camera || !m_SceneRoot) return;

    // Prepare frame
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    glViewport(0, 0, width, height);

    // Execute GPU culling
    ExecuteGPUCulling();

    // Collect renderable objects based on culling results
    CollectRenderableObjects();

    // Update light data
    UpdateLightData();

    // Execute render passes based on mode
    switch (m_RenderMode) {
        case RenderMode::DeferredOnly:
            ExecuteGeometryPass();
            ExecuteLightingPass();
            break;
        case RenderMode::ForwardOnly:
            ExecuteGeometryPass();  // Reuse for forward
            ExecuteTransparentPass();
            break;
        case RenderMode::HybridOptimized:
            ExecuteGeometryPass();
            ExecuteLightingPass();
            ExecuteTransparentPass();
            ExecutePostProcessing();
            break;
        case RenderMode::HybridDebug:
            ExecuteGeometryPass();
            ExecuteLightingPass();
            ExecuteTransparentPass();
            RenderDebugVis();
            break;
    }

    // Final composition
    ExecuteCompositePass();
}

void HybridRenderer::ExecuteShadowPass() {
    // TODO: Implement cascaded shadow map rendering
}

void HybridRenderer::ExecuteGPUCulling() {
    if (!m_GPUCullingEnabled || !m_CullingSystem) return;

    // Setup culling parameters
    m_CullingSystem->SetupCulling(
        glm::make_mat4(m_Camera->GetViewMatrix().m),
        glm::make_mat4(m_Camera->GetProjectionMatrix().m),
        glm::vec3(m_Camera->GetPosition().x, m_Camera->GetPosition().y, m_Camera->GetPosition().z),
        m_RenderableObjects.size()
    );

    // Populate cull data from scene
    std::vector<GPUCullingSystem::CullData> cullData;
    if (m_SceneRoot) {
        // Flatten scene hierarchy and extract bounding volumes
        std::vector<std::shared_ptr<GameObject>> queue;
        queue.push_back(m_SceneRoot);

        while (!queue.empty()) {
            auto obj = queue.back();
            queue.pop_back();

            // Extract bounding volume
            GPUCullingSystem::CullData data = {};
            data.modelMatrix = glm::make_mat4(obj->GetTransform().GetModelMatrix().m);
            
            // Placeholder bounding sphere/AABB
            data.boundingSphere = glm::vec4(0, 0, 0, 1.0f);
            data.aabbMin = glm::vec4(-1, -1, -1, 0);
            data.aabbMax = glm::vec4(1, 1, 1, 0);
            data.meshletCount = 1;
            data.lodLevel = 0;
            data.isVisible = 1;
            data.screenCoverage = 100;

            cullData.push_back(data);

            for (auto& child : obj->GetChildren()) {
                queue.push_back(child);
            }
        }
    }

    if (!cullData.empty()) {
        m_CullingSystem->SetCullData(cullData);
        m_CullingSystem->ExecuteAll(m_GBuffer->GetDepthTexture());
    }
}

void HybridRenderer::ExecuteGeometryPass() {
    // Bind G-Buffer for writing
    m_GBuffer->BindForWriting();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!m_GeometryShader) return;
    m_GeometryShader->Use();

    // Set uniforms
    m_GeometryShader->SetMat4("u_View", m_Camera->GetViewMatrix().m);
    m_GeometryShader->SetMat4("u_Projection", m_Camera->GetProjectionMatrix().m);

    // Render visible objects
    if (m_SceneRoot) {
        m_SceneRoot->Draw(m_GeometryShader.get(), 
                         m_Camera->GetViewMatrix(), 
                         m_Camera->GetProjectionMatrix());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void HybridRenderer::ExecuteLightingPass() {
    // Bind lighting output framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_MainFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!m_DeferredLightingShader) return;
    m_DeferredLightingShader->Use();

    // Bind G-Buffer textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_GBuffer->GetPositionTexture());
    m_DeferredLightingShader->SetInt("u_GBufferPosition", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_GBuffer->GetNormalTexture());
    m_DeferredLightingShader->SetInt("u_GBufferNormal", 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_GBuffer->GetAlbedoSpecTexture());
    m_DeferredLightingShader->SetInt("u_GBufferAlbedo", 2);

    // Full-screen quad rendering for lighting computation
    // TODO: Use compute shader version (deferred_lighting.comp)
}

void HybridRenderer::ExecuteTransparentPass() {
    // Forward rendering of transparent objects
    if (!m_TransparentShader) return;

    glBindFramebuffer(GL_FRAMEBUFFER, m_MainFramebuffer);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_TransparentShader->Use();
    m_TransparentShader->SetMat4("u_View", m_Camera->GetViewMatrix().m);
    m_TransparentShader->SetMat4("u_Projection", m_Camera->GetProjectionMatrix().m);

    // Render transparent objects (particles, billboards, etc.)
    if (m_ParticleSystem) {
        // Render particles
    }

    glDisable(GL_BLEND);
}

void HybridRenderer::ExecutePostProcessing() {
    // Apply post-processing effects (SSAO, SSR, TAA, bloom, fog)
    if (!m_PostProcessing) return;

    // TODO: Execute post-processing pipeline
}

void HybridRenderer::ExecuteCompositePass() {
    // Final composition to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!m_CompositeShader) return;
    m_CompositeShader->Use();

    // Copy/composite to screen
    // TODO: Implement full-screen quad rendering
}

void HybridRenderer::CollectRenderableObjects() {
    m_RenderableObjects.clear();

    if (!m_SceneRoot) return;

    // Simple depth-first traversal to collect objects
    std::vector<std::shared_ptr<GameObject>> queue;
    queue.push_back(m_SceneRoot);

    while (!queue.empty()) {
        auto obj = queue.back();
        queue.pop_back();

        if (obj) {
            RenderableObject renderObj = {};
            renderObj.gameObject = obj.get();
            renderObj.worldMatrix = glm::make_mat4(obj->GetTransform().GetModelMatrix().m);
            renderObj.lodLevel = 0;  // TODO: Get from culling results
            renderObj.isVisible = true;

            m_RenderableObjects.push_back(renderObj);

            for (auto& child : obj->GetChildren()) {
                queue.push_back(child);
            }
        }
    }
}

void HybridRenderer::UpdateLightData() {
    // Upload light data to GPU UBO
    glBindBuffer(GL_UNIFORM_BUFFER, m_LightDataUBO);

    struct LightData {
        glm::vec4 position;
        glm::vec4 direction;
        glm::vec4 colorIntensity;
        glm::vec4 params;
    };

    std::vector<LightData> lightData;
    for (const auto& light : m_Lights) {
        LightData ld = {};
        ld.position = glm::vec4(light.position.x, light.position.y, light.position.z, static_cast<float>(light.type));
        ld.direction = glm::vec4(light.direction.x, light.direction.y, light.direction.z, 0.0f);
        ld.colorIntensity = glm::vec4(light.color.x, light.color.y, light.color.z, light.intensity);
        ld.params = glm::vec4(light.range, light.cutOff, light.outerCutOff, light.linear);
        lightData.push_back(ld);
    }

    if (!lightData.empty()) {
        glBufferSubData(GL_UNIFORM_BUFFER, 0, lightData.size() * sizeof(LightData), lightData.data());
    }

    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void HybridRenderer::RenderDebugVis() {
    if (m_ShowCullingBounds) {
        // Render culling bounds (bounding boxes)
    }
    if (m_ShowGBuffer) {
        // Visualize G-Buffer contents
    }
    if (m_ShowLightHeatmap) {
        // Visualize light coverage heatmap
    }
}

void HybridRenderer::SetupDefaultShaders() {
    // Load shader programs
    m_GeometryShader = std::make_unique<Shader>();
    m_GeometryShader->LoadFromFiles("shaders/geometry.vert", "shaders/geometry.frag");
    m_DeferredLightingShader = std::make_unique<Shader>();
    m_DeferredLightingShader->LoadComputeShader("shaders/deferred_lighting.comp");
    m_TransparentShader = std::make_unique<Shader>();
    m_TransparentShader->LoadFromFiles("shaders/transparent.vert", "shaders/transparent.frag");
    m_CompositeShader = std::make_unique<Shader>();
    m_CompositeShader->LoadFromFiles("shaders/composite.vert", "shaders/composite.frag");
    m_ShadowShader = std::make_unique<Shader>();
    m_ShadowShader->LoadFromFiles("shaders/shadow.vert", "shaders/shadow.frag");
    m_DebugVisShader = std::make_unique<Shader>();
    m_DebugVisShader->LoadFromFiles("shaders/debug_vis.vert", "shaders/debug_vis.frag");
}

void HybridRenderer::SetupGPUBuffers() {
    // Light data UBO
    glGenBuffers(1, &m_LightDataUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, m_LightDataUBO);
    glBufferStorage(GL_UNIFORM_BUFFER, MAX_LIGHTS * sizeof(glm::vec4) * 4 + sizeof(int), 
                   nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // Create main and transparency framebuffers
    glGenFramebuffers(1, &m_MainFramebuffer);
    glGenFramebuffers(1, &m_TransparencyFramebuffer);
}

void HybridRenderer::CleanupGPUBuffers() {
    if (m_LightDataUBO) glDeleteBuffers(1, &m_LightDataUBO);
    if (m_LightGridSSBO) glDeleteBuffers(1, &m_LightGridSSBO);
    if (m_LightListSSBO) glDeleteBuffers(1, &m_LightListSSBO);
}

void HybridRenderer::RecreateDefaultPipeline() {
    m_Pipeline = std::make_unique<RenderPipeline>("Default");

    // Add passes in order
    // Shadow pass, Geometry pass, Lighting pass, Transparent pass, Post-processing, Composite
    // TODO: Implement concrete RenderPass subclasses for each stage
}
