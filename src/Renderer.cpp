#define NOMINMAX
#include "Renderer.h"
#include "Camera.h"
#include "GLExtensions.h"
#include "Material.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

Renderer::Renderer() 
    : m_Camera(nullptr)
{
    m_TextureManager = std::make_unique<TextureManager>();
    m_Root = std::make_shared<GameObject>("Root");
}

Renderer::~Renderer() {
    Shutdown();
}

bool Renderer::CheckCollision(const AABB& bounds) {
    if (m_Root) {
        return m_Root->CheckCollision(bounds);
    }
    return false;
}

void Renderer::SaveScene(const std::string& filename) {
    // Simplified save: only saves direct children of root
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open scene file for saving: " << filename << std::endl;
        return;
    }

    if (m_Root) {
        for (auto& child : m_Root->GetChildren()) {
            // Determine type based on name or mesh (simplified)
            std::string source = "cube";
            if (child->GetName() == "Pyramid") source = "assets/pyramid.obj";
            
            const Transform& t = child->GetTransform();
            file << source << " "
                 << t.position.x << " " << t.position.y << " " << t.position.z << " "
                 << t.rotation.x << " " << t.rotation.y << " " << t.rotation.z << " "
                 << t.scale.x << " " << t.scale.y << " " << t.scale.z << "\n";
        }
    }
    
    std::cout << "Scene saved to " << filename << std::endl;
}

void Renderer::LoadScene(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open scene file for loading: " << filename << std::endl;
        return;
    }

    // Clear current scene
    if (m_Root) {
        m_Root->GetChildren().clear();
    } else {
        m_Root = std::make_shared<GameObject>("Root");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string source;
        Vec3 pos, rot, scale;

        ss >> source 
           >> pos.x >> pos.y >> pos.z 
           >> rot.x >> rot.y >> rot.z 
           >> scale.x >> scale.y >> scale.z;

        auto obj = std::make_shared<GameObject>(source == "cube" ? "Cube" : "Pyramid");
        if (source == "cube") {
            obj->SetMesh(Mesh::CreateCube());
        } else {
            obj->SetMesh(Mesh::LoadFromOBJ(source));
        }
        
        obj->GetTransform() = Transform(pos, rot, scale);
        
        // Create default material for loaded object
        auto mat = std::make_shared<Material>();
        mat->texture = m_Texture;
        mat->specularMap = m_Texture;
        if (source != "cube") mat->diffuse = Vec3(1.0f, 1.0f, 0.0f); // Yellow for pyramid
        obj->SetMaterial(mat);
        
        m_Root->AddChild(obj);
    }

    std::cout << "Scene loaded from " << filename << std::endl;
}

void Renderer::AddCube(const Transform& transform) {
    auto cube = std::make_shared<GameObject>("Cube");
    cube->SetMesh(Mesh::CreateCube());
    cube->GetTransform() = transform;
    
    auto mat = std::make_shared<Material>();
    mat->texture = m_Texture;
    mat->specularMap = m_Texture;
    cube->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(cube);
    
    std::cout << "Added cube at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::AddPyramid(const Transform& transform) {
    auto pyramid = std::make_shared<GameObject>("Pyramid");
    pyramid->SetMesh(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    pyramid->GetTransform() = transform;
    
    auto mat = std::make_shared<Material>();
    mat->diffuse = Vec3(1.0f, 1.0f, 0.0f);
    mat->texture = m_Texture;
    mat->specularMap = m_Texture;
    pyramid->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(pyramid);
    
    std::cout << "Added pyramid at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::RemoveObject(size_t index) {
    if (m_Root && index < m_Root->GetChildren().size()) {
        auto child = m_Root->GetChildren()[index];
        m_Root->RemoveChild(child);
        std::cout << "Removed object at index " << index << std::endl;
    }
}

void Renderer::SetupScene() {
    // Add some default objects
    AddCube(Transform(Vec3(0, 0, 0)));
    AddCube(Transform(Vec3(2, 0, 0)));
    AddPyramid(Transform(Vec3(-2, 0, 0)));
    
    // Add a floor
    auto floor = std::make_shared<GameObject>("Floor");
    floor->SetMesh(Mesh::CreateCube()); // Use cube as floor for now
    floor->GetTransform() = Transform(Vec3(0, -2, 0), Vec3(0, 0, 0), Vec3(10, 0.1f, 10));
    
    auto mat = std::make_shared<Material>();
    mat->texture = m_Texture;
    mat->specularMap = m_Texture;
    floor->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(floor);
    
    // Add lights
    AddLight(Light(Vec3(0, 5, 0), Vec3(1, 1, 1), 1.0f));
    AddLight(Light(Vec3(-5, 5, -5), Vec3(1, 0, 0), 1.0f)); // Red light
}

bool Renderer::Init() {
    // Create and load shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/textured.vert", "shaders/textured.frag")) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }

    // Load texture using manager
    m_Texture = m_TextureManager->LoadTexture("assets/brick.png");
    if (!m_Texture) {
        std::cerr << "Failed to load default texture" << std::endl;
        return false;
    }

    // Setup scene with multiple cubes
    SetupScene();

    // Initialize Skybox
    m_Skybox = std::make_unique<Skybox>();
    std::vector<std::string> faces;
    // Use the same texture for all faces for now
    for(int i=0; i<6; i++) faces.push_back("assets/brick.png");
    
    if (!m_Skybox->Init(faces)) {
        std::cerr << "Failed to initialize skybox" << std::endl;
        // Don't return false, just continue without skybox
    }

    // Initialize Shadow Mapping
    m_DepthShader = std::make_unique<Shader>();
    if (!m_DepthShader->LoadFromFiles("shaders/depth.vert", "shaders/depth.frag")) {
        std::cerr << "Failed to load depth shaders" << std::endl;
        return false;
    }

    m_CSM = std::make_unique<CascadedShadowMap>();
    if (!m_CSM->Init(2048, 2048)) {
        std::cerr << "Failed to initialize cascaded shadow map" << std::endl;
        return false;
    }
    m_CascadeSplits = { 25.0f, 100.0f }; // Splits at 25m and 100m (plus far plane)

    // Initialize Point Light Shadows
    m_PointShadowShader = std::make_unique<Shader>();
    if (!m_PointShadowShader->LoadFromFiles("shaders/point_shadow.vert", "shaders/point_shadow.frag", "shaders/point_shadow.geom")) {
        std::cerr << "Failed to load point shadow shaders" << std::endl;
        return false;
    }

    // Create 4 cubemap shadows
    for (int i = 0; i < 4; ++i) {
        auto shadow = std::make_unique<CubemapShadow>();
        if (!shadow->Init(1024, 1024)) {
            std::cerr << "Failed to initialize cubemap shadow " << i << std::endl;
            return false;
        }
        m_PointShadows.push_back(std::move(shadow));
    }

    // Create 4 spot light shadow maps
    for (int i = 0; i < 4; ++i) {
        auto shadow = std::make_unique<ShadowMap>();
        if (!shadow->Init(1024, 1024)) {
            std::cerr << "Failed to initialize spot shadow map " << i << std::endl;
            return false;
        }
        m_SpotShadows.push_back(std::move(shadow));
    }

    // Initialize GBuffer for deferred rendering
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    m_GBuffer = std::make_unique<GBuffer>();
    if (!m_GBuffer->Init(width, height)) {
        std::cerr << "Failed to initialize GBuffer" << std::endl;
        return false;
    }

    // Load geometry pass shader
    m_GeometryShader = std::make_unique<Shader>();
    if (!m_GeometryShader->LoadFromFiles("shaders/geometry_pass.vert", "shaders/geometry_pass.frag")) {
        std::cerr << "Failed to load geometry pass shaders" << std::endl;
        return false;
    }

    // Load lighting pass shader
    m_LightingShader = std::make_unique<Shader>();
    if (!m_LightingShader->LoadFromFiles("shaders/lighting_pass.vert", "shaders/lighting_pass.frag")) {
        std::cerr << "Failed to load lighting pass shaders" << std::endl;
        return false;
    }

    // Setup fullscreen quad for lighting pass
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &m_QuadVAO);
    glGenBuffers(1, &m_QuadVBO);
    glBindVertexArray(m_QuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Initialize post-processing
    m_PostProcessing = std::make_unique<PostProcessing>();
    if (!m_PostProcessing->Init(width, height)) {
        std::cerr << "Failed to initialize post-processing" << std::endl;
        return false;
    }

    return true;
}

void Renderer::Render() {
    if (!m_Camera) return;

    // Update scene graph
    if (m_Root) {
        m_Root->Update(Mat4::Identity());
    }

    // ===== PASS 0: Render point light shadows =====
    m_PointShadowShader->Use();
    int shadowIndex = 0;
    for (const auto& light : m_Lights) {
        if (light.type == LightType::Point && light.castsShadows && shadowIndex < m_PointShadows.size()) {
            std::vector<Mat4> shadowTransforms;
            float farPlane;
            m_PointShadows[shadowIndex]->CalculateViewMatrices(light.position, shadowTransforms, farPlane);
            
            for (int i = 0; i < 6; ++i) {
                m_PointShadowShader->SetMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i].m);
            }
            m_PointShadowShader->SetFloat("far_plane", farPlane);
            m_PointShadowShader->SetVec3("lightPos", light.position.x, light.position.y, light.position.z);
            m_PointShadowShader->SetMat4("u_Model", Mat4::Identity().m); // Simplified, should be per object
            
            m_PointShadows[shadowIndex]->BindForWriting();
            
            // Draw scene for shadow map
            if (m_Root) {
                // Note: We need a simpler Draw method that just sends model matrices
                // For now, reusing standard Draw but ignoring view/proj since shader doesn't use them
                m_Root->Draw(m_PointShadowShader.get(), Mat4::Identity(), Mat4::Identity());
            }
            
            shadowIndex++;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 1: Render spot light shadows =====
    m_DepthShader->Use();
    int spotShadowIndex = 0;
    std::vector<Mat4> spotLightMatrices;
    
    for (size_t i = 0; i < m_Lights.size(); ++i) {
        if (m_Lights[i].type == LightType::Spot && m_Lights[i].castsShadows && spotShadowIndex < m_SpotShadows.size()) {
            Mat4 spotLightMatrix = GetSpotLightMatrix(m_Lights[i]);
            spotLightMatrices.push_back(spotLightMatrix);
            
            m_DepthShader->SetMat4("u_LightSpaceMatrix", spotLightMatrix.m);
            
            glViewport(0, 0, m_SpotShadows[spotShadowIndex]->GetWidth(), m_SpotShadows[spotShadowIndex]->GetHeight());
            m_SpotShadows[spotShadowIndex]->BindForWriting();
            glClear(GL_DEPTH_BUFFER_BIT);
            
            if (m_Root) {
                m_Root->Draw(m_DepthShader.get(), Mat4::Identity(), spotLightMatrix);
            }
            
            spotShadowIndex++;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 2: Calculate light space matrix for shadow mapping (first directional light only)
    std::vector<Mat4> lightSpaceMatrices;
    if (m_Lights.size() > 0 && m_Lights[0].castsShadows) {
        lightSpaceMatrices = GetLightSpaceMatrices();
        
        m_DepthShader->Use();
        glViewport(0, 0, m_CSM->GetWidth(), m_CSM->GetHeight());
        
        for (unsigned int i = 0; i < 3; ++i) {
            m_DepthShader->SetMat4("u_LightSpaceMatrix", lightSpaceMatrices[i].m);
            m_CSM->BindForWriting(i);
            glClear(GL_DEPTH_BUFFER_BIT);
            
            if (m_Root) {
                m_Root->Draw(m_DepthShader.get(), Mat4::Identity(), lightSpaceMatrices[i]);
            }
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Get framebuffer size
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);

    // ===== PASS 2: Geometry Pass - Render to G-Buffer =====
    glViewport(0, 0, width, height);
    m_GBuffer->BindForWriting();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_GeometryShader->Use();

    // Render scene to G-Buffer
    if (m_Root) {
        m_Root->Draw(m_GeometryShader.get(), m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 3: Lighting Pass - Render to HDR framebuffer =====
    m_PostProcessing->BeginHDR();

    m_LightingShader->Use();

    // Bind G-Buffer textures
    m_GBuffer->BindForReading();
    m_LightingShader->SetInt("gPosition", 0);
    m_LightingShader->SetInt("gNormal", 1);
    m_LightingShader->SetInt("gAlbedoSpec", 2);

    // Bind shadow map
    m_CSM->BindForReading(3);
    m_LightingShader->SetInt("shadowMap", 3);
    
    // Upload cascade data
    if (lightSpaceMatrices.size() == 3) {
        for (int i = 0; i < 3; ++i) {
            m_LightingShader->SetMat4("cascadeLightSpaceMatrices[" + std::to_string(i) + "]", lightSpaceMatrices[i].m);
        }
        m_LightingShader->SetFloat("cascadePlaneDistances[0]", m_CascadeSplits[0]);
        m_LightingShader->SetFloat("cascadePlaneDistances[1]", m_CascadeSplits[1]);
        m_LightingShader->SetFloat("cascadePlaneDistances[2]", m_Camera->GetFarPlane());
    }

    // Bind spot shadow maps
    spotShadowIndex = 0;
    for (size_t i = 0; i < m_Lights.size(); ++i) {
        if (m_Lights[i].type == LightType::Spot && m_Lights[i].castsShadows && spotShadowIndex < m_SpotShadows.size()) {
            m_SpotShadows[spotShadowIndex]->BindForReading(8 + spotShadowIndex);
            m_LightingShader->SetInt("spotShadowMaps[" + std::to_string(spotShadowIndex) + "]", 8 + spotShadowIndex);
            m_LightingShader->SetMat4("spotLightSpaceMatrices[" + std::to_string(spotShadowIndex) + "]", spotLightMatrices[spotShadowIndex].m);
            spotShadowIndex++;
        }
    }

    // Bind point shadow maps
    int pointShadowIndex = 0;
    for (size_t i = 0; i < m_Lights.size(); ++i) {
        if (m_Lights[i].type == LightType::Point && m_Lights[i].castsShadows && pointShadowIndex < m_PointShadows.size()) {
            m_PointShadows[pointShadowIndex]->BindForReading(4 + pointShadowIndex);
            m_LightingShader->SetInt("pointShadowMaps[" + std::to_string(pointShadowIndex) + "]", 4 + pointShadowIndex);
            pointShadowIndex++;
        }
    }

    // Light setup
    int lightCount = (std::min)(static_cast<int>(m_Lights.size()), MAX_LIGHTS);
    m_LightingShader->SetInt("u_LightCount", lightCount);
    
    for (size_t i = 0; i < lightCount; ++i) {
        std::string base = "u_Lights[" + std::to_string(i) + "]";
        m_LightingShader->SetInt(base + ".type", static_cast<int>(m_Lights[i].type));
        m_LightingShader->SetVec3(base + ".position", m_Lights[i].position.x, m_Lights[i].position.y, m_Lights[i].position.z);
        m_LightingShader->SetVec3(base + ".direction", m_Lights[i].direction.x, m_Lights[i].direction.y, m_Lights[i].direction.z);
        m_LightingShader->SetVec3(base + ".color", m_Lights[i].color.x, m_Lights[i].color.y, m_Lights[i].color.z);
        m_LightingShader->SetFloat(base + ".intensity", m_Lights[i].intensity);
        
        m_LightingShader->SetFloat(base + ".constant", m_Lights[i].constant);
        m_LightingShader->SetFloat(base + ".linear", m_Lights[i].linear);
        m_LightingShader->SetFloat(base + ".quadratic", m_Lights[i].quadratic);
        
        m_LightingShader->SetFloat(base + ".cutOff", std::cos(m_Lights[i].cutOff * 3.14159f / 180.0f));
        m_LightingShader->SetFloat(base + ".outerCutOff", std::cos(m_Lights[i].outerCutOff * 3.14159f / 180.0f));
        
        m_LightingShader->SetFloat(base + ".range", m_Lights[i].range);
        m_LightingShader->SetFloat(base + ".shadowSoftness", m_Lights[i].shadowSoftness);
        m_LightingShader->SetInt(base + ".castsShadows", m_Lights[i].castsShadows ? 1 : 0);
        m_LightingShader->SetFloat(base + ".lightSize", m_Lights[i].lightSize);
    }
    
    Vec3 camPos = m_Camera->GetPosition();
    m_LightingShader->SetVec3("u_ViewPos", camPos.x, camPos.y, camPos.z);
    m_LightingShader->SetMat4("view", m_Camera->GetViewMatrix().m);

    // Render fullscreen quad
    RenderQuad();

    // ===== PASS 4: Forward Pass for Skybox (still in HDR) =====
    // Copy depth buffer from G-Buffer to HDR framebuffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_GBuffer->GetPositionTexture());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // Will be handled by post-processing
    
    // Draw Skybox to HDR framebuffer
    if (m_Skybox) {
        m_Skybox->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
    }
    
    // ===== PASS 5: Post-Processing =====
    // Apply bloom, tone mapping, and other effects
    m_PostProcessing->ApplyEffects();
}

void Renderer::RenderQuad() {
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Renderer::Shutdown() {
    if (m_QuadVAO) {
        glDeleteVertexArrays(1, &m_QuadVAO);
        m_QuadVAO = 0;
    }
    if (m_QuadVBO) {
        glDeleteBuffers(1, &m_QuadVBO);
        m_QuadVBO = 0;
    }
    m_Root.reset();
}

std::vector<Vec4> Renderer::GetFrustumCornersWorldSpace(const Mat4& proj, const Mat4& view) {
    const auto inv = (proj * view).Inverse();
    
    std::vector<Vec4> frustumCorners;
    for (unsigned int x = 0; x < 2; ++x) {
        for (unsigned int y = 0; y < 2; ++y) {
            for (unsigned int z = 0; z < 2; ++z) {
                const Vec4 pt = inv * Vec4(
                    2.0f * x - 1.0f,
                    2.0f * y - 1.0f,
                    2.0f * z - 1.0f,
                    1.0f);
                frustumCorners.push_back(pt / pt.w);
            }
        }
    }
    
    return frustumCorners;
}

Mat4 Renderer::GetLightSpaceMatrix(const float nearPlane, const float farPlane) {
    const auto proj = Mat4::Perspective(m_Camera->GetFOV(), m_Camera->GetAspectRatio(), nearPlane, farPlane);
    const auto corners = GetFrustumCornersWorldSpace(proj, m_Camera->GetViewMatrix());

    Vec3 center = Vec3(0, 0, 0);
    for (const auto& v : corners) {
        center = center + Vec3(v.x, v.y, v.z);
    }
    center = center / static_cast<float>(corners.size());

    const auto lightView = Mat4::LookAt(center + m_Lights[0].direction * -1.0f, center, Vec3(0.0f, 1.0f, 0.0f));

    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();

    for (const auto& v : corners) {
        const auto trf = lightView * v;
        minX = (std::min)(minX, trf.x);
        maxX = (std::max)(maxX, trf.x);
        minY = (std::min)(minY, trf.y);
        maxY = (std::max)(maxY, trf.y);
        minZ = (std::min)(minZ, trf.z);
        maxZ = (std::max)(maxZ, trf.z);
    }

    // Tune this parameter for your scene
    constexpr float zMult = 10.0f;
    if (minZ < 0) {
        minZ *= zMult;
    } else {
        minZ /= zMult;
    }
    if (maxZ < 0) {
        maxZ /= zMult;
    } else {
        maxZ *= zMult;
    }

    const Mat4 lightProjection = Mat4::Orthographic(minX, maxX, minY, maxY, minZ, maxZ);
    return lightProjection * lightView;
}

std::vector<Mat4> Renderer::GetLightSpaceMatrices() {
    std::vector<Mat4> ret;
    for (size_t i = 0; i < m_CascadeSplits.size() + 1; ++i) {
        if (i == 0) {
            ret.push_back(GetLightSpaceMatrix(m_Camera->GetNearPlane(), m_CascadeSplits[i]));
        } else if (i < m_CascadeSplits.size()) {
            ret.push_back(GetLightSpaceMatrix(m_CascadeSplits[i - 1], m_CascadeSplits[i]));
        } else {
            ret.push_back(GetLightSpaceMatrix(m_CascadeSplits[i - 1], m_Camera->GetFarPlane()));
        }
    }
    return ret;
}

Mat4 Renderer::GetSpotLightMatrix(const Light& light) {
    // Calculate perspective projection based on spot light's cutoff angle
    // Use outer cutoff for FOV to ensure entire cone is covered
    float fov = light.outerCutOff * 2.0f * 3.14159f / 180.0f; // Convert to radians and double for full cone
    float aspect = 1.0f; // Square shadow map
    float nearPlane = 0.1f;
    float farPlane = light.range > 0.0f ? light.range : 50.0f;
    
    Mat4 projection = Mat4::Perspective(fov, aspect, nearPlane, farPlane);
    
    // Calculate view matrix from light's position looking in light's direction
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    // If light direction is parallel to up vector, use different up vector
    float dotProduct = light.direction.Dot(up);
    if (dotProduct < 0.0f) dotProduct = -dotProduct; // Manual abs to avoid macro
    if (dotProduct > 0.99f) {
        up = Vec3(1.0f, 0.0f, 0.0f);
    }
    
    Mat4 view = Mat4::LookAt(light.position, light.position + light.direction, up);
    
    return projection * view;
}
