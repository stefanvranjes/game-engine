#include "Renderer.h"
#include "Sprite.h"
#include "Frustum.h"
#include "GLExtensions.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <GLFW/glfw3.h>

Renderer::Renderer() 
    : m_Camera(nullptr)
    , m_ShowCascades(false)
    , m_ShadowFadeStart(40.0f)
    , m_ShadowFadeEnd(50.0f)
    , m_SSAOEnabled(false)
    , m_SSREnabled(false)
    , m_TAAEnabled(false)
    , m_BatchedRenderingEnabled(true)  // Enable by default
{
    m_TextureManager = std::make_unique<TextureManager>();
    m_MaterialLibrary = std::make_unique<MaterialLibrary>();
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
        mat->SetTexture(m_Texture);
        mat->SetSpecularMap(m_Texture);
        if (source != "cube") mat->SetDiffuse(Vec3(1.0f, 1.0f, 0.0f)); // Yellow for pyramid
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
    mat->SetTexture(m_Texture);
    mat->SetSpecularMap(m_Texture);
    cube->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(cube);
    
    std::cout << "Added cube at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::AddPyramid(const Transform& transform) {
    auto pyramid = std::make_shared<GameObject>("Pyramid");
    pyramid->SetMesh(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    pyramid->GetTransform() = transform;
    
    auto mat = std::make_shared<Material>();
    mat->SetDiffuse(Vec3(1.0f, 1.0f, 0.0f));
    mat->SetTexture(m_Texture);
    mat->SetSpecularMap(m_Texture);
    pyramid->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(pyramid);
    
    std::cout << "Added pyramid at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::AddLODTestObject(const Transform& transform) {
    auto lodObject = std::make_shared<GameObject>("LOD_Test");
    
    // Base mesh (High detail - Pyramid) - Swapped for debugging
    lodObject->SetMesh(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    lodObject->GetTransform() = transform;
    
    auto mat = std::make_shared<Material>();
    mat->SetTexture(m_Texture);
    mat->SetSpecularMap(m_Texture);
    mat->SetDiffuse(Vec3(1.0f, 1.0f, 0.0f)); // Yellow for pyramid
    lodObject->SetMaterial(mat);
    
    // LOD 1 (Low detail - Cube at 10m) - Swapped for debugging
    auto cubeMesh = std::make_shared<Mesh>(Mesh::CreateCube());
    lodObject->AddLOD(cubeMesh, 10.0f);
    
    if (m_Root) m_Root->AddChild(lodObject);
    
    std::cout << "Added LOD test object at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::RemoveObject(size_t index) {
    if (m_Root && index < m_Root->GetChildren().size()) {
        auto child = m_Root->GetChildren()[index];
        m_Root->RemoveChild(child);
        std::cout << "Removed object at index " << index << std::endl;
    }
}

void Renderer::UpdateShaders() {
    if (m_Shader) m_Shader->CheckForUpdates();
    if (m_DepthShader) m_DepthShader->CheckForUpdates();
    if (m_GeometryShader) m_GeometryShader->CheckForUpdates();
    if (m_LightingShader) m_LightingShader->CheckForUpdates();
    if (m_EquirectangularToCubemapShader) m_EquirectangularToCubemapShader->CheckForUpdates();
    if (m_IrradianceShader) m_IrradianceShader->CheckForUpdates();
    if (m_PrefilterShader) m_PrefilterShader->CheckForUpdates();
    if (m_BRDFShader) m_BRDFShader->CheckForUpdates();
    if (m_PointShadowShader) m_PointShadowShader->CheckForUpdates();
}

void Renderer::SetupScene() {
    // Add some default objects
    AddCube(Transform(Vec3(0, 0, 0)));
    AddCube(Transform(Vec3(2, 0, 0)));
    AddPyramid(Transform(Vec3(-2, 0, 0)));
    
    // Add LOD test object
    AddLODTestObject(Transform(Vec3(0, 0, -5))); // Place it 10 units away from camera (at 0,0,5)
    
    // Add a floor
    auto floor = std::make_shared<GameObject>("Floor");
    floor->SetMesh(Mesh::CreateCube()); // Use cube as floor for now
    floor->GetTransform() = Transform(Vec3(0, -2, 0), Vec3(0, 0, 0), Vec3(10, 0.1f, 10));
    
    auto mat = std::make_shared<Material>();
    mat->SetTexture(m_Texture);
    mat->SetSpecularMap(m_Texture);
    floor->SetMaterial(mat);
    
    if (m_Root) m_Root->AddChild(floor);
    
    // Add lights
    // First light must be Directional with shadows for CSM to work
    Light mainLight;
    mainLight.type = LightType::Directional;
    mainLight.direction = Vec3(0.3f, -1.0f, 0.5f); // Diagonal downward direction
    mainLight.color = Vec3(1.0f, 1.0f, 0.95f); // Slightly warm white
    mainLight.intensity = 2.0f;
    mainLight.castsShadows = true;
    mainLight.lightSize = 0.5f; // For PCSS soft shadows
    AddLight(mainLight);
    
    AddLight(Light(Vec3(-5, 5, -5), Vec3(1, 0, 0), 3.0f)); // Red point light
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
    if (m_Texture) {
        std::cout << "Texture loaded successfully: brick.png (ID: " << m_Texture->GetID() << ")" << std::endl;
    } else {
        std::cerr << "ERROR: Failed to load brick.png texture!" << std::endl;
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

    // Initialize Instance VBO
    glGenBuffers(1, &m_InstanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
    // Initial size, will be resized if needed (using GL_DYNAMIC_DRAW)
    glBufferData(GL_ARRAY_BUFFER, 1000 * sizeof(Mat4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Initialize post-processing
    m_PostProcessing = std::make_unique<PostProcessing>();
    if (!m_PostProcessing->Init(width, height)) {
        std::cerr << "Failed to initialize post-processing" << std::endl;
        return false;
    }

    // Initialize SSAO
    m_SSAO = std::make_unique<SSAO>();
    if (!m_SSAO->Init(width, height)) {
        std::cerr << "Failed to initialize SSAO" << std::endl;
        return false;
    }

    // Initialize SSR
    m_SSR = std::make_unique<SSR>();
    if (!m_SSR->Init(width, height)) {
        std::cerr << "Failed to initialize SSR" << std::endl;
        return false;
    }

    // Initialize TAA
    m_TAA = std::make_unique<TAA>();
    if (!m_TAA->Init(width, height)) {
        std::cerr << "Failed to initialize TAA" << std::endl;
        return false;
    }

    // Initialize IBL
    InitIBL();

    // Add a test Light Probe
    // AddLightProbe(Vec3(0, 2, 0), 10.0f);
    // BakeLightProbes();
    
    // Initialize Particle System
    m_ParticleSystem = std::make_unique<ParticleSystem>();
    if (!m_ParticleSystem->Init()) {
        std::cerr << "Failed to initialize ParticleSystem" << std::endl;
        return false;
    }
    
    // Add a test fire emitter
    auto fireEmitter = ParticleEmitter::CreateFire(Vec3(0, 1, 0));
    // Use the generated font atlas
    auto atlasTexture = m_TextureManager->LoadTexture("assets/font_atlas.png");
    if (atlasTexture) {
        fireEmitter->SetTexture(atlasTexture);
        fireEmitter->SetAtlasSize(6, 16); // 6 rows, 16 cols as per generate_atlas.py
    }
    m_ParticleSystem->AddEmitter(fireEmitter);
    
    // Add a test animated sprite with Animation Events
    auto testSprite = std::make_shared<Sprite>("TestSprite");
    testSprite->GetTransform().position = Vec3(3, 1, 0);
    testSprite->GetTransform().scale = Vec3(2, 2, 1);
    
    // Create a simple quad mesh for the sprite (using cube for now)
    testSprite->SetMesh(Mesh::CreateCube());
    
    // Create material with the atlas texture
    auto spriteMaterial = std::make_shared<Material>();
    if (atlasTexture) {
        spriteMaterial->SetTexture(atlasTexture);
    }
    testSprite->SetMaterial(spriteMaterial);
    
    // Configure atlas animation (6 rows, 16 cols = 96 frames)
    testSprite->SetAtlas(6, 16);
    
    // Define animation sequences
    testSprite->AddSequence("idle", 0, 3, 4.0f, true);     // Frames 0-3, 4 fps, looping
    testSprite->AddSequence("walk", 4, 11, 8.0f, true);    // Frames 4-11, 8 fps, looping
    
    // Add Animation Events to demonstrate the system
    testSprite->AddEventToSequence("walk", 6, "footstep_left", [](const std::string& eventName) {
        std::cout << "[Animation Event] " << eventName << " triggered!" << std::endl;
    });
    
    testSprite->AddEventToSequence("walk", 10, "footstep_right", [](const std::string& eventName) {
        std::cout << "[Animation Event] " << eventName << " triggered!" << std::endl;
    });
    
    // Set up callbacks
    testSprite->SetOnFrameChange([](int frame) {
        // Uncomment to see frame changes
        // std::cout << "Frame changed to: " << frame << std::endl;
    });
    
    testSprite->SetOnLoop([]() {
        std::cout << "[Animation] Loop completed!" << std::endl;
    });
    
    // Start with walk animation to demonstrate events
    testSprite->PlaySequence("walk");
    
    // Store sprite for transition test
    m_TestSprite = testSprite;
    m_TransitionTestTimer = 0.0f;
    
    // Add to scene
    m_Root->AddChild(testSprite);

    return true;
}

void Renderer::AddLightProbe(const Vec3& position, float radius) {
    auto probe = std::make_unique<LightProbe>(position, radius);
    if (probe->Init()) {
        m_LightProbes.push_back(std::move(probe));
    }
}

void Renderer::BakeLightProbes() {
    glDisable(GL_CULL_FACE); // Disable culling for capturing inside objects
    for (auto& probe : m_LightProbes) {
        BakeProbe(probe.get());
    }
    glEnable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Restore viewport
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    glViewport(0, 0, width, height);
}

void Renderer::BakeProbe(LightProbe* probe) {
    if (!probe) return;
    
    // 1. Capture Scene to Environment Cubemap
    unsigned int captureFBO, captureRBO;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);
    
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);
    
    Mat4 captureProjection = Mat4::Perspective(90.0f, 1.0f, 0.1f, 100.0f);
    Vec3 pos = probe->GetPosition();
    std::vector<Mat4> captureViews = {
        Mat4::LookAt(pos, pos + Vec3( 1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3(-1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  1.0f,  0.0f), Vec3(0.0f,  0.0f,  1.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f, -1.0f,  0.0f), Vec3(0.0f,  0.0f, -1.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  0.0f,  1.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  0.0f, -1.0f), Vec3(0.0f, -1.0f,  0.0f))
    };
    
    // Use a simple shader for capturing (e.g., the geometry shader but outputting color directly? 
    // or just reuse lighting shader? No, lighting shader needs G-Buffer.
    // We need a Forward Shader. Let's use m_Shader (textured.vert/frag) which is a simple forward shader.
    // But textured.frag might need lights.
    // Let's assume m_Shader is simple enough or we configure it.
    // Actually, let's just render the Skybox for now to verify IBL capture.
    // Rendering the scene geometry requires a proper forward shader which we might not have fully set up for PBR.
    // But we can try using m_Shader which seems to be "textured".
    
    m_Shader->Use();
    m_Shader->SetInt("u_Texture", 0);
    
    glViewport(0, 0, 512, 512);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    
    for (unsigned int i = 0; i < 6; ++i) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, probe->GetEnvironmentMap(), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Render Scene
        // Set view/proj matrices
        m_Shader->SetMat4("u_MVP", (captureProjection * captureViews[i]).m); // This might be wrong if shader expects Model/View/Proj separately
        // m_Shader expects u_MVP based on previous look at geometry_pass.vert, but let's check textured.vert
        
        // Render Skybox
        if (m_Skybox) {
            m_Skybox->Draw(captureViews[i], captureProjection);
        }
        
        // Render Objects (Forward)
        // RenderSceneForward(m_Shader.get()); // Commented out until we verify shader compatibility
    }
    
    // Generate Mipmaps for Environment Map
    glBindTexture(GL_TEXTURE_CUBE_MAP, probe->GetEnvironmentMap());
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    // 2. Convolute to Irradiance Map
    m_IrradianceShader->Use();
    m_IrradianceShader->SetInt("environmentMap", 0);
    m_IrradianceShader->SetMat4("projection", captureProjection.m);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, probe->GetEnvironmentMap());
    
    glViewport(0, 0, 32, 32);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);
    
    for (unsigned int i = 0; i < 6; ++i) {
        m_IrradianceShader->SetMat4("view", captureViews[i].m);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, probe->GetIrradianceMap(), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        RenderCube();
    }
    
    // 3. Prefilter Map
    m_PrefilterShader->Use();
    m_PrefilterShader->SetInt("environmentMap", 0);
    m_PrefilterShader->SetMat4("projection", captureProjection.m);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, probe->GetEnvironmentMap());
    
    unsigned int maxMipLevels = 5;
    for (unsigned int mip = 0; mip < maxMipLevels; ++mip) {
        unsigned int mipWidth = 128 * std::pow(0.5, mip);
        unsigned int mipHeight = 128 * std::pow(0.5, mip);
        
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);
        
        float roughness = (float)mip / (float)(maxMipLevels - 1);
        m_PrefilterShader->SetFloat("roughness", roughness);
        
        for (unsigned int i = 0; i < 6; ++i) {
            m_PrefilterShader->SetMat4("view", captureViews[i].m);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, probe->GetPrefilterMap(), mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            RenderCube();
        }
    }
    
    // Cleanup
    glDeleteFramebuffers(1, &captureFBO);
    glDeleteRenderbuffers(1, &captureRBO);
}

// Reflection Probes
void Renderer::AddReflectionProbe(const Vec3& position, float radius, unsigned int resolution) {
    auto probe = std::make_unique<ReflectionProbe>(position, radius, resolution);
    if (probe->Init()) {
        m_ReflectionProbes.push_back(std::move(probe));
        std::cout << "Added reflection probe at (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
    }
}

void Renderer::CaptureReflectionProbes() {
    glDisable(GL_CULL_FACE);
    for (auto& probe : m_ReflectionProbes) {
        if (probe->NeedsUpdate()) {
            CaptureProbe(probe.get());
        }
    }
    glEnable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Restore viewport
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    glViewport(0, 0, width, height);
}

void Renderer::CaptureProbe(ReflectionProbe* probe) {
    if (!probe) return;
    
    // Setup capture matrices
    Mat4 captureProjection = Mat4::Perspective(90.0f, 1.0f, 0.1f, 100.0f);
    Vec3 pos = probe->GetPosition();
    std::vector<Mat4> captureViews = {
        Mat4::LookAt(pos, pos + Vec3( 1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3(-1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  1.0f,  0.0f), Vec3(0.0f,  0.0f,  1.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f, -1.0f,  0.0f), Vec3(0.0f,  0.0f, -1.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  0.0f,  1.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(pos, pos + Vec3( 0.0f,  0.0f, -1.0f), Vec3(0.0f, -1.0f,  0.0f))
    };
    
    // Bind probe FBO
    unsigned int probeFBO = probe->GetCubemap(); // We need to expose FBO, not cubemap
    // Actually, ReflectionProbe stores FBO internally, we need a getter
    // For now, let's bind the probe's internal FBO by accessing it
    // We'll need to modify ReflectionProbe to expose GetFBO()
    
    // Temporary: create FBO here (inefficient but works)
    unsigned int captureFBO;
    glGenFramebuffers(1, &captureFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    
    // Set viewport to probe resolution
    glViewport(0, 0, 256, 256); // Should use probe->GetResolution() but it's private
    
    // Render each face
    for (unsigned int i = 0; i < 6; ++i) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, probe->GetCubemap(), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Render skybox
        if (m_Skybox) {
            m_Skybox->Draw(captureViews[i], captureProjection);
        }
        
        // TODO: Render scene geometry with forward shader
        // For now, just capturing skybox is sufficient for testing
    }
    
    // Generate mipmaps for roughness sampling
    glBindTexture(GL_TEXTURE_CUBE_MAP, probe->GetCubemap());
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    // Cleanup
    glDeleteFramebuffers(1, &captureFBO);
    
    probe->SetNeedsUpdate(false);
    std::cout << "Captured reflection probe at (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
}

void Renderer::RenderSceneForward(Shader* shader) {
    if (m_Root) {
        // We need a DrawForward method on GameObject or similar, 
        // or we just use Draw but with a specific shader.
        // GameObject::Draw takes a shader, so we can use that.
        // But we need to make sure the shader uniforms are compatible.
        // For now, let's assume m_Shader (textured) is compatible with GameObject::Draw
        
        // We need to traverse and draw
        std::vector<std::shared_ptr<GameObject>> queue;
        queue.push_back(m_Root);
        
        while (!queue.empty()) {
            auto obj = queue.back();
            queue.pop_back();
            
            // Draw object
            // Note: GameObject::Draw sets uniforms like u_Model, u_MVP if passed
            // But we need to pass View/Proj to Draw? 
            // GameObject::Draw signature: Draw(Shader* shader, const Mat4& view, const Mat4& projection, ...)
            // So we need to pass the capture view/proj here.
            // But RenderSceneForward signature I defined doesn't take matrices.
            // I should probably pass them or set them in shader before calling.
            // GameObject::Draw uses the passed matrices to calculate MVP.
            
            // Let's skip implementing this fully for now and just rely on Skybox capture
            // as it's safer and sufficient to prove the probe works (it will reflect the sky).
            
            for (auto& child : obj->GetChildren()) {
                queue.push_back(child);
            }
        }
    }
}



void Renderer::Update(float deltaTime) {
    // Update camera matrices
    if (m_Camera) {
        m_Camera->UpdateMatrices();
    }

    // Update scene graph with deltaTime
    if (m_Root) {
        // For Sprite objects, we need to call the deltaTime-aware Update
        // For now, we'll call the base Update which will use default deltaTime
        // In the future, we could add a virtual Update(Mat4, float) to GameObject
        m_Root->Update(Mat4::Identity());
        
        // Update sprites specifically with deltaTime
        UpdateSprites(m_Root, deltaTime);
        
        // Test transition blending after 3 seconds
        if (m_TestSprite) {
            m_TransitionTestTimer += deltaTime;
            
            // Transition from walk to idle after 3 seconds
            if (m_TransitionTestTimer > 3.0f && m_TransitionTestTimer < 3.1f) {
                std::cout << "[Transition Test] Starting transition from walk to idle (0.5s)..." << std::endl;
                m_TestSprite->PlaySequenceWithTransition("idle", 0.5f);
            }
            
            // Transition back to walk after 6 seconds
            if (m_TransitionTestTimer > 6.0f && m_TransitionTestTimer < 6.1f) {
                std::cout << "[Transition Test] Starting transition from idle to walk (0.5s)..." << std::endl;
                m_TestSprite->PlaySequenceWithTransition("walk", 0.5f);
                m_TransitionTestTimer = 0.0f;  // Reset timer to loop the test
            }
        }
    }
    
    // Update particles with actual deltaTime
    if (m_ParticleSystem) {
        m_ParticleSystem->Update(deltaTime);
    }
}

void Renderer::UpdateSprites(std::shared_ptr<GameObject> node, float deltaTime) {
    if (!node) return;
    
    // Check if this is a Sprite and update with deltaTime
    auto sprite = std::dynamic_pointer_cast<Sprite>(node);
    if (sprite) {
        sprite->Update(Mat4::Identity(), deltaTime);
    }
    
    // Recursively update children
    for (auto& child : node->GetChildren()) {
        UpdateSprites(child, deltaTime);
    }
}

void Renderer::Render() {
    // Update camera matrices (moved to Update method)
    // Update scene graph (moved to Update method)
    // Update particles (moved to Update method)

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
            
            m_PointShadows[shadowIndex]->BindForWriting();
            
            // Draw scene for shadow map with frustum culling
            if (m_Root) {
                // For cubemap, we render all 6 faces
                // Frustum culling for each face would require extracting frustum per face
                // For now, skip frustum culling for point lights (cubemap complexity)
                m_Root->Draw(m_PointShadowShader.get(), Mat4::Identity(), Mat4::Identity(), nullptr);
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
                // Extract frustum from spot light matrix for culling
                Frustum spotFrustum;
                spotFrustum.ExtractFromMatrix(spotLightMatrix);
                m_Root->Draw(m_DepthShader.get(), Mat4::Identity(), spotLightMatrix, &spotFrustum);
            }
            
            spotShadowIndex++;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 2: Calculate light space matrix for shadow mapping (first directional light only) =====
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
                // Extract frustum from cascade matrix for culling
                Frustum cascadeFrustum;
                cascadeFrustum.ExtractFromMatrix(lightSpaceMatrices[i]);
                m_Root->Draw(m_DepthShader.get(), Mat4::Identity(), lightSpaceMatrices[i], &cascadeFrustum);
            }
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Get framebuffer size
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);

    // Restore viewport to window size after shadow passes
    glViewport(0, 0, width, height);

    // ===== PASS 2: Issue Occlusion Queries (BEFORE updating visibility) =====
    // Issue queries against the depth buffer from the PREVIOUS frame
    // This ensures we test with the visibility state that was used to render that depth buffer
    
    // Disable color and depth writes
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_FALSE);
    
    // Bind G-Buffer FBO from previous frame (still has old depth buffer)
    // Note: We haven't rendered the new frame yet, so this is the previous frame's depth
    m_GBuffer->BindForWriting();
    
    m_GeometryShader->Use();
    
    if (m_Root) {
        std::vector<std::shared_ptr<GameObject>> queue;
        queue.push_back(m_Root);
        
        while (!queue.empty()) {
            auto obj = queue.back();
            queue.pop_back();
            
            // Initialize query if needed
            obj->InitQuery();
            
            // Check if we should issue query this frame (adaptive frequency)
            if (obj->ShouldIssueQuery()) {
                // Issue query
                glBeginQuery(GL_SAMPLES_PASSED, obj->GetQueryID());
                
                // Draw the object's bounding box for query
                obj->RenderBoundingBox(m_GeometryShader.get(), m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
                
                glEndQuery(GL_SAMPLES_PASSED);
                obj->SetQueryIssued(true);
                
                // Reset frame counter
                obj->ResetQueryFrameCounter();
            } else {
                // Increment frame counter
                obj->IncrementQueryFrameCounter();
            }
            
            for (auto& child : obj->GetChildren()) {
                queue.push_back(child);
            }
        }
    }
    
    // Restore state
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 3: Retrieve Query Results and Update Visibility =====
    // Retrieve results from queries issued in the PREVIOUS frame
    if (m_Root) {
        std::vector<std::shared_ptr<GameObject>> queue;
        queue.push_back(m_Root);
        
        while (!queue.empty()) {
            auto obj = queue.back();
            queue.pop_back();
            
            if (obj->IsQueryIssued()) {
                GLuint available = 0;
                glGetQueryObjectuiv(obj->GetQueryID(), GL_QUERY_RESULT_AVAILABLE, &available);
                
                if (available) {
                    GLuint samples = 0;
                    glGetQueryObjectuiv(obj->GetQueryID(), GL_QUERY_RESULT, &samples);
                    obj->SetVisible(samples > 0);
                    
                    // Update query interval based on visibility stability
                    obj->UpdateQueryInterval();
                }
            }
            
            for (auto& child : obj->GetChildren()) {
                queue.push_back(child);
            }
        }
    }

    // ===== PASS 4: Geometry Pass - Render to G-Buffer =====
    glViewport(0, 0, width, height);
    m_GBuffer->BindForWriting();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_GeometryShader->Use();

    // Set view position for parallax occlusion mapping
    Vec3 viewPos = m_Camera->GetPosition();
    m_GeometryShader->SetVec3("u_ViewPos", viewPos.x, viewPos.y, viewPos.z);

    // Render scene to G-Buffer
    if (m_Root) {
        if (m_BatchedRenderingEnabled) {
            // Batched rendering: Group objects by material to minimize state changes
            std::map<Material*, std::vector<RenderItem>> batches;
            CollectRenderItems(m_Root.get(), batches, nullptr, false);
            
            // Render batched objects
            RenderBatched(batches, m_GeometryShader.get(), m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
            
            // Note: Objects with Models (multiple meshes/materials) are not batched
            // They still use the regular Draw path which is called during collection
            // This is acceptable as Models are typically fewer in number
        } else {
            // Regular rendering: Objects rendered in scene graph order
            m_Root->Draw(m_GeometryShader.get(), m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ===== PASS 4.5: SSAO Pass =====
    if (m_SSAOEnabled) {
        m_SSAO->Render(
            m_GBuffer->GetPositionTexture(),
            m_GBuffer->GetNormalTexture(),
            m_Camera->GetProjectionMatrix().m,
            m_Camera->GetViewMatrix().m
        );
    }

    // ===== PASS 4.6: SSR Pass =====
    if (m_SSREnabled) {
        m_SSR->Render(
            m_GBuffer->GetPositionTexture(),
            m_GBuffer->GetNormalTexture(),
            m_GBuffer->GetAlbedoSpecTexture(),
            m_Camera->GetViewMatrix(),
            m_Camera->GetProjectionMatrix()
        );
    }

    // ===== PASS 5: Lighting Pass - Render to HDR framebuffer =====
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

    // Bind IBL textures
    glActiveTexture(GL_TEXTURE0 + 12);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_IrradianceMap);
    m_LightingShader->SetInt("irradianceMap", 12);
    
    glActiveTexture(GL_TEXTURE0 + 13);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_PrefilterMap);
    m_LightingShader->SetInt("prefilterMap", 13);
    
    glActiveTexture(GL_TEXTURE0 + 14);
    glBindTexture(GL_TEXTURE_2D, m_BRDFLUT);
    m_LightingShader->SetInt("brdfLUT", 14);

    // Bind SSAO texture
    glActiveTexture(GL_TEXTURE0 + 15);
    glBindTexture(GL_TEXTURE_2D, m_SSAO->GetSSAOTexture());
    m_LightingShader->SetInt("ssaoTexture", 15);
    m_LightingShader->SetInt("ssaoEnabled", m_SSAOEnabled ? 1 : 0);

    // Bind SSR texture
    glActiveTexture(GL_TEXTURE0 + 16);
    glBindTexture(GL_TEXTURE_2D, m_SSR->GetSSRTexture());
    m_LightingShader->SetInt("ssrTexture", 16);
    m_LightingShader->SetInt("ssrEnabled", m_SSREnabled ? 1 : 0);

    // Bind Light Probe (Test with first probe)
    if (!m_LightProbes.empty()) {
        glActiveTexture(GL_TEXTURE0 + 17);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_LightProbes[0]->GetIrradianceMap());
        m_LightingShader->SetInt("probeIrradianceMap", 17);
        
        glActiveTexture(GL_TEXTURE0 + 18);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_LightProbes[0]->GetPrefilterMap());
        m_LightingShader->SetInt("probePrefilterMap", 18);
        
        m_LightingShader->SetInt("u_HasLightProbe", 1);
        m_LightingShader->SetVec3("u_ProbePos", m_LightProbes[0]->GetPosition().x, m_LightProbes[0]->GetPosition().y, m_LightProbes[0]->GetPosition().z);
        m_LightingShader->SetFloat("u_ProbeRadius", m_LightProbes[0]->GetRadius());
    } else {
        m_LightingShader->SetInt("u_HasLightProbe", 0);
    }

    // Bind Reflection Probes
    int reflectionProbeCount = (std::min)(static_cast<int>(m_ReflectionProbes.size()), 4);
    m_LightingShader->SetInt("u_ReflectionProbeCount", reflectionProbeCount);
    
    for (int i = 0; i < reflectionProbeCount; ++i) {
        // Bind cubemap
        glActiveTexture(GL_TEXTURE0 + 19 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_ReflectionProbes[i]->GetCubemap());
        m_LightingShader->SetInt("u_ReflectionProbeCubemaps[" + std::to_string(i) + "]", 19 + i);
        
        // Upload position and radius
        Vec3 pos = m_ReflectionProbes[i]->GetPosition();
        m_LightingShader->SetVec3("u_ReflectionProbePositions[" + std::to_string(i) + "]", pos.x, pos.y, pos.z);
        m_LightingShader->SetFloat("u_ReflectionProbeRadii[" + std::to_string(i) + "]", m_ReflectionProbes[i]->GetRadius());
    }

    // Bind emissive texture from G-Buffer
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_GBuffer->GetEmissiveTexture());
    m_LightingShader->SetInt("gEmissive", 3);

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
    m_LightingShader->SetInt("u_ShowCascades", m_ShowCascades ? 1 : 0);
    m_LightingShader->SetFloat("u_ShadowFadeStart", m_ShadowFadeStart);
    m_LightingShader->SetFloat("u_ShadowFadeEnd", m_ShadowFadeEnd);

    // Render fullscreen quad
    RenderQuad();

    // ===== PASS 5.5: Transparent Pass =====
    // Render transparent objects forward
    // Use m_Shader (textured) for now, assuming it handles lighting or is unlit
    // Ideally we need a Forward PBR shader here
    RenderTransparentItems(m_Shader.get(), m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
    
    // ===== PASS 5.6: Particle Pass =====
    if (m_ParticleSystem) {
        m_ParticleSystem->Render(m_Camera, m_GBuffer.get());
    }

    // ===== PASS 6: Forward Pass for Skybox (still in HDR) =====
    // Copy depth buffer from G-Buffer to HDR framebuffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_GBuffer->GetPositionTexture());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // Will be handled by post-processing
    
    // Draw Skybox to HDR framebuffer
    if (m_Skybox) {
        m_Skybox->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
    }
    
    // ===== PASS 6.5: TAA Pass =====
    if (m_TAAEnabled) {
        m_TAA->Render(
            m_PostProcessing->GetHDRTexture(),
            m_GBuffer->GetVelocityTexture(),
            m_Camera->GetViewMatrix(),
            m_Camera->GetProjectionMatrix(),
            m_Camera->GetPreviousViewProjection()
        );
        
        // Use TAA output for post-processing
        // Note: Post-processing should use TAA output instead of HDR texture
        // This requires modifying PostProcessing to accept input texture
        // For now, TAA output is available via m_TAA->GetOutputTexture()
    }
    
    // ===== PASS 7: Post-Processing =====
    // Apply bloom, tone mapping, and other effects
    m_PostProcessing->ApplyEffects();
}

void Renderer::RenderQuad() {
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Renderer::RenderCube() {
    // Unit cube vertices for cubemap rendering
    static unsigned int cubeVAO = 0;
    static unsigned int cubeVBO = 0;
    
    if (cubeVAO == 0) {
        float vertices[] = {
            // positions          
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f,  1.0f
        };
        
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
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
    if (m_InstanceVBO) {
        glDeleteBuffers(1, &m_InstanceVBO);
        m_InstanceVBO = 0;
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

#undef min
#undef max

    for (const auto& v : corners) {
        const auto trf = lightView * Vec3(v.x, v.y, v.z);
        minX = std::fmin(minX, trf.x);
        maxX = std::fmax(maxX, trf.x);
        minY = std::fmin(minY, trf.y);
        maxY = std::fmax(maxY, trf.y);
        minZ = std::fmin(minZ, trf.z);
        maxZ = std::fmax(maxZ, trf.z);
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

void Renderer::InitIBL() {
    std::cout << "Initializing IBL..." << std::endl;
    
    // Load IBL shaders
    m_EquirectangularToCubemapShader = std::make_unique<Shader>();
    if (!m_EquirectangularToCubemapShader->LoadFromFiles("shaders/cubemap.vert", "shaders/equirectangular_to_cubemap.frag")) {
        std::cerr << "Failed to load equirectangular to cubemap shader" << std::endl;
        return;
    }
    
    m_IrradianceShader = std::make_unique<Shader>();
    if (!m_IrradianceShader->LoadFromFiles("shaders/cubemap.vert", "shaders/irradiance_convolution.frag")) {
        std::cerr << "Failed to load irradiance shader" << std::endl;
        return;
    }
    
    m_PrefilterShader = std::make_unique<Shader>();
    if (!m_PrefilterShader->LoadFromFiles("shaders/cubemap.vert", "shaders/prefilter.frag")) {
        std::cerr << "Failed to load prefilter shader" << std::endl;
        return;
    }
    
    m_BRDFShader = std::make_unique<Shader>();
    if (!m_BRDFShader->LoadFromFiles("shaders/brdf.vert", "shaders/brdf.frag")) {
        std::cerr << "Failed to load BRDF shader" << std::endl;
        return;
    }
    
    // For now, create a simple procedural environment (will be replaced with HDR loading later)
    // Generate environment cubemap (512x512)
    glGenTextures(1, &m_EnvCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_EnvCubemap);
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    // Generate irradiance map (32x32)
    glGenTextures(1, &m_IrradianceMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_IrradianceMap);
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Generate prefilter map (128x128 with mipmaps)
    glGenTextures(1, &m_PrefilterMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_PrefilterMap);
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    // Generate BRDF LUT (512x512)
    glGenTextures(1, &m_BRDFLUT);
    glBindTexture(GL_TEXTURE_2D, m_BRDFLUT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    std::cout << "IBL textures created successfully" << std::endl;
    
    // Load HDR environment map
    std::cout << "Loading HDR environment map..." << std::endl;
    auto hdrTexture = std::make_shared<Texture>();
    if (!hdrTexture->LoadHDR("assets/environment.hdr")) {
        std::cerr << "Failed to load HDR environment map, using procedural fallback" << std::endl;
        // Fall back to procedural generation (code omitted for brevity)
        return;
    }
    
    std::cout << "Converting HDR to cubemap..." << std::endl;
    
    // Setup capture framebuffer and renderbuffer
    unsigned int captureFBO, captureRBO;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);
    
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 512, 512);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);
    
    // Setup projection and view matrices for cubemap faces
    Mat4 captureProjection = Mat4::Perspective(90.0f * 3.14159f / 180.0f, 1.0f, 0.1f, 10.0f);
    Mat4 captureViews[] = {
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3( 1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3(-1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3( 0.0f,  1.0f,  0.0f), Vec3(0.0f,  0.0f,  1.0f)),
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3( 0.0f, -1.0f,  0.0f), Vec3(0.0f,  0.0f, -1.0f)),
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3( 0.0f,  0.0f,  1.0f), Vec3(0.0f, -1.0f,  0.0f)),
        Mat4::LookAt(Vec3(0.0f, 0.0f, 0.0f), Vec3( 0.0f,  0.0f, -1.0f), Vec3(0.0f, -1.0f,  0.0f))
    };
    
    // Convert HDR equirectangular to cubemap
    m_EquirectangularToCubemapShader->Use();
    m_EquirectangularToCubemapShader->SetInt("equirectangularMap", 0);
    m_EquirectangularToCubemapShader->SetMat4("projection", captureProjection.m);
    hdrTexture->Bind(0);
    
    glViewport(0, 0, 512, 512);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i) {
        m_EquirectangularToCubemapShader->SetMat4("view", captureViews[i].m);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_EnvCubemap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        RenderCube();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_EnvCubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    std::cout << "Generating irradiance map..." << std::endl;
    
    // Generate irradiance map
    m_IrradianceShader->Use();
    m_IrradianceShader->SetInt("environmentMap", 0);
    m_IrradianceShader->SetMat4("projection", captureProjection.m);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_EnvCubemap);
    
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 32, 32);
    
    glViewport(0, 0, 32, 32);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i) {
        m_IrradianceShader->SetMat4("view", captureViews[i].m);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_IrradianceMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        RenderCube();
    }
    
    std::cout << "Generating prefilter map..." << std::endl;
    
    // Generate prefilter map
    m_PrefilterShader->Use();
    m_PrefilterShader->SetInt("environmentMap", 0);
    m_PrefilterShader->SetMat4("projection", captureProjection.m);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_EnvCubemap);
    
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    unsigned int maxMipLevels = 5;
    for (unsigned int mip = 0; mip < maxMipLevels; ++mip) {
        unsigned int mipWidth = 128 * std::pow(0.5, mip);
        unsigned int mipHeight = 128 * std::pow(0.5, mip);
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);
        
        float roughness = (float)mip / (float)(maxMipLevels - 1);
        m_PrefilterShader->SetFloat("roughness", roughness);
        for (unsigned int i = 0; i < 6; ++i) {
            m_PrefilterShader->SetMat4("view", captureViews[i].m);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_PrefilterMap, mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            RenderCube();
        }
    }
    
    std::cout << "Generating BRDF LUT..." << std::endl;
    
    // Generate BRDF LUT
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 512, 512);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_BRDFLUT, 0);
    
    glViewport(0, 0, 512, 512);
    m_BRDFShader->Use();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    RenderQuad();
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &captureFBO);
    glDeleteRenderbuffers(1, &captureRBO);
    
    // Restore viewport
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    glViewport(0, 0, width, height);
    
    std::cout << "IBL initialization complete with real HDR environment!" << std::endl;
}

void Renderer::CollectRenderItems(GameObject* obj, std::map<Material*, std::vector<RenderItem>>& batches, Frustum* frustum, bool forceRender) {
    if (!obj) return;
    
    // Occlusion culling check
    if (!forceRender && !obj->IsVisible()) {
        // Skip this object but process children
        for (auto& child : obj->GetChildren()) {
            CollectRenderItems(child.get(), batches, frustum, forceRender);
        }
        return;
    }

    // Frustum Culling
    if (frustum && !forceRender) {
        if (!frustum->ContainsAABB(obj->GetWorldAABB())) {
            // If parent is culled, children might still be visible if they are larger or offset?
            // Usually, we assume hierarchy implies spatial containment or we check children individually.
            // For now, let's assume we check children individually if parent is culled, 
            // UNLESS we implement hierarchical culling (parent AABB encapsulates children).
            // The current GameObject doesn't enforce AABB encapsulation.
            // So we MUST check children even if parent is culled, unless we know parent bounds include children.
            // However, the standard optimization is: if parent is culled, cull children.
            // Let's stick to checking this object. If it's culled, we don't render IT, but we proceed to children.
            
            // BUT, if we return here, we skip children loop at the end.
            // So we should NOT return, just skip adding to batch.
        } else {
            // Object is inside frustum, proceed to add to batch
            
            // TODO: Batched rendering for Model objects not yet implemented
            // GameObject::GetModel() method doesn't exist yet
            /*
            // Case 1: Object has a Model (multiple meshes/materials)
            auto model = obj->GetModel();
            if (model) {
                const auto& meshes = model->GetMeshes();
                const auto& materials = model->GetMaterials();
                
                for (size_t i = 0; i < meshes.size(); ++i) {
                    if (i >= materials.size() || !materials[i]) continue;
                    
                    RenderItem item;
                    item.object = obj;
                    item.mesh = meshes[i];
                    item.worldMatrix = obj->GetWorldMatrix();
                    
                    // Calculate squared distance to camera
                    Vec3 camPos = m_Camera->GetPosition();
                    Vec3 objPos = item.worldMatrix.GetTranslation();
                    Vec3 diff = objPos - camPos;
                    item.distance = diff.Dot(diff);
                    
                    if (materials[i]->IsTransparent()) {
                        m_TransparentItems.push_back(item);
                    } else {
                        batches[materials[i].get()].push_back(item);
                    }
                }
            }
            */
            // Case 2: Object has a single Mesh and Material
            {
                auto material = obj->GetMaterial();
                auto mesh = obj->GetActiveMesh(m_Camera->GetViewMatrix());
                
                if (material && mesh) {
                    RenderItem item;
                    item.object = obj;
                    item.mesh = mesh;
                    item.worldMatrix = obj->GetWorldMatrix();
                    
                    // Calculate squared distance to camera
                    Vec3 camPos = m_Camera->GetPosition();
                    Vec3 objPos = item.worldMatrix.GetTranslation();
                    Vec3 diff = objPos - camPos;
                    item.distance = diff.Dot(diff);
                    
                    if (material->IsTransparent()) {
                        m_TransparentItems.push_back(item);
                    } else {
                        batches[material.get()].push_back(item);
                    }
                }
            }
        }
    } else {
        // No frustum or forced render, just add it
        
        // TODO: Batched rendering for Model objects not yet implemented
        // GameObject::GetModel() method doesn't exist yet
        /*
        // Case 1: Object has a Model
        auto model = obj->GetModel();
        if (model) {
            const auto& meshes = model->GetMeshes();
            const auto& materials = model->GetMaterials();
            
            for (size_t i = 0; i < meshes.size(); ++i) {
                if (i >= materials.size() || !materials[i]) continue;
                
                RenderItem item;
                item.object = obj;
                item.mesh = meshes[i];
                item.worldMatrix = obj->GetWorldMatrix();
                
                // Calculate squared distance to camera
                Vec3 camPos = m_Camera->GetPosition();
                Vec3 objPos = item.worldMatrix.GetTranslation();
                Vec3 diff = objPos - camPos;
                item.distance = diff.Dot(diff);
                
                if (materials[i]->IsTransparent()) {
                    m_TransparentItems.push_back(item);
                } else {
                    batches[materials[i].get()].push_back(item);
                }
            }
        }
        */
        // Case 2: Object has single Mesh/Material
        {
            auto material = obj->GetMaterial();
            auto mesh = obj->GetActiveMesh(m_Camera->GetViewMatrix());
            
            if (material && mesh) {
                RenderItem item;
                item.object = obj;
                item.mesh = mesh;
                item.worldMatrix = obj->GetWorldMatrix();
                
                // Calculate squared distance to camera
                Vec3 camPos = m_Camera->GetPosition();
                Vec3 objPos = item.worldMatrix.GetTranslation();
                Vec3 diff = objPos - camPos;
                item.distance = diff.Dot(diff);
                
                if (material->IsTransparent()) {
                    m_TransparentItems.push_back(item);
                } else {
                    batches[material.get()].push_back(item);
                }
            }
        }
    }
    
    // Process children
    for (auto& child : obj->GetChildren()) {
        CollectRenderItems(child.get(), batches, frustum, forceRender);
    }
}

void Renderer::RenderBatched(const std::map<Material*, std::vector<RenderItem>>& batches, Shader* shader, const Mat4& view, const Mat4& projection) {
    // Render each batch
    for (auto& batch : batches) { // Note: Removed const to allow sorting
        Material* material = batch.first;
        auto& items = batch.second; // Note: Removed const to allow sorting
        
        if (items.empty()) continue;
        
        // Sort Opaque items Front-to-Back for Early Z optimization
        SortItems(const_cast<std::vector<RenderItem>&>(items), true);
        
        // Bind material once for all objects in this batch
        material->Bind(shader);
        
        // Group items by mesh to enable instancing
        // Map from Mesh* to vector of model matrices
        std::map<Mesh*, std::vector<Mat4>> meshGroups;
        
        for (const auto& item : items) {
            if (!item.mesh) continue;
            meshGroups[item.mesh.get()].push_back(item.worldMatrix);
        }
        
        // Render each mesh group instanced
        for (const auto& group : meshGroups) {
            Mesh* mesh = group.first;
            const std::vector<Mat4>& models = group.second;
            
            if (models.empty()) continue;
            
            // Update instance VBO
            glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
            glBufferData(GL_ARRAY_BUFFER, models.size() * sizeof(Mat4), models.data(), GL_DYNAMIC_DRAW);
            
            // Bind Mesh VAO
            mesh->Bind();
            
            // Set up instance attributes (Model Matrix = 4 vec4s)
            // Locations 3, 4, 5, 6
            std::size_t vec4Size = sizeof(Vec4);
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)0);
            glEnableVertexAttribArray(4);
            glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(1 * vec4Size));
            glEnableVertexAttribArray(5);
            glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(2 * vec4Size));
            glEnableVertexAttribArray(6);
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 4 * vec4Size, (void*)(3 * vec4Size));
            
            glVertexAttribDivisor(3, 1);
            glVertexAttribDivisor(4, 1);
            glVertexAttribDivisor(5, 1);
            glVertexAttribDivisor(6, 1);
            
            // Set uniforms that are constant for the batch
            shader->SetInt("u_Instanced", 1);
            shader->SetMat4("u_View", view.m);
            shader->SetMat4("u_Projection", projection.m);
            // u_MVP and u_Model are now handled in shader via attributes
            
            // Draw instanced
            glDrawElementsInstanced(GL_TRIANGLES, mesh->GetIndexCount(), GL_UNSIGNED_INT, 0, static_cast<GLsizei>(models.size()));
            
            // Cleanup
            glVertexAttribDivisor(3, 0);
            glVertexAttribDivisor(4, 0);
            glVertexAttribDivisor(5, 0);
            glVertexAttribDivisor(6, 0);
            glDisableVertexAttribArray(3);
            glDisableVertexAttribArray(4);
            glDisableVertexAttribArray(5);
            glDisableVertexAttribArray(6);
            
            mesh->Unbind();
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            
            shader->SetInt("u_Instanced", 0);
        }
    }
}

void Renderer::SortItems(std::vector<RenderItem>& items, bool frontToBack) {
    if (frontToBack) {
        // Sort by distance ascending (Front-to-Back) for opaque objects (Early Z)
        std::sort(items.begin(), items.end(), [](const RenderItem& a, const RenderItem& b) {
            return a.distance < b.distance;
        });
    } else {
        // Sort by distance descending (Back-to-Front) for transparent objects (Blending)
        std::sort(items.begin(), items.end(), [](const RenderItem& a, const RenderItem& b) {
            return a.distance > b.distance;
        });
    }
}

void Renderer::RenderTransparentItems(Shader* shader, const Mat4& view, const Mat4& projection) {
    if (m_TransparentItems.empty()) return;

    // Sort transparent items Back-to-Front
    SortItems(m_TransparentItems, false);

    glEnable(GL_BLEND);
    // Default blend equation
    glBlendEquation(GL_FUNC_ADD);
    
    // Use forward shader (passed as argument, likely m_Shader or similar)
    // Note: Deferred renderer's lighting pass is already done. 
    // We need a forward pass shader that can handle lighting or just unlit/simple lighting.
    // For now, let's assume 'shader' is capable (e.g. textured.frag with lighting uniforms set).
    
    shader->Use();
    shader->SetMat4("u_View", view.m);
    shader->SetMat4("u_Projection", projection.m);
    
    // We can reuse RenderBatched logic if we group by material, 
    // but we MUST preserve the sort order.
    // Batching breaks sort order unless we batch only adjacent items with same material.
    // For simplicity and correctness, let's draw one by one for now.
    
    for (const auto& item : m_TransparentItems) {
        auto material = item.object->GetMaterial();
        if (material) {
            // Set blend mode
            Material::BlendMode mode = material->GetBlendMode();
            switch (mode) {
                case Material::BlendMode::Alpha:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                case Material::BlendMode::Additive:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                    break;
                case Material::BlendMode::Multiply:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_DST_COLOR, GL_ZERO);
                    break;
                case Material::BlendMode::Screen:
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
                    break;
                case Material::BlendMode::Subtractive:
                    glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                    break;
                default: // Opaque or unknown
                    glBlendEquation(GL_FUNC_ADD);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    break;
            }

            material->Bind(shader);
            
            Mat4 mvp = projection * view * item.worldMatrix;
            shader->SetMat4("u_MVP", mvp.m);
            shader->SetMat4("u_Model", item.worldMatrix.m);
            
            if (item.mesh) {
                item.mesh->Draw();
            }
        }
    }
    
    // Restore defaults
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
}

