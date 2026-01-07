#include "Application.h"
#include "ImGuiManager.h"
#include "GLTFLoader.h"
#include "BlendTreeEditor.h"
#include "AudioSystem.h"
#include <iostream>
#include <algorithm>
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"

Application::Application()
    : m_LastFrameTime(0.0f)
    , m_FPS(0.0f)
    , m_FrameCount(0.0f)
    , m_FPSTimer(0.0f)
    , m_SelectedObjectIndex(-1)
{
}

Application::~Application() {
    if (m_PhysicsSystem) {
        m_PhysicsSystem->Shutdown();
    }
    AudioSystem::Get().Shutdown();
    RemoteProfiler::Instance().Shutdown();
}

void Application::Shutdown() {
    if (m_PhysicsSystem) {
        m_PhysicsSystem->Shutdown();
    }
    AudioSystem::Get().Shutdown();
    RemoteProfiler::Instance().Shutdown();
    m_Running = false;
}

bool Application::Init() {
    // Initialize remote profiler (telemetry server on port 8080)
    RemoteProfiler::Instance().Initialize(8080);
    std::cout << "Remote Profiler initialized. View at: http://localhost:8080" << std::endl;
    
    // Create window
    m_Window = std::make_unique<Window>(800, 600, "Game Engine");
    
    if (!m_Window->Init()) {
        std::cerr << "Failed to initialize window" << std::endl;
        return false;
    }

    // Create and initialize renderer
    m_Renderer = std::make_unique<Renderer>();
    if (!m_Renderer->Init()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return false;
    }

    // Create camera
    m_Camera = std::make_unique<Camera>(Vec3(0, 0, 5), 45.0f, 800.0f / 600.0f);
    m_Renderer->SetCamera(m_Camera.get());

    // Initialize Gizmo Manager
    m_GizmoManager = std::make_shared<GizmoManager>();
    m_Renderer->SetGizmoManager(m_GizmoManager);
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Initialize text rendering
    m_Text = std::make_unique<Text>();
    if (!m_Text->Init("assets/font_atlas.png", 800, 600)) {
        std::cerr << "Failed to initialize text system" << std::endl;
        return false;
    }

    // Initialize ImGui
    m_ImGui = std::make_unique<ImGuiManager>();
    if (!m_ImGui->Init(m_Window->GetGLFWWindow())) {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        return false;
    }

    // Initialize Preview Renderer
    m_PreviewRenderer = std::make_unique<PreviewRenderer>();
    if (!m_PreviewRenderer->Init(512, 512)) {
        std::cerr << "Failed to initialize PreviewRenderer" << std::endl;
        return false;
    }

    // Initialize Blend Tree Editor
    m_BlendTreeEditor = std::make_unique<BlendTreeEditor>();

    // Audio System
    if (!AudioSystem::Get().Initialize()) {
        std::cerr << "Failed to initialize Audio System" << std::endl;
        // Check if critical? maybe just log
    }

    // Initialize Physics System
    m_PhysicsSystem = std::make_unique<PhysicsSystem>();
    m_PhysicsSystem->Initialize(Vec3(0, -9.81f, 0)); // Standard Earth gravity
    std::cout << "Physics System initialized with Bullet3D" << std::endl;

    // Initialize ECS Manager
    m_EntityManager = std::make_unique<EntityManager>();
    // Register core systems
    m_EntityManager->AddSystem<TransformSystem>();
    m_EntityManager->AddSystem<PhysicsSystem>();
    m_EntityManager->AddSystem<RenderSystem>();
    m_EntityManager->AddSystem<CollisionSystem>();
    m_EntityManager->AddSystem<LifetimeSystem>();
    std::cout << "ECS Manager initialized with core systems" << std::endl;

    // Initialize Asset Hot-Reload Manager
    m_HotReloadManager = std::make_unique<AssetHotReloadManager>();
    m_HotReloadManager->Initialize(m_Renderer.get(), m_Renderer->GetTextureManager());
    m_HotReloadManager->SetEnabled(true);  // Enable for editor
    m_HotReloadManager->WatchShaderDirectory("shaders/");
    m_HotReloadManager->WatchTextureDirectory("assets/");
    std::cout << "Asset Hot-Reload Manager initialized" << std::endl;

    m_Running = true;
    return true;
}

void Application::Run() {
    m_LastFrameTime = static_cast<float>(glfwGetTime());
    
    while (m_Running && !m_Window->ShouldClose()) {
        // Begin frame profiling
        Profiler::Instance().BeginFrame();
        
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - m_LastFrameTime;
        m_LastFrameTime = currentTime;

        Update(deltaTime);
        Render();
        
        // End frame profiling and update telemetry
        Profiler::Instance().EndFrame();
        PerformanceMonitor::Instance().Update();
        RemoteProfiler::Instance().Update();
        
        m_Window->SwapBuffers();
        m_Window->PollEvents();
    }
}

void Application::Update(float deltaTime) {
    SCOPED_PROFILE("Application::Update");
    
    // Update hot-reload system
    if (m_HotReloadManager) {
        m_HotReloadManager->Update();
    }
    
    // Calculate FPS
    m_FrameCount++;
    m_FPSTimer += deltaTime;
    if (m_FPSTimer >= 1.0f) {
        m_FPS = m_FrameCount / m_FPSTimer;
        m_FrameCount = 0.0f;
        m_FPSTimer = 0.0f;
    }

    // Update Gizmo Manager
    if (m_GizmoManager && m_Camera) {
        m_GizmoManager->Update(deltaTime);
        // Process Input for Gizmos (intercepts if dragging)
        m_GizmoManager->ProcessInput(m_Window->GetGLFWWindow(), deltaTime, *m_Camera);
        
        // Disable camera input if using gizmo
        if (m_GizmoManager->IsDragging()) {
             // We can hack this by "eating" the delta time for camera or similar?
             // Or explicitly disabling camera input. 
             // Camera::ProcessInput doesn't have an enable/disable flag visible here.
             // But we can check before calling Camera::ProcessInput below.
        }
    }

    // Update ECS (core game logic)
    {
        SCOPED_PROFILE("ECS::Update");
        if (m_EntityManager) {
            m_EntityManager->Update(deltaTime);
        }
    }

    // Update camera with collision detection
    {
        SCOPED_PROFILE("Camera::Update");
        if (m_Camera) {
        Vec3 oldPos = m_Camera->GetPosition();
        
        // Only process camera input if NOT dragging a gizmo
        bool isGizmoDragging = m_GizmoManager && m_GizmoManager->IsDragging();
        if (!isGizmoDragging) {
            m_Camera->ProcessInput(m_Window->GetGLFWWindow(), deltaTime);
        }
        
        Vec3 newPos = m_Camera->GetPosition();

        // Create player AABB (size 0.5)
        Vec3 playerSize(0.25f, 0.25f, 0.25f); // Half-extents
        AABB playerBounds(newPos - playerSize, newPos + playerSize);

        // Check collision
        if (m_Renderer->CheckCollision(playerBounds)) {
            // Revert position if collision detected
            // Simple response: just revert to old position
            // Ideally we would slide along the wall, but this prevents walking through objects
        if (m_Renderer->CheckCollision(playerBounds)) {
            // Revert position if collision detected
            // Simple response: just revert to old position
            // Ideally we would slide along the wall, but this prevents walking through objects
            m_Camera->SetPosition(oldPos);
        }
        
        // Input: Duplication (Ctrl + D)
        static bool s_CtrlDPressed = false;
        bool ctrlD = (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || 
                      glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) &&
                      glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_D) == GLFW_PRESS;
                      
        if (ctrlD && !s_CtrlDPressed) {
            // Trigger duplication
            if (m_GizmoManager) {
                // We need to get the selected object from GizmoManager? 
                // Currently GizmoManager has m_SelectedObject but no public getter?
                // Actually Application tracks m_SelectedObjectIndex.
                // But better to ask GizmoManager or trust the index if it's synced.
                // Let's assume m_SelectedObjectIndex is reliable for root objects.
                // Wait, GizmoManager supports picking ANY object. m_SelectedObjectIndex in Application.cpp was for the flat list in UI.
                
                // Let's rely on the Selection capability.
                // If GizmoManager has selection, use that.
                // But GizmoManager interface doesn't expose GetSelectedObject().
                
                // Let's use the m_SelectedObjectIndex for now as it's what the UI uses.
                // BUT, if we picked via 3D ray (future), the UI index might be stale if not synced.
                // Let's stick to the UI list for MVP.
                
                auto root = m_Renderer->GetRoot();
                if (root && m_SelectedObjectIndex >= 0) {
                     auto& children = root->GetChildren();
                     if (m_SelectedObjectIndex < children.size()) {
                         auto original = children[m_SelectedObjectIndex];
                         auto clone = original->Clone();
                         m_Renderer->GetRoot()->AddChild(clone);
                         
                         // Select new object
                         // This is tricky because indices shift or append.
                         m_SelectedObjectIndex = (int)children.size() - 1; 
                         m_GizmoManager->SetSelectedObject(clone);
                     }
                }
            }
        }
        s_CtrlDPressed = ctrlD;
        
        Vec3 listenerPos = m_Camera->GetPosition();
        Vec3 listenerFwd = m_Camera->GetFront();
        Vec3 listenerVel = Vec3(0,0,0);
        
        // Calculate camera velocity for fallback
        if (deltaTime > 0.0001f) {
            listenerVel = (listenerPos - m_LastCameraPosition) / deltaTime;
        }
        m_LastCameraPosition = listenerPos;

        // Check for active override
        AudioListener* activeListener = AudioSystem::Get().GetActiveListener();
        if (activeListener && activeListener->IsEnabled()) {
             listenerPos = activeListener->GetPosition();
             listenerFwd = activeListener->GetForward();
             listenerVel = activeListener->GetVelocity();
        }

        AudioSystem::Get().SetListenerPosition(listenerPos);
        AudioSystem::Get().SetListenerDirection(listenerFwd);
        AudioSystem::Get().SetListenerVelocity(listenerVel);
    }
    }
    
    // Update scene (sprites, particles, etc.) with deltaTime
    {
        SCOPED_PROFILE("Renderer::Update");
        if (m_Renderer) {
            m_Renderer->Update(deltaTime);
        }
    }

    // Update physics simulation
    {
        SCOPED_PROFILE("Physics::Update");
        if (m_PhysicsSystem) {
            m_PhysicsSystem->Update(deltaTime);
            
            // Sync rigid bodies to GameObjects
            if (m_Renderer) {
                auto& gameObjects = m_Renderer->GetGameObjects();
                for (auto& obj : gameObjects) {
                    if (obj && obj->GetRigidBody()) {
                        Vec3 physicsPos;
                        Quat physicsRot;
                        obj->GetRigidBody()->SyncTransformFromPhysics(physicsPos, physicsRot);
                        obj->GetTransform().SetPosition(physicsPos);
                        obj->GetTransform().SetRotation(physicsRot);
                    }
                    if (obj && obj->GetKinematicController()) {
                        Vec3 physicsPos;
                        obj->GetKinematicController()->SyncTransformFromPhysics(physicsPos);
                        obj->GetTransform().SetPosition(physicsPos);
                    }
                }
            }
        }
    }

    // Scene Management Input
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F5) == GLFW_PRESS) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F9) == GLFW_PRESS) {
        m_Renderer->LoadScene("assets/scene.txt");
    }
    
    // Shader Hot-Reload (Poll every 1.0s)
    static float shaderTimer = 0.0f;
    shaderTimer += deltaTime;
    if (shaderTimer >= 1.0f) {
        m_Renderer->UpdateShaders();
        shaderTimer = 0.0f;
    }
}

void Application::Render() {
    SCOPED_PROFILE("Application::Render");
    
    // Clear the screen and depth buffer
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render scene
    {
        PROFILE_GPU("Renderer::Render");
        m_Renderer->Render();
        
        // Render Gizmos on top of scene
        if (m_Renderer && m_Camera) {
             m_Renderer->RenderGizmos(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
        }
    }

    // Render UI (HUD)
    {
        SCOPED_PROFILE("UI::Render");
        if (m_Text && m_Camera) {
            // FPS counter
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(m_FPS));
            m_Text->RenderText(fpsText, 10.0f, 10.0f, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

            // Camera position
            Vec3 camPos = m_Camera->GetPosition();
            std::string posText = "Pos: (" + 
                std::to_string(static_cast<int>(camPos.x)) + ", " +
                std::to_string(static_cast<int>(camPos.y)) + ", " +
                std::to_string(static_cast<int>(camPos.z)) + ")";
            m_Text->RenderText(posText, 10.0f, 30.0f, 1.0f, Vec3(1.0f, 1.0f, 1.0f));
        }
    }

    // Render ImGui Editor UI
    {
        SCOPED_PROFILE("ImGui::Render");
        if (m_ImGui) {
            m_ImGui->BeginFrame();
            RenderEditorUI();
            m_ImGui->EndFrame();
        }
    }
}

void Application::RenderEditorUI() {
    // Scene Hierarchy Panel
    ImGui::Begin("Scene Hierarchy");
    
    auto root = m_Renderer->GetRoot();
    if (root) {
        auto& children = root->GetChildren();
        for (size_t i = 0; i < children.size(); ++i) {
            std::string label = children[i]->GetName() + " ##" + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), m_SelectedObjectIndex == static_cast<int>(i))) {
                m_SelectedObjectIndex = static_cast<int>(i);
                
                // Update Gizmo Selection
                if (m_GizmoManager) {
                    m_GizmoManager->SetSelectedObject(children[i]);
                }
            }
        }
        
        ImGui::Separator();
        
        if (ImGui::Button("Add Cube")) {
            m_Renderer->AddCube(Transform(Vec3(0, 0, 0)));
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Pyramid")) {
            m_Renderer->AddPyramid(Transform(Vec3(0, 0, 0)));
        }
        
        if (m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(children.size())) {
            if (ImGui::Button("Delete Selected")) {
                m_Renderer->RemoveObject(m_SelectedObjectIndex);
                m_SelectedObjectIndex = -1;
            }
        }
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Save Scene (F5)")) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (ImGui::Button("Load Scene (F9)")) {
        m_Renderer->LoadScene("assets/scene.txt");
        m_SelectedObjectIndex = -1;
    }
    
    ImGui::Separator();
    if (ImGui::Button("Load Cornell Box")) {
        LoadCornellBox();
    }
    
    ImGui::End();
    
    // Object Inspector Panel
    if (root && m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(root->GetChildren().size())) {
        ImGui::Begin("Object Inspector");
        
        auto object = root->GetChildren()[m_SelectedObjectIndex];
        Transform& transform = object->GetTransform();
        
        ImGui::Text("Object: %s", object->GetName().c_str());
        ImGui::Separator();
        
        // Position
        ImGui::Text("Position");
        ImGui::DragFloat("X##pos", &transform.position.x, 0.1f);
        ImGui::DragFloat("Y##pos", &transform.position.y, 0.1f);
        ImGui::DragFloat("Z##pos", &transform.position.z, 0.1f);
        
        ImGui::Separator();
        
        // Rotation
        ImGui::Text("Rotation");
        ImGui::DragFloat("X##rot", &transform.rotation.x, 1.0f);
        ImGui::DragFloat("Y##rot", &transform.rotation.y, 1.0f);
        ImGui::DragFloat("Z##rot", &transform.rotation.z, 1.0f);
        
        ImGui::Separator();
        
        // Scale
        ImGui::Text("Scale");
        ImGui::DragFloat("X##scl", &transform.scale.x, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Y##scl", &transform.scale.y, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Z##scl", &transform.scale.z, 0.1f, 0.1f, 10.0f);
        
        ImGui::Separator();

            // Audio System Global Settings (Reverb)
            if (ImGui::CollapsingHeader("Audio Environment")) {
                static float roomSize = 0.5f;
                static float damping = 0.5f;
                static float wet = 0.3f;
                static float dry = 1.0f;
                
                bool changed = false;
                changed |= ImGui::SliderFloat("Reverb Room Size", &roomSize, 0.0f, 1.0f);
                changed |= ImGui::SliderFloat("Reverb Damping", &damping, 0.0f, 1.0f);
                changed |= ImGui::SliderFloat("Reverb Mix (Wet)", &wet, 0.0f, 1.0f);
                changed |= ImGui::SliderFloat("Dry Level", &dry, 0.0f, 1.0f);
                
                if (changed) {
                    AudioSystem::ReverbProperties props;
                    props.roomSize = roomSize;
                    props.damping = damping;
                    props.wetVolume = wet;
                    props.dryVolume = dry;
                    AudioSystem::Get().SetReverbProperties(props);
                }
            }

        // Material Inspector
        auto mat = object->GetMaterial();
        if (mat) {
            // Render and display preview
            if (m_PreviewRenderer) {
                // Create a simple shader for preview (we'll use a basic approach)
                // For now, render the preview - the PreviewRenderer will handle shader internally
                m_PreviewRenderer->RenderPreview(object.get(), nullptr);
                
                // Display preview image
                ImGui::Text("Material Preview");
                ImVec2 previewSize(256, 256);
                ImGui::Image(
                    (void*)(intptr_t)m_PreviewRenderer->GetTextureID(),
                    previewSize,
                    ImVec2(0, 1), // UV coordinates flipped for OpenGL
                    ImVec2(1, 0)
                );
                ImGui::Separator();
            }
            
            ImGui::Text("Material Properties");
            ImGui::Separator();
            
            // Basic Colors
            if (ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen)) {
                Vec3 ambient = mat->GetAmbient();
                if (ImGui::ColorEdit3("Ambient", &ambient.x)) {
                    mat->SetAmbient(ambient);
                }
                
                Vec3 diffuse = mat->GetDiffuse();
                if (ImGui::ColorEdit3("Diffuse", &diffuse.x)) {
                    mat->SetDiffuse(diffuse);
                }
                
                Vec3 specular = mat->GetSpecular();
                if (ImGui::ColorEdit3("Specular", &specular.x)) {
                    mat->SetSpecular(specular);
                }
                
                Vec3 emissive = mat->GetEmissiveColor();
                if (ImGui::ColorEdit3("Emissive", &emissive.x)) {
                    mat->SetEmissiveColor(emissive);
                }
            }
            
            // PBR Properties
            if (ImGui::CollapsingHeader("PBR Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                float shininess = mat->GetShininess();
                if (ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f)) {
                    mat->SetShininess(shininess);
                }
                
                float roughness = mat->GetRoughness();
                if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f)) {
                    mat->SetRoughness(roughness);
                }
                
                float metallic = mat->GetMetallic();
                if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f)) {
                    mat->SetMetallic(metallic);
                }
                
                float heightScale = mat->GetHeightScale();
                if (ImGui::SliderFloat("Height Scale", &heightScale, 0.0f, 1.0f)) {
                    mat->SetHeightScale(heightScale);
                }
                
                float opacity = mat->GetOpacity();
                if (ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f)) {
                    mat->SetOpacity(opacity);
                }
                
                bool isTransparent = mat->IsTransparent();
                if (ImGui::Checkbox("Transparent", &isTransparent)) {
                    mat->SetIsTransparent(isTransparent);
                }
            }
            
            // Texture Maps
            if (ImGui::CollapsingHeader("Texture Maps")) {
                auto texManager = m_Renderer->GetTextureManager();
                if (texManager) {
                    auto textureNames = texManager->GetTextureNames();
                    
                    // Helper lambda for texture selection
                    auto renderTextureCombo = [&](const char* label, std::shared_ptr<Texture> currentTex, auto setter) {
                        std::string currentName = currentTex ? "Loaded Texture" : "None";
                        if (ImGui::BeginCombo(label, currentName.c_str())) {
                            if (ImGui::Selectable("None", !currentTex)) {
                                (mat.get()->*setter)(nullptr);
                            }
                            for (const auto& name : textureNames) {
                                bool isSelected = (currentTex && currentTex == texManager->GetTexture(name));
                                if (ImGui::Selectable(name.c_str(), isSelected)) {
                                    (mat.get()->*setter)(texManager->GetTexture(name));
                                }
                            }
                            ImGui::EndCombo();
                        }
                    };
                    
                    renderTextureCombo("Albedo/Diffuse", mat->GetTexture(), &Material::SetTexture);
                    renderTextureCombo("Normal Map", mat->GetNormalMap(), &Material::SetNormalMap);
                    renderTextureCombo("Specular Map", mat->GetSpecularMap(), &Material::SetSpecularMap);
                    renderTextureCombo("Roughness Map", mat->GetRoughnessMap(), &Material::SetRoughnessMap);
                    renderTextureCombo("Metallic Map", mat->GetMetallicMap(), &Material::SetMetallicMap);
                    renderTextureCombo("AO Map", mat->GetAOMap(), &Material::SetAOMap);
                    renderTextureCombo("ORM Map", mat->GetORMMap(), &Material::SetORMMap);
                    renderTextureCombo("Height Map", mat->GetHeightMap(), &Material::SetHeightMap);
                    renderTextureCombo("Emissive Map", mat->GetEmissiveMap(), &Material::SetEmissiveMap);
                }
            }
            
            // Material Presets
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Material Presets")) {
                static char presetName[128] = "my_material";
                
                // Save Preset
                ImGui::InputText("Preset Name", presetName, sizeof(presetName));
                if (ImGui::Button("Save Preset")) {
                    std::string filepath = "assets/materials/" + std::string(presetName) + ".mat";
                    if (mat->SaveToFile(filepath)) {
                        std::cout << "Preset saved successfully!" << std::endl;
                    }
                }
                
                ImGui::Separator();
                
                // Load Preset - scan for .mat files
                static std::vector<std::string> presetFiles;
                static int selectedPreset = -1;
                
                if (ImGui::Button("Refresh Presets")) {
                    presetFiles.clear();
                    selectedPreset = -1;
                    
                    // Simple directory scan (Windows-specific for now)
                    #ifdef _WIN32
                    WIN32_FIND_DATAA findData;
                    HANDLE hFind = FindFirstFileA("assets/materials/*.mat", &findData);
                    if (hFind != INVALID_HANDLE_VALUE) {
                        do {
                            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                                presetFiles.push_back(findData.cFileName);
                            }
                        } while (FindNextFileA(hFind, &findData));
                        FindClose(hFind);
                    }
                    #endif
                }
                
                if (!presetFiles.empty()) {
                    ImGui::Text("Available Presets:");
                    for (size_t i = 0; i < presetFiles.size(); ++i) {
                        if (ImGui::Selectable(presetFiles[i].c_str(), selectedPreset == (int)i)) {
                            selectedPreset = (int)i;
                        }
                    }
                    
                    if (selectedPreset >= 0 && ImGui::Button("Load Selected Preset")) {
                        std::string filepath = "assets/materials/" + presetFiles[selectedPreset];
                        if (mat->LoadFromFile(filepath, m_Renderer->GetTextureManager())) {
                            std::cout << "Preset loaded successfully!" << std::endl;
                        }
                    }
                } else {
                    ImGui::TextDisabled("No presets found. Click 'Refresh Presets'.");
                }
            }
        }

        ImGui::End();
    }

    // Light Inspector Panel
    ImGui::Begin("Light Inspector");
    
    auto& lights = m_Renderer->GetLights();
    static int selectedLightIndex = -1;

    if (ImGui::Button("Add Light")) {
        m_Renderer->AddLight(Light(Vec3(0, 5, 0)));
    }
    
    bool showCascades = m_Renderer->GetShowCascades();
    if (ImGui::Checkbox("Show CSM Cascades", &showCascades)) {
        m_Renderer->SetShowCascades(showCascades);
    }
    
    float fadeStart = m_Renderer->GetShadowFadeStart();
    if (ImGui::SliderFloat("Shadow Fade Start", &fadeStart, 10.0f, 100.0f)) {
        m_Renderer->SetShadowFadeStart(fadeStart);
    }
    
    float fadeEnd = m_Renderer->GetShadowFadeEnd();
    if (ImGui::SliderFloat("Shadow Fade End", &fadeEnd, 10.0f, 100.0f)) {
        m_Renderer->SetShadowFadeEnd(fadeEnd);
    }
    
    ImGui::Separator();

    for (size_t i = 0; i < lights.size(); ++i) {
        std::string label = "Light " + std::to_string(i);
        if (ImGui::Selectable(label.c_str(), selectedLightIndex == static_cast<int>(i))) {
            selectedLightIndex = static_cast<int>(i);
        }
    }

    if (selectedLightIndex >= 0 && selectedLightIndex < static_cast<int>(lights.size())) {
        ImGui::Separator();
        Light& light = lights[selectedLightIndex];
        
        ImGui::Text("Light Properties");
        
        // Light Type
        const char* lightTypes[] = { "Directional", "Point", "Spot" };
        int currentType = static_cast<int>(light.type);
        if (ImGui::Combo("Type", &currentType, lightTypes, IM_ARRAYSIZE(lightTypes))) {
            light.type = static_cast<LightType>(currentType);
        }

        bool castsShadows = light.castsShadows;
        if (ImGui::Checkbox("Cast Shadows", &castsShadows)) {
            light.castsShadows = castsShadows;
        }
        if (light.castsShadows) {
            if (light.type == LightType::Point) {
                ImGui::SliderFloat("Shadow Softness", &light.shadowSoftness, 1.0f, 5.0f);
            } else {
                ImGui::SliderFloat("Light Size (PCSS)", &light.lightSize, 0.0f, 5.0f);
            }
        }

        if (light.type != LightType::Directional) {
            ImGui::DragFloat("Range", &light.range, 0.5f, 1.0f, 100.0f);
        }
        
        ImGui::DragFloat3("Position", &light.position.x, 0.1f);
        if (light.type != LightType::Point) {
            ImGui::DragFloat3("Direction", &light.direction.x, 0.1f);
        }
        
        ImGui::ColorEdit3("Color", &light.color.x);
        ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 10.0f);
        
        if (light.type != LightType::Directional) {
            ImGui::Text("Attenuation (Advanced)");
            ImGui::DragFloat("Constant", &light.constant, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Linear", &light.linear, 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("Quadratic", &light.quadratic, 0.001f, 0.0f, 1.0f);
        }
        
        if (light.type == LightType::Spot) {
            ImGui::Text("Spotlight");
            ImGui::DragFloat("Cutoff", &light.cutOff, 0.1f, 0.0f, 90.0f);
            ImGui::DragFloat("Outer Cutoff", &light.outerCutOff, 0.1f, 0.0f, 90.0f);
        }
        
        if (ImGui::Button("Delete Light")) {
            m_Renderer->RemoveLight(selectedLightIndex);
            selectedLightIndex = -1;
        }
    }

    ImGui::End();
    
    // Post-Processing Panel
    ImGui::Begin("Post-Processing");
    
    auto postProcessing = m_Renderer->GetPostProcessing();
    if (postProcessing) {
        // Bloom settings
        ImGui::Text("Bloom");
        bool bloomEnabled = postProcessing->IsBloomEnabled();
        if (ImGui::Checkbox("Enable Bloom", &bloomEnabled)) {
            postProcessing->SetBloomEnabled(bloomEnabled);
        }
        
        if (bloomEnabled) {
            float bloomIntensity = postProcessing->GetBloomIntensity();
            if (ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 2.0f)) {
                postProcessing->SetBloomIntensity(bloomIntensity);
            }
            
            float bloomThreshold = postProcessing->GetBloomThreshold();
            if (ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.0f, 5.0f)) {
                postProcessing->SetBloomThreshold(bloomThreshold);
            }
        }
        
        ImGui::Separator();
        
        // Tone mapping settings
        ImGui::Text("Tone Mapping");
        const char* toneMappingModes[] = { "Reinhard", "ACES Filmic" };
        int toneMappingMode = postProcessing->GetToneMappingMode();
        if (ImGui::Combo("Mode", &toneMappingMode, toneMappingModes, IM_ARRAYSIZE(toneMappingModes))) {
            postProcessing->SetToneMappingMode(toneMappingMode);
        }
        
        float exposure = postProcessing->GetExposure();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f)) {
            postProcessing->SetExposure(exposure);
        }
        
        float gamma = postProcessing->GetGamma();
        if (ImGui::SliderFloat("Gamma", &gamma, 1.8f, 2.4f)) {
            postProcessing->SetGamma(gamma);
        }
        
        ImGui::Separator();
        
        // SSAO settings
        ImGui::Text("SSAO (Ambient Occlusion)");
        bool ssaoEnabled = m_Renderer->GetSSAOEnabled();
        if (ImGui::Checkbox("Enable SSAO", &ssaoEnabled)) {
            m_Renderer->SetSSAOEnabled(ssaoEnabled);
        }
        
        if (ssaoEnabled) {
            auto ssao = m_Renderer->GetSSAO();
            if (ssao) {
                float radius = ssao->GetRadius();
                if (ImGui::SliderFloat("SSAO Radius", &radius, 0.1f, 2.0f)) {
                    ssao->SetRadius(radius);
                }
                
                float bias = ssao->GetBias();
                if (ImGui::SliderFloat("SSAO Bias", &bias, 0.001f, 0.1f)) {
                    ssao->SetBias(bias);
                }
            }
        }
    }
    
    
    // Blend Tree Editor
    if (m_BlendTreeEditor) {
        // Try to update binding if selection changed
        if (root && m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(root->GetChildren().size())) {
            auto object = root->GetChildren()[m_SelectedObjectIndex];
            m_BlendTreeEditor->SetAnimator(object->GetAnimator().get());
        } else {
             m_BlendTreeEditor->SetAnimator(nullptr);
        }
        
        m_BlendTreeEditor->Render();
    }
    
    ImGui::End();
    
    // Physics Demos Panel
    ImGui::Begin("Physics Demos (Advanced)");
    
    static int clothWidth = 20;
    static int clothHeight = 20;
    static float clothSpacing = 0.2f;
    static float clothStiffness = 100.0f; // High stiffness for stability
    static float clothDamping = 0.5f;
    
    ImGui::Text("Cloth Simulation");
    ImGui::SliderInt("Width", &clothWidth, 10, 100);
    ImGui::SliderInt("Height", &clothHeight, 10, 100);
    ImGui::SliderFloat("Spacing", &clothSpacing, 0.05f, 0.5f);
    ImGui::SliderFloat("Stiffness", &clothStiffness, 10.0f, 500.0f);
    ImGui::SliderFloat("Damping", &clothDamping, 0.0f, 2.0f);
    
    if (ImGui::Button("Spawn Cloth")) {
        // Create emitter at a high position
        auto emitter = std::make_shared<ParticleEmitter>(Vec3(0, 10, 0), clothWidth * clothHeight);
        
        // Use default texture if available
        auto tex = m_Renderer->GetTextureManager()->GetTexture("assets/brick.png"); // Or some particle tex
        if (tex) emitter->SetTexture(tex);
        
        emitter->SetColorRange(Vec4(1, 1, 1, 1), Vec4(1, 1, 1, 1));
        emitter->SetSizeRange(0.1f, 0.1f);
        
        emitter->InitCloth(clothWidth, clothHeight, clothSpacing, clothStiffness, clothDamping);
        m_Renderer->GetParticleSystem()->AddEmitter(emitter);
        std::cout << "Spawned Cloth Emitter" << std::endl;
    }
    
    ImGui::Separator();
    
    static int sbWidth = 10;
    static int sbHeight = 10;
    static int sbDepth = 10;
    
    ImGui::Text("Soft Body Simulation");
    ImGui::SliderInt("SB Width", &sbWidth, 5, 20);
    ImGui::SliderInt("SB Height", &sbHeight, 5, 20);
    ImGui::SliderInt("SB Depth", &sbDepth, 5, 20);
    
    if (ImGui::Button("Spawn Soft Body")) {
        auto emitter = std::make_shared<ParticleEmitter>(Vec3(2, 10, 0), sbWidth * sbHeight * sbDepth);
         // Use default texture if available
        auto tex = m_Renderer->GetTextureManager()->GetTexture("assets/brick.png"); 
        if (tex) emitter->SetTexture(tex);
        
        emitter->SetColorRange(Vec4(0, 1, 0, 1), Vec4(0, 1, 0, 1)); // Green
        emitter->SetSizeRange(0.1f, 0.1f);
        
        emitter->InitSoftBody(sbWidth, sbHeight, sbDepth, 0.2f, 200.0f, 0.5f);
        m_Renderer->GetParticleSystem()->AddEmitter(emitter);
        std::cout << "Spawned Soft Body Emitter" << std::endl;
    }
    
    ImGui::Separator();
    
    ImGui::Text("Fluid Simulation (SPH)");
    static int fluidParticles = 4096;
    ImGui::SliderInt("Particle Count", &fluidParticles, 1024, 65536);
    
    if (ImGui::Button("Spawn Fluid")) {
         auto emitter = std::make_shared<ParticleEmitter>(Vec3(-2, 5, 0), fluidParticles);
         
         auto tex = m_Renderer->GetTextureManager()->GetTexture("assets/brick.png"); 
         if (tex) emitter->SetTexture(tex);
         
         emitter->SetPhysicsMode(PhysicsMode::SPHFluid);
         emitter->SetColorRange(Vec4(0, 0.5f, 1.0f, 0.8f), Vec4(0, 0.5f, 1.0f, 0.8f)); // Blue
         emitter->SetSizeRange(0.1f, 0.1f);
         emitter->SetSpawnRate(5000.0f); // Dump them fast
         emitter->SetParticleLifetime(99999.0f);
         emitter->SetUseGPUCompute(true);
         
         m_Renderer->GetParticleSystem()->AddEmitter(emitter);
         std::cout << "Spawned Fluid Emitter" << std::endl;
    }
    
    ImGui::End();

    // Particle System Manager
    ImGui::Begin("Particle Manager");
    
    auto particleSystem = m_Renderer->GetParticleSystem();
    if (particleSystem) {
        // Global Stats & Settings
        ImGui::Text("Global Active Particles: %d", particleSystem->GetTotalActiveParticles());
        int globalLimit = particleSystem->GetGlobalParticleLimit();
        if (ImGui::DragInt("Global Limit", &globalLimit, 1000, 1000, 2000000)) {
            particleSystem->SetGlobalParticleLimit(globalLimit);
        }
        
        // Physics Global Env
        ImGui::Separator();
        ImGui::Text("Global Physics");
        Vec3 wind = particleSystem->GetGlobalWind();
        if (ImGui::DragFloat3("Global Wind", &wind.x, 0.1f)) {
            particleSystem->SetGlobalWind(wind);
        }
        
        // Attractors
        auto& attractors = particleSystem->GetAttractors();
        if (ImGui::TreeNode("Attractors")) {
            for (size_t i = 0; i < attractors.size(); ++i) {
                ImGui::PushID((int)i);
                ImGui::Text("Attractor %d", (int)i);
                ImGui::DragFloat3("Pos", &attractors[i].position.x, 0.1f);
                ImGui::DragFloat("Strength", &attractors[i].strength, 1.0f, -100.0f, 100.0f);
                if (ImGui::Button("Remove")) {
                    attractors.erase(attractors.begin() + i);
                    i--;
                }
                ImGui::PopID();
            }
            if (ImGui::Button("Add Attractor")) {
                particleSystem->AddAttractor({Vec3(0,5,0), 10.0f});
            }
            ImGui::TreePop();
        }
        
        ImGui::Separator();
        
        auto& emitters = particleSystem->GetEmitters();
        static int selectedEmitterIndex = -1;
        
        ImGui::Text("Active Emitters: %d", (int)emitters.size());
        
        // List Emitters
        for (size_t i = 0; i < emitters.size(); ++i) {
            std::string label = "Emitter " + std::to_string(i);
            if (emitters[i]->GetParent()) {
                label += " (Attached to " + emitters[i]->GetParent()->GetName() + ")";
            }
            if (ImGui::Selectable(label.c_str(), selectedEmitterIndex == (int)i)) {
                selectedEmitterIndex = (int)i;
            }
        }
        
        ImGui::Separator();
        
        if (selectedEmitterIndex >= 0 && selectedEmitterIndex < (int)emitters.size()) {
            auto emitter = emitters[selectedEmitterIndex];
            
            // Pooling & Priority Settings
            ImGui::Text("Emitter Settings");
            int priority = emitter->GetPriority();
            if (ImGui::SliderInt("Priority", &priority, 0, 10)) {
                emitter->SetPriority(priority);
            }
            
            int maxParticles = emitter->GetMaxParticles();
            if (ImGui::InputInt("Max Particles", &maxParticles)) {
                 // Debounce resizing ideally, but for now apply on enter
                 if (maxParticles > 0) emitter->SetMaxParticles(maxParticles);
            if (ImGui::InputInt("Max Particles", &maxParticles)) {
                 // Debounce resizing ideally, but for now apply on enter
                 if (maxParticles > 0) emitter->SetMaxParticles(maxParticles);
            }
            
            // Physics Props
            float velInherit = emitter->GetVelocityInheritance();
            if (ImGui::SliderFloat("Velocity Inheritance", &velInherit, 0.0f, 5.0f)) {
                emitter->SetVelocityInheritance(velInherit);
            }
            
            // Spawning
            static int burstCount = 50;
            ImGui::InputInt("Burst Count", &burstCount);
            ImGui::SameLine();
            if (ImGui::Button("Burst!")) {
                emitter->Burst(burstCount);
            }
            
                emitter->Burst(burstCount);
            }
            
            // Visuals
            if (ImGui::TreeNode("Visuals")) {
                // Gradient
                ImGui::Text("Color Gradient");
                auto& gradient = emitter->GetGradient();
                for (size_t i = 0; i < gradient.size(); ++i) {
                     ImGui::PushID((int)i);
                     ImGui::DragFloat("T", &gradient[i].t, 0.01f, 0.0f, 1.0f);
                     ImGui::ColorEdit4("Color", &gradient[i].color.r);
                     if (ImGui::Button("X")) {
                         gradient.erase(gradient.begin() + i);
                         i--;
                     }
                     ImGui::PopID();
                }
                if (ImGui::Button("Add Stop")) {
                    emitter->AddGradientStop(1.0f, Vec4(1,1,1,1));
                }
                
                ImGui::Separator();
                
                // Size Curve
                ImGui::Text("Size Curve (Equidistant)");
                auto& curve = emitter->GetSizeCurve();
                // Plot
                if (!curve.empty()) {
                    ImGui::PlotLines("Curve", curve.data(), (int)curve.size(), 0, nullptr, 0.0f, 5.0f, ImVec2(0, 80));
                }
                // Edit points
                for (size_t i = 0; i < curve.size(); ++i) {
                     ImGui::PushID((int)i + 100);
                     std::string label = "Pt " + std::to_string(i);
                     ImGui::SliderFloat(label.c_str(), &curve[i], 0.0f, 5.0f);
                     ImGui::PopID();
                }
                if (ImGui::Button("Add Point")) {
                    emitter->AddSizeCurvePoint(1.0f);
                }
                if (ImGui::Button("Clear Curve")) {
                    curve.clear();
                }
                
                ImGui::TreePop();
            }
            
            // Sub Emitters
            if (ImGui::TreeNode("Sub-Emitters")) {
                auto currentDeath = emitter->GetSubEmitterDeath();
                std::string currentName = currentDeath ? "Linked" : "None";
                if (ImGui::BeginCombo("On Death", currentName.c_str())) {
                    if (ImGui::Selectable("None", !currentDeath)) {
                        emitter->SetSubEmitterDeath(nullptr);
                    }
                    for (size_t i = 0; i < emitters.size(); ++i) {
                        if (emitters[i] == emitter) continue; // Don't link self (infinite loop risk)
                        std::string label = "Emitter " + std::to_string(i);
                        if (ImGui::Selectable(label.c_str(), currentDeath == emitters[i])) {
                            emitter->SetSubEmitterDeath(emitters[i]);
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::TreePop();
            }
            
            bool useLOD = emitter->GetUseLOD();
            if (ImGui::Checkbox("Use LOD System", &useLOD)) {
                emitter->SetUseLOD(useLOD);
            }
            
            if (useLOD) {
                ImGui::Text("Current Distance: %.1f m", emitter->GetCurrentDistance());
                ImGui::Text("Current Level: %d", emitter->GetCurrentLODLevel());
                
                ImGui::Separator();
                ImGui::Text("LOD Levels (Distance -> Settings)");
                
                auto& lods = emitter->GetLODs();
                for (size_t i = 0; i < lods.size(); ++i) {
                    ImGui::PushID((int)i);
                    ImGui::Text("Level %d", (int)i);
                    ImGui::DragFloat("Distance >", &lods[i].distance, 1.0f, 0.0f, 1000.0f);
                    ImGui::SliderFloat("Emission Mult", &lods[i].emissionMultiplier, 0.0f, 1.0f);
                    ImGui::Checkbox("Turbulence", &lods[i].enableTurbulence);
                    ImGui::SameLine();
                    ImGui::Checkbox("Collisions", &lods[i].enableCollisions);
                    ImGui::Checkbox("Trails", &lods[i].enableTrails);
                    ImGui::SameLine();
                    ImGui::Checkbox("Shadows", &lods[i].enableShadows);
                    
                    if (ImGui::Button("Remove")) {
                        lods.erase(lods.begin() + i);
                        i--;
                    }
                    ImGui::Separator();
                    ImGui::PopID();
                }
                
                if (ImGui::Button("Add LOD Level")) {
                    EmitterLOD newLOD;
                    // Auto-guess distance
                    if (!lods.empty()) newLOD.distance = lods.back().distance + 10.0f;
                    else newLOD.distance = 10.0f;
                    
                    lods.push_back(newLOD);
                }
                
                ImGui::SameLine();
                if (ImGui::Button("Sort by Distance")) {
                    std::sort(lods.begin(), lods.end(), [](const EmitterLOD& a, const EmitterLOD& b) {
                        return a.distance < b.distance;
                    });
                }
            } else {
                 ImGui::TextDisabled("Enable LOD to configure levels.");
            }
        }
    }
    
    ImGui::End();

    // Asset Hot-Reload Panel
    ImGui::Begin("Asset Hot-Reload");
    
    if (m_HotReloadManager) {
        bool hotReloadEnabled = m_HotReloadManager->IsEnabled();
        if (ImGui::Checkbox("Enable Hot-Reload##main", &hotReloadEnabled)) {
            m_HotReloadManager->SetEnabled(hotReloadEnabled);
        }
        
        ImGui::Separator();
        ImGui::Text("Status:");
        ImGui::SameLine();
        if (m_HotReloadManager->IsEnabled()) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "ACTIVE");
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "INACTIVE");
        }
        
        ImGui::Separator();
        ImGui::Text("Statistics:");
        ImGui::BulletText("Watched Files: %zu", m_HotReloadManager->GetWatchedFileCount());
        ImGui::BulletText("Reloads: %u", m_HotReloadManager->GetReloadCount());
        
        ImGui::Separator();
        if (ImGui::Button("Reset Reload Count")) {
            m_HotReloadManager->ResetReloadCount();
        }
        
        ImGui::Separator();
        ImGui::Text("Watching:");
        ImGui::BulletText("shaders/");
        ImGui::BulletText("assets/");
        
        ImGui::Separator();
        ImGui::TextWrapped("Hot-reload monitors shader and texture files for changes. "
                          "Edit files and they will reload automatically in the editor.");
    }
    
    ImGui::End();
}
