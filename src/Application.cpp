#include "Application.h"
#include "ImGuiManager.h"
#include "GLTFLoader.h"
#include "BlendTreeEditor.h"
#include "AudioSystem.h"
#include "AudioListener.h"
#include "PhysicsSystem.h"
#include "ScriptSystem.h" // Defines LuaScriptSystem (should be renamed ideally)
#include "PythonScriptSystem.h"
#include "CustomScriptSystem.h"
#include "ScriptDebugger.h"
#include <iostream>
#include <algorithm>
#include <GLFW/glfw3.h>
#include <imgui.h>

#ifdef USE_PHYSX
#include "PhysXRigidBody.h"
#include "PhysXShape.h"
#endif
#ifdef USE_BOX2D
#include "Box2DBackend.h"
#endif

static Application* s_Instance = nullptr;

Application::Application()
    : m_LastFrameTime(0.0f)
    , m_FPS(0.0f)
    , m_FrameCount(0.0f)
    , m_FPSTimer(0.0f)
    , m_SelectedObjectIndex(-1)
{
    s_Instance = this;
}

Application::~Application() {
    if (m_PhysicsSystem) {
        m_PhysicsSystem->Shutdown();
    }
    s_Instance = nullptr;
#ifdef USE_PHYSX
    if (m_PhysXBackend) {
        m_PhysXBackend->Shutdown();
    }
#endif
#ifdef USE_BOX2D
    if (m_Box2DBackend) {
        m_Box2DBackend->Shutdown();
    }
#endif
    // ScriptSystem::GetInstance().Shutdown();
}

Application& Application::Get() {
    return *s_Instance;
}

void Application::Shutdown() {
    if (m_PhysicsSystem) {
        m_PhysicsSystem->Shutdown();
    }
    
    // Shutdown Asset Pipeline
    if (m_AssetPipeline) {
        m_AssetPipeline->Shutdown();
        m_AssetPipeline.reset();
    }
    
    AudioSystem::Get().Shutdown();
    RemoteProfiler::Instance().Shutdown();
    // LuaScriptSystem::GetInstance().Shutdown();
    // PythonScriptSystem::GetInstance().Shutdown();
    CustomScriptSystem::GetInstance().Shutdown();
    m_Running = false;
}

bool Application::Init() {
    RemoteProfiler::Instance().Initialize(8080);
    Profiler::Instance().SetEnabled(true);
    std::cout << "Remote Profiler initialized. View at: http://localhost:8080" << std::endl;
    
    // Initialize Script System (Choose one: Lua or Python or C# or Custom)
    // LuaScriptSystem::GetInstance().Init();
    // PythonScriptSystem::GetInstance().Init();
    CustomScriptSystem::GetInstance().Init();

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

    // Initialize Script Debugger UI
    m_ScriptDebuggerUI = std::make_unique<ScriptDebuggerUI>();
    m_ScriptDebuggerUI->Init();
    ScriptDebugger::GetInstance().Init();
    std::cout << "Script Debugger initialized" << std::endl;

    // Initialize Scripting Profiler UI
    m_ScriptingProfilerUI = std::make_unique<ScriptingProfilerUI>();
    m_ScriptingProfilerUI->Init();
    std::cout << "Scripting Profiler UI initialized" << std::endl;

    // Initialize Blend Tree Editor
    // m_BlendTreeEditor = std::make_unique<BlendTreeEditor>();

    // Audio System
    if (!AudioSystem::Get().Initialize()) {
        std::cerr << "Failed to initialize Audio System" << std::endl;
        // Check if critical? maybe just log
    }

    // Initialize Physics System
    m_PhysicsSystem = std::make_unique<PhysicsSystem>();
    m_PhysicsSystem->Initialize(Vec3(0, -9.81f, 0)); // Standard Earth gravity
    std::cout << "Physics System initialized with Bullet3D" << std::endl;

#ifdef USE_PHYSX
    // Initialize PhysX Backend
    m_PhysXBackend = std::make_unique<PhysXBackend>();
    m_PhysXBackend->Initialize(Vec3(0, -9.81f, 0));
    std::cout << "PhysX Backend initialized" << std::endl;
    
    // Pass PhysX Backend to Renderer (for serialization)
    m_Renderer->SetPhysXBackend(m_PhysXBackend.get());
#endif

#ifdef USE_BOX2D
    // Initialize Box2D Backend
    m_Box2DBackend = std::make_unique<Box2DBackend>();
    m_Box2DBackend->Initialize(Vec3(0, -9.81f, 0));
    std::cout << "Box2D Backend initialized" << std::endl;
#endif

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

    // Initialize Asset Pipeline
    m_AssetPipeline = std::make_unique<AssetPipeline>();
    AssetPipeline::Config config;
    config.assetSourceDir = "assets";
    config.assetOutputDir = "assets_processed";
    config.databasePath = "asset_database.json";
    config.maxThreads = 4;
    config.verbose = true;
    
    if (!m_AssetPipeline->Initialize(config)) {
        std::cerr << "Failed to initialize Asset Pipeline" << std::endl;
        return false;
    }
    
    // Scan and process assets
    if (!m_AssetPipeline->ScanAssetDirectory(config.assetSourceDir)) {
        std::cerr << "Failed to scan asset directory" << std::endl;
        return false;
    }
    
    // Set up progress callback for UI display
    m_AssetPipeline->SetProgressCallback([this](float progress, const std::string& desc) {
        // Can be used for UI progress display
        if (progress >= 1.0f) {
            std::cout << "Asset Pipeline: " << desc << " [Complete]" << std::endl;
        }
    });
    
    std::cout << "Asset Pipeline initialized" << std::endl;

    // Initialize Editor UI Components (Phase 1 Enhancement)
    m_EditorMenuBar = std::make_unique<EditorMenuBar>();
    m_EditorHierarchy = std::make_unique<EditorHierarchy>();
    m_EditorPropertyPanel = std::make_unique<EditorPropertyPanel>();
    
    // Initialize Gizmo Tools Panel (Phase 3 Enhancement)
    m_GizmoToolsPanel = std::make_unique<GizmoToolsPanel>();
    std::cout << "Gizmo Tools Panel initialized" << std::endl;
    
    // Initialize Editor Docking Manager (Phase 2 Enhancement)
    m_DockingManager = std::make_unique<EditorDockingManager>();
    m_DockingManager->Initialize();
    
    // Link Docking Manager to Menu Bar
    if (m_EditorMenuBar) {
        m_EditorMenuBar->SetDockingManager(m_DockingManager.get());
    }
    
    std::cout << "Editor Docking Manager initialized" << std::endl;
    
    // Wire up menu bar callbacks
    m_EditorMenuBar->SetOnNewScene([this]() { 
        std::cout << "New Scene requested" << std::endl;
        m_Renderer->ClearScene();
    });
    m_EditorMenuBar->SetOnSaveScene([this]() { 
        m_Renderer->SaveScene("assets/current_scene.json");
    });
    m_EditorMenuBar->SetOnOpenScene([this]() { 
        m_Renderer->LoadScene("assets/current_scene.json");
    });
    m_EditorMenuBar->SetOnExit([this]() { 
        m_Running = false;
    });
    m_EditorMenuBar->SetOnDelete([this]() {
        auto selected = m_EditorHierarchy->GetSelectedObject();
        if (selected) {
            m_Renderer->RemoveObject(selected);
            m_EditorHierarchy->ClearSelection();
        }
    });
    
    // Wire up hierarchy callbacks
    m_EditorHierarchy->SetOnObjectSelected([this](std::shared_ptr<GameObject> obj) {
        if (m_GizmoManager) {
            m_GizmoManager->SetSelectedObject(obj);
        }
    });
    m_EditorHierarchy->SetOnObjectDeleted([this](std::shared_ptr<GameObject> obj) {
        m_Renderer->RemoveObject(obj);
    });
    
    std::cout << "Editor UI Components initialized" << std::endl;

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
        RemoteProfiler::Instance().Update(deltaTime);
        
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
    
    // Update asset pipeline (non-blocking, processes queued assets)
    if (m_AssetPipeline) {
        {
            SCOPED_PROFILE("AssetPipeline::Update");
            // Process dirty assets incrementally (this will return immediately if no assets to process)
            m_AssetPipeline->Update();
        }
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
        // Check collision
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

    // Update physics simulation (Double-Buffered Fixed Timestep)
    {
        SCOPED_PROFILE("Physics::Update");
        
        m_PhysicsAccumulator += deltaTime;
        
        // Safety clamp to prevent spiral of death
        if (m_PhysicsAccumulator > 0.25f) m_PhysicsAccumulator = 0.25f;

        while (m_PhysicsAccumulator >= m_PhysicsStepSize) {
            
            // 1. Step Bullet Physics
            if (m_PhysicsSystem) {
                m_PhysicsSystem->Update(m_PhysicsStepSize);
            }
            
            // 2. Step PhysX
#ifdef USE_PHYSX
            if (m_PhysXBackend) {
                m_PhysXBackend->Update(m_PhysicsStepSize);
                
                // 3. Sync Transforms
                // Get active bodies that moved this step
                static std::vector<IPhysicsRigidBody*> activeBodies;
                m_PhysXBackend->GetActiveRigidBodies(activeBodies);
                
                for (IPhysicsRigidBody* body : activeBodies) {
                    if (!body) continue;
                    
                    GameObject* owner = static_cast<GameObject*>(body->GetUserData());
                    if (owner) {
                         Vec3 pos;
                         Quat rot;
                         body->SyncTransformFromPhysics(pos, rot);
                         
                         // Update Physics Buffer (Prev = Cur, Cur = New)
                         owner->UpdatePhysicsState(pos, rot);
                    }
                }
            }
#endif

#ifdef USE_BOX2D
            // 2b. Step Box2D
            if (m_Box2DBackend) {
                m_Box2DBackend->Update(m_PhysicsStepSize);
                
                // Sync Transforms
                // Box2DBackend might not have GetActiveRigidBodies fully implemented or optimized yet, 
                // but checking it would be consistent.
                // Assuming we can iterate all bodies or similar.
                // For now, let's assume active bodies or just rely on Update doing callbacks? 
                // Box2D rigid bodies hold pointers to GameObjects, and SyncTransformFromPhysics works.
                // We'd need to iterate bodies. Does backend expose them?
                
                // Implementation in Box2DBackend.cpp doesn't seem to expose a "GetActiveRigidBodies" method 
                // (interface has GetActiveRigidBodies? IPhysicsBackend has it? Let's check interface).
                // IPhysicsBackend has: virtual void GetActiveRigidBodies(std::vector<IPhysicsRigidBody*>& outBodies) = 0;
                
                // So calling GetActiveRigidBodies on m_Box2DBackend should work if implemented.
                // Assuming it is implemented.
                
                static std::vector<IPhysicsRigidBody*> activeBox2DBodies;
                m_Box2DBackend->GetActiveRigidBodies(activeBox2DBodies);
                
                for (IPhysicsRigidBody* body : activeBox2DBodies) {
                    if (!body) continue;
                    
                    GameObject* owner = static_cast<GameObject*>(body->GetUserData());
                    if (owner) {
                         Vec3 pos;
                         Quat rot;
                         body->SyncTransformFromPhysics(pos, rot);
                         
                         owner->UpdatePhysicsState(pos, rot);
                    }
                }
            }
#endif
            
            m_PhysicsAccumulator -= m_PhysicsStepSize;
        }
        
        // 4. Interpolate for Rendering
        float alpha = m_PhysicsAccumulator / m_PhysicsStepSize;
        
        // Ideally we only interpolate active objects.
        // But for now, let's iterate root children or active list?
        // Issue: activeBodies is local to the loop.
        // We need to iterate ALL objects that *might* have physics.
        // Or better: The Scene Graph update relies on m_Transform.
        // We need to make sure m_Transform is set to the interpolated value.
        
        // Optimization: Create a list of "PhysicsActiveGameObjects"? 
        // For strict correctness, we should traverse the scene.
        // But if we only care about PhysX objects:
        // We can't use activeBodies here because that's only for the *last* step.
        // Objects that are sleeping don't need interpolation (Prev == Cur).
        // So actually, iterating active objects from the LAST step is "okay" IF we assume sleeping objects don't move between frames.
        // But let's be safe: The scene graph Update (line 311) propagates transforms.
        // We should update transforms BEFORE that.
        
        // Let's iterate all PhysX bodies known to backend?
        // PhysXBackend doesn't expose all bodies easily.
        // Let's iterate scene graph? Slow.
        
        // COMPROMISE:
        // For the visual test (Active Rigid Body count is high), we rely on activeBodies.
        // But we need to update *every frame* for interpolation, not just on physics steps.
        // Use a persistent list of "ActivePhysicsObjects"?
        
        // Let's rely on the fact that if an object is sleeping, Prev == Cur, so Lerp(Prev, Cur, alpha) == Cur.
        // So we just need to touch objects that *are* moving or *were* moving.
        // Implementing a "RegisteredPhysicsObjects" list in Application or Renderer would be best.
        // For now, let's just stick to the specific test case needs or basic iteration?
        // Actually, let's assume we can get ALL rigid bodies from backend if needed.
        // But iterating 1000s of static bodies is bad.
        
        // Correct approach: 
        // Iterate only *interpolatable* objects.
        // Since we don't have that list, and we want "Active Rigid Body" optimization...
        // We might be missing the interpolation if we don't track them.
        
        // Hack for now: Repopulate activeBodies list for interpolation? No that's physics state.
        
        // Let's use `m_PhysXBackend->GetActiveRigidBodies` again? 
        // No, that returns PxScene::getActiveActors which resets after fetchResults.
        // Since we are outside the loop, we might not get valid actors if 0 steps ran this frame.
        
        // OK, critical logic:
        // If 0 physics steps ran this frame (high FPS), we still need to interpolate.
        // We need the data from the LAST rigid body update.
        // So we need to store `std::vector<GameObject*> m_InterpolatedObjects`?
        // Let's add that to Application later if needed. 
        // For now, let's iterate the Root's children as a heuristic (since test scene is flat).
        
        if (m_Renderer) {
             auto root = m_Renderer->GetRoot();
             if (root) {
                 // Recursive function to interpolate
                 std::function<void(std::shared_ptr<GameObject>)> interpolateRecursive = 
                    [&](std::shared_ptr<GameObject> obj) {
                        if (obj->GetPhysicsRigidBody() != nullptr) {
                            obj->InterpolatePhysicsState(alpha);
                        }
                        for (auto& child : obj->GetChildren()) {
                            interpolateRecursive(child);
                        }
                    };
                 interpolateRecursive(root);
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
#ifdef USE_BOX2D
             if (m_Box2DBackend && m_Box2DBackend->IsDebugDrawEnabled()) {
                 m_Box2DBackend->DebugDraw(m_Renderer.get());
             }
#endif
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
    // PHASE 2: DOCKABLE LAYOUT SYSTEM
    
    // Begin dockspace
    if (m_DockingManager) {
        ImGuiID dockspace_id = m_DockingManager->BeginDockspace();
    }
    
    // Main Menu Bar with File, Edit, View, Window, Help menus
    if (m_EditorMenuBar) {
        m_EditorMenuBar->Render();
    }

    // Scene Hierarchy Panel - Docked
    if (m_EditorMenuBar->IsHierarchyVisible()) {
        if (ImGui::Begin("Scene Hierarchy", nullptr, ImGuiWindowFlags_None)) {
            ImGui::SetWindowPos(ImVec2(0, 19), ImGuiCond_FirstUseEver);
            ImGui::SetWindowSize(ImVec2(250, 400), ImGuiCond_FirstUseEver);
            
            if (m_EditorHierarchy && m_Renderer) {
                m_EditorHierarchy->Render(m_Renderer->GetRoot());
                
                // Get selected object from hierarchy
                auto selected = m_EditorHierarchy->GetSelectedObject();
                if (selected && m_GizmoManager) {
                    m_GizmoManager->SetSelectedObject(selected);
                }
            }
        }
        ImGui::End();
    }

    // Property Inspector Panel - Docked
    if (m_EditorMenuBar->IsInspectorVisible()) {
        if (ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_None)) {
            ImGui::SetWindowPos(ImVec2(1000, 19), ImGuiCond_FirstUseEver);
            ImGui::SetWindowSize(ImVec2(250, 400), ImGuiCond_FirstUseEver);
            
            if (m_EditorPropertyPanel) {
                auto selectedObj = m_EditorHierarchy->GetSelectedObject();
                m_EditorPropertyPanel->Render(selectedObj);
            }
        }
        ImGui::End();
    }

    // Gizmo Tools Panel - Docked (Phase 3 Enhancement)
    if (ImGui::Begin("Gizmo Tools", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(1000, 430), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);
        
        if (m_GizmoToolsPanel && m_GizmoManager) {
            m_GizmoToolsPanel->Render(m_GizmoManager);
        }
    }
    ImGui::End();

    // Asset Browser Panel (when visible) - Docked
    if (m_EditorMenuBar->IsAssetBrowserVisible()) {
        if (ImGui::Begin("Asset Browser", nullptr, ImGuiWindowFlags_None)) {
            ImGui::SetWindowPos(ImVec2(0, 450), ImGuiCond_FirstUseEver);
            ImGui::SetWindowSize(ImVec2(500, 150), ImGuiCond_FirstUseEver);
            ImGui::TextDisabled("Asset Browser - Phase 2");
        }
        ImGui::End();
    }

    // Viewport Window - Docked
    if (ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(250, 19), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(750, 450), ImGuiCond_FirstUseEver);
        ImGui::Text("Viewport - Game View");
        ImGui::TextDisabled("(Render target would go here)");
    }
    ImGui::End();

    // Profiler Panel (when visible) - Docked
    if (m_EditorMenuBar->IsProfilerVisible()) {
        if (ImGui::Begin("Performance Profiler", nullptr, ImGuiWindowFlags_None)) {
            ImGui::SetWindowPos(ImVec2(1000, 450), ImGuiCond_FirstUseEver);
            ImGui::SetWindowSize(ImVec2(250, 150), ImGuiCond_FirstUseEver);
            
            // Display FPS and frame time
            ImGuiIO& io = ImGui::GetIO();
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Text("Frame Time: %.2f ms", 1000.0f / io.Framerate);
            
            ImGui::Separator();
            
            // Display profiler stats
            auto& profiler = Profiler::Instance();
            ImGui::Text("Profiler Statistics:");
            ImGui::TextDisabled("(Profiler data)");
        }
        ImGui::End();
    }

    // Tools Panel - Script Debugger and Profiler UI - Docked
    if (ImGui::Begin("Tools", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(10, 100), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);
        
        if (ImGui::CollapsingHeader("Debug Tools", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Script Debugger##ToolsBtn", ImVec2(-1, 0))) {
                if (m_ScriptDebuggerUI) {
                    m_ScriptDebuggerUI->Toggle();
                }
            }
            
            if (ImGui::Button("Scripting Profiler##ToolsBtn", ImVec2(-1, 0))) {
                if (m_ScriptingProfilerUI) {
                    m_ScriptingProfilerUI->Toggle();
                }
            }
        }
        
        ImGui::Separator();
        
        // Quick Scene Actions
        if (ImGui::CollapsingHeader("Scene Actions", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Load Cornell Box", ImVec2(-1, 0))) {
                LoadCornellBox();
            }
            
            if (ImGui::Button("Add Cube", ImVec2(-1, 0))) {
                if (m_Renderer) {
                    m_Renderer->AddCube(Transform(Vec3(0, 0, 0)));
                }
            }
            
            if (ImGui::Button("Add Pyramid", ImVec2(-1, 0))) {
                if (m_Renderer) {
                    m_Renderer->AddPyramid(Transform(Vec3(0, 0, 0)));
                }
            }
        }
    }
    ImGui::End();

    // Light Inspector Panel - Docked
    if (ImGui::Begin("Light Inspector", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(1200, 100), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(300, 600), ImGuiCond_FirstUseEver);
        
        if (m_Renderer) {
            auto& lights = m_Renderer->GetLights();
            static int selectedLightIndex = -1;

            if (ImGui::Button("Add Light")) {
                m_Renderer->AddLight(Light(Vec3(0, 5, 0)));
            }
            
            bool showCascades = m_Renderer->GetShowCascades();
            if (ImGui::Checkbox("Show CSM Cascades", &showCascades)) {
                m_Renderer->SetShowCascades(showCascades);
            }
            
            for (size_t i = 0; i < lights.size(); ++i) {
                std::string label = "Light " + std::to_string(i);
                if (ImGui::Selectable(label.c_str(), selectedLightIndex == static_cast<int>(i))) {
                    selectedLightIndex = static_cast<int>(i);
                }
            }
        }
    }
    ImGui::End();

    // Post-Processing Panel
    if (ImGui::Begin("Post-Processing", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(920, 100), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(280, 300), ImGuiCond_FirstUseEver);
        
        if (m_Renderer) {
            auto postProcessing = m_Renderer->GetPostProcessing();
            if (postProcessing) {
                bool bloomEnabled = postProcessing->IsBloomEnabled();
                if (ImGui::Checkbox("Enable Bloom", &bloomEnabled)) {
                    postProcessing->SetBloomEnabled(bloomEnabled);
                }
                
                if (bloomEnabled) {
                    float bloomIntensity = postProcessing->GetBloomIntensity();
                    if (ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 2.0f)) {
                        postProcessing->SetBloomIntensity(bloomIntensity);
                    }
                }
                
                ImGui::Separator();
                
                float exposure = postProcessing->GetExposure();
                if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f)) {
                    postProcessing->SetExposure(exposure);
                }
            }
        }
    }
    ImGui::End();

    // Simulation Controls Panel
    if (ImGui::Begin("Simulation Controls", nullptr, ImGuiWindowFlags_None)) {
        ImGui::SetWindowPos(ImVec2(100, 600), ImGuiCond_FirstUseEver);
        ImGui::SetWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

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
            
            // Visuals
            if (ImGui::TreeNode("Visuals")) {
                // Gradient
                ImGui::Text("Color Gradient");
                auto& gradient = emitter->GetGradient();
                for (size_t i = 0; i < gradient.size(); ++i) {
                     ImGui::PushID((int)i);
                     ImGui::DragFloat("T", &gradient[i].t, 0.01f, 0.0f, 1.0f);
                     // ImGui::ColorEdit4("Color", &gradient[i].color.r);
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
    
#ifdef USE_PHYSX
    ImGui::Separator();
    if (ImGui::Button("Run GPU Rigid Body Test")) {
        LoadGpuTestScene();
    }
#endif

#ifdef USE_BOX2D
    ImGui::Separator();
    if (m_Box2DBackend) {
        bool debugDraw = m_Box2DBackend->IsDebugDrawEnabled();
        if (ImGui::Checkbox("Box2D Debug Draw", &debugDraw)) {
            m_Box2DBackend->SetDebugDrawEnabled(debugDraw);
        }
        if (ImGui::Button("Run Joint Test (Pendulum)")) {
            LoadBox2DJointTest();
        }
        if (ImGui::Button("Run Character Test")) {
            LoadBox2DCharacterTest();
        }

        // Plane Selector
        const char* planes[] = { "XY (Side-Scroller)", "XZ (Top-Down)" };
        int currentPlane = (int)m_Box2DBackend->GetPlane();
        if (ImGui::Combo("Simulation Plane", &currentPlane, planes, IM_ARRAYSIZE(planes))) {
            m_Box2DBackend->SetPlane((Box2DBackend::Plane2D)currentPlane);
            
            // Auto-set gravity based on plane for convenience?
            if (currentPlane == (int)Box2DBackend::Plane2D::XY) {
                m_Box2DBackend->SetGravity(Vec3(0, -9.81f, 0));
            } else {
                m_Box2DBackend->SetGravity(Vec3(0, 0, 0)); // Top down usually no gravity
            }
        }
    }
#endif
    
    ImGui::End();

    // Render Script Debugger UI
    if (m_ScriptDebuggerUI) {
        m_ScriptDebuggerUI->Update(m_LastFrameTime);
        m_ScriptDebuggerUI->Render();
    }

    // Render Scripting Profiler UI
    if (m_ScriptingProfilerUI) {
        m_ScriptingProfilerUI->Update(m_LastFrameTime);
        m_ScriptingProfilerUI->Render();
    }
    
    // End dockspace
    if (m_DockingManager) {
        m_DockingManager->EndDockspace();
    }
}

// Helper function to toggle script debugger
static bool g_ShowScriptDebugger = false;

#ifdef USE_PHYSX
void Application::LoadGpuTestScene() {
    if (!m_PhysXBackend || !m_Renderer) return;

    // Clear existing scene
    auto root = m_Renderer->GetRoot();
    root->GetChildren().clear();
    m_SelectedObjectIndex = -1;
    
    // Create Floor
    {
        auto floor = std::make_shared<GameObject>("Floor");
        floor->GetTransform().position = Vec3(0, -1, 0);
        floor->GetTransform().scale = Vec3(50, 1, 50);
        floor->SetMesh(Mesh::CreateCube());
        
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(Vec3(0.5f, 0.5f, 0.5f));
        floor->SetMaterial(mat);
        
        // PhysX Static Body
        auto body = std::make_shared<PhysXRigidBody>(m_PhysXBackend.get());
        auto shape = PhysXShape::CreateBox(m_PhysXBackend.get(), Vec3(25, 0.5f, 25)); // Half-extents
        
        if (shape) {
            body->Initialize(BodyType::Static, 0.0f, shape); 
            floor->SetPhysicsRigidBody(body);
        }
        root->AddChild(floor);
    }
    
    // Spawn Dynamic Cubes
    int gridSize = 10;
    float spacing = 1.2f;
    float startHeight = 10.0f;
    
    auto cubeMesh = Mesh::CreateCube();
    auto cubeMat = std::make_shared<Material>();
    cubeMat->SetDiffuse(Vec3(1.0f, 0.2f, 0.2f)); // Red
    
    // Shared shape for optimization? 
    // PhysX shapes can be shared between rigid bodies if attached to multiple? 
    // Actually PxShape can be shared if we use PxRigidActor::attachShape.
    // But PhysXRigidBody::Initialize creates its own actor and attaches the shape.
    // PhysXShape w wrapper holds a unique PxShape? Let's check PhysXShape.cpp later. 
    // Assuming we create new wrappers for now to be safe.
    
    for (int x = 0; x < gridSize; ++x) {
        for (int z = 0; z < gridSize; ++z) {
            for (int y = 0; y < 5; ++y) {
                 auto cube = std::make_shared<GameObject>("GpuCube");
                 
                 float px = (x - gridSize/2.0f) * spacing;
                 float py = startHeight + y * spacing;
                 float pz = (z - gridSize/2.0f) * spacing;
                 
                 cube->GetTransform().position = Vec3(px, py, pz);
                 cube->SetMesh(cubeMesh); // Share mesh
                 cube->SetMaterial(cubeMat); // Share material
                 
                 // Create Rigid Body
                 auto body = std::make_shared<PhysXRigidBody>(m_PhysXBackend.get());
                 auto shape = PhysXShape::CreateBox(m_PhysXBackend.get(), Vec3(0.5f, 0.5f, 0.5f));
                 
                 if (shape) {
                     body->Initialize(BodyType::Dynamic, 1.0f, shape);
                     cube->SetPhysicsRigidBody(body); // This sets userData to 'cube'
                 }
                 
                 root->AddChild(cube);
            }
        }
    }
    
    std::cout << "Loaded GPU Rigid Body Test Scene with " << (gridSize*gridSize*5) << " cubes." << std::endl;
}
#endif

#ifdef USE_BOX2D
void Application::LoadBox2DJointTest() {
    if (!m_Box2DBackend || !m_Renderer) return;

    // Clear scene
    auto root = m_Renderer->GetRoot();
    root->GetChildren().clear();
    m_SelectedObjectIndex = -1;

    // 1. Static Anchor (Ceiling)
    auto ceiling = std::make_shared<GameObject>("CeilingAnchor");
    ceiling->GetTransform().position = Vec3(0, 10, 0); // High up
    ceiling->GetTransform().scale = Vec3(2, 0.5f, 2);
    ceiling->SetMesh(Mesh::CreateCube());
    
    auto mat = std::make_shared<Material>();
    mat->SetDiffuse(Vec3(0.5f, 0.5f, 0.5f));
    ceiling->SetMaterial(mat);

    auto rbCeiling = std::make_shared<Box2DRigidBody>(m_Box2DBackend.get());
    auto shapeCeiling = Box2DShape::CreateBox(m_Box2DBackend.get(), Vec3(1, 0.25f, 1));
    if (shapeCeiling) {
        rbCeiling->Initialize(BodyType::Static, 0.0f, shapeCeiling);
        ceiling->SetPhysicsRigidBody(rbCeiling);
    }
    root->AddChild(ceiling);

    // 2. Dynamic Pendulum Body
    auto bob = std::make_shared<GameObject>("PendulumBob");
    bob->GetTransform().position = Vec3(5, 10, 0); // To the right, same height (start horizontal)
    bob->GetTransform().scale = Vec3(1, 1, 1);
    bob->SetMesh(Mesh::CreateCube());
    
    auto matRed = std::make_shared<Material>();
    matRed->SetDiffuse(Vec3(1.0f, 0.0f, 0.0f));
    bob->SetMaterial(matRed);

    auto rbBob = std::make_shared<Box2DRigidBody>(m_Box2DBackend.get());
    auto shapeBob = Box2DShape::CreateBox(m_Box2DBackend.get(), Vec3(0.5f, 0.5f, 0.5f));
    if (shapeBob) {
        rbBob->Initialize(BodyType::Dynamic, 10.0f, shapeBob);
        bob->SetPhysicsRigidBody(rbBob);
    }
    root->AddChild(bob);

    // 3. Create Joint (Revolute)
    JointDef jDef;
    jDef.type = JointType::Revolute;
    jDef.bodyA = rbCeiling.get();
    jDef.bodyB = rbBob.get();
    jDef.anchorA = Vec3(0, 10, 0); // Pivot at ceiling center
    jDef.collideConnected = false;
    
    // Optional: Motor
    // jDef.enableMotor = true;
    // jDef.motorSpeed = 1.0f;
    // jDef.maxMotorTorque = 1000.0f;

    m_Box2DBackend->CreateJoint(jDef);
    
    std::cout << "Created Box2D Joint Test (Pendulum)" << std::endl;
}

void Application::LoadBox2DCharacterTest() {
    if (!m_Box2DBackend || !m_Renderer) return;

    // Clear scene
    auto root = m_Renderer->GetRoot();
    root->GetChildren().clear();
    m_SelectedObjectIndex = -1;

    // 1. Static Floor
    auto floor = std::make_shared<GameObject>("Floor");
    floor->GetTransform().position = Vec3(0, -2, 0);
    floor->GetTransform().scale = Vec3(20, 1, 20);
    floor->SetMesh(Mesh::CreateCube());
    
    auto mat = std::make_shared<Material>();
    mat->SetDiffuse(Vec3(0.5f, 0.5f, 0.5f));
    floor->SetMaterial(mat);

    auto rbFloor = std::make_shared<Box2DRigidBody>(m_Box2DBackend.get());
    auto shapeFloor = Box2DShape::CreateBox(m_Box2DBackend.get(), Vec3(10, 0.5f, 10)); // Half extents
    if (shapeFloor) {
        rbFloor->Initialize(BodyType::Static, 0.0f, shapeFloor);
        floor->SetPhysicsRigidBody(rbFloor);
    }
    root->AddChild(floor);

    // 2. Character
    auto charObj = std::make_shared<GameObject>("Character");
    charObj->GetTransform().position = Vec3(0, 5, 0); // Drop from height
    charObj->GetTransform().scale = Vec3(0.5f, 1.0f, 0.5f); // 1x2x1 approx
    charObj->SetMesh(Mesh::CreateCube());
    
    auto matChar = std::make_shared<Material>();
    matChar->SetDiffuse(Vec3(0.0f, 0.8f, 0.2f)); // Green
    charObj->SetMaterial(matChar);

    // Create CCT
    auto cct = m_Box2DBackend->CreateCharacterController();
    if (cct) {
        cct->Initialize(nullptr, 60.0f, 0.5f); // Default shape (box), 60kg
        cct->SetPosition(charObj->GetTransform().position);
        charObj->SetPhysicsCharacterController(cct);
        
        // Give it some initial push? No, let user control?
        // We don't have input hooked up to selected object easily here without modifying Update.
        // It will just fall and land.
    }
    root->AddChild(charObj);
    
    std::cout << "Created Box2D Character Test" << std::endl;
}
#endif
void Application::LoadCornellBox() {
    if (!m_Renderer) return;
    
    m_Renderer->ClearScene();
    std::cout << "Loading Cornell Box scene..." << std::endl;
    
    // Floor
    auto floor = std::make_shared<GameObject>("Floor");
    floor->SetTransform(Transform(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(10, 0.1f, 10)));
    auto whiteMat = std::make_shared<Material>();
    whiteMat->SetDiffuse(Vec3(1, 1, 1));
    floor->SetMaterial(whiteMat);
    m_Renderer->GetRoot()->AddChild(floor);
    
    // Back Wall
    auto backWall = std::make_shared<GameObject>("BackWall");
    backWall->SetTransform(Transform(Vec3(0, 5, -5), Vec3(0, 0, 0), Vec3(10, 10, 0.1f)));
    backWall->SetMaterial(whiteMat);
    m_Renderer->GetRoot()->AddChild(backWall);
    
    // Left Wall (Red)
    auto leftWall = std::make_shared<GameObject>("LeftWall");
    leftWall->SetTransform(Transform(Vec3(-5, 5, 0), Vec3(0, 0, 0), Vec3(0.1f, 10, 10)));
    auto redMat = std::make_shared<Material>();
    redMat->SetDiffuse(Vec3(1, 0, 0));
    leftWall->SetMaterial(redMat);
    m_Renderer->GetRoot()->AddChild(leftWall);
    
    // Right Wall (Green)
    auto rightWall = std::make_shared<GameObject>("RightWall");
    rightWall->SetTransform(Transform(Vec3(5, 5, 0), Vec3(0, 0, 0), Vec3(0.1f, 10, 10)));
    auto greenMat = std::make_shared<Material>();
    greenMat->SetDiffuse(Vec3(0, 1, 0));
    rightWall->SetMaterial(greenMat);
    m_Renderer->GetRoot()->AddChild(rightWall);
    
    // Ceiling
    auto ceiling = std::make_shared<GameObject>("Ceiling");
    ceiling->SetTransform(Transform(Vec3(0, 10, 0), Vec3(0, 0, 0), Vec3(10, 0.1f, 10)));
    ceiling->SetMaterial(whiteMat);
    m_Renderer->GetRoot()->AddChild(ceiling);
    
    // Large Box
    auto box1 = std::make_shared<GameObject>("LargeBox");
    box1->SetTransform(Transform(Vec3(-1.5f, 1.5f, -1.0f), Vec3(0, 15, 0), Vec3(3, 3, 3)));
    box1->SetMaterial(whiteMat);
    m_Renderer->GetRoot()->AddChild(box1);
    
    // Small Box
    auto box2 = std::make_shared<GameObject>("SmallBox");
    box2->SetTransform(Transform(Vec3(1.5f, 1.0f, 1.0f), Vec3(0, -15, 0), Vec3(2, 2, 2)));
    box2->SetMaterial(whiteMat);
    m_Renderer->GetRoot()->AddChild(box2);
    
    // Light
    m_Renderer->AddLight(Light(Vec3(0, 9.8f, 0), Vec3(1, 1, 1), 1.0f));
}
