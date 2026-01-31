#pragma once

#include "Window.h"
#include "Renderer.h"
#include "Camera.h"
#include "Text.h"
#include "ImGuiManager.h"
#include "GizmoManager.h"
#include "PreviewRenderer.h"
#include "Profiler.h"
#include "TelemetryServer.h"
#include "PhysicsSystem.h"
#include "ScriptDebuggerUI.h"
#ifdef USE_PHYSX
#include "PhysXBackend.h"
#endif

#ifdef USE_BOX2D
#include "Box2DBackend.h"
#endif
#include "ECS.h"
#include "AssetHotReloadManager.h"
#include "AssetPipeline.h"
#include <memory>

class Application {
public:
    Application();
    ~Application();

    bool Init();
    void Run();
    void Shutdown();
    void LoadCornellBox(); // Test scene
#ifdef USE_PHYSX
    void LoadGpuTestScene(); // GPU Rigid Body test scene
#endif
#ifdef USE_BOX2D
    void LoadBox2DJointTest();
    void LoadBox2DCharacterTest();
#endif

#ifdef USE_PHYSX
    class PhysXBackend* GetPhysXBackend() { return m_PhysXBackend.get(); }
#endif

    // Access singleton
    static Application& Get();

private:
    void Update(float deltaTime);
    void Render();
    void RenderEditorUI();

    std::unique_ptr<Window> m_Window;
    std::unique_ptr<Renderer> m_Renderer;
    std::unique_ptr<Camera> m_Camera;
    std::unique_ptr<Text> m_Text;
    std::unique_ptr<ImGuiManager> m_ImGui;
    std::unique_ptr<PreviewRenderer> m_PreviewRenderer;
    std::unique_ptr<PhysicsSystem> m_PhysicsSystem;
    std::unique_ptr<ScriptDebuggerUI> m_ScriptDebuggerUI;
#ifdef USE_PHYSX
    std::unique_ptr<class PhysXBackend> m_PhysXBackend;
#endif

#ifdef USE_BOX2D
    std::unique_ptr<class Box2DBackend> m_Box2DBackend;
#endif
    
    // Asset Hot-Reload
    std::unique_ptr<AssetHotReloadManager> m_HotReloadManager;
    
    // Asset Pipeline
    std::unique_ptr<AssetPipeline> m_AssetPipeline;
    
    // ECS Architecture
    std::unique_ptr<EntityManager> m_EntityManager;
    std::shared_ptr<class GizmoManager> m_GizmoManager; // Shared with Renderer
    
    // Editor State
    int m_SelectedObjectIndex;
    
    float m_LastFrameTime;
    float m_FPS;
    float m_FrameCount;
    float m_FPSTimer;
    bool m_Running;
    
    // Audio Listener
    Vec3 m_LastCameraPosition;

    // Physics Interpolation
    float m_PhysicsAccumulator = 0.0f;
    const float m_PhysicsStepSize = 1.0f / 60.0f;
};

