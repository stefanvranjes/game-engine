#pragma once

#include "Window.h"
#include "Renderer.h"
#include "Camera.h"
#include "Text.h"
#include "ImGuiManager.h"
#include "PreviewRenderer.h"
#include "Profiler.h"
#include "TelemetryServer.h"
#include <memory>

class Application {
public:
    Application();
    ~Application();

    bool Init();
    void Run();
    void Shutdown();

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
    
    int m_SelectedObjectIndex;
    
    float m_LastFrameTime;
    float m_FPS;
    float m_FrameCount;
    float m_FPSTimer;
    bool m_Running;
    
    // Audio Listener
    Vec3 m_LastCameraPosition;
};
