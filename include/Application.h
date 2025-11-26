#pragma once

#include "Window.h"
#include "Renderer.h"
#include "Camera.h"
#include "Text.h"
#include <memory>

class Application {
public:
    Application();
    ~Application();

    bool Init();
    void Run();

private:
    void Update(float deltaTime);
    void Render();

    std::unique_ptr<Window> m_Window;
    std::unique_ptr<Renderer> m_Renderer;
    std::unique_ptr<Camera> m_Camera;
    std::unique_ptr<Text> m_Text;
    
    float m_LastFrameTime;
    float m_FPS;
    float m_FrameCount;
    float m_FPSTimer;
    bool m_Running;
};
