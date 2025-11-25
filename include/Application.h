#pragma once

#include "Window.h"
#include "Renderer.h"
#include "Camera.h"
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
    
    float m_LastFrameTime;
    bool m_Running;
};
