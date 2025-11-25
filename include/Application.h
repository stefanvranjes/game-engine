#pragma once

#include "Window.h"
#include "Renderer.h"
#include <memory>

class Application {
public:
    Application();
    ~Application();

    bool Init();
    void Run();

private:
    void Update();
    void Render();

    std::unique_ptr<Window> m_Window;
    std::unique_ptr<Renderer> m_Renderer;
    bool m_Running;
};
