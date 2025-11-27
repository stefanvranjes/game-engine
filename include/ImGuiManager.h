#pragma once

#include <GLFW/glfw3.h>

class ImGuiManager {
public:
    ImGuiManager();
    ~ImGuiManager();

    bool Init(GLFWwindow* window);
    void BeginFrame();
    void EndFrame();
    void Shutdown();

private:
    bool m_Initialized;
};
