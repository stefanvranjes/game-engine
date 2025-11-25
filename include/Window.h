#pragma once

#include <string>

struct GLFWwindow;

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    bool Init();
    bool ShouldClose() const;
    void SwapBuffers();
    void PollEvents();
    GLFWwindow* GetNativeWindow() const { return m_Window; }

    int GetWidth() const { return m_Width; }
    int GetHeight() const { return m_Height; }

private:
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);

    GLFWwindow* m_Window;
    int m_Width;
    int m_Height;
    std::string m_Title;
};
