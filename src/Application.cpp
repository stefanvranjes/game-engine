#include "Application.h"
#include <GLFW/glfw3.h>
#include <iostream>

Application::Application() : m_Running(false) {
}

Application::~Application() {
}

bool Application::Init() {
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

    m_Running = true;
    return true;
}

void Application::Run() {
    if (!Init()) {
        return;
    }

    std::cout << "Game Engine Running..." << std::endl;

    // Main game loop
    while (m_Running && !m_Window->ShouldClose()) {
        Update();
        Render();
        
        m_Window->SwapBuffers();
        m_Window->PollEvents();
    }

    std::cout << "Game Engine Shutting Down..." << std::endl;
}

void Application::Update() {
    // Update game logic here
}

void Application::Render() {
    // Clear the screen
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render triangle
    m_Renderer->Render();
}
