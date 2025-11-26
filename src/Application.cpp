#include "Application.h"
#include <GLFW/glfw3.h>
#include <iostream>

Application::Application() 
    : m_Running(false)
    , m_LastFrameTime(0.0f)
    , m_FPS(0.0f)
    , m_FrameCount(0.0f)
    , m_FPSTimer(0.0f)
{
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

    // Create camera
    m_Camera = std::make_unique<Camera>(Vec3(0, 0, 5), 45.0f, 800.0f / 600.0f);
    m_Renderer->SetCamera(m_Camera.get());

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Initialize text rendering
    m_Text = std::make_unique<Text>();
    if (!m_Text->Init("assets/font_atlas.png", 800, 600)) {
        std::cerr << "Failed to initialize text system" << std::endl;
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
    std::cout << "Controls: WASD to move, Arrow keys to rotate, Space/Shift for up/down" << std::endl;

    // Main game loop
    while (m_Running && !m_Window->ShouldClose()) {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - m_LastFrameTime;
        m_LastFrameTime = currentTime;

        Update(deltaTime);
        Render();
        
        m_Window->SwapBuffers();
        m_Window->PollEvents();
    }

    std::cout << "Game Engine Shutting Down..." << std::endl;
}

void Application::Update(float deltaTime) {
    // Calculate FPS
    m_FrameCount++;
    m_FPSTimer += deltaTime;
    if (m_FPSTimer >= 1.0f) {
        m_FPS = m_FrameCount / m_FPSTimer;
        m_FrameCount = 0.0f;
        m_FPSTimer = 0.0f;
    }

    // Update camera
    if (m_Camera) {
        m_Camera->ProcessInput(m_Window->GetGLFWWindow(), deltaTime);
    }
}

void Application::Render() {
    // Clear the screen and depth buffer
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render scene
    m_Renderer->Render();

    // Render UI (HUD)
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
