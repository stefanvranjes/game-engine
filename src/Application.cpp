#include "Application.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "Math/AABB.h"
#include "imgui/imgui.h"

Application::Application() 
    : m_Running(false)
    , m_LastFrameTime(0.0f)
    , m_FPS(0.0f)
    , m_FrameCount(0.0f)
    , m_FPSTimer(0.0f)
    , m_SelectedObjectIndex(-1)
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

    // Initialize ImGui
    m_ImGui = std::make_unique<ImGuiManager>();
    if (!m_ImGui->Init(m_Window->GetGLFWWindow())) {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        return false;
    }

    m_Running = true;
    return true;
}

void Application::Run() {
    m_LastFrameTime = static_cast<float>(glfwGetTime());
    
    while (m_Running && !m_Window->ShouldClose()) {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - m_LastFrameTime;
        m_LastFrameTime = currentTime;

        Update(deltaTime);
        Render();
        
        m_Window->SwapBuffers();
        m_Window->PollEvents();
    }
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

    // Update camera with collision detection
    if (m_Camera) {
        Vec3 oldPos = m_Camera->GetPosition();
        m_Camera->ProcessInput(m_Window->GetGLFWWindow(), deltaTime);
        Vec3 newPos = m_Camera->GetPosition();

        // Create player AABB (size 0.5)
        Vec3 playerSize(0.25f, 0.25f, 0.25f); // Half-extents
        AABB playerBounds(newPos - playerSize, newPos + playerSize);

        // Check collision
        if (m_Renderer->CheckCollision(playerBounds)) {
            // Revert position if collision detected
            // Simple response: just revert to old position
            // Ideally we would slide along the wall, but this prevents walking through objects
            m_Camera->SetPosition(oldPos);
        }
    }

    // Scene Management Input
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F5) == GLFW_PRESS) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F9) == GLFW_PRESS) {
        m_Renderer->LoadScene("assets/scene.txt");
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

    // Render ImGui Editor UI
    if (m_ImGui) {
        m_ImGui->BeginFrame();
        RenderEditorUI();
        m_ImGui->EndFrame();
    }
}

void Application::RenderEditorUI() {
    // Scene Hierarchy Panel
    ImGui::Begin("Scene Hierarchy");
    
    auto& meshes = m_Renderer->GetMeshes();
    auto& transforms = m_Renderer->GetTransforms();
    
    for (size_t i = 0; i < meshes.size(); ++i) {
        std::string label = meshes[i].GetSource() + " ##" + std::to_string(i);
        if (ImGui::Selectable(label.c_str(), m_SelectedObjectIndex == static_cast<int>(i))) {
            m_SelectedObjectIndex = static_cast<int>(i);
        }
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Add Cube")) {
        m_Renderer->AddCube(Transform(Vec3(0, 0, 0)));
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Pyramid")) {
        m_Renderer->AddPyramid(Transform(Vec3(0, 0, 0)));
    }
    
    if (m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(meshes.size())) {
        if (ImGui::Button("Delete Selected")) {
            m_Renderer->RemoveObject(m_SelectedObjectIndex);
            m_SelectedObjectIndex = -1;
        }
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Save Scene (F5)")) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (ImGui::Button("Load Scene (F9)")) {
        m_Renderer->LoadScene("assets/scene.txt");
        m_SelectedObjectIndex = -1;
    }
    
    ImGui::End();
    
    // Object Inspector Panel
    if (m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(transforms.size())) {
        ImGui::Begin("Object Inspector");
        
        Transform& transform = transforms[m_SelectedObjectIndex];
        
        ImGui::Text("Object: %s", meshes[m_SelectedObjectIndex].GetSource().c_str());
        ImGui::Separator();
        
        // Position
        ImGui::Text("Position");
        ImGui::DragFloat("X##pos", &transform.position.x, 0.1f);
        ImGui::DragFloat("Y##pos", &transform.position.y, 0.1f);
        ImGui::DragFloat("Z##pos", &transform.position.z, 0.1f);
        
        ImGui::Separator();
        
        // Rotation
        ImGui::Text("Rotation");
        ImGui::DragFloat("X##rot", &transform.rotation.x, 1.0f);
        ImGui::DragFloat("Y##rot", &transform.rotation.y, 1.0f);
        ImGui::DragFloat("Z##rot", &transform.rotation.z, 1.0f);
        
        ImGui::Separator();
        
        // Scale
        ImGui::Text("Scale");
        ImGui::DragFloat("X##scl", &transform.scale.x, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Y##scl", &transform.scale.y, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Z##scl", &transform.scale.z, 0.1f, 0.1f, 10.0f);
        
        ImGui::End();
    }
}
