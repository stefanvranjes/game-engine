#include "Renderer.h"
#include "Camera.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>

Renderer::Renderer() : m_Camera(nullptr) {
}

Renderer::~Renderer() {
    Shutdown();
}

void Renderer::SetupScene() {
    // Create multiple cubes at different positions
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());
    m_Meshes.push_back(Mesh::CreateCube());

    // Create transforms for each cube
    m_Transforms.push_back(Transform(Vec3(0, 0, 0), Vec3(0, 0, 0)));
    m_Transforms.push_back(Transform(Vec3(2, 0, 0), Vec3(0, 45, 0)));
    m_Transforms.push_back(Transform(Vec3(-2, 0, 0), Vec3(0, -45, 0)));
    m_Transforms.push_back(Transform(Vec3(0, 2, 0), Vec3(45, 0, 0)));
    m_Transforms.push_back(Transform(Vec3(0, -2, 0), Vec3(-45, 0, 0)));
    m_Transforms.push_back(Transform(Vec3(1.5, 1.5, -1), Vec3(30, 30, 0)));
    m_Transforms.push_back(Transform(Vec3(-1.5, -1.5, 1), Vec3(-30, -30, 0)));

    std::cout << "Scene setup complete with " << m_Meshes.size() << " cubes" << std::endl;
}

bool Renderer::Init() {
    // Create and load shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/textured.vert", "shaders/textured.frag")) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }

    // Load texture
    m_Texture = std::make_unique<Texture>();
    if (!m_Texture->LoadFromFile("assets/brick.ppm")) {
        std::cerr << "Failed to load texture" << std::endl;
        return false;
    }

    // Setup scene with multiple cubes
    SetupScene();

    return true;
}

void CheckOpenGLError(const char* stmt, const char* fname, int line) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error " << err << ", at " << fname << ":" << line << " - for " << stmt << std::endl;
    }
}

#define GL_CHECK(stmt) do { \
        stmt; \
        CheckOpenGLError(#stmt, __FILE__, __LINE__); \
    } while (0)

void Renderer::Render() {
    m_Shader->Use();
    
    // Bind texture once for all objects
    if (m_Texture) {
        m_Texture->Bind(0);
        m_Shader->SetInt("u_Texture", 0);
    }

    // Light setup
    Vec3 lightPos(2.0f, 2.0f, 2.0f);
    Vec3 lightColor(1.0f, 1.0f, 1.0f);
    m_Shader->SetVec3("u_LightPos", lightPos.x, lightPos.y, lightPos.z);
    m_Shader->SetVec3("u_LightColor", lightColor.x, lightColor.y, lightColor.z);
    
    if (m_Camera) {
        Vec3 camPos = m_Camera->GetPosition();
        m_Shader->SetVec3("u_ViewPos", camPos.x, camPos.y, camPos.z);
    }

    // Render each mesh with its transform
    for (size_t i = 0; i < m_Meshes.size() && i < m_Transforms.size(); ++i) {
        // Calculate MVP matrix for this object
        if (m_Camera) {
            Mat4 model = m_Transforms[i].GetModelMatrix();
            Mat4 view = m_Camera->GetViewMatrix();
            Mat4 projection = m_Camera->GetProjectionMatrix();
            Mat4 mvp = projection * view * model;
            
            m_Shader->SetMat4("u_MVP", mvp.m);
            m_Shader->SetMat4("u_Model", model.m); // Pass model matrix for world space calculations
        }

        // Draw the mesh
        m_Meshes[i].Draw();
    }

}

void Renderer::Shutdown() {
    m_Meshes.clear();
    m_Transforms.clear();
}
