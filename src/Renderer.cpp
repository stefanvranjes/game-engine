#include "Renderer.h"
#include "Camera.h"
#include "GLExtensions.h"
#include "Material.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>

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

    // Create materials for each cube
    for (int i = 0; i < 7; ++i) {
        auto mat = std::make_shared<Material>();
        // Vary colors slightly
        if (i % 2 == 0) mat->diffuse = Vec3(1.0f, 0.5f, 0.5f); // Reddish
        else if (i % 3 == 0) mat->diffuse = Vec3(0.5f, 1.0f, 0.5f); // Greenish
        else mat->diffuse = Vec3(0.5f, 0.5f, 1.0f); // Blueish
        
        mat->texture = m_Texture; // Use default texture
        m_Materials.push_back(mat);
    }

    // Load pyramid from OBJ
    m_Meshes.push_back(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    m_Transforms.push_back(Transform(Vec3(0, 2, 0), Vec3(0, 0, 0), Vec3(2, 2, 2)));
    
    auto pyramidMat = std::make_shared<Material>();
    pyramidMat->diffuse = Vec3(1.0f, 1.0f, 0.0f); // Yellow
    pyramidMat->texture = m_Texture;
    m_Materials.push_back(pyramidMat);

    // Add initial lights
    m_Lights.push_back(Light(Vec3(2.0f, 2.0f, 2.0f), Vec3(1.0f, 1.0f, 1.0f), 1.0f, true)); // White light with shadows
    m_Lights.push_back(Light(Vec3(-2.0f, 2.0f, -2.0f), Vec3(1.0f, 0.0f, 0.0f), 1.0f, false)); // Red light
    m_Lights.push_back(Light(Vec3(0.0f, 4.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f), 0.8f, false)); // Blue light

    std::cout << "Scene setup complete with " << m_Meshes.size() << " objects and " << m_Lights.size() << " lights" << std::endl;
}

bool Renderer::CheckCollision(const AABB& bounds) {
    for (size_t i = 0; i < m_Meshes.size(); ++i) {
        const AABB& localBounds = m_Meshes[i].GetBounds();
        Mat4 model = m_Transforms[i].GetModelMatrix();

        // Transform all 8 corners of the AABB
        Vec3 corners[8];
        corners[0] = model * Vec3(localBounds.min.x, localBounds.min.y, localBounds.min.z);
        corners[1] = model * Vec3(localBounds.max.x, localBounds.min.y, localBounds.min.z);
        corners[2] = model * Vec3(localBounds.min.x, localBounds.max.y, localBounds.min.z);
        corners[3] = model * Vec3(localBounds.min.x, localBounds.min.y, localBounds.max.z);
        corners[4] = model * Vec3(localBounds.max.x, localBounds.max.y, localBounds.min.z);
        corners[5] = model * Vec3(localBounds.max.x, localBounds.min.y, localBounds.max.z);
        corners[6] = model * Vec3(localBounds.min.x, localBounds.max.y, localBounds.max.z);
        corners[7] = model * Vec3(localBounds.max.x, localBounds.max.y, localBounds.max.z);

        // Calculate new world AABB
        Vec3 minBounds = corners[0];
        Vec3 maxBounds = corners[0];

        for (int j = 1; j < 8; ++j) {
            if (corners[j].x < minBounds.x) minBounds.x = corners[j].x;
            if (corners[j].y < minBounds.y) minBounds.y = corners[j].y;
            if (corners[j].z < minBounds.z) minBounds.z = corners[j].z;

            if (corners[j].x > maxBounds.x) maxBounds.x = corners[j].x;
            if (corners[j].y > maxBounds.y) maxBounds.y = corners[j].y;
            if (corners[j].z > maxBounds.z) maxBounds.z = corners[j].z;
        }

        AABB worldBounds(minBounds, maxBounds);
        if (bounds.Intersects(worldBounds)) {
            return true;
        }
    }
    return false;
}

void Renderer::SaveScene(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open scene file for saving: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < m_Meshes.size(); ++i) {
        const Transform& t = m_Transforms[i];
        file << m_Meshes[i].GetSource() << " "
             << t.position.x << " " << t.position.y << " " << t.position.z << " "
             << t.rotation.x << " " << t.rotation.y << " " << t.rotation.z << " "
             << t.scale.x << " " << t.scale.y << " " << t.scale.z << "\n";
    }
    
    std::cout << "Scene saved to " << filename << std::endl;
}

void Renderer::LoadScene(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open scene file for loading: " << filename << std::endl;
        return;
    }

    // Clear current scene
    m_Meshes.clear();
    m_Transforms.clear();
    m_Materials.clear();
    // Keep lights for now, or clear if we save them too (future task)

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string source;
        Vec3 pos, rot, scale;

        ss >> source 
           >> pos.x >> pos.y >> pos.z 
           >> rot.x >> rot.y >> rot.z 
           >> scale.x >> scale.y >> scale.z;

        if (source == "cube") {
            m_Meshes.push_back(Mesh::CreateCube());
        } else {
            m_Meshes.push_back(Mesh::LoadFromOBJ(source));
        }
        
        m_Transforms.push_back(Transform(pos, rot, scale));
        
        // Create default material for loaded object
        auto mat = std::make_shared<Material>();
        mat->texture = m_Texture;
        m_Materials.push_back(mat);
    }

    std::cout << "Scene loaded from " << filename << " with " << m_Meshes.size() << " objects" << std::endl;
}

void Renderer::AddCube(const Transform& transform) {
    m_Meshes.push_back(Mesh::CreateCube());
    m_Transforms.push_back(transform);
    
    auto mat = std::make_shared<Material>();
    mat->texture = m_Texture;
    m_Materials.push_back(mat);
    
    std::cout << "Added cube at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::AddPyramid(const Transform& transform) {
    m_Meshes.push_back(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    m_Transforms.push_back(transform);
    
    auto mat = std::make_shared<Material>();
    mat->diffuse = Vec3(1.0f, 1.0f, 0.0f);
    mat->texture = m_Texture;
    m_Materials.push_back(mat);
    
    std::cout << "Added pyramid at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::RemoveObject(size_t index) {
    if (index < m_Meshes.size()) {
        m_Meshes.erase(m_Meshes.begin() + index);
        m_Transforms.erase(m_Transforms.begin() + index);
        if (index < m_Materials.size()) {
            m_Materials.erase(m_Materials.begin() + index);
        }
        std::cout << "Removed object at index " << index << std::endl;
    }
}

bool Renderer::Init() {
    // Create and load shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/textured.vert", "shaders/textured.frag")) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }

    // Load texture
    m_Texture = std::make_shared<Texture>();
    if (!m_Texture->LoadFromFile("assets/brick.png")) {
        std::cerr << "Failed to load texture" << std::endl;
        return false;
    }

    // Setup scene with multiple cubes
    SetupScene();

    // Initialize Skybox
    m_Skybox = std::make_unique<Skybox>();
    std::vector<std::string> faces;
    // Use the same texture for all faces for now
    for(int i=0; i<6; i++) faces.push_back("assets/brick.png");
    
    if (!m_Skybox->Init(faces)) {
        std::cerr << "Failed to initialize skybox" << std::endl;
        // Don't return false, just continue without skybox
    }

    // Initialize Shadow Mapping
    m_DepthShader = std::make_unique<Shader>();
    if (!m_DepthShader->LoadFromFiles("shaders/depth.vert", "shaders/depth.frag")) {
        std::cerr << "Failed to load depth shaders" << std::endl;
        return false;
    }

    m_ShadowMap = std::make_unique<ShadowMap>();
    if (!m_ShadowMap->Init(1024, 1024)) {
        std::cerr << "Failed to initialize shadow map" << std::endl;
        return false;
    }

    return true;
}

void Renderer::Render() {
    if (!m_Camera) return;

    // Calculate light space matrix for shadow mapping (first light only)
    Mat4 lightSpaceMatrix;
    if (m_Lights.size() > 0 && m_Lights[0].castsShadows) {
        // Simple orthographic projection from light's perspective
        float near_plane = 1.0f, far_plane = 15.0f;
        Mat4 lightProjection = Mat4::Orthographic(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
        Mat4 lightView = Mat4::LookAt(m_Lights[0].position, Vec3(0, 0, 0), Vec3(0, 1, 0));
        lightSpaceMatrix = lightProjection * lightView;

        // ===== PASS 1: Render depth map from light's perspective =====
        m_DepthShader->Use();
        m_DepthShader->SetMat4("u_LightSpaceMatrix", lightSpaceMatrix.m);

        glViewport(0, 0, m_ShadowMap->GetWidth(), m_ShadowMap->GetHeight());
        m_ShadowMap->BindForWriting();
        glClear(GL_DEPTH_BUFFER_BIT);

        for (size_t i = 0; i < m_Meshes.size() && i < m_Transforms.size(); ++i) {
            Mat4 model = m_Transforms[i].GetModelMatrix();
            m_DepthShader->SetMat4("u_Model", model.m);
            m_Meshes[i].Draw();
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // ===== PASS 2: Render scene normally with shadows =====
    int width, height;
    glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_Shader->Use();
    
    // Bind shadow map
    m_ShadowMap->BindForReading(1);
    m_Shader->SetInt("shadowMap", 1);
    m_Shader->SetMat4("u_LightSpaceMatrix", lightSpaceMatrix.m);

    // Bind texture
    if (m_Texture) {
        m_Texture->Bind(0);
        m_Shader->SetInt("u_Texture", 0);
    }

    // Light setup
    m_Shader->SetInt("u_LightCount", static_cast<int>(m_Lights.size()));
    
    for (size_t i = 0; i < m_Lights.size(); ++i) {
        std::string base = "u_Lights[" + std::to_string(i) + "]";
        m_Shader->SetVec3(base + ".position", m_Lights[i].position.x, m_Lights[i].position.y, m_Lights[i].position.z);
        m_Shader->SetVec3(base + ".color", m_Lights[i].color.x, m_Lights[i].color.y, m_Lights[i].color.z);
        m_Shader->SetFloat(base + ".intensity", m_Lights[i].intensity);
    }
    
    Vec3 camPos = m_Camera->GetPosition();
    m_Shader->SetVec3("u_ViewPos", camPos.x, camPos.y, camPos.z);

    // Render each mesh
    for (size_t i = 0; i < m_Meshes.size() && i < m_Transforms.size(); ++i) {
        Mat4 model = m_Transforms[i].GetModelMatrix();
        Mat4 view = m_Camera->GetViewMatrix();
        Mat4 projection = m_Camera->GetProjectionMatrix();
        Mat4 mvp = projection * view * model;
        
        m_Shader->SetMat4("u_MVP", mvp.m);
        m_Shader->SetMat4("u_Model", model.m);

        // Bind material
        if (i < m_Materials.size() && m_Materials[i]) {
            m_Materials[i]->Bind(m_Shader.get());
        } else {
            Material defaultMat;
            defaultMat.texture = m_Texture;
            defaultMat.Bind(m_Shader.get());
        }

        m_Meshes[i].Draw();
    }

    // Draw Skybox last
    if (m_Skybox) {
        m_Skybox->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix());
    }
}

void Renderer::Shutdown() {
    m_Meshes.clear();
    m_Transforms.clear();
}
