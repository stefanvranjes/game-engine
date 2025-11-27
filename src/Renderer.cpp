#include "Renderer.h"
#include "Camera.h"
#include "GLExtensions.h"
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

    // Load pyramid from OBJ
    m_Meshes.push_back(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    m_Transforms.push_back(Transform(Vec3(0, 2, 0), Vec3(0, 0, 0), Vec3(2, 2, 2)));

    std::cout << "Scene setup complete with " << m_Meshes.size() << " objects" << std::endl;
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
    }

    std::cout << "Scene loaded from " << filename << " with " << m_Meshes.size() << " objects" << std::endl;
}

void Renderer::AddCube(const Transform& transform) {
    m_Meshes.push_back(Mesh::CreateCube());
    m_Transforms.push_back(transform);
    std::cout << "Added cube at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::AddPyramid(const Transform& transform) {
    m_Meshes.push_back(Mesh::LoadFromOBJ("assets/pyramid.obj"));
    m_Transforms.push_back(transform);
    std::cout << "Added pyramid at position (" << transform.position.x << ", " << transform.position.y << ", " << transform.position.z << ")" << std::endl;
}

void Renderer::RemoveObject(size_t index) {
    if (index < m_Meshes.size()) {
        m_Meshes.erase(m_Meshes.begin() + index);
        m_Transforms.erase(m_Transforms.begin() + index);
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
    m_Texture = std::make_unique<Texture>();
    if (!m_Texture->LoadFromFile("assets/brick.png")) {
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
