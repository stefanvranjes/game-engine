#pragma once

#include "Shader.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"
#include "Math/AABB.h"
#include <memory>
#include <vector>
#include <string>

class Camera;

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool Init();
    void Render();
    void Shutdown();
    
    void SetCamera(Camera* camera) { m_Camera = camera; }
    bool CheckCollision(const AABB& bounds);

    void SaveScene(const std::string& filename);
    void LoadScene(const std::string& filename);

    // Scene manipulation for editor
    std::vector<Mesh>& GetMeshes() { return m_Meshes; }
    std::vector<Transform>& GetTransforms() { return m_Transforms; }
    void AddCube(const Transform& transform);
    void AddPyramid(const Transform& transform);
    void RemoveObject(size_t index);

private:
    void SetupScene();

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Texture> m_Texture;
    Camera* m_Camera;
    
    std::vector<Mesh> m_Meshes;
    std::vector<Transform> m_Transforms;
};
