#pragma once

#include "Shader.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"
#include "Math/AABB.h"
#include "Material.h"
#include "Light.h"
#include "Skybox.h"
#include "ShadowMap.h"
#include <memory>
#include <vector>
#include <string>

class Camera;
struct Material;

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
    std::vector<std::shared_ptr<Material>>& GetMaterials() { return m_Materials; }
    std::vector<Light>& GetLights() { return m_Lights; }
    
    void AddCube(const Transform& transform);
    void AddPyramid(const Transform& transform);
    void RemoveObject(size_t index);
    
    void AddLight(const Light& light) { m_Lights.push_back(light); }
    void RemoveLight(size_t index) { 
        if (index < m_Lights.size()) m_Lights.erase(m_Lights.begin() + index); 
    }

private:
    void SetupScene();

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Shader> m_DepthShader;
    std::shared_ptr<Texture> m_Texture; // Default texture
    Camera* m_Camera;
    std::unique_ptr<Skybox> m_Skybox;
    std::unique_ptr<ShadowMap> m_ShadowMap;
    
    std::vector<Mesh> m_Meshes;
    std::vector<Transform> m_Transforms;
    std::vector<std::shared_ptr<Material>> m_Materials;
    std::vector<Light> m_Lights;
};
