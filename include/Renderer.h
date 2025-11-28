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
#include "TextureManager.h"
#include "GameObject.h"
#include "GBuffer.h"
#include "PostProcessing.h"
#include <memory>
#include <vector>
#include <string>

class Camera;
struct Material;

class Renderer {
public:
    static const int MAX_LIGHTS = 32;

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
    std::shared_ptr<GameObject> GetRoot() { return m_Root; }
    std::vector<Light>& GetLights() { return m_Lights; }

    TextureManager* GetTextureManager() { return m_TextureManager.get(); }
    PostProcessing* GetPostProcessing() { return m_PostProcessing.get(); }
    
    void AddCube(const Transform& transform);
    void AddPyramid(const Transform& transform);
    void RemoveObject(size_t index);
    
    void AddLight(const Light& light) { m_Lights.push_back(light); }
    void RemoveLight(size_t index) { 
        if (index < m_Lights.size()) m_Lights.erase(m_Lights.begin() + index); 
    }

private:
    void SetupScene();
    void RenderQuad(); // For fullscreen quad in lighting pass

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Shader> m_DepthShader;
    std::unique_ptr<Shader> m_GeometryShader;
    std::unique_ptr<Shader> m_LightingShader;
    std::shared_ptr<Texture> m_Texture; // Default texture
    Camera* m_Camera;
    std::unique_ptr<Skybox> m_Skybox;
    std::unique_ptr<ShadowMap> m_ShadowMap;
    std::unique_ptr<TextureManager> m_TextureManager;
    std::unique_ptr<GBuffer> m_GBuffer;
    std::unique_ptr<PostProcessing> m_PostProcessing;
    
    unsigned int m_QuadVAO, m_QuadVBO;
    
    std::shared_ptr<GameObject> m_Root;
    std::vector<Light> m_Lights;
};
