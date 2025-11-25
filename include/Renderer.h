#pragma once

#include "Shader.h"
#include "Texture.h"
#include "Mesh.h"
#include "Transform.h"
#include <memory>
#include <vector>

class Camera;

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool Init();
    void Render();
    void Shutdown();
    
    void SetCamera(Camera* camera) { m_Camera = camera; }

private:
    void SetupScene();

    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Texture> m_Texture;
    Camera* m_Camera;
    
    std::vector<Mesh> m_Meshes;
    std::vector<Transform> m_Transforms;
};
