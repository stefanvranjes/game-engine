#pragma once

#include "Shader.h"
#include "Texture.h"
#include <memory>

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool Init();
    void Render();
    void Shutdown();

private:
    void SetupQuad();

    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_EBO;
    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Texture> m_Texture;
};
