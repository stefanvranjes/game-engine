#pragma once

#include "Shader.h"
#include <memory>

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool Init();
    void Render();
    void Shutdown();

private:
    void SetupTriangle();

    unsigned int m_VAO;
    unsigned int m_VBO;
    std::unique_ptr<Shader> m_Shader;
};
