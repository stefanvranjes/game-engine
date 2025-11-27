#pragma once

#include "Shader.h"
#include "Math/Mat4.h"
#include <vector>
#include <string>
#include <memory>

class Skybox {
public:
    Skybox();
    ~Skybox();

    bool Init(const std::vector<std::string>& faces);
    void Draw(const Mat4& view, const Mat4& projection);

private:
    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_TextureID;
    std::unique_ptr<Shader> m_Shader;
};
