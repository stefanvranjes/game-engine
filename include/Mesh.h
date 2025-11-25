#pragma once

#include <vector>

class Mesh {
public:
    Mesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);
    ~Mesh();

    void Draw() const;
    
    static Mesh CreateCube();

private:
    void SetupMesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);

    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_EBO;
    unsigned int m_IndexCount;
};
