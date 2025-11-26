#pragma once

#include <vector>

class Mesh {
public:
    Mesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);
    ~Mesh();

    // Delete copy constructor and copy assignment
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // Move constructor and move assignment
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void Draw() const;
    
    static Mesh CreateCube();

    unsigned int GetIndexCount() const { return m_IndexCount; }

private:
    void SetupMesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);

    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_EBO;
    unsigned int m_IndexCount;
};
