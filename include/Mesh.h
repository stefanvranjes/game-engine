#pragma once

#include <vector>
#include <string>
#include "Math/AABB.h"

class Mesh {
public:
    Mesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, const std::string& source = "");
    ~Mesh();

    // Delete copy constructor and copy assignment
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // Move constructor and move assignment
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void Draw() const;
    void DrawInstanced(unsigned int count) const;
    
    void Bind() const;
    void Unbind() const;
    
    static Mesh CreateCube();
    static Mesh LoadFromOBJ(const std::string& filename);
    
    unsigned int GetIndexCount() const { return m_IndexCount; }
    const AABB& GetBounds() const { return m_Bounds; }
    const std::string& GetSource() const { return m_Source; }

private:
    void SetupMesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);

    std::string m_Source;
    unsigned int m_VAO;
    unsigned int m_VBO;
    unsigned int m_EBO;
    unsigned int m_IndexCount;
    AABB m_Bounds;
};
