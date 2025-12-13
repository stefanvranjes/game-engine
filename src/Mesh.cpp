#include "Mesh.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "Math/Vec3.h"
#include "Math/Vec2.h"

Mesh::Mesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, const std::string& source)
    : m_VAO(0), m_VBO(0), m_EBO(0), m_IndexCount(0), m_Source(source),
      m_Vertices(vertices), m_Indices(indices)
{
    SetupMesh(vertices, indices);
}

Mesh::~Mesh() {
    if (m_VAO != 0) glDeleteVertexArrays(1, &m_VAO);
    if (m_VBO != 0) glDeleteBuffers(1, &m_VBO);
    if (m_EBO != 0) glDeleteBuffers(1, &m_EBO);
}

// Move constructor
Mesh::Mesh(Mesh&& other) noexcept
    : m_VAO(other.m_VAO)
    , m_VBO(other.m_VBO)
    , m_EBO(other.m_EBO)
    , m_IndexCount(other.m_IndexCount)
    , m_Bounds(other.m_Bounds)
    , m_Source(std::move(other.m_Source))
    , m_Vertices(std::move(other.m_Vertices))
    , m_Indices(std::move(other.m_Indices))
{
    // Invalidate the source object
    other.m_VAO = 0;
    other.m_VBO = 0;
    other.m_EBO = 0;
    other.m_IndexCount = 0;
    other.m_Vertices.clear();
    other.m_Indices.clear();
}

// Move assignment
Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (m_VAO != 0) glDeleteVertexArrays(1, &m_VAO);
        if (m_VBO != 0) glDeleteBuffers(1, &m_VBO);
        if (m_EBO != 0) glDeleteBuffers(1, &m_EBO);

        // Transfer ownership
        m_VAO = other.m_VAO;
        m_VBO = other.m_VBO;
        m_EBO = other.m_EBO;
        m_IndexCount = other.m_IndexCount;
        m_Bounds = other.m_Bounds;
        m_Source = std::move(other.m_Source);
        m_Vertices = std::move(other.m_Vertices);
        m_Indices = std::move(other.m_Indices);

        // Invalidate source
        other.m_VAO = 0;
        other.m_VBO = 0;
        other.m_EBO = 0;
        other.m_IndexCount = 0;
        other.m_Vertices.clear();
        other.m_Indices.clear();
    }
    return *this;
}

void Mesh::SetupMesh(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) {
    m_IndexCount = static_cast<unsigned int>(indices.size());

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);

    // Calculate AABB bounds
    if (!vertices.empty()) {
        Vec3 minBounds(vertices[0], vertices[1], vertices[2]);
        Vec3 maxBounds(vertices[0], vertices[1], vertices[2]);

        for (size_t i = 0; i < vertices.size(); i += 16) {
            float x = vertices[i];
            float y = vertices[i + 1];
            float z = vertices[i + 2];

            if (x < minBounds.x) minBounds.x = x;
            if (y < minBounds.y) minBounds.y = y;
            if (z < minBounds.z) minBounds.z = z;

            if (x > maxBounds.x) maxBounds.x = x;
            if (y > maxBounds.y) maxBounds.y = y;
            if (z > maxBounds.z) maxBounds.z = z;
        }
        m_Bounds = AABB(minBounds, maxBounds);
    }

    glBindVertexArray(m_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Texture coordinate attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Bone IDs attribute (location 3)
    glVertexAttribIPointer(3, 4, GL_INT, 16 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);
    
    // Bone Weights attribute (location 4)
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)(12 * sizeof(float)));
    glEnableVertexAttribArray(4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Mesh::Draw() const {
    Bind();
    glDrawElements(GL_TRIANGLES, m_IndexCount, GL_UNSIGNED_INT, 0);
    Unbind();
}

void Mesh::DrawInstanced(unsigned int count) const {
    Bind();
    glDrawElementsInstanced(GL_TRIANGLES, m_IndexCount, GL_UNSIGNED_INT, 0, count);
    Unbind();
}

void Mesh::Bind() const {
    glBindVertexArray(m_VAO);
}

void Mesh::Unbind() const {
    glBindVertexArray(0);
}

Mesh Mesh::CreateCube() {
    // Cube vertices: position (x, y, z), normal (nx, ny, nz), texture coords (u, v)
    // 24 vertices (4 per face for proper texturing and normals)
    std::vector<float> vertices = {
        // Front face (Normal: 0, 0, 1)
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,

        // Back face (Normal: 0, 0, -1)
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,

        // Left face (Normal: -1, 0, 0)
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,

        // Right face (Normal: 1, 0, 0)
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,

        // Top face (Normal: 0, 1, 0)
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,

        // Bottom face (Normal: 0, -1, 0)
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f
    };

    // 36 indices (6 faces * 2 triangles * 3 vertices)
    std::vector<unsigned int> indices = {
        0,  1,  2,   2,  3,  0,   // Front
        4,  5,  6,   6,  7,  4,   // Back
        8,  9,  10,  10, 11, 8,   // Left
        12, 13, 14,  14, 15, 12,  // Right
        16, 17, 18,  18, 19, 16,  // Top
        20, 21, 22,  22, 23, 20   // Bottom
    };

    return Mesh(vertices, indices, "cube");
}

Mesh Mesh::LoadFromOBJ(const std::string& filename) {
    std::vector<Vec3> temp_vertices;
    std::vector<Vec2> temp_uvs;
    std::vector<Vec3> temp_normals;

    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    unsigned int indexOffset = 0;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filename << std::endl;
        return CreateCube(); // Fallback
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            Vec3 v;
            ss >> v.x >> v.y >> v.z;
            temp_vertices.push_back(v);
        } else if (prefix == "vt") {
            Vec2 vt;
            ss >> vt.x >> vt.y;
            temp_uvs.push_back(vt);
        } else if (prefix == "vn") {
            Vec3 vn;
            ss >> vn.x >> vn.y >> vn.z;
            temp_normals.push_back(vn);
        } else if (prefix == "f") {
            std::string vertexStr;
            for (int i = 0; i < 3; ++i) { // Assume triangles
                ss >> vertexStr;
                
                std::stringstream vss(vertexStr);
                std::string segment;
                std::vector<std::string> segments;
                
                while (std::getline(vss, segment, '/')) {
                    segments.push_back(segment);
                }

                // Parse indices (1-based)
                int vIdx = std::stoi(segments[0]) - 1;
                int vtIdx = (segments.size() > 1 && !segments[1].empty()) ? std::stoi(segments[1]) - 1 : 0;
                int vnIdx = (segments.size() > 2) ? std::stoi(segments[2]) - 1 : 0;

                // Position
                Vec3 p = temp_vertices[vIdx];
                vertices.push_back(p.x);
                vertices.push_back(p.y);
                vertices.push_back(p.z);

                // Normal
                if (vnIdx >= 0 && vnIdx < temp_normals.size()) {
                    Vec3 n = temp_normals[vnIdx];
                    vertices.push_back(n.x);
                    vertices.push_back(n.y);
                    vertices.push_back(n.z);
                } else {
                    vertices.push_back(0.0f);
                    vertices.push_back(1.0f);
                    vertices.push_back(0.0f);
                }

                // UV
                if (vtIdx >= 0 && vtIdx < temp_uvs.size()) {
                    Vec2 uv = temp_uvs[vtIdx];
                    vertices.push_back(uv.x);
                    vertices.push_back(uv.y);
                } else {
                    vertices.push_back(0.0f);
                    vertices.push_back(0.0f);
                }

                indices.push_back(indexOffset++);
            }
        }
    }

    std::cout << "Loaded OBJ: " << filename << " with " << vertices.size() / 8 << " vertices" << std::endl;
    return Mesh(vertices, indices, filename);
}

// Automatic LOD Generation Implementation
#include <map>
#include <algorithm>

Mesh Mesh::GenerateLOD(int gridSize) const {
    if (m_Vertices.empty() || m_Indices.empty()) return Mesh({}, {});

    // Vertex Clustering Algorithm
    
    // 1. Calculate Bounding Box
    Vec3 minBounds = m_Bounds.min;
    Vec3 maxBounds = m_Bounds.max;
    Vec3 size = maxBounds - minBounds;
    
    // Avoid division by zero
    if (size.x < 0.0001f) size.x = 1.0f;
    if (size.y < 0.0001f) size.y = 1.0f;
    if (size.z < 0.0001f) size.z = 1.0f;
    
    float maxSide = std::max(size.x, std::max(size.y, size.z));
    // gridSize = cells per axis? No, cells per maxSide ideally.
    if (gridSize <= 0) gridSize = 1;
    
    struct VertexData {
        Vec3 position;
        Vec3 normal;
        Vec2 uv;
        int count = 0;
    };

    // Map: Cell ID -> Accumulated Vertex Data
    std::map<unsigned long long, VertexData> cellMap;

    auto GetCellID = [&](const Vec3& pos) -> unsigned long long {
        int x = static_cast<int>((pos.x - minBounds.x) / maxSide * gridSize);
        int y = static_cast<int>((pos.y - minBounds.y) / maxSide * gridSize);
        int z = static_cast<int>((pos.z - minBounds.z) / maxSide * gridSize);
        
        // Clamp to grid
        if (x >= gridSize) x = gridSize - 1;
        if (y >= gridSize) y = gridSize - 1;
        if (z >= gridSize) z = gridSize - 1;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (z < 0) z = 0;
        
        // Pack into 64-bit int (20 bits each)
        return (static_cast<unsigned long long>(x) << 40) | 
               (static_cast<unsigned long long>(y) << 20) | 
               static_cast<unsigned long long>(z);
    };

    // 2. Cluster Vertices
    const int stride = 8; 

    // Mapping: Old Vertex Index -> New Cell ID
    std::vector<unsigned long long> vertexToCell(m_Vertices.size() / stride);

    for (size_t i = 0; i < m_Vertices.size(); i += stride) {
        Vec3 pos(m_Vertices[i], m_Vertices[i+1], m_Vertices[i+2]);
        Vec3 norm(m_Vertices[i+3], m_Vertices[i+4], m_Vertices[i+5]);
        Vec2 uv(m_Vertices[i+6], m_Vertices[i+7]);

        unsigned long long cellID = GetCellID(pos);
        vertexToCell[i / stride] = cellID;

        VertexData& data = cellMap[cellID];
        data.position += pos;
        data.normal += norm;
        data.uv += uv;
        data.count++;
    }

    // 3. Generate New Vertices
    std::vector<float> newVertices;
    std::map<unsigned long long, unsigned int> cellToNewIndex;
    unsigned int currentIndex = 0;

    for (auto& pair : cellMap) {
        VertexData& data = pair.second;
        float invCount = 1.0f / data.count;
        
        data.position *= invCount;
        data.normal = (data.normal * invCount).Normalized();
        data.uv *= invCount;

        // Push 8 floats
        newVertices.push_back(data.position.x);
        newVertices.push_back(data.position.y);
        newVertices.push_back(data.position.z);
        newVertices.push_back(data.normal.x);
        newVertices.push_back(data.normal.y);
        newVertices.push_back(data.normal.z);
        newVertices.push_back(data.uv.x);
        newVertices.push_back(data.uv.y);

        cellToNewIndex[pair.first] = currentIndex++;
    }

    // 4. Generate New Indices
    std::vector<unsigned int> newIndices;
    for (size_t i = 0; i < m_Indices.size(); i += 3) {
        unsigned int idx0 = m_Indices[i];
        unsigned int idx1 = m_Indices[i+1];
        unsigned int idx2 = m_Indices[i+2];

        // Ensure indices are within bounds (sanity check)
        if (idx0 * stride >= m_Vertices.size() || 
            idx1 * stride >= m_Vertices.size() || 
            idx2 * stride >= m_Vertices.size()) continue;

        unsigned long long cell0 = vertexToCell[idx0];
        unsigned long long cell1 = vertexToCell[idx1];
        unsigned long long cell2 = vertexToCell[idx2];

        // If simple triangle (vertices in different cells), add it
        // Note: strictly speaking, we might want to check if they are all distinct or just not all same?
        // A triangle is degenerate if 2 or more vertices are same.
        // We only want non-degenerate triangles.
        if (cell0 != cell1 && cell1 != cell2 && cell0 != cell2) {
            newIndices.push_back(cellToNewIndex[cell0]);
            newIndices.push_back(cellToNewIndex[cell1]);
            newIndices.push_back(cellToNewIndex[cell2]);
        }
    }
    
    // Debug info
    std::cout << "LOD Gen: " << m_Vertices.size() / stride << " -> " << newVertices.size() / stride 
              << " vertices. " << m_Indices.size() / 3 << " -> " << newIndices.size() / 3 << " tris." << std::endl;

    return Mesh(newVertices, newIndices, m_Source + "_LOD");
}
