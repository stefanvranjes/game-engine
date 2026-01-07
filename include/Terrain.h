#pragma once

#include "Math/Vec3.h"
#include "Math/Vec2.h"
#include "Math/Mat4.h"
#include "GLExtensions.h"
#include <vector>
#include <memory>
#include <string>
#include <array>

class Shader;
class Texture;
class Vegetation;

/**
 * @brief AABB for terrain chunk culling
 */
struct TerrainAABB {
    Vec3 min;
    Vec3 max;
    
    bool Intersects(const Vec3& point, float radius) const {
        Vec3 closest;
        closest.x = std::max(min.x, std::min(point.x, max.x));
        closest.y = std::max(min.y, std::min(point.y, max.y));
        closest.z = std::max(min.z, std::min(point.z, max.z));
        float dist = (closest - point).Length();
        return dist < radius;
    }
};

/**
 * @brief A single terrain chunk with its own mesh and LOD
 */
struct TerrainChunk {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLuint EBO = 0;
    int indexCount = 0;
    int lodLevel = 0;
    TerrainAABB bounds;
    int gridX = 0;
    int gridZ = 0;
};

/**
 * @brief Terrain layer for texture splatting
 */
struct TerrainLayer {
    std::shared_ptr<Texture> albedo;
    std::shared_ptr<Texture> normal;
    std::shared_ptr<Texture> roughnessAO;  // R = roughness, G = AO
    float tiling = 10.0f;
    float metallicValue = 0.0f;
};

/**
 * @brief Heightmap-based terrain with LOD and texture splatting
 */
class Terrain {
public:
    Terrain();
    ~Terrain();

    // ========== Initialization ==========
    
    /**
     * @brief Load terrain from a heightmap image
     * @param heightmapPath Path to grayscale heightmap image
     * @param width World width of terrain
     * @param depth World depth of terrain
     * @param heightScale Maximum height of terrain
     * @return true if successful
     */
    bool LoadHeightmap(const std::string& heightmapPath, float width, float depth, float heightScale);

    /**
     * @brief Generate terrain procedurally using noise
     * @param resolution Grid resolution (vertices per side)
     * @param width World width
     * @param depth World depth
     * @param heightScale Maximum height
     */
    void GenerateFromNoise(int resolution, float width, float depth, float heightScale);

    /**
     * @brief Generate the terrain mesh geometry
     * @param chunkSize Vertices per chunk side
     */
    void GenerateMesh(int chunkSize = 32);

    // ========== Runtime ==========

    /**
     * @brief Update LOD levels based on camera position
     * @param cameraPos Camera world position
     */
    void UpdateLOD(const Vec3& cameraPos);

    /**
     * @brief Render the terrain
     * @param shader Terrain shader to use
     * @param view View matrix
     * @param projection Projection matrix
     */
    void Render(Shader* shader, const Mat4& view, const Mat4& projection);

    // ========== Queries ==========

    /**
     * @brief Get terrain height at world position
     * @param x World X coordinate
     * @param z World Z coordinate
     * @return Height at position (interpolated)
     */
    float GetHeightAt(float x, float z) const;

    /**
     * @brief Get terrain normal at world position
     */
    Vec3 GetNormalAt(float x, float z) const;

    // ========== Texturing ==========

    /**
     * @brief Set a texture layer for splatting
     * @param index Layer index (0-3)
     * @param layer Layer data
     */
    void SetLayer(int index, const TerrainLayer& layer);

    /**
     * @brief Set splatmap texture (RGBA = weights for 4 layers)
     */
    void SetSplatmap(std::shared_ptr<Texture> splatmap);

    /**
     * @brief Generate splatmap based on height and slope
     */
    void GenerateSplatmap();

    // ========== Properties ==========

    float GetWidth() const { return m_WorldWidth; }
    float GetDepth() const { return m_WorldDepth; }
    float GetHeightScale() const { return m_HeightScale; }
    int GetResolution() const { return m_Resolution; }

    void SetPosition(const Vec3& pos) { m_Position = pos; }
    Vec3 GetPosition() const { return m_Position; }

    // LOD distances
    std::array<float, 4> m_LODDistances = { 50.0f, 100.0f, 200.0f, 400.0f };
    
    // Vegetation
    std::shared_ptr<Vegetation> m_Vegetation;
    void SetVegetation(std::shared_ptr<Vegetation> veg) { m_Vegetation = veg; }
    std::shared_ptr<Vegetation> GetVegetation() const { return m_Vegetation; }

    // ========== Editing ==========

    /**
     * @brief Set height at heightmap coordinate
     * @param x Heightmap X coordinate
     * @param z Heightmap Z coordinate
     * @param height New height (in world units, not normalized)
     */
    void SetHeightAt(int x, int z, float height);

    /**
     * @brief Modify height at heightmap coordinate
     * @param x Heightmap X coordinate
     * @param z Heightmap Z coordinate
     * @param delta Height change amount
     */
    void ModifyHeight(int x, int z, float delta);

    /**
     * @brief Paint splatmap at heightmap coordinate
     * @param x Heightmap X coordinate
     * @param z Heightmap Z coordinate
     * @param channel Splatmap channel (0-3)
     * @param weight Weight to add (will be normalized)
     */
    void PaintSplatmap(int x, int z, int channel, float weight);

    /**
     * @brief Upload heightmap changes to GPU
     */
    void UpdateHeightmapTexture();

    /**
     * @brief Upload splatmap changes to GPU
     */
    void UpdateSplatmapTexture();

    /**
     * @brief Regenerate mesh after height editing
     */
    void RegenerateMesh();

private:
    void CreateChunkMesh(TerrainChunk& chunk, int startX, int startZ, int size, int lodLevel);
    void CleanupChunks();
    float SampleHeight(int x, int z) const;

    // Heightmap data
    std::vector<float> m_HeightData;
    int m_Resolution = 0;          // Vertices per side
    float m_WorldWidth = 100.0f;
    float m_WorldDepth = 100.0f;
    float m_HeightScale = 50.0f;
    Vec3 m_Position = Vec3(0, 0, 0);

    // Chunks
    std::vector<TerrainChunk> m_Chunks;
    int m_ChunkSize = 32;
    int m_ChunksPerSide = 0;

    // Textures
    GLuint m_HeightmapTexture = 0;
    std::shared_ptr<Texture> m_Splatmap;
    std::array<TerrainLayer, 4> m_Layers;
    
    // Splatmap editing data
    std::vector<unsigned char> m_SplatmapData;
    GLuint m_SplatmapTexture = 0;

    // GPU resources
    bool m_Initialized = false;
};
