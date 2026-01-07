#include "Terrain.h"
#include "Shader.h"
#include "Texture.h"
#include "NoiseGenerator.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION_GUARD
#ifndef STB_IMAGE_INCLUDED
#include "stb_image.h"
#define STB_IMAGE_INCLUDED
#endif

Terrain::Terrain() {
    // Initialize default layers
    for (auto& layer : m_Layers) {
        layer.tiling = 10.0f;
        layer.metallicValue = 0.0f;
    }
}

Terrain::~Terrain() {
    CleanupChunks();
    if (m_HeightmapTexture) {
        glDeleteTextures(1, &m_HeightmapTexture);
    }
}

void Terrain::CleanupChunks() {
    for (auto& chunk : m_Chunks) {
        if (chunk.VAO) glDeleteVertexArrays(1, &chunk.VAO);
        if (chunk.VBO) glDeleteBuffers(1, &chunk.VBO);
        if (chunk.EBO) glDeleteBuffers(1, &chunk.EBO);
    }
    m_Chunks.clear();
}

bool Terrain::LoadHeightmap(const std::string& heightmapPath, float width, float depth, float heightScale) {
    int imgWidth, imgHeight, channels;
    unsigned char* data = stbi_load(heightmapPath.c_str(), &imgWidth, &imgHeight, &channels, 1);
    
    if (!data) {
        std::cerr << "Failed to load heightmap: " << heightmapPath << std::endl;
        return false;
    }

    m_Resolution = imgWidth; // Assume square
    m_WorldWidth = width;
    m_WorldDepth = depth;
    m_HeightScale = heightScale;

    // Convert to float heightmap
    m_HeightData.resize(m_Resolution * m_Resolution);
    for (int i = 0; i < m_Resolution * m_Resolution; ++i) {
        m_HeightData[i] = static_cast<float>(data[i]) / 255.0f;
    }

    stbi_image_free(data);

    // Create GPU heightmap texture
    glGenTextures(1, &m_HeightmapTexture);
    glBindTexture(GL_TEXTURE_2D, m_HeightmapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_Resolution, m_Resolution, 0, GL_RED, GL_FLOAT, m_HeightData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    std::cout << "Loaded heightmap: " << m_Resolution << "x" << m_Resolution << std::endl;
    return true;
}

void Terrain::GenerateFromNoise(int resolution, float width, float depth, float heightScale) {
    m_Resolution = resolution;
    m_WorldWidth = width;
    m_WorldDepth = depth;
    m_HeightScale = heightScale;

    m_HeightData.resize(resolution * resolution);

    // Generate using multiple octaves of Perlin noise
    float frequency = 4.0f;
    float amplitude = 1.0f;
    float maxHeight = 0.0f;

    for (int z = 0; z < resolution; ++z) {
        for (int x = 0; x < resolution; ++x) {
            float nx = static_cast<float>(x) / resolution;
            float nz = static_cast<float>(z) / resolution;

            float height = 0.0f;
            float freq = frequency;
            float amp = amplitude;

            // 6 octaves of noise
            for (int oct = 0; oct < 6; ++oct) {
                // Simple noise approximation (replace with proper Perlin if NoiseGenerator is available)
                float noiseVal = std::sin(nx * freq * 6.28f + oct) * 
                                 std::cos(nz * freq * 6.28f + oct * 0.5f);
                noiseVal = (noiseVal + 1.0f) * 0.5f; // Normalize to 0-1
                
                height += noiseVal * amp;
                maxHeight += amp;
                
                freq *= 2.0f;
                amp *= 0.5f;
            }

            m_HeightData[z * resolution + x] = height / maxHeight;
            maxHeight = amplitude + amplitude * 0.5f + amplitude * 0.25f + 
                        amplitude * 0.125f + amplitude * 0.0625f + amplitude * 0.03125f;
        }
    }

    // Normalize
    float minH = *std::min_element(m_HeightData.begin(), m_HeightData.end());
    float maxH = *std::max_element(m_HeightData.begin(), m_HeightData.end());
    float range = maxH - minH;
    if (range > 0.001f) {
        for (auto& h : m_HeightData) {
            h = (h - minH) / range;
        }
    }

    // Create GPU texture
    glGenTextures(1, &m_HeightmapTexture);
    glBindTexture(GL_TEXTURE_2D, m_HeightmapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_Resolution, m_Resolution, 0, GL_RED, GL_FLOAT, m_HeightData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    std::cout << "Generated procedural terrain: " << resolution << "x" << resolution << std::endl;
}

void Terrain::GenerateMesh(int chunkSize) {
    if (m_HeightData.empty()) {
        std::cerr << "No heightmap data to generate mesh from" << std::endl;
        return;
    }

    CleanupChunks();
    m_ChunkSize = chunkSize;
    m_ChunksPerSide = (m_Resolution - 1) / (chunkSize - 1);
    if (m_ChunksPerSide < 1) m_ChunksPerSide = 1;

    // Create chunks
    for (int cz = 0; cz < m_ChunksPerSide; ++cz) {
        for (int cx = 0; cx < m_ChunksPerSide; ++cx) {
            TerrainChunk chunk;
            chunk.gridX = cx;
            chunk.gridZ = cz;
            
            int startX = cx * (chunkSize - 1);
            int startZ = cz * (chunkSize - 1);
            
            CreateChunkMesh(chunk, startX, startZ, chunkSize, 0);
            m_Chunks.push_back(chunk);
        }
    }

    m_Initialized = true;
    std::cout << "Generated terrain mesh with " << m_Chunks.size() << " chunks" << std::endl;
}

void Terrain::CreateChunkMesh(TerrainChunk& chunk, int startX, int startZ, int size, int lodLevel) {
    int step = 1 << lodLevel; // LOD step: 1, 2, 4, 8...
    int verticesPerSide = (size / step) + 1;
    
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    float cellWidth = m_WorldWidth / (m_Resolution - 1);
    float cellDepth = m_WorldDepth / (m_Resolution - 1);

    // Calculate bounds
    chunk.bounds.min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    chunk.bounds.max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // Generate vertices
    for (int z = 0; z < verticesPerSide; ++z) {
        for (int x = 0; x < verticesPerSide; ++x) {
            int hx = std::min(startX + x * step, m_Resolution - 1);
            int hz = std::min(startZ + z * step, m_Resolution - 1);

            float height = SampleHeight(hx, hz) * m_HeightScale;
            float worldX = hx * cellWidth + m_Position.x;
            float worldZ = hz * cellDepth + m_Position.z;
            float worldY = height + m_Position.y;

            // Position
            vertices.push_back(worldX);
            vertices.push_back(worldY);
            vertices.push_back(worldZ);

            // Texture coordinates
            float u = static_cast<float>(hx) / (m_Resolution - 1);
            float v = static_cast<float>(hz) / (m_Resolution - 1);
            vertices.push_back(u);
            vertices.push_back(v);

            // Calculate normal from neighbors
            float hL = SampleHeight(std::max(0, hx - step), hz) * m_HeightScale;
            float hR = SampleHeight(std::min(m_Resolution - 1, hx + step), hz) * m_HeightScale;
            float hD = SampleHeight(hx, std::max(0, hz - step)) * m_HeightScale;
            float hU = SampleHeight(hx, std::min(m_Resolution - 1, hz + step)) * m_HeightScale;

            Vec3 normal(hL - hR, 2.0f * step * cellWidth, hD - hU);
            normal = normal.Normalized();
            vertices.push_back(normal.x);
            vertices.push_back(normal.y);
            vertices.push_back(normal.z);

            // Update bounds
            chunk.bounds.min.x = std::min(chunk.bounds.min.x, worldX);
            chunk.bounds.min.y = std::min(chunk.bounds.min.y, worldY);
            chunk.bounds.min.z = std::min(chunk.bounds.min.z, worldZ);
            chunk.bounds.max.x = std::max(chunk.bounds.max.x, worldX);
            chunk.bounds.max.y = std::max(chunk.bounds.max.y, worldY);
            chunk.bounds.max.z = std::max(chunk.bounds.max.z, worldZ);
        }
    }

    // Generate indices
    for (int z = 0; z < verticesPerSide - 1; ++z) {
        for (int x = 0; x < verticesPerSide - 1; ++x) {
            int topLeft = z * verticesPerSide + x;
            int topRight = topLeft + 1;
            int bottomLeft = (z + 1) * verticesPerSide + x;
            int bottomRight = bottomLeft + 1;

            // Two triangles per quad
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    chunk.indexCount = static_cast<int>(indices.size());
    chunk.lodLevel = lodLevel;

    // Create GPU buffers
    glGenVertexArrays(1, &chunk.VAO);
    glGenBuffers(1, &chunk.VBO);
    glGenBuffers(1, &chunk.EBO);

    glBindVertexArray(chunk.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, chunk.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coords
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Normal
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

float Terrain::SampleHeight(int x, int z) const {
    if (x < 0) x = 0;
    if (z < 0) z = 0;
    if (x >= m_Resolution) x = m_Resolution - 1;
    if (z >= m_Resolution) z = m_Resolution - 1;
    return m_HeightData[z * m_Resolution + x];
}

float Terrain::GetHeightAt(float x, float z) const {
    if (m_HeightData.empty()) return 0.0f;

    // Convert world to local coordinates
    float localX = x - m_Position.x;
    float localZ = z - m_Position.z;

    // Convert to heightmap coordinates
    float cellWidth = m_WorldWidth / (m_Resolution - 1);
    float cellDepth = m_WorldDepth / (m_Resolution - 1);

    float hx = localX / cellWidth;
    float hz = localZ / cellDepth;

    // Clamp to valid range
    hx = std::max(0.0f, std::min(hx, static_cast<float>(m_Resolution - 2)));
    hz = std::max(0.0f, std::min(hz, static_cast<float>(m_Resolution - 2)));

    int x0 = static_cast<int>(hx);
    int z0 = static_cast<int>(hz);
    float fx = hx - x0;
    float fz = hz - z0;

    // Bilinear interpolation
    float h00 = SampleHeight(x0, z0);
    float h10 = SampleHeight(x0 + 1, z0);
    float h01 = SampleHeight(x0, z0 + 1);
    float h11 = SampleHeight(x0 + 1, z0 + 1);

    float h0 = h00 * (1 - fx) + h10 * fx;
    float h1 = h01 * (1 - fx) + h11 * fx;
    float height = h0 * (1 - fz) + h1 * fz;

    return height * m_HeightScale + m_Position.y;
}

Vec3 Terrain::GetNormalAt(float x, float z) const {
    float delta = m_WorldWidth / m_Resolution;
    float hL = GetHeightAt(x - delta, z);
    float hR = GetHeightAt(x + delta, z);
    float hD = GetHeightAt(x, z - delta);
    float hU = GetHeightAt(x, z + delta);

    Vec3 normal(hL - hR, 2.0f * delta, hD - hU);
    return normal.Normalized();
}

void Terrain::UpdateLOD(const Vec3& cameraPos) {
    // TODO: Implement dynamic LOD switching based on distance
    // For now, all chunks use LOD 0
}

void Terrain::Render(Shader* shader, const Mat4& view, const Mat4& projection) {
    if (!m_Initialized || !shader) return;

    shader->Use();
    shader->SetMat4("u_View", view.m);
    shader->SetMat4("u_Projection", projection.m);

    // Bind heightmap
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_HeightmapTexture);
    shader->SetInt("u_Heightmap", 0);

    // Bind splatmap
    if (m_Splatmap) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_Splatmap->GetID());
        shader->SetInt("u_Splatmap", 1);
    }

    // Bind layer textures
    for (int i = 0; i < 4; ++i) {
        if (m_Layers[i].albedo) {
            glActiveTexture(GL_TEXTURE2 + i * 2);
            glBindTexture(GL_TEXTURE_2D, m_Layers[i].albedo->GetID());
            shader->SetInt("u_Layer" + std::to_string(i) + "_Albedo", 2 + i * 2);
        }
        if (m_Layers[i].normal) {
            glActiveTexture(GL_TEXTURE3 + i * 2);
            glBindTexture(GL_TEXTURE_2D, m_Layers[i].normal->GetID());
            shader->SetInt("u_Layer" + std::to_string(i) + "_Normal", 3 + i * 2);
        }
        shader->SetFloat("u_Layer" + std::to_string(i) + "_Tiling", m_Layers[i].tiling);
    }

    // Terrain uniforms
    shader->SetFloat("u_HeightScale", m_HeightScale);
    shader->SetVec3("u_TerrainPos", m_Position.x, m_Position.y, m_Position.z);

    // Render chunks
    for (const auto& chunk : m_Chunks) {
        Mat4 model;
        model.SetIdentity();
        shader->SetMat4("u_Model", model.m);

        glBindVertexArray(chunk.VAO);
        glDrawElements(GL_TRIANGLES, chunk.indexCount, GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
}

void Terrain::SetLayer(int index, const TerrainLayer& layer) {
    if (index >= 0 && index < 4) {
        m_Layers[index] = layer;
    }
}

void Terrain::SetSplatmap(std::shared_ptr<Texture> splatmap) {
    m_Splatmap = splatmap;
}

void Terrain::GenerateSplatmap() {
    if (m_HeightData.empty()) return;

    std::vector<unsigned char> splatData(m_Resolution * m_Resolution * 4);

    for (int z = 0; z < m_Resolution; ++z) {
        for (int x = 0; x < m_Resolution; ++x) {
            int idx = (z * m_Resolution + x) * 4;
            float height = m_HeightData[z * m_Resolution + x];
            Vec3 normal = GetNormalAt(
                x * (m_WorldWidth / (m_Resolution - 1)) + m_Position.x,
                z * (m_WorldDepth / (m_Resolution - 1)) + m_Position.z
            );
            float slope = 1.0f - normal.y; // 0 = flat, 1 = vertical

            // Layer 0: Grass (low areas, flat)
            float grass = std::max(0.0f, 1.0f - height * 2.0f) * (1.0f - slope * 2.0f);
            
            // Layer 1: Dirt (mid heights, some slope)
            float dirt = std::max(0.0f, 1.0f - std::abs(height - 0.4f) * 3.0f) * (1.0f - slope);
            
            // Layer 2: Rock (steep slopes)
            float rock = std::max(0.0f, slope * 2.0f - 0.3f);
            
            // Layer 3: Snow (high areas)
            float snow = std::max(0.0f, (height - 0.7f) * 3.0f) * (1.0f - slope);

            // Normalize weights
            float total = grass + dirt + rock + snow;
            if (total > 0.001f) {
                grass /= total;
                dirt /= total;
                rock /= total;
                snow /= total;
            } else {
                grass = 1.0f;
            }

            splatData[idx + 0] = static_cast<unsigned char>(grass * 255);
            splatData[idx + 1] = static_cast<unsigned char>(dirt * 255);
            splatData[idx + 2] = static_cast<unsigned char>(rock * 255);
            splatData[idx + 3] = static_cast<unsigned char>(snow * 255);
        }
    }

    // Create splatmap texture
    auto splatTex = std::make_shared<Texture>();
    // Note: This requires Texture class to support creation from raw data
    // For now, we'll create it directly
    GLuint texId;
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Resolution, m_Resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, splatData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Store the texture ID (simplified - ideally would wrap in Texture class)
    std::cout << "Generated splatmap texture" << std::endl;
}

void Terrain::SetHeightAt(int x, int z, float height) {
    if (x < 0 || x >= m_Resolution || z < 0 || z >= m_Resolution) return;
    
    // Convert world height to normalized height
    float normalizedHeight = (height - m_Position.y) / m_HeightScale;
    normalizedHeight = std::max(0.0f, std::min(1.0f, normalizedHeight));
    
    m_HeightData[z * m_Resolution + x] = normalizedHeight;
}

void Terrain::ModifyHeight(int x, int z, float delta) {
    if (x < 0 || x >= m_Resolution || z < 0 || z >= m_Resolution) return;
    
    float& h = m_HeightData[z * m_Resolution + x];
    h += delta / m_HeightScale;
    h = std::max(0.0f, std::min(1.0f, h));
}

void Terrain::PaintSplatmap(int x, int z, int channel, float weight) {
    if (x < 0 || x >= m_Resolution || z < 0 || z >= m_Resolution) return;
    if (channel < 0 || channel > 3) return;
    
    // Ensure splatmap data exists
    if (m_SplatmapData.empty()) {
        m_SplatmapData.resize(m_Resolution * m_Resolution * 4, 0);
        // Initialize with first channel
        for (int i = 0; i < m_Resolution * m_Resolution; ++i) {
            m_SplatmapData[i * 4] = 255;
        }
    }
    
    int idx = (z * m_Resolution + x) * 4;
    
    // Add weight to target channel
    float current = static_cast<float>(m_SplatmapData[idx + channel]) / 255.0f;
    current += weight;
    current = std::max(0.0f, std::min(1.0f, current));
    m_SplatmapData[idx + channel] = static_cast<unsigned char>(current * 255);
    
    // Normalize all channels
    float total = 0.0f;
    for (int c = 0; c < 4; ++c) {
        total += m_SplatmapData[idx + c];
    }
    if (total > 0.001f) {
        for (int c = 0; c < 4; ++c) {
            m_SplatmapData[idx + c] = static_cast<unsigned char>(
                (m_SplatmapData[idx + c] / total) * 255
            );
        }
    }
}

void Terrain::UpdateHeightmapTexture() {
    if (m_HeightData.empty() || m_HeightmapTexture == 0) return;
    
    glBindTexture(GL_TEXTURE_2D, m_HeightmapTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Resolution, m_Resolution, 
                    GL_RED, GL_FLOAT, m_HeightData.data());
}

void Terrain::UpdateSplatmapTexture() {
    if (m_SplatmapData.empty()) return;
    
    if (!m_SplatmapTexture) {
        glGenTextures(1, &m_SplatmapTexture);
        glBindTexture(GL_TEXTURE_2D, m_SplatmapTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Resolution, m_Resolution, 0, 
                     GL_RGBA, GL_UNSIGNED_BYTE, m_SplatmapData.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, m_SplatmapTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Resolution, m_Resolution, 
                        GL_RGBA, GL_UNSIGNED_BYTE, m_SplatmapData.data());
    }
}

void Terrain::RegenerateMesh() {
    if (m_HeightData.empty()) return;
    
    // Store chunk size and regenerate
    int chunkSize = m_ChunkSize > 0 ? m_ChunkSize : 32;
    GenerateMesh(chunkSize);
}
