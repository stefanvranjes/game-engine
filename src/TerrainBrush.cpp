#include "TerrainBrush.h"
#include "Terrain.h"
#include <cmath>
#include <algorithm>

TerrainBrush::TerrainBrush() {
}

float TerrainBrush::CalculateFalloff(float distance) const {
    if (distance >= m_Radius) return 0.0f;
    if (m_Radius <= 0.001f) return 1.0f;

    float normalizedDist = distance / m_Radius;
    
    // Apply falloff curve based on type
    float falloffStart = 1.0f - m_Falloff; // Where falloff begins (0 = edge, 1 = center)
    
    if (normalizedDist <= falloffStart) {
        return 1.0f; // Full strength in inner area
    }

    // Calculate falloff in outer ring
    float falloffT = (normalizedDist - falloffStart) / (1.0f - falloffStart);
    falloffT = std::max(0.0f, std::min(1.0f, falloffT));

    switch (m_FalloffType) {
        case BrushFalloff::Linear:
            return 1.0f - falloffT;
            
        case BrushFalloff::Smooth:
            // Smooth cosine falloff
            return 0.5f * (1.0f + std::cos(falloffT * 3.14159f));
            
        case BrushFalloff::Sphere:
            // Spherical falloff (like a hemisphere)
            return std::sqrt(std::max(0.0f, 1.0f - falloffT * falloffT));
            
        case BrushFalloff::Tip:
            // Quadratic falloff (sharp tip)
            return (1.0f - falloffT) * (1.0f - falloffT);
            
        default:
            return 1.0f - falloffT;
    }
}

void TerrainBrush::Apply(Terrain* terrain, float worldX, float worldZ, float deltaTime) {
    if (!terrain) return;

    Vec3 terrainPos = terrain->GetPosition();
    float terrainWidth = terrain->GetWidth();
    float terrainDepth = terrain->GetDepth();
    int resolution = terrain->GetResolution();
    
    if (resolution <= 0) return;

    float cellWidth = terrainWidth / (resolution - 1);
    float cellDepth = terrainDepth / (resolution - 1);

    // Convert world position to heightmap coordinates
    float localX = worldX - terrainPos.x;
    float localZ = worldZ - terrainPos.z;
    
    int centerHX = static_cast<int>(localX / cellWidth);
    int centerHZ = static_cast<int>(localZ / cellDepth);

    // Calculate brush radius in heightmap cells
    int radiusCells = static_cast<int>(std::ceil(m_Radius / cellWidth)) + 1;

    // For smooth mode, collect neighbors first
    std::vector<float> neighborHeights;

    // Iterate over affected cells
    for (int hz = centerHZ - radiusCells; hz <= centerHZ + radiusCells; ++hz) {
        for (int hx = centerHX - radiusCells; hx <= centerHX + radiusCells; ++hx) {
            if (hx < 0 || hx >= resolution || hz < 0 || hz >= resolution) continue;

            // World position of this cell
            float cellWorldX = terrainPos.x + hx * cellWidth;
            float cellWorldZ = terrainPos.z + hz * cellDepth;

            // Distance from brush center
            float dx = cellWorldX - worldX;
            float dz = cellWorldZ - worldZ;
            float dist = std::sqrt(dx * dx + dz * dz);

            if (dist > m_Radius) continue;

            // Calculate brush influence
            float influence = CalculateFalloff(dist) * m_Strength * deltaTime;

            // Get current height
            float currentHeight = terrain->GetHeightAt(cellWorldX, cellWorldZ);

            float newHeight = currentHeight;

            switch (m_Mode) {
                case BrushMode::Raise:
                    newHeight = currentHeight + influence * terrain->GetHeightScale() * 0.1f;
                    break;

                case BrushMode::Lower:
                    newHeight = currentHeight - influence * terrain->GetHeightScale() * 0.1f;
                    break;

                case BrushMode::SetHeight:
                    newHeight = currentHeight + (m_TargetHeight - currentHeight) * influence;
                    break;

                case BrushMode::Flatten: {
                    // Flatten toward target height
                    float diff = m_TargetHeight - currentHeight;
                    newHeight = currentHeight + diff * influence;
                    break;
                }

                case BrushMode::Smooth: {
                    // Average with neighbors
                    float avgHeight = 0.0f;
                    int count = 0;
                    for (int nz = -1; nz <= 1; ++nz) {
                        for (int nx = -1; nx <= 1; ++nx) {
                            int nhx = hx + nx;
                            int nhz = hz + nz;
                            if (nhx >= 0 && nhx < resolution && nhz >= 0 && nhz < resolution) {
                                avgHeight += terrain->GetHeightAt(
                                    terrainPos.x + nhx * cellWidth,
                                    terrainPos.z + nhz * cellDepth
                                );
                                count++;
                            }
                        }
                    }
                    if (count > 0) {
                        avgHeight /= count;
                        newHeight = currentHeight + (avgHeight - currentHeight) * influence;
                    }
                    break;
                }

                case BrushMode::Paint: {
                    // Paint splatmap - handled separately
                    terrain->PaintSplatmap(hx, hz, m_PaintChannel, influence * m_PaintWeight);
                    continue; // Skip height modification
                }
            }

            // Apply height change
            terrain->SetHeightAt(hx, hz, newHeight);
        }
    }

    // Update GPU resources
    terrain->UpdateHeightmapTexture();
    if (m_Mode == BrushMode::Paint) {
        terrain->UpdateSplatmapTexture();
    }
}

void TerrainBrush::ApplyCustom(Terrain* terrain, float worldX, float worldZ, float deltaTime,
                               std::function<float(float currentHeight, float brushInfluence)> func) {
    if (!terrain || !func) return;

    Vec3 terrainPos = terrain->GetPosition();
    float terrainWidth = terrain->GetWidth();
    float terrainDepth = terrain->GetDepth();
    int resolution = terrain->GetResolution();
    
    if (resolution <= 0) return;

    float cellWidth = terrainWidth / (resolution - 1);
    float cellDepth = terrainDepth / (resolution - 1);

    float localX = worldX - terrainPos.x;
    float localZ = worldZ - terrainPos.z;
    
    int centerHX = static_cast<int>(localX / cellWidth);
    int centerHZ = static_cast<int>(localZ / cellDepth);
    int radiusCells = static_cast<int>(std::ceil(m_Radius / cellWidth)) + 1;

    for (int hz = centerHZ - radiusCells; hz <= centerHZ + radiusCells; ++hz) {
        for (int hx = centerHX - radiusCells; hx <= centerHX + radiusCells; ++hx) {
            if (hx < 0 || hx >= resolution || hz < 0 || hz >= resolution) continue;

            float cellWorldX = terrainPos.x + hx * cellWidth;
            float cellWorldZ = terrainPos.z + hz * cellDepth;

            float dx = cellWorldX - worldX;
            float dz = cellWorldZ - worldZ;
            float dist = std::sqrt(dx * dx + dz * dz);

            if (dist > m_Radius) continue;

            float influence = CalculateFalloff(dist) * m_Strength * deltaTime;
            float currentHeight = terrain->GetHeightAt(cellWorldX, cellWorldZ);
            float newHeight = func(currentHeight, influence);

            terrain->SetHeightAt(hx, hz, newHeight);
        }
    }

    terrain->UpdateHeightmapTexture();
}

void TerrainBrush::GetPreviewCircle(float centerX, float centerZ, std::vector<Vec3>& outPoints, int segments) {
    outPoints.clear();
    outPoints.reserve(segments + 1);

    float angleStep = 6.28318f / segments;
    
    for (int i = 0; i <= segments; ++i) {
        float angle = i * angleStep;
        Vec3 point;
        point.x = centerX + std::cos(angle) * m_Radius;
        point.y = 0.0f; // Will be set to terrain height by caller
        point.z = centerZ + std::sin(angle) * m_Radius;
        outPoints.push_back(point);
    }
}
