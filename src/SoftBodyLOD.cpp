#include "SoftBodyLOD.h"
#include "TetrahedralMeshSimplifier.h"
#include <algorithm>
#include <cmath>
#include <iostream>

SoftBodyLODConfig::SoftBodyLODConfig()
    : m_Hysteresis(2.0f)  // 2 meter hysteresis by default
{
}

void SoftBodyLODConfig::AddLODLevel(const SoftBodyLODLevel& level) {
    m_Levels.push_back(level);
    
    // Sort by distance (should already be sorted, but ensure it)
    std::sort(m_Levels.begin(), m_Levels.end(),
        [](const SoftBodyLODLevel& a, const SoftBodyLODLevel& b) {
            return a.minDistance < b.minDistance;
        });
}

int SoftBodyLODConfig::GetLODForDistance(float distance, int currentLOD) const {
    if (m_Levels.empty()) {
        return 0;
    }
    
    // Find appropriate LOD level
    int selectedLOD = 0;
    for (int i = static_cast<int>(m_Levels.size()) - 1; i >= 0; --i) {
        if (distance >= m_Levels[i].minDistance) {
            selectedLOD = i;
            break;
        }
    }
    
    // Apply hysteresis if we have a current LOD
    if (currentLOD >= 0 && currentLOD < static_cast<int>(m_Levels.size())) {
        // If transitioning to higher LOD (lower quality), add hysteresis
        if (selectedLOD > currentLOD) {
            float threshold = m_Levels[selectedLOD].minDistance + m_Hysteresis;
            if (distance < threshold) {
                selectedLOD = currentLOD;  // Stay at current LOD
            }
        }
        // If transitioning to lower LOD (higher quality), subtract hysteresis
        else if (selectedLOD < currentLOD) {
            float threshold = m_Levels[currentLOD].minDistance - m_Hysteresis;
            if (distance > threshold) {
                selectedLOD = currentLOD;  // Stay at current LOD
            }
        }
    }
    
    return selectedLOD;
}

const SoftBodyLODLevel* SoftBodyLODConfig::GetLODLevel(int index) const {
    if (index < 0 || index >= static_cast<int>(m_Levels.size())) {
        return nullptr;
    }
    return &m_Levels[index];
}

void SoftBodyLODConfig::GenerateLODMeshes(
    const std::vector<Vec3>& basePositions,
    const std::vector<int>& baseTetrahedra)
{
    if (basePositions.empty() || baseTetrahedra.empty()) {
        std::cerr << "SoftBodyLODConfig: Cannot generate LOD meshes from empty base mesh" << std::endl;
        return;
    }
    
    std::cout << "SoftBodyLODConfig: Generating LOD meshes from base mesh with " 
              << basePositions.size() << " vertices..." << std::endl;
    
    for (auto& level : m_Levels) {
        if (level.lodIndex == 0) {
            // LOD 0 uses original mesh
            level.vertexPositions = basePositions;
            level.tetrahedronIndices = baseTetrahedra;
            level.vertexCount = static_cast<int>(basePositions.size());
            level.tetrahedronCount = static_cast<int>(baseTetrahedra.size()) / 4;
            level.hasMeshData = true;
            
            // Identity mapping for LOD 0
            level.vertexMapping.resize(level.vertexCount);
            for (int i = 0; i < level.vertexCount; ++i) {
                level.vertexMapping[i] = i;
            }
            
            std::cout << "  LOD 0: " << level.vertexCount << " vertices (original)" << std::endl;
        } 
        else if (level.isFrozen) {
            // Frozen LOD doesn't need mesh data
            level.vertexCount = 0;
            level.tetrahedronCount = 0;
            level.hasMeshData = false;
            
            std::cout << "  LOD " << level.lodIndex << ": Frozen (no mesh)" << std::endl;
        }
        else {
            // Simplify mesh for this LOD
            float ratio = static_cast<float>(level.vertexCount) / static_cast<float>(basePositions.size());
            
            // Ensure we don't try to simplify to too few vertices
            int targetCount = std::max(8, level.vertexCount);  // Minimum 8 vertices for tetrahedra
            
            auto result = TetrahedralMeshSimplifier::Simplify(
                basePositions, 
                baseTetrahedra, 
                targetCount
            );
            
            if (result.success) {
                level.vertexPositions = result.positions;
                level.tetrahedronIndices = result.indices;
                level.vertexCount = result.simplifiedVertexCount;
                level.tetrahedronCount = result.simplifiedTetrahedronCount;
                level.vertexMapping = result.vertexMapping;
                level.hasMeshData = true;
                
                std::cout << "  LOD " << level.lodIndex << ": " 
                          << level.vertexCount << " vertices ("
                          << (ratio * 100.0f) << "% of original)" << std::endl;
            } else {
                std::cerr << "  LOD " << level.lodIndex << ": Simplification failed" << std::endl;
                level.hasMeshData = false;
            }
        }
    }
    
    std::cout << "SoftBodyLODConfig: LOD mesh generation complete" << std::endl;
}

SoftBodyLODConfig SoftBodyLODConfig::CreateDefault(int baseVertexCount, int baseTetrahedronCount) {
    SoftBodyLODConfig config;
    config.SetHysteresis(2.0f);
    
    // LOD 0: Full quality (0-15m)
    SoftBodyLODLevel lod0;
    lod0.lodIndex = 0;
    lod0.minDistance = 0.0f;
    lod0.vertexCount = baseVertexCount;
    lod0.tetrahedronCount = baseTetrahedronCount;
    lod0.solverIterations = 10;
    lod0.substeps = 1;
    lod0.updateFrequency = 1;
    lod0.isFrozen = false;
    config.AddLODLevel(lod0);
    
    // LOD 1: Medium quality (15-35m)
    SoftBodyLODLevel lod1;
    lod1.lodIndex = 1;
    lod1.minDistance = 15.0f;
    lod1.vertexCount = baseVertexCount / 2;
    lod1.tetrahedronCount = baseTetrahedronCount / 2;
    lod1.solverIterations = 5;
    lod1.substeps = 1;
    lod1.updateFrequency = 1;
    lod1.isFrozen = false;
    config.AddLODLevel(lod1);
    
    // LOD 2: Low quality (35-60m)
    SoftBodyLODLevel lod2;
    lod2.lodIndex = 2;
    lod2.minDistance = 35.0f;
    lod2.vertexCount = baseVertexCount / 4;
    lod2.tetrahedronCount = baseTetrahedronCount / 4;
    lod2.solverIterations = 3;
    lod2.substeps = 1;
    lod2.updateFrequency = 2;  // Update every 2 frames
    lod2.isFrozen = false;
    config.AddLODLevel(lod2);
    
    // LOD 3: Frozen (60m+)
    SoftBodyLODLevel lod3;
    lod3.lodIndex = 3;
    lod3.minDistance = 60.0f;
    lod3.vertexCount = 0;
    lod3.tetrahedronCount = 0;
    lod3.solverIterations = 0;
    lod3.substeps = 0;
    lod3.updateFrequency = 0;
    lod3.isFrozen = true;
    config.AddLODLevel(lod3);
    
    return config;
}

// ============================================================================
// Serialization Methods
// ============================================================================

nlohmann::json SoftBodyLODConfig::Serialize() const {
    nlohmann::json j;
    
    j["hysteresis"] = m_Hysteresis;
    j["levelCount"] = static_cast<int>(m_Levels.size());
    
    nlohmann::json levels = nlohmann::json::array();
    for (const auto& level : m_Levels) {
        nlohmann::json levelJson;
        levelJson["lodIndex"] = level.lodIndex;
        levelJson["minDistance"] = level.minDistance;
        levelJson["vertexCount"] = level.vertexCount;
        levelJson["tetrahedronCount"] = level.tetrahedronCount;
        levelJson["solverIterations"] = level.solverIterations;
        levelJson["substeps"] = level.substeps;
        levelJson["updateFrequency"] = level.updateFrequency;
        levelJson["isFrozen"] = level.isFrozen;
        
        levels.push_back(levelJson);
    }
    
    j["levels"] = levels;
    
    return j;
}

SoftBodyLODConfig SoftBodyLODConfig::Deserialize(const nlohmann::json& j) {
    SoftBodyLODConfig config;
    
    if (j.contains("hysteresis")) {
        config.SetHysteresis(j["hysteresis"].get<float>());
    }
    
    if (j.contains("levels") && j["levels"].is_array()) {
        for (const auto& levelJson : j["levels"]) {
            SoftBodyLODLevel level;
            
            level.lodIndex = levelJson.value("lodIndex", 0);
            level.minDistance = levelJson.value("minDistance", 0.0f);
            level.vertexCount = levelJson.value("vertexCount", 0);
            level.tetrahedronCount = levelJson.value("tetrahedronCount", 0);
            level.solverIterations = levelJson.value("solverIterations", 10);
            level.substeps = levelJson.value("substeps", 1);
            level.updateFrequency = levelJson.value("updateFrequency", 1);
            level.isFrozen = levelJson.value("isFrozen", false);
            
            config.AddLODLevel(level);
        }
    }
    
    return config;
}
