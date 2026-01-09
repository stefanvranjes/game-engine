#include "SoftBodySerializer.h"
#include "SoftBodySerializationHelpers.h"
#include "PhysXSoftBody.h"
#include "SoftBodyLOD.h"
#include "TearResistanceMap.h"
#include <fstream>
#include <iostream>

using namespace SoftBodySerializationHelpers;
using json = nlohmann::json;

// ============================================================================
// Main Serialization Functions
// ============================================================================

json SoftBodySerializer::SerializeToJson(const PhysXSoftBody* softBody) {
    if (!softBody) {
        std::cerr << "SoftBodySerializer: Cannot serialize null soft body" << std::endl;
        return json();
    }
    
    json result;
    result["version"] = GetCurrentVersion();
    
    // Serialize different sections
    result["basic"] = SerializeBasicProperties(softBody);
    result["mesh"] = SerializeMeshData(softBody);
    result["lod"] = SerializeLODConfig(softBody);
    result["material"] = SerializeMaterialProperties(softBody);
    result["tearSystem"] = SerializeTearSystem(softBody);
    
    return result;
}

bool SoftBodySerializer::DeserializeFromJson(const json& j, PhysXSoftBody* softBody) {
    if (!softBody) {
        std::cerr << "SoftBodySerializer: Cannot deserialize to null soft body" << std::endl;
        return false;
    }
    
    // Validate version
    if (!j.contains("version")) {
        std::cerr << "SoftBodySerializer: Missing version field" << std::endl;
        return false;
    }
    
    std::string version = j["version"].get<std::string>();
    if (!ValidateVersion(version, 1)) {
        std::cerr << "SoftBodySerializer: Incompatible version " << version << std::endl;
        return false;
    }
    
    // Deserialize sections
    bool success = true;
    
    if (j.contains("basic")) {
        success &= DeserializeBasicProperties(j["basic"], softBody);
    }
    
    if (j.contains("mesh")) {
        success &= DeserializeMeshData(j["mesh"], softBody);
    }
    
    if (j.contains("lod")) {
        success &= DeserializeLODConfig(j["lod"], softBody);
    }
    
    if (j.contains("material")) {
        success &= DeserializeMaterialProperties(j["material"], softBody);
    }
    
    if (j.contains("tearSystem")) {
        success &= DeserializeTearSystem(j["tearSystem"], softBody);
    }
    
    return success;
}

bool SoftBodySerializer::SaveToFile(const PhysXSoftBody* softBody, const std::string& filename) {
    if (!softBody) {
        std::cerr << "SoftBodySerializer: Cannot save null soft body" << std::endl;
        return false;
    }
    
    try {
        json j = SerializeToJson(softBody);
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "SoftBodySerializer: Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write with pretty printing (4 space indent)
        file << j.dump(4);
        file.close();
        
        std::cout << "SoftBodySerializer: Saved soft body to " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Exception while saving: " << e.what() << std::endl;
        return false;
    }
}

bool SoftBodySerializer::LoadFromFile(const std::string& filename, PhysXSoftBody* softBody) {
    if (!softBody) {
        std::cerr << "SoftBodySerializer: Cannot load to null soft body" << std::endl;
        return false;
    }
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "SoftBodySerializer: Failed to open file for reading: " << filename << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        file.close();
        
        bool success = DeserializeFromJson(j, softBody);
        
        if (success) {
            std::cout << "SoftBodySerializer: Loaded soft body from " << filename << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Exception while loading: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Basic Properties Serialization
// ============================================================================

json SoftBodySerializer::SerializeBasicProperties(const PhysXSoftBody* softBody) {
    json j;
    
    // Counts
    j["vertexCount"] = softBody->GetVertexCount();
    j["tetrahedronCount"] = 0; // TODO: Add getter for tetrahedron count
    
    // Stiffness
    j["volumeStiffness"] = softBody->GetVolumeStiffness();
    j["shapeStiffness"] = softBody->GetShapeStiffness();
    j["deformationStiffness"] = softBody->GetDeformationStiffness();
    
    // Mass
    j["totalMass"] = softBody->GetTotalMass();
    
    // Collision
    j["collisionMargin"] = 0.01f; // TODO: Add getter
    
    // State
    j["enabled"] = softBody->IsEnabled();
    j["tearable"] = false; // TODO: Add getter
    
    // Tear threshold
    j["tearThreshold"] = softBody->GetTearThreshold();
    
    return j;
}

bool SoftBodySerializer::DeserializeBasicProperties(const json& j, PhysXSoftBody* softBody) {
    try {
        // Apply stiffness
        if (j.contains("volumeStiffness")) {
            softBody->SetVolumeStiffness(j["volumeStiffness"].get<float>());
        }
        
        if (j.contains("shapeStiffness")) {
            softBody->SetShapeStiffness(j["shapeStiffness"].get<float>());
        }
        
        if (j.contains("deformationStiffness")) {
            softBody->SetDeformationStiffness(j["deformationStiffness"].get<float>());
        }
        
        // Apply state
        if (j.contains("enabled")) {
            softBody->SetEnabled(j["enabled"].get<bool>());
        }
        
        if (j.contains("tearThreshold")) {
            softBody->SetTearThreshold(j["tearThreshold"].get<float>());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Error deserializing basic properties: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Mesh Data Serialization
// ============================================================================

json SoftBodySerializer::SerializeMeshData(const PhysXSoftBody* softBody) {
    json j;
    
    int vertexCount = softBody->GetVertexCount();
    
    // Get vertex positions
    std::vector<Vec3> positions(vertexCount);
    softBody->GetVertexPositions(positions.data());
    
    j["vertexPositions"] = SerializeVec3Array(positions);
    
    // TODO: Serialize tetrahedral indices when getter is available
    // j["tetrahedronIndices"] = SerializeIntArray(tetrahedra);
    
    return j;
}

bool SoftBodySerializer::DeserializeMeshData(const json& j, PhysXSoftBody* softBody) {
    try {
        // Vertex positions
        if (j.contains("vertexPositions")) {
            std::vector<Vec3> positions = DeserializeVec3Array(j["vertexPositions"]);
            if (!positions.empty()) {
                softBody->SetVertexPositions(positions.data());
            }
        }
        
        // TODO: Deserialize tetrahedral indices when setter is available
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Error deserializing mesh data: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// LOD Configuration Serialization
// ============================================================================

json SoftBodySerializer::SerializeLODConfig(const PhysXSoftBody* softBody) {
    json j;
    
    const SoftBodyLODConfig* lodConfig = softBody->GetLODConfig();
    if (!lodConfig) {
        j["enabled"] = false;
        return j;
    }
    
    j["enabled"] = softBody->IsLODEnabled();
    j["currentLOD"] = softBody->GetCurrentLOD();
    j["hysteresis"] = lodConfig->GetHysteresis();
    j["levelCount"] = lodConfig->GetLODCount();
    
    // Serialize each LOD level
    json levels = json::array();
    for (int i = 0; i < lodConfig->GetLODCount(); ++i) {
        const SoftBodyLODLevel* level = lodConfig->GetLODLevel(i);
        if (!level) continue;
        
        json levelJson;
        levelJson["lodIndex"] = level->lodIndex;
        levelJson["minDistance"] = level->minDistance;
        levelJson["vertexCount"] = level->vertexCount;
        levelJson["tetrahedronCount"] = level->tetrahedronCount;
        levelJson["solverIterations"] = level->solverIterations;
        levelJson["substeps"] = level->substeps;
        levelJson["updateFrequency"] = level->updateFrequency;
        levelJson["isFrozen"] = level->isFrozen;
        levelJson["hasMeshData"] = level->hasMeshData;
        
        // Optionally serialize mesh data for each LOD
        if (level->hasMeshData) {
            levelJson["vertexPositions"] = SerializeVec3Array(level->vertexPositions);
            levelJson["tetrahedronIndices"] = SerializeIntArray(level->tetrahedronIndices);
            levelJson["vertexMapping"] = SerializeIntArray(level->vertexMapping);
        }
        
        levels.push_back(levelJson);
    }
    
    j["levels"] = levels;
    
    return j;
}

bool SoftBodySerializer::DeserializeLODConfig(const json& j, PhysXSoftBody* softBody) {
    try {
        if (!j.contains("enabled") || !j["enabled"].get<bool>()) {
            return true; // LOD not enabled, skip
        }
        
        SoftBodyLODConfig config;
        
        if (j.contains("hysteresis")) {
            config.SetHysteresis(j["hysteresis"].get<float>());
        }
        
        // Deserialize LOD levels
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
                level.hasMeshData = levelJson.value("hasMeshData", false);
                
                // Deserialize mesh data if present
                if (level.hasMeshData && levelJson.contains("vertexPositions")) {
                    level.vertexPositions = DeserializeVec3Array(levelJson["vertexPositions"]);
                    level.tetrahedronIndices = DeserializeIntArray(levelJson["tetrahedronIndices"]);
                    level.vertexMapping = DeserializeIntArray(levelJson["vertexMapping"]);
                }
                
                config.AddLODLevel(level);
            }
        }
        
        // Apply LOD configuration
        softBody->SetLODConfig(config);
        
        if (j.contains("enabled")) {
            softBody->SetLODEnabled(j["enabled"].get<bool>());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Error deserializing LOD config: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Material Properties Serialization
// ============================================================================

json SoftBodySerializer::SerializeMaterialProperties(const PhysXSoftBody* softBody) {
    json j;
    
    // Tear resistance map
    const TearResistanceMap& resistanceMap = softBody->GetResistanceMap();
    
    json resistanceJson;
    resistanceJson["tetrahedronCount"] = 0; // TODO: Get from resistance map
    // TODO: Serialize resistance values when getter is available
    
    j["resistanceMap"] = resistanceJson;
    
    // TODO: Serialize anisotropic material when available
    j["anisotropic"] = json::object();
    
    return j;
}

bool SoftBodySerializer::DeserializeMaterialProperties(const json& j, PhysXSoftBody* softBody) {
    try {
        // TODO: Deserialize resistance map when setters are available
        // TODO: Deserialize anisotropic material when available
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Error deserializing material properties: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Tear System Serialization
// ============================================================================

json SoftBodySerializer::SerializeTearSystem(const PhysXSoftBody* softBody) {
    json j;
    
    // Healing settings
    j["healingEnabled"] = false; // TODO: Add getter
    j["healingRate"] = 0.0f; // TODO: Add getter
    j["healingDelay"] = 0.0f; // TODO: Add getter
    
    // Plasticity settings
    j["plasticityEnabled"] = false; // TODO: Add getter
    j["plasticThreshold"] = 0.0f; // TODO: Add getter
    j["plasticityRate"] = 0.0f; // TODO: Add getter
    
    // TODO: Serialize fracture lines when available
    j["fractureLines"] = json::array();
    
    // TODO: Serialize tear patterns when available
    j["tearPatterns"] = json::array();
    
    return j;
}

bool SoftBodySerializer::DeserializeTearSystem(const json& j, PhysXSoftBody* softBody) {
    try {
        // Apply healing settings
        if (j.contains("healingEnabled")) {
            softBody->SetHealingEnabled(j["healingEnabled"].get<bool>());
        }
        
        if (j.contains("healingRate")) {
            softBody->SetHealingRate(j["healingRate"].get<float>());
        }
        
        if (j.contains("healingDelay")) {
            softBody->SetHealingDelay(j["healingDelay"].get<float>());
        }
        
        // Apply plasticity settings
        if (j.contains("plasticityEnabled")) {
            softBody->SetPlasticityEnabled(j["plasticityEnabled"].get<bool>());
        }
        
        if (j.contains("plasticThreshold")) {
            softBody->SetPlasticThreshold(j["plasticThreshold"].get<float>());
        }
        
        if (j.contains("plasticityRate")) {
            softBody->SetPlasticityRate(j["plasticityRate"].get<float>());
        }
        
        // TODO: Deserialize fracture lines when available
        // TODO: Deserialize tear patterns when available
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "SoftBodySerializer: Error deserializing tear system: " << e.what() << std::endl;
        return false;
    }
}
