#pragma once

#include "Math/Vec3.h"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>

class PhysXSoftBody;
class SoftBodyLODConfig;

/**
 * @brief Serializer for PhysX soft body configurations
 * 
 * Provides JSON-based serialization for complete soft body state including:
 * - Basic properties (stiffness, damping, collision)
 * - Mesh data (vertices, tetrahedra)
 * - LOD configuration
 * - Material properties (anisotropic, resistance map)
 * - Tear system (patterns, fracture lines, healing)
 */
class SoftBodySerializer {
public:
    /**
     * @brief Serialize soft body to JSON
     * @param softBody Soft body to serialize
     * @return JSON object containing complete configuration
     */
    static nlohmann::json SerializeToJson(const PhysXSoftBody* softBody);
    
    /**
     * @brief Deserialize soft body from JSON
     * @param json JSON object containing configuration
     * @param softBody Soft body to populate (must be initialized)
     * @return True if successful
     */
    static bool DeserializeFromJson(const nlohmann::json& json, PhysXSoftBody* softBody);
    
    /**
     * @brief Save soft body configuration to file
     * @param softBody Soft body to save
     * @param filename Output file path
     * @return True if successful
     */
    static bool SaveToFile(const PhysXSoftBody* softBody, const std::string& filename);
    
    /**
     * @brief Load soft body configuration from file
     * @param filename Input file path
     * @param softBody Soft body to populate (must be initialized)
     * @return True if successful
     */
    static bool LoadFromFile(const std::string& filename, PhysXSoftBody* softBody);
    
private:
    // Serialization helpers for different sections
    static nlohmann::json SerializeBasicProperties(const PhysXSoftBody* softBody);
    static nlohmann::json SerializeMeshData(const PhysXSoftBody* softBody);
    static nlohmann::json SerializeLODConfig(const PhysXSoftBody* softBody);
    static nlohmann::json SerializeMaterialProperties(const PhysXSoftBody* softBody);
    static nlohmann::json SerializeTearSystem(const PhysXSoftBody* softBody);
    
    // Deserialization helpers
    static bool DeserializeBasicProperties(const nlohmann::json& json, PhysXSoftBody* softBody);
    static bool DeserializeMeshData(const nlohmann::json& json, PhysXSoftBody* softBody);
    static bool DeserializeLODConfig(const nlohmann::json& json, PhysXSoftBody* softBody);
    static bool DeserializeMaterialProperties(const nlohmann::json& json, PhysXSoftBody* softBody);
    static bool DeserializeTearSystem(const nlohmann::json& json, PhysXSoftBody* softBody);
};
