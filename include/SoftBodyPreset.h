#pragma once

#include <string>
#include <nlohmann/json.hpp>

class PhysXSoftBody;

/**
 * @brief Represents a named soft body configuration preset
 * 
 * Presets allow quick application of predefined soft body properties
 * for common material types (jello, rubber, metal, etc.)
 */
class SoftBodyPreset {
public:
    std::string name;
    std::string description;
    std::string category;
    nlohmann::json configuration;
    
    /**
     * @brief Create preset from existing soft body
     * @param softBody Soft body to capture configuration from
     * @param name Preset name
     * @param description Optional description
     * @return New preset
     */
    static SoftBodyPreset CreateFromSoftBody(
        const PhysXSoftBody* softBody,
        const std::string& name,
        const std::string& description = ""
    );
    
    /**
     * @brief Apply preset to soft body
     * @param softBody Soft body to modify
     */
    void ApplyToSoftBody(PhysXSoftBody* softBody) const;
    
    /**
     * @brief Serialize preset to JSON
     */
    nlohmann::json Serialize() const;
    
    /**
     * @brief Deserialize preset from JSON
     */
    static SoftBodyPreset Deserialize(const nlohmann::json& json);
    
    /**
     * @brief Save preset to file
     */
    bool SaveToFile(const std::string& path) const;
    
    /**
     * @brief Load preset from file
     */
    static SoftBodyPreset LoadFromFile(const std::string& path);
    
    // Default presets
    static SoftBodyPreset CreateJelloPreset();
    static SoftBodyPreset CreateRubberPreset();
    static SoftBodyPreset CreateMetalPreset();
    static SoftBodyPreset CreateClothPreset();
    static SoftBodyPreset CreateFleshPreset();
};
