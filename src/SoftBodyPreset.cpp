#include "SoftBodyPreset.h"
#include "PhysXSoftBody.h"
#include "SoftBodySerializer.h"
#include <fstream>
#include <iostream>

SoftBodyPreset SoftBodyPreset::CreateFromSoftBody(
    const PhysXSoftBody* softBody,
    const std::string& name,
    const std::string& description)
{
    SoftBodyPreset preset;
    preset.name = name;
    preset.description = description;
    preset.category = "Custom";
    
    // Serialize soft body configuration
    preset.configuration = SoftBodySerializer::SerializeToJson(softBody);
    
    return preset;
}

void SoftBodyPreset::ApplyToSoftBody(PhysXSoftBody* softBody) const {
    if (!softBody) return;
    
    // Deserialize configuration to soft body
    SoftBodySerializer::DeserializeFromJson(configuration, softBody);
}

nlohmann::json SoftBodyPreset::Serialize() const {
    nlohmann::json j;
    j["name"] = name;
    j["description"] = description;
    j["category"] = category;
    j["configuration"] = configuration;
    return j;
}

SoftBodyPreset SoftBodyPreset::Deserialize(const nlohmann::json& json) {
    SoftBodyPreset preset;
    preset.name = json.value("name", "Unnamed");
    preset.description = json.value("description", "");
    preset.category = json.value("category", "Custom");
    preset.configuration = json.value("configuration", nlohmann::json());
    return preset;
}

bool SoftBodyPreset::SaveToFile(const std::string& path) const {
    try {
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << path << std::endl;
            return false;
        }
        
        nlohmann::json j = Serialize();
        file << j.dump(4);
        file.close();
        
        std::cout << "Saved preset '" << name << "' to " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving preset: " << e.what() << std::endl;
        return false;
    }
}

SoftBodyPreset SoftBodyPreset::LoadFromFile(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for reading: " << path << std::endl;
            return SoftBodyPreset();
        }
        
        nlohmann::json j;
        file >> j;
        file.close();
        
        return Deserialize(j);
    } catch (const std::exception& e) {
        std::cerr << "Error loading preset: " << e.what() << std::endl;
        return SoftBodyPreset();
    }
}

// ============================================================================
// Default Presets
// ============================================================================

SoftBodyPreset SoftBodyPreset::CreateJelloPreset() {
    SoftBodyPreset preset;
    preset.name = "Jello";
    preset.description = "Soft, jiggly material with high deformation";
    preset.category = "Soft";
    
    preset.configuration = {
        {"basic", {
            {"volumeStiffness", 0.1f},
            {"shapeStiffness", 0.05f},
            {"deformationStiffness", 0.05f},
            {"totalMass", 5.0f}
        }},
        {"tearSystem", {
            {"tearThreshold", 5.0f},
            {"healingEnabled", true},
            {"healingRate", 0.2f},
            {"healingDelay", 2.0f}
        }}
    };
    
    return preset;
}

SoftBodyPreset SoftBodyPreset::CreateRubberPreset() {
    SoftBodyPreset preset;
    preset.name = "Rubber";
    preset.description = "Elastic material that returns to shape";
    preset.category = "Elastic";
    
    preset.configuration = {
        {"basic", {
            {"volumeStiffness", 0.6f},
            {"shapeStiffness", 0.4f},
            {"deformationStiffness", 0.5f},
            {"totalMass", 10.0f}
        }},
        {"tearSystem", {
            {"tearThreshold", 3.0f},
            {"healingEnabled", false}
        }}
    };
    
    return preset;
}

SoftBodyPreset SoftBodyPreset::CreateMetalPreset() {
    SoftBodyPreset preset;
    preset.name = "Metal";
    preset.description = "Rigid material with minimal deformation";
    preset.category = "Rigid";
    
    preset.configuration = {
        {"basic", {
            {"volumeStiffness", 0.95f},
            {"shapeStiffness", 0.9f},
            {"deformationStiffness", 0.95f},
            {"totalMass", 50.0f}
        }},
        {"tearSystem", {
            {"tearThreshold", 1.0f},
            {"plasticityEnabled", true},
            {"plasticThreshold", 2.0f},
            {"plasticityRate", 0.05f}
        }}
    };
    
    return preset;
}

SoftBodyPreset SoftBodyPreset::CreateClothPreset() {
    SoftBodyPreset preset;
    preset.name = "Cloth";
    preset.description = "Flexible fabric-like material";
    preset.category = "Flexible";
    
    preset.configuration = {
        {"basic", {
            {"volumeStiffness", 0.2f},
            {"shapeStiffness", 0.1f},
            {"deformationStiffness", 0.15f},
            {"totalMass", 2.0f}
        }},
        {"tearSystem", {
            {"tearThreshold", 2.5f},
            {"healingEnabled", false}
        }}
    };
    
    return preset;
}

SoftBodyPreset SoftBodyPreset::CreateFleshPreset() {
    SoftBodyPreset preset;
    preset.name = "Flesh";
    preset.description = "Organic material with healing";
    preset.category = "Organic";
    
    preset.configuration = {
        {"basic", {
            {"volumeStiffness", 0.4f},
            {"shapeStiffness", 0.3f},
            {"deformationStiffness", 0.35f},
            {"totalMass", 15.0f}
        }},
        {"tearSystem", {
            {"tearThreshold", 2.0f},
            {"healingEnabled", true},
            {"healingRate", 0.1f},
            {"healingDelay", 1.0f},
            {"plasticityEnabled", true},
            {"plasticThreshold", 1.5f},
            {"plasticityRate", 0.1f}
        }}
    };
    
    return preset;
}
