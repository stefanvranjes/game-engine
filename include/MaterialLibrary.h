#pragma once

#include "MaterialNew.h"
#include <map>
#include <string>
#include <memory>

class MaterialLibrary {
public:
    std::shared_ptr<Material> CreateMaterial(const std::string& name);

    // Get an existing material by name
    std::shared_ptr<Material> GetMaterial(const std::string& name);

    // Create a material instance (child) from a parent material
    std::shared_ptr<Material> CreateInstance(const std::string& parentName, const std::string& instanceName);

    // Check if a material exists
    bool HasMaterial(const std::string& name) const;

    // Clear all materials
    void Clear();
    
    // Resource Listing
    std::vector<std::string> GetMaterialNames() const;

private:
    std::map<std::string, std::shared_ptr<Material>> m_Materials;
};
