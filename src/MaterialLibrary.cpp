#include "MaterialLibrary.h"
#include <iostream>

MaterialLibrary::MaterialLibrary() {
}

MaterialLibrary::~MaterialLibrary() {
    Clear();
}

std::shared_ptr<Material> MaterialLibrary::CreateMaterial(const std::string& name) {
    // Check if material already exists
    if (m_Materials.find(name) != m_Materials.end()) {
        std::cerr << "Material '" << name << "' already exists!" << std::endl;
        return m_Materials[name];
    }

    // Create new material
    auto material = std::make_shared<Material>();
    m_Materials[name] = material;
    
    std::cout << "Created material: " << name << std::endl;
    return material;
}

std::shared_ptr<Material> MaterialLibrary::GetMaterial(const std::string& name) {
    auto it = m_Materials.find(name);
    if (it != m_Materials.end()) {
        return it->second;
    }
    
    std::cerr << "Material '" << name << "' not found!" << std::endl;
    return nullptr;
}

std::shared_ptr<Material> MaterialLibrary::CreateInstance(const std::string& parentName, const std::string& instanceName) {
    // Get parent material
    auto parent = GetMaterial(parentName);
    if (!parent) {
        std::cerr << "Cannot create instance: parent material '" << parentName << "' not found!" << std::endl;
        return nullptr;
    }

    // Check if instance already exists
    if (m_Materials.find(instanceName) != m_Materials.end()) {
        std::cerr << "Material instance '" << instanceName << "' already exists!" << std::endl;
        return m_Materials[instanceName];
    }

    // Create instance
    auto instance = std::make_shared<Material>();
    instance->SetParent(parent);
    m_Materials[instanceName] = instance;
    
    std::cout << "Created material instance: " << instanceName << " (parent: " << parentName << ")" << std::endl;
    return instance;
}

bool MaterialLibrary::HasMaterial(const std::string& name) const {
    return m_Materials.find(name) != m_Materials.end();
}

void MaterialLibrary::Clear() {
    m_Materials.clear();
}

std::vector<std::string> MaterialLibrary::GetMaterialNames() const {
    std::vector<std::string> names;
    for (const auto& pair : m_Materials) {
        names.push_back(pair.first);
    }
    return names;
}
