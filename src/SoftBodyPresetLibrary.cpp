#include "SoftBodyPresetLibrary.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

SoftBodyPresetLibrary::SoftBodyPresetLibrary() {
    LoadDefaultPresets();
}

void SoftBodyPresetLibrary::LoadPresetsFromDirectory(const std::string& directory) {
    try {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Preset directory does not exist: " << directory << std::endl;
            return;
        }
        
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".preset") {
                SoftBodyPreset preset = SoftBodyPreset::LoadFromFile(entry.path().string());
                if (!preset.name.empty()) {
                    AddPreset(preset);
                }
            }
        }
        
        std::cout << "Loaded presets from " << directory << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading presets from directory: " << e.what() << std::endl;
    }
}

void SoftBodyPresetLibrary::AddPreset(const SoftBodyPreset& preset) {
    m_Presets[preset.name] = preset;
}

const SoftBodyPreset* SoftBodyPresetLibrary::GetPreset(const std::string& name) const {
    auto it = m_Presets.find(name);
    if (it != m_Presets.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> SoftBodyPresetLibrary::GetPresetNames() const {
    std::vector<std::string> names;
    names.reserve(m_Presets.size());
    
    for (const auto& pair : m_Presets) {
        names.push_back(pair.first);
    }
    
    return names;
}

std::vector<std::string> SoftBodyPresetLibrary::GetCategories() const {
    std::vector<std::string> categories;
    
    for (const auto& pair : m_Presets) {
        const std::string& category = pair.second.category;
        if (std::find(categories.begin(), categories.end(), category) == categories.end()) {
            categories.push_back(category);
        }
    }
    
    return categories;
}

std::vector<std::string> SoftBodyPresetLibrary::GetPresetsInCategory(const std::string& category) const {
    std::vector<std::string> presets;
    
    for (const auto& pair : m_Presets) {
        if (pair.second.category == category) {
            presets.push_back(pair.first);
        }
    }
    
    return presets;
}

void SoftBodyPresetLibrary::RemovePreset(const std::string& name) {
    m_Presets.erase(name);
}

void SoftBodyPresetLibrary::Clear() {
    m_Presets.clear();
}

void SoftBodyPresetLibrary::LoadDefaultPresets() {
    AddPreset(SoftBodyPreset::CreateJelloPreset());
    AddPreset(SoftBodyPreset::CreateRubberPreset());
    AddPreset(SoftBodyPreset::CreateMetalPreset());
    AddPreset(SoftBodyPreset::CreateClothPreset());
    AddPreset(SoftBodyPreset::CreateFleshPreset());
    
    std::cout << "Loaded " << m_Presets.size() << " default presets" << std::endl;
}
