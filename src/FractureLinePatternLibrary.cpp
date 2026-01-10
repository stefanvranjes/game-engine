#include "FractureLinePatternLibrary.h"
#include "FractureLineToPattern.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

FractureLinePatternLibrary::FractureLinePatternLibrary() {
}

bool FractureLinePatternLibrary::SavePreset(
    const std::string& name,
    const FractureLine& line,
    const std::string& description)
{
    if (!IsValidPresetName(name)) {
        std::cerr << "Invalid preset name: " << name << std::endl;
        return false;
    }
    
    if (HasPreset(name)) {
        std::cerr << "Preset already exists: " << name << std::endl;
        return false;
    }
    
    PatternPreset preset;
    preset.name = name;
    preset.description = description;
    preset.line = line;
    
    // Auto-detect pattern type
    preset.type = FractureLineToPattern::EstimatePatternType(line);
    preset.curvature = FractureLineToPattern::CalculateCurvature(line);
    
    m_Presets[name] = preset;
    
    std::cout << "Saved preset: " << name << std::endl;
    return true;
}

bool FractureLinePatternLibrary::LoadPreset(const std::string& name, FractureLine& outLine) {
    auto it = m_Presets.find(name);
    if (it == m_Presets.end()) {
        std::cerr << "Preset not found: " << name << std::endl;
        return false;
    }
    
    outLine = it->second.line;
    return true;
}

const FractureLinePatternLibrary::PatternPreset* FractureLinePatternLibrary::GetPreset(const std::string& name) const {
    auto it = m_Presets.find(name);
    if (it == m_Presets.end()) {
        return nullptr;
    }
    return &it->second;
}

bool FractureLinePatternLibrary::DeletePreset(const std::string& name) {
    auto it = m_Presets.find(name);
    if (it == m_Presets.end()) {
        return false;
    }
    
    m_Presets.erase(it);
    std::cout << "Deleted preset: " << name << std::endl;
    return true;
}

bool FractureLinePatternLibrary::RenamePreset(const std::string& oldName, const std::string& newName) {
    if (!IsValidPresetName(newName)) {
        std::cerr << "Invalid new preset name: " << newName << std::endl;
        return false;
    }
    
    auto it = m_Presets.find(oldName);
    if (it == m_Presets.end()) {
        std::cerr << "Preset not found: " << oldName << std::endl;
        return false;
    }
    
    if (HasPreset(newName)) {
        std::cerr << "Preset already exists: " << newName << std::endl;
        return false;
    }
    
    PatternPreset preset = it->second;
    preset.name = newName;
    
    m_Presets.erase(it);
    m_Presets[newName] = preset;
    
    std::cout << "Renamed preset: " << oldName << " -> " << newName << std::endl;
    return true;
}

std::vector<std::string> FractureLinePatternLibrary::GetPresetNames() const {
    std::vector<std::string> names;
    names.reserve(m_Presets.size());
    
    for (const auto& pair : m_Presets) {
        names.push_back(pair.first);
    }
    
    // Sort alphabetically
    std::sort(names.begin(), names.end());
    
    return names;
}

bool FractureLinePatternLibrary::HasPreset(const std::string& name) const {
    return m_Presets.find(name) != m_Presets.end();
}

void FractureLinePatternLibrary::Clear() {
    m_Presets.clear();
    std::cout << "Cleared all presets" << std::endl;
}

bool FractureLinePatternLibrary::SaveToFile(const std::string& filename) const {
    try {
        json j = json::array();
        
        for (const auto& pair : m_Presets) {
            const PatternPreset& preset = pair.second;
            
            json presetJson;
            presetJson["name"] = preset.name;
            presetJson["description"] = preset.description;
            presetJson["type"] = static_cast<int>(preset.type);
            presetJson["curvature"] = preset.curvature;
            
            // Serialize fracture line
            json lineJson;
            lineJson["weakness"] = preset.line.GetWeaknessMultiplier();
            lineJson["width"] = preset.line.GetWidth();
            
            json pointsJson = json::array();
            const auto& points = preset.line.GetPoints();
            for (const Vec3& point : points) {
                pointsJson.push_back({point.x, point.y, point.z});
            }
            lineJson["points"] = pointsJson;
            
            presetJson["line"] = lineJson;
            
            j.push_back(presetJson);
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        file << j.dump(2);  // Pretty print with 2-space indent
        file.close();
        
        std::cout << "Saved library to: " << filename << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving library: " << e.what() << std::endl;
        return false;
    }
}

bool FractureLinePatternLibrary::LoadFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        file.close();
        
        // Clear existing presets
        m_Presets.clear();
        
        for (const auto& presetJson : j) {
            PatternPreset preset;
            
            preset.name = presetJson["name"].get<std::string>();
            preset.description = presetJson.value("description", "");
            preset.type = static_cast<SoftBodyTearPattern::PatternType>(presetJson["type"].get<int>());
            preset.curvature = presetJson.value("curvature", 0.5f);
            
            // Deserialize fracture line
            const json& lineJson = presetJson["line"];
            float weakness = lineJson["weakness"].get<float>();
            float width = lineJson["width"].get<float>();
            
            FractureLine line(weakness);
            line.SetWidth(width);
            
            const json& pointsJson = lineJson["points"];
            for (const auto& pointJson : pointsJson) {
                Vec3 point;
                point.x = pointJson[0].get<float>();
                point.y = pointJson[1].get<float>();
                point.z = pointJson[2].get<float>();
                line.AddPoint(point);
            }
            
            preset.line = line;
            
            m_Presets[preset.name] = preset;
        }
        
        std::cout << "Loaded " << m_Presets.size() << " presets from: " << filename << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading library: " << e.what() << std::endl;
        return false;
    }
}

std::string FractureLinePatternLibrary::GetDefaultLibraryPath() {
    return "fracture_line_patterns.json";
}

bool FractureLinePatternLibrary::IsValidPresetName(const std::string& name) const {
    if (name.empty()) {
        return false;
    }
    
    // Check for invalid characters
    for (char c : name) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || 
            c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
    }
    
    return true;
}
