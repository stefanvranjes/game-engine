#include "MaterialNew.h"
#include "TextureManager.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool Material::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }

    file << "# Material Preset\n";
    
    // Save colors
    Vec3 ambient = GetAmbient();
    Vec3 diffuse = GetDiffuse();
    Vec3 specular = GetSpecular();
    Vec3 emissive = GetEmissiveColor();
    
    file << "ambient=" << ambient.x << "," << ambient.y << "," << ambient.z << "\n";
    file << "diffuse=" << diffuse.x << "," << diffuse.y << "," << diffuse.z << "\n";
    file << "specular=" << specular.x << "," << specular.y << "," << specular.z << "\n";
    file << "emissive=" << emissive.x << "," << emissive.y << "," << emissive.z << "\n";
    
    // Save PBR properties
    file << "shininess=" << GetShininess() << "\n";
    file << "roughness=" << GetRoughness() << "\n";
    file << "metallic=" << GetMetallic() << "\n";
    file << "heightScale=" << GetHeightScale() << "\n";
    file << "opacity=" << GetOpacity() << "\n";
    file << "transparent=" << (IsTransparent() ? "1" : "0") << "\n";
    
    // Note: Texture paths are not saved as they may not be portable
    // Users should reassign textures after loading presets
    
    file.close();
    std::cout << "Material preset saved to: " << filepath << std::endl;
    return true;
}

bool Material::LoadFromFile(const std::string& filepath, TextureManager* texManager) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Parse key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Parse Vec3 values (format: x,y,z)
        auto parseVec3 = [](const std::string& str) -> Vec3 {
            Vec3 result;
            std::stringstream ss(str);
            char comma;
            ss >> result.x >> comma >> result.y >> comma >> result.z;
            return result;
        };
        
        // Set properties based on key
        if (key == "ambient") {
            SetAmbient(parseVec3(value));
        } else if (key == "diffuse") {
            SetDiffuse(parseVec3(value));
        } else if (key == "specular") {
            SetSpecular(parseVec3(value));
        } else if (key == "emissive") {
            SetEmissiveColor(parseVec3(value));
        } else if (key == "shininess") {
            SetShininess(std::stof(value));
        } else if (key == "roughness") {
            SetRoughnessX(std::stof(value));
        } else if (key == "metallic") {
            SetMetalnessX(std::stof(value));
        } else if (key == "heightScale") {
            SetParallaxX(std::stof(value));
        } else if (key == "opacity") {
            SetMyAlpha(std::stof(value));
        } else if (key == "transparent") {
            SetMyTrans(value == "1");
        }
    }
    
    file.close();
    std::cout << "Material preset loaded from: " << filepath << std::endl;
    return true;
}

void Material::SetAmbient(const Vec3& v) { m_Ambient = v; m_Overrides |= PropAmbient; }
void Material::SetDiffuse(const Vec3& v) { m_Diffuse = v; m_Overrides |= PropDiffuse; }
void Material::SetSpecular(const Vec3& v) { m_Specular = v; m_Overrides |= PropSpecular; }
void Material::SetShininess(float v) { m_Shininess = v; m_Overrides |= PropShininess; }
void Material::SetRoughnessX(float v) { m_Roughness = v; m_Overrides |= PropRoughness; }
void Material::SetMetalnessX(float v) { m_Metallic = v; m_Overrides |= PropMetallic; }
void Material::SetParallaxX(float v) { m_HeightScale = v; m_Overrides |= PropHeightScale; }
void Material::SetEmissiveColor(const Vec3& v) { m_EmissiveColor = v; m_Overrides |= PropEmissiveColor; }
void Material::SetMyAlpha(float opacity) { m_Opacity = opacity; m_Overrides |= PropOpacity; }
void Material::SetMyTrans(bool transparent) { m_IsTransparent = transparent; m_Overrides |= PropTransparent; }
