#include "ClothTearPatternLibrary.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// ============================================================================
// Singleton
// ============================================================================

ClothTearPatternLibrary& ClothTearPatternLibrary::GetInstance() {
    static ClothTearPatternLibrary instance;
    return instance;
}

ClothTearPatternLibrary::ClothTearPatternLibrary() {
    // Initialize with built-in patterns
    InitializeBuiltInPatterns();
}

// ============================================================================
// Pattern Management
// ============================================================================

void ClothTearPatternLibrary::RegisterPattern(
    const std::string& name,
    std::shared_ptr<ClothTearPattern> pattern,
    const std::string& category
) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    PatternEntry entry;
    entry.pattern = pattern;
    entry.category = category;
    
    m_Patterns[name] = entry;
}

std::shared_ptr<ClothTearPattern> ClothTearPatternLibrary::GetPattern(const std::string& name) const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    auto it = m_Patterns.find(name);
    if (it != m_Patterns.end()) {
        // Return a clone to prevent modification of library patterns
        return it->second.pattern->Clone();
    }
    
    return nullptr;
}

bool ClothTearPatternLibrary::HasPattern(const std::string& name) const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Patterns.find(name) != m_Patterns.end();
}

bool ClothTearPatternLibrary::RemovePattern(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    auto it = m_Patterns.find(name);
    if (it != m_Patterns.end()) {
        m_Patterns.erase(it);
        return true;
    }
    
    return false;
}

std::vector<std::string> ClothTearPatternLibrary::GetPatternNames(const std::string& category) const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::vector<std::string> names;
    
    for (const auto& pair : m_Patterns) {
        if (category.empty() || pair.second.category == category) {
            names.push_back(pair.first);
        }
    }
    
    return names;
}

std::vector<std::string> ClothTearPatternLibrary::GetCategories() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::vector<std::string> categories;
    
    for (const auto& pair : m_Patterns) {
        const std::string& cat = pair.second.category;
        if (!cat.empty() && std::find(categories.begin(), categories.end(), cat) == categories.end()) {
            categories.push_back(cat);
        }
    }
    
    return categories;
}

std::vector<std::shared_ptr<ClothTearPattern>> ClothTearPatternLibrary::GetPatternsInCategory(
    const std::string& category
) const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    std::vector<std::shared_ptr<ClothTearPattern>> patterns;
    
    for (const auto& pair : m_Patterns) {
        if (pair.second.category == category) {
            patterns.push_back(pair.second.pattern->Clone());
        }
    }
    
    return patterns;
}

void ClothTearPatternLibrary::Clear() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Patterns.clear();
}

// ============================================================================
// File I/O
// ============================================================================

bool ClothTearPatternLibrary::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open pattern library file: " << filepath << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    // Simple JSON-like parsing (basic implementation)
    // Format: Each pattern is a block of key=value pairs separated by blank lines
    
    std::string line;
    std::map<std::string, std::string> currentPattern;
    std::string currentName;
    std::string currentCategory;
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) {
            // End of pattern block
            if (!currentPattern.empty() && !currentName.empty()) {
                auto pattern = CreatePatternFromType(currentPattern["type"]);
                if (pattern) {
                    pattern->Deserialize(currentPattern);
                    
                    PatternEntry entry;
                    entry.pattern = pattern;
                    entry.category = currentCategory;
                    m_Patterns[currentName] = entry;
                }
                
                currentPattern.clear();
                currentName.clear();
                currentCategory.clear();
            }
            continue;
        }
        
        // Parse key=value
        size_t equalPos = line.find('=');
        if (equalPos != std::string::npos) {
            std::string key = line.substr(0, equalPos);
            std::string value = line.substr(equalPos + 1);
            
            // Trim key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "patternName") {
                currentName = value;
            } else if (key == "category") {
                currentCategory = value;
            } else {
                currentPattern[key] = value;
            }
        }
    }
    
    // Handle last pattern
    if (!currentPattern.empty() && !currentName.empty()) {
        auto pattern = CreatePatternFromType(currentPattern["type"]);
        if (pattern) {
            pattern->Deserialize(currentPattern);
            
            PatternEntry entry;
            entry.pattern = pattern;
            entry.category = currentCategory;
            m_Patterns[currentName] = entry;
        }
    }
    
    file.close();
    return true;
}

bool ClothTearPatternLibrary::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to create pattern library file: " << filepath << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    for (const auto& pair : m_Patterns) {
        file << "patternName=" << pair.first << "\n";
        file << "category=" << pair.second.category << "\n";
        
        auto data = pair.second.pattern->Serialize();
        for (const auto& dataPair : data) {
            file << dataPair.first << "=" << dataPair.second << "\n";
        }
        
        file << "\n"; // Blank line separator
    }
    
    file.close();
    return true;
}

// ============================================================================
// Built-in Patterns
// ============================================================================

void ClothTearPatternLibrary::InitializeBuiltInPatterns() {
    // Combat category
    
    // Bullet impact - small
    auto bulletSmall = std::make_shared<RadialTearPattern>(0.05f, 4);
    bulletSmall->SetName("Bullet Impact (Small)");
    bulletSmall->SetDescription("Small bullet impact with 4 rays");
    bulletSmall->SetRayWidth(0.005f);
    RegisterPattern("bullet_impact_small", bulletSmall, "combat");
    
    // Bullet impact - large
    auto bulletLarge = std::make_shared<RadialTearPattern>(0.15f, 8);
    bulletLarge->SetName("Bullet Impact (Large)");
    bulletLarge->SetDescription("Large caliber bullet impact with 8 rays");
    bulletLarge->SetRayWidth(0.01f);
    RegisterPattern("bullet_impact_large", bulletLarge, "combat");
    
    // Sword slash
    auto swordSlash = std::make_shared<LinearTearPattern>(0.3f, 0.02f);
    swordSlash->SetName("Sword Slash");
    swordSlash->SetDescription("Clean sword cut");
    RegisterPattern("sword_slash", swordSlash, "combat");
    
    // Knife cut
    auto knifeCut = std::make_shared<LinearTearPattern>(0.15f, 0.005f);
    knifeCut->SetName("Knife Cut");
    knifeCut->SetDescription("Thin knife cut");
    RegisterPattern("knife_cut", knifeCut, "combat");
    
    // Claw mark (triple slash)
    auto clawMark = std::make_shared<CrossTearPattern>(0.2f, 0.01f, 30.0f);
    clawMark->SetName("Claw Mark");
    clawMark->SetDescription("Triple claw slash pattern");
    RegisterPattern("claw_mark", clawMark, "combat");
    
    // Environmental category
    
    // Explosion
    auto explosion = std::make_shared<RadialTearPattern>(0.5f, 12);
    explosion->SetName("Explosion");
    explosion->SetDescription("Large explosion tear with 12 rays");
    explosion->SetRayWidth(0.02f);
    RegisterPattern("explosion", explosion, "environmental");
    
    // Shrapnel
    auto shrapnel = std::make_shared<RadialTearPattern>(0.3f, 16);
    shrapnel->SetName("Shrapnel");
    shrapnel->SetDescription("Shrapnel impact with many small tears");
    shrapnel->SetRayWidth(0.005f);
    RegisterPattern("shrapnel", shrapnel, "environmental");
    
    // Artistic category
    
    // Cross tear
    auto crossTear = std::make_shared<CrossTearPattern>(0.25f, 0.015f, 90.0f);
    crossTear->SetName("Cross Tear");
    crossTear->SetDescription("Perfect cross-shaped tear");
    RegisterPattern("cross_tear", crossTear, "artistic");
    
    // Star tear
    auto starTear = std::make_shared<RadialTearPattern>(0.2f, 5);
    starTear->SetName("Star Tear");
    starTear->SetDescription("5-pointed star tear");
    starTear->SetRayWidth(0.01f);
    RegisterPattern("star_tear", starTear, "artistic");
    
    std::cout << "Initialized " << m_Patterns.size() << " built-in tear patterns" << std::endl;
}

// ============================================================================
// Helper Methods
// ============================================================================

std::shared_ptr<ClothTearPattern> ClothTearPatternLibrary::CreatePatternFromType(const std::string& type) const {
    if (type == "Linear") {
        return std::make_shared<LinearTearPattern>();
    } else if (type == "Radial") {
        return std::make_shared<RadialTearPattern>();
    } else if (type == "Cross") {
        return std::make_shared<CrossTearPattern>();
    } else if (type == "CustomPath") {
        return std::make_shared<CustomPathTearPattern>();
    } else if (type == "StressBased") {
        return std::make_shared<StressBasedTearPattern>();
    }
    
    std::cerr << "Unknown pattern type: " << type << std::endl;
    return nullptr;
}
