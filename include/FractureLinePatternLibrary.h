#pragma once

#include "FractureLine.h"
#include "SoftBodyTearPattern.h"
#include <string>
#include <map>
#include <vector>
#include <memory>

/**
 * @brief Pattern library for saving and loading fracture line presets
 */
class FractureLinePatternLibrary {
public:
    /**
     * @brief Pattern preset data
     */
    struct PatternPreset {
        std::string name;
        std::string description;
        FractureLine line;
        SoftBodyTearPattern::PatternType type;
        float curvature;  // For curved patterns
        
        PatternPreset() 
            : curvature(0.5f)
            , type(SoftBodyTearPattern::PatternType::Straight) 
        {}
    };
    
    FractureLinePatternLibrary();
    
    /**
     * @brief Save fracture line as preset
     * @param name Preset name (must be unique)
     * @param line Fracture line to save
     * @param description Optional description
     * @return True if saved successfully
     */
    bool SavePreset(
        const std::string& name, 
        const FractureLine& line,
        const std::string& description = ""
    );
    
    /**
     * @brief Load preset by name
     * @param name Preset name
     * @param outLine Output fracture line
     * @return True if loaded successfully
     */
    bool LoadPreset(const std::string& name, FractureLine& outLine);
    
    /**
     * @brief Get preset by name
     */
    const PatternPreset* GetPreset(const std::string& name) const;
    
    /**
     * @brief Delete preset by name
     */
    bool DeletePreset(const std::string& name);
    
    /**
     * @brief Rename preset
     */
    bool RenamePreset(const std::string& oldName, const std::string& newName);
    
    /**
     * @brief Get all preset names
     */
    std::vector<std::string> GetPresetNames() const;
    
    /**
     * @brief Get number of presets
     */
    int GetPresetCount() const { return static_cast<int>(m_Presets.size()); }
    
    /**
     * @brief Check if preset exists
     */
    bool HasPreset(const std::string& name) const;
    
    /**
     * @brief Clear all presets
     */
    void Clear();
    
    /**
     * @brief Save library to file
     * @param filename Output file path
     * @return True if saved successfully
     */
    bool SaveToFile(const std::string& filename) const;
    
    /**
     * @brief Load library from file
     * @param filename Input file path
     * @return True if loaded successfully
     */
    bool LoadFromFile(const std::string& filename);
    
    /**
     * @brief Get default library path
     */
    static std::string GetDefaultLibraryPath();

private:
    std::map<std::string, PatternPreset> m_Presets;
    
    /**
     * @brief Validate preset name
     */
    bool IsValidPresetName(const std::string& name) const;
};
