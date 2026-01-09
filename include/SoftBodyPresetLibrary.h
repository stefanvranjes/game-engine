#pragma once

#include "SoftBodyPreset.h"
#include <map>
#include <vector>
#include <string>

/**
 * @brief Manages a library of soft body presets
 * 
 * Provides centralized access to predefined and custom presets,
 * with support for loading from directories and categorization.
 */
class SoftBodyPresetLibrary {
public:
    SoftBodyPresetLibrary();
    
    /**
     * @brief Load all presets from a directory
     * @param directory Directory path containing .preset files
     */
    void LoadPresetsFromDirectory(const std::string& directory);
    
    /**
     * @brief Add a preset to the library
     * @param preset Preset to add
     */
    void AddPreset(const SoftBodyPreset& preset);
    
    /**
     * @brief Get preset by name
     * @param name Preset name
     * @return Pointer to preset, or nullptr if not found
     */
    const SoftBodyPreset* GetPreset(const std::string& name) const;
    
    /**
     * @brief Get all preset names
     */
    std::vector<std::string> GetPresetNames() const;
    
    /**
     * @brief Get all category names
     */
    std::vector<std::string> GetCategories() const;
    
    /**
     * @brief Get presets in a specific category
     */
    std::vector<std::string> GetPresetsInCategory(const std::string& category) const;
    
    /**
     * @brief Remove preset by name
     */
    void RemovePreset(const std::string& name);
    
    /**
     * @brief Clear all presets
     */
    void Clear();
    
    /**
     * @brief Load default presets
     */
    void LoadDefaultPresets();
    
private:
    std::map<std::string, SoftBodyPreset> m_Presets;
};
