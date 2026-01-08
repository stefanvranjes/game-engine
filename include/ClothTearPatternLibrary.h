#pragma once

#include "ClothTearPattern.h"
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <mutex>

/**
 * @brief Pattern library manager for cloth tear patterns
 * 
 * Provides centralized management of tear patterns including:
 * - Pattern registration and lookup
 * - Category-based organization
 * - Built-in pattern presets
 * - File I/O for pattern persistence
 * - Thread-safe access
 */
class ClothTearPatternLibrary {
public:
    /**
     * @brief Get singleton instance
     */
    static ClothTearPatternLibrary& GetInstance();

    /**
     * @brief Register a pattern with the library
     * @param name Unique pattern name
     * @param pattern Pattern to register
     * @param category Optional category for organization
     */
    void RegisterPattern(
        const std::string& name,
        std::shared_ptr<ClothTearPattern> pattern,
        const std::string& category = ""
    );

    /**
     * @brief Get a pattern by name
     * @param name Pattern name
     * @return Pattern or nullptr if not found
     */
    std::shared_ptr<ClothTearPattern> GetPattern(const std::string& name) const;

    /**
     * @brief Check if pattern exists
     * @param name Pattern name
     * @return True if pattern exists
     */
    bool HasPattern(const std::string& name) const;

    /**
     * @brief Remove a pattern from the library
     * @param name Pattern name
     * @return True if pattern was removed
     */
    bool RemovePattern(const std::string& name);

    /**
     * @brief Get all pattern names
     * @param category Optional category filter (empty = all)
     * @return List of pattern names
     */
    std::vector<std::string> GetPatternNames(const std::string& category = "") const;

    /**
     * @brief Get all categories
     * @return List of category names
     */
    std::vector<std::string> GetCategories() const;

    /**
     * @brief Get patterns in a category
     * @param category Category name
     * @return List of patterns in category
     */
    std::vector<std::shared_ptr<ClothTearPattern>> GetPatternsInCategory(
        const std::string& category
    ) const;

    /**
     * @brief Load pattern library from JSON file
     * @param filepath Path to JSON file
     * @return True if successful
     */
    bool LoadFromFile(const std::string& filepath);

    /**
     * @brief Save pattern library to JSON file
     * @param filepath Path to JSON file
     * @return True if successful
     */
    bool SaveToFile(const std::string& filepath) const;

    /**
     * @brief Clear all patterns
     */
    void Clear();

    /**
     * @brief Initialize built-in patterns
     * 
     * Registers default patterns:
     * - bullet_impact_small
     * - bullet_impact_large
     * - sword_slash
     * - knife_cut
     * - explosion
     * - claw_mark
     */
    void InitializeBuiltInPatterns();

private:
    ClothTearPatternLibrary();
    ~ClothTearPatternLibrary() = default;

    // Prevent copying
    ClothTearPatternLibrary(const ClothTearPatternLibrary&) = delete;
    ClothTearPatternLibrary& operator=(const ClothTearPatternLibrary&) = delete;

    struct PatternEntry {
        std::shared_ptr<ClothTearPattern> pattern;
        std::string category;
    };

    mutable std::mutex m_Mutex;
    std::map<std::string, PatternEntry> m_Patterns;

    // Helper to create pattern from serialized data
    std::shared_ptr<ClothTearPattern> CreatePatternFromType(const std::string& type) const;
};
