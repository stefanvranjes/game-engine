#pragma once

#include "GameObject.h"
#include "SceneSerializer.h"
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @class Prefab
 * @brief A reusable blueprint for GameObjects that can be instantiated multiple times.
 * 
 * Prefabs store the complete configuration of a GameObject including:
 * - Mesh and material data
 * - Transform and hierarchy
 * - Components (animator, physics, etc.)
 * - All child GameObjects
 * 
 * Prefabs support:
 * - Nested prefabs (prefabs within prefabs)
 * - Property overrides on instances
 * - Automatic syncing with source prefabs
 */
class Prefab {
public:
    struct Metadata {
        std::string name;
        std::string version = "1.0";
        std::string description;
        std::string author;
        std::string created; // ISO 8601 timestamp
        std::string modified; // ISO 8601 timestamp
        std::vector<std::string> tags; // For categorization and searching
    };

    /**
     * @brief Create prefab from a GameObject
     * @param sourceObject GameObject to create prefab from
     * @param metadata Prefab metadata
     */
    Prefab(std::shared_ptr<GameObject> sourceObject, const Metadata& metadata = Metadata());

    /**
     * @brief Create empty prefab
     */
    Prefab();

    ~Prefab() = default;

    /**
     * @brief Instantiate this prefab as a new GameObject
     * @param name Name for the new instance
     * @param applyTransform If true, applies the prefab's transform to the instance
     * @return New GameObject instance (fully independent copy)
     */
    std::shared_ptr<GameObject> Instantiate(
        const std::string& name = "",
        bool applyTransform = true
    ) const;

    /**
     * @brief Instantiate with custom transform override
     * @param position World position for instance
     * @param rotation World rotation for instance
     * @param scale World scale for instance
     * @param name Name for the new instance
     * @return New GameObject instance
     */
    std::shared_ptr<GameObject> InstantiateAt(
        const Vec3& position,
        const Vec3& rotation = Vec3(0),
        const Vec3& scale = Vec3(1),
        const std::string& name = ""
    ) const;

    /**
     * @brief Update prefab from existing instance
     * (Recreates the blueprint from current GameObject state)
     * @param sourceObject GameObject to update from
     * @param updateMetadata Whether to update modification time
     */
    void UpdateFromInstance(
        std::shared_ptr<GameObject> sourceObject,
        bool updateMetadata = true
    );

    /**
     * @brief Sync an instance with current prefab data
     * (Applies prefab changes to existing instance)
     * @param instance GameObject instance to sync
     * @param preserveLocalChanges If true, preserves certain properties
     */
    void ApplyToInstance(
        std::shared_ptr<GameObject> instance,
        bool preserveLocalChanges = false
    );

    // ===== Getters/Setters =====
    const Metadata& GetMetadata() const { return m_Metadata; }
    void SetMetadata(const Metadata& metadata) { m_Metadata = metadata; }

    const std::string& GetName() const { return m_Metadata.name; }
    void SetName(const std::string& name) { m_Metadata.name = name; }

    const std::string& GetDescription() const { return m_Metadata.description; }
    void SetDescription(const std::string& desc) { m_Metadata.description = desc; }

    std::shared_ptr<GameObject> GetSourceObject() const { return m_SourceObject; }

    const json& GetSerializedData() const { return m_SerializedData; }

    bool IsValid() const { return !m_SerializedData.is_null(); }

    /**
     * @brief Get full path where this prefab is stored
     * @return Prefab file path
     */
    const std::string& GetFilePath() const { return m_FilePath; }

private:
    Metadata m_Metadata;
    std::shared_ptr<GameObject> m_SourceObject;
    json m_SerializedData; // Serialized representation of the prefab
    std::string m_FilePath; // Path to prefab file on disk

    friend class PrefabManager; // Allow PrefabManager to set filepath
};

/**
 * @class PrefabManager
 * @brief Manages creation, storage, and instantiation of prefabs.
 * 
 * Features:
 * - Save/load prefabs to disk (JSON or binary format)
 * - Prefab registry for quick lookup
 * - Bulk operations on prefabs
 * - Prefab searching and filtering
 * - Version management and migration
 */
class PrefabManager {
public:
    PrefabManager(const std::string& prefabDirectory = "assets/prefabs");
    ~PrefabManager() = default;

    // ===== Prefab Creation & Management =====
    /**
     * @brief Create and register a prefab from GameObject
     * @param sourceObject GameObject to create prefab from
     * @param prefabName Unique name for the prefab
     * @param metadata Optional metadata
     * @return Shared pointer to created prefab
     */
    std::shared_ptr<Prefab> CreatePrefab(
        std::shared_ptr<GameObject> sourceObject,
        const std::string& prefabName,
        const Prefab::Metadata& metadata = Prefab::Metadata()
    );

    /**
     * @brief Save prefab to disk
     * @param prefab Prefab to save
     * @param filename Optional override filename (without extension)
     * @param format Serialization format
     * @return true if successful
     */
    bool SavePrefab(
        std::shared_ptr<Prefab> prefab,
        const std::string& filename = "",
        SceneSerializer::SerializationFormat format = SceneSerializer::SerializationFormat::JSON
    );

    /**
     * @brief Load prefab from disk
     * @param filename Prefab file path
     * @return Loaded prefab, or nullptr on failure
     */
    std::shared_ptr<Prefab> LoadPrefab(const std::string& filename);

    /**
     * @brief Register loaded prefab in manager
     * @param prefab Prefab to register
     * @param name Registration name
     */
    void RegisterPrefab(std::shared_ptr<Prefab> prefab, const std::string& name);

    /**
     * @brief Unregister prefab from manager
     * @param name Prefab name to unregister
     * @return true if prefab was registered
     */
    bool UnregisterPrefab(const std::string& name);

    /**
     * @brief Get registered prefab by name
     * @param name Prefab name
     * @return Prefab pointer, or nullptr if not found
     */
    std::shared_ptr<Prefab> GetPrefab(const std::string& name);

    /**
     * @brief Check if prefab is registered
     * @param name Prefab name
     * @return true if registered
     */
    bool HasPrefab(const std::string& name) const;

    /**
     * @brief Get all registered prefab names
     * @return Vector of prefab names
     */
    std::vector<std::string> GetPrefabNames() const;

    // ===== Batch Operations =====
    /**
     * @brief Save all registered prefabs
     * @param format Serialization format
     * @return Number of successfully saved prefabs
     */
    int SaveAllPrefabs(SceneSerializer::SerializationFormat format = SceneSerializer::SerializationFormat::JSON);

    /**
     * @brief Load all prefabs from directory
     * @return Number of successfully loaded prefabs
     */
    int LoadAllPrefabs();

    /**
     * @brief Clear all registered prefabs
     */
    void Clear();

    // ===== Searching & Filtering =====
    /**
     * @brief Search prefabs by name (substring match)
     * @param nameFilter Substring to search for
     * @return Vector of matching prefab names
     */
    std::vector<std::string> SearchByName(const std::string& nameFilter) const;

    /**
     * @brief Search prefabs by tags
     * @param tag Tag to search for
     * @return Vector of prefab names with matching tag
     */
    std::vector<std::string> SearchByTag(const std::string& tag) const;

    /**
     * @brief Get prefab count
     * @return Number of registered prefabs
     */
    size_t GetPrefabCount() const { return m_Prefabs.size(); }

    /**
     * @brief Get prefab directory
     * @return Base directory for prefab storage
     */
    const std::string& GetPrefabDirectory() const { return m_PrefabDirectory; }

    /**
     * @brief Set prefab directory
     * @param directory New directory path
     */
    void SetPrefabDirectory(const std::string& directory) { m_PrefabDirectory = directory; }

    /**
     * @brief Get human-readable error message from last operation
     * @return Error string
     */
    const std::string& GetLastError() const { return m_LastError; }

#ifdef USE_PHYSX
    void SetPhysXBackend(class PhysXBackend* backend);
#endif

private:
    std::string m_PrefabDirectory;
    std::map<std::string, std::shared_ptr<Prefab>> m_Prefabs; // Registry of loaded prefabs
    std::string m_LastError;

    // ===== Helper Methods =====
    void SetError(const std::string& error);
    std::string BuildPrefabPath(const std::string& filename, const std::string& ext = ".json");
    bool DirectoryExists(const std::string& directory) const;
    void CreateDirectory(const std::string& directory) const;

    SceneSerializer m_Serializer;
};
