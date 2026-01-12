#include "Prefab.h"
#include "MaterialNew.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iostream>
#include <filesystem>
#include <algorithm>

// Helper function for C++17 compatibility (ends_with is C++20)
static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

// ============================================================================
// Prefab Implementation
// ============================================================================

Prefab::Prefab(std::shared_ptr<GameObject> sourceObject, const Metadata& metadata)
    : m_SourceObject(sourceObject), m_Metadata(metadata)
{
    if (!metadata.name.empty()) {
        m_Metadata.name = metadata.name;
    }
    
    // Set timestamps if not provided
    if (m_Metadata.created.empty()) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
        m_Metadata.created = ss.str();
    }
    
    // Serialize source object
    if (m_SourceObject) {
        SceneSerializer serializer;
        m_SerializedData = serializer.SerializeGameObjectToJson(m_SourceObject, true);
    }
}

Prefab::Prefab() : m_SourceObject(nullptr)
{
}

std::shared_ptr<GameObject> Prefab::Instantiate(
    const std::string& name,
    bool applyTransform) const
{
    if (!IsValid()) {
        std::cerr << "Cannot instantiate invalid prefab" << std::endl;
        return nullptr;
    }

    SceneSerializer serializer;
    auto instance = serializer.DeserializeGameObjectFromJson(m_SerializedData);
    
    if (!instance) {
        std::cerr << "Failed to instantiate prefab" << std::endl;
        return nullptr;
    }

    // Set instance name
    std::string instanceName = name.empty() ? 
        (m_Metadata.name + "_Instance") : 
        name;
    
    // Create a fresh copy to avoid shared state
    auto result = std::make_shared<GameObject>(instanceName);
    result->GetTransform() = instance->GetTransform();
    result->SetMaterial(instance->GetMaterial());
    result->SetAnimator(instance->GetAnimator());
    result->SetRigidBody(instance->GetRigidBody());
    result->SetKinematicController(instance->GetKinematicController());
    result->SetUVOffset(instance->GetUVOffset());
    result->SetUVScale(instance->GetUVScale());
    result->SetVisible(instance->IsVisible());

    // Copy children
    for (auto& child : instance->GetChildren()) {
        result->AddChild(child);
    }

    return result;
}

std::shared_ptr<GameObject> Prefab::InstantiateAt(
    const Vec3& position,
    const Vec3& rotation,
    const Vec3& scale,
    const std::string& name) const
{
    auto instance = Instantiate(name, false);
    if (instance) {
        instance->GetTransform().position = position;
        instance->GetTransform().rotation = rotation;
        instance->GetTransform().scale = scale;
    }
    return instance;
}

void Prefab::UpdateFromInstance(
    std::shared_ptr<GameObject> sourceObject,
    bool updateMetadata)
{
    if (!sourceObject) {
        std::cerr << "Cannot update prefab from null GameObject" << std::endl;
        return;
    }

    m_SourceObject = sourceObject;

    // Re-serialize the source object
    SceneSerializer serializer;
    m_SerializedData = serializer.SerializeGameObjectToJson(m_SourceObject, true);

    if (updateMetadata) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
        m_Metadata.modified = ss.str();
    }
}

void Prefab::ApplyToInstance(
    std::shared_ptr<GameObject> instance,
    bool preserveLocalChanges)
{
    if (!IsValid() || !instance) {
        std::cerr << "Cannot apply invalid prefab or to null instance" << std::endl;
        return;
    }

    // Update instance from current prefab data
    // Preserve instance name
    std::string instanceName = instance->GetName();
    
    SceneSerializer serializer;
    auto prefabData = serializer.DeserializeGameObjectFromJson(m_SerializedData);
    
    if (!prefabData) {
        std::cerr << "Failed to apply prefab to instance" << std::endl;
        return;
    }

    // Copy prefab data to instance
    if (!preserveLocalChanges) {
        instance->GetTransform() = prefabData->GetTransform();
    }
    instance->SetMaterial(prefabData->GetMaterial());
    instance->SetAnimator(prefabData->GetAnimator());
    instance->SetRigidBody(prefabData->GetRigidBody());
    instance->SetKinematicController(prefabData->GetKinematicController());
    instance->SetUVOffset(prefabData->GetUVOffset());
    instance->SetUVScale(prefabData->GetUVScale());
    instance->SetVisible(prefabData->IsVisible());

    // Restore instance name
    // (Note: This would require a way to set the name in GameObject)
}

// ============================================================================
// PrefabManager Implementation
// ============================================================================

PrefabManager::PrefabManager(const std::string& prefabDirectory)
    : m_PrefabDirectory(prefabDirectory)
{
    if (!DirectoryExists(m_PrefabDirectory)) {
        try {
            CreateDirectory(m_PrefabDirectory);
        }
        catch (const std::exception& e) {
            SetError("Failed to create prefab directory: " + std::string(e.what()));
        }
    }
}

std::shared_ptr<Prefab> PrefabManager::CreatePrefab(
    std::shared_ptr<GameObject> sourceObject,
    const std::string& prefabName,
    const Prefab::Metadata& metadata)
{
    if (!sourceObject) {
        SetError("Cannot create prefab from null GameObject");
        return nullptr;
    }

    auto prefabMetadata = metadata;
    if (prefabMetadata.name.empty()) {
        prefabMetadata.name = prefabName;
    }

    auto prefab = std::make_shared<Prefab>(sourceObject, prefabMetadata);
    RegisterPrefab(prefab, prefabName);

    return prefab;
}

bool PrefabManager::SavePrefab(
    std::shared_ptr<Prefab> prefab,
    const std::string& filename,
    SceneSerializer::SerializationFormat format)
{
    if (!prefab || !prefab->IsValid()) {
        SetError("Cannot save invalid prefab");
        return false;
    }

    try {
        std::string saveFilename = filename.empty() ? prefab->GetName() : filename;
        std::string filepath = BuildPrefabPath(
            saveFilename,
            format == SceneSerializer::SerializationFormat::JSON ? ".json" : ".bin"
        );

        // Create metadata JSON
        json metadataJson;
        metadataJson["name"] = prefab->GetMetadata().name;
        metadataJson["version"] = prefab->GetMetadata().version;
        metadataJson["description"] = prefab->GetMetadata().description;
        metadataJson["author"] = prefab->GetMetadata().author;
        metadataJson["created"] = prefab->GetMetadata().created;
        metadataJson["modified"] = prefab->GetMetadata().modified;
        metadataJson["tags"] = prefab->GetMetadata().tags;

        // Combine metadata with serialized data
        json fullData;
        fullData["prefabMetadata"] = metadataJson;
        fullData["gameObject"] = prefab->GetSerializedData();

        if (format == SceneSerializer::SerializationFormat::JSON) {
            std::ofstream file(filepath);
            if (!file.is_open()) {
                SetError("Failed to open file for writing: " + filepath);
                return false;
            }
            file << fullData.dump(4);
            file.close();
        }
        else {
            // Binary format
            std::vector<uint8_t> buffer;
            
            // Write magic and version
            uint32_t magic = 0x50524642; // "PRFB" in hex
            buffer.insert(buffer.end(), (uint8_t*)&magic, (uint8_t*)&magic + 4);
            uint32_t version = 1;
            buffer.insert(buffer.end(), (uint8_t*)&version, (uint8_t*)&version + 4);

            // Write JSON data
            std::string jsonStr = fullData.dump();
            uint32_t jsonLen = jsonStr.length();
            buffer.insert(buffer.end(), (uint8_t*)&jsonLen, (uint8_t*)&jsonLen + 4);
            buffer.insert(buffer.end(), jsonStr.begin(), jsonStr.end());

            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                SetError("Failed to open file for writing: " + filepath);
                return false;
            }
            file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            file.close();
        }

        std::cout << "Prefab saved: " << filepath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        SetError(std::string("Exception while saving prefab: ") + e.what());
        return false;
    }
}

std::shared_ptr<Prefab> PrefabManager::LoadPrefab(const std::string& filename)
{
    try {
        if (!std::filesystem::exists(filename)) {
            SetError("Prefab file not found: " + filename);
            return nullptr;
        }

        json fullData;
        
        // Determine format and load
        if (ends_with(filename, ".json")) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                SetError("Failed to open prefab file: " + filename);
                return nullptr;
            }
            file >> fullData;
            file.close();
        }
        else if (ends_with(filename, ".bin")) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                SetError("Failed to open prefab file: " + filename);
                return nullptr;
            }

            std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
            file.close();

            if (buffer.size() < 12) {
                SetError("Invalid prefab binary format (too small)");
                return nullptr;
            }

            size_t offset = 0;
            uint32_t magic = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
            if (magic != 0x50524642) { // "PRFB"
                SetError("Invalid prefab magic number");
                return nullptr;
            }
            offset += 4;

            uint32_t version = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
            offset += 4;

            uint32_t jsonLen = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
            offset += 4;

            if (offset + jsonLen > buffer.size()) {
                SetError("Corrupted prefab binary format");
                return nullptr;
            }

            std::string jsonStr(buffer.begin() + offset, buffer.begin() + offset + jsonLen);
            fullData = json::parse(jsonStr);
        }
        else {
            SetError("Unknown prefab file format: " + filename);
            return nullptr;
        }

        // Parse metadata
        Prefab::Metadata metadata;
        if (fullData.contains("prefabMetadata")) {
            auto& meta = fullData["prefabMetadata"];
            if (meta.contains("name")) metadata.name = meta["name"];
            if (meta.contains("version")) metadata.version = meta["version"];
            if (meta.contains("description")) metadata.description = meta["description"];
            if (meta.contains("author")) metadata.author = meta["author"];
            if (meta.contains("created")) metadata.created = meta["created"];
            if (meta.contains("modified")) metadata.modified = meta["modified"];
            if (meta.contains("tags")) metadata.tags = meta["tags"].get<std::vector<std::string>>();
        }

        // Create prefab
        auto prefab = std::make_shared<Prefab>();
        prefab->m_Metadata = metadata;
        prefab->m_FilePath = filename;
        
        if (fullData.contains("gameObject")) {
            prefab->m_SerializedData = fullData["gameObject"];
        }

        if (!prefab->IsValid()) {
            SetError("Loaded prefab is invalid");
            return nullptr;
        }

        std::cout << "Prefab loaded: " << filename << std::endl;
        return prefab;
    }
    catch (const std::exception& e) {
        SetError(std::string("Exception while loading prefab: ") + e.what());
        return nullptr;
    }
}

void PrefabManager::RegisterPrefab(std::shared_ptr<Prefab> prefab, const std::string& name)
{
    if (!prefab) {
        SetError("Cannot register null prefab");
        return;
    }
    m_Prefabs[name] = prefab;
}

bool PrefabManager::UnregisterPrefab(const std::string& name)
{
    auto it = m_Prefabs.find(name);
    if (it != m_Prefabs.end()) {
        m_Prefabs.erase(it);
        return true;
    }
    return false;
}

std::shared_ptr<Prefab> PrefabManager::GetPrefab(const std::string& name)
{
    auto it = m_Prefabs.find(name);
    if (it != m_Prefabs.end()) {
        return it->second;
    }
    return nullptr;
}

bool PrefabManager::HasPrefab(const std::string& name) const
{
    return m_Prefabs.find(name) != m_Prefabs.end();
}

std::vector<std::string> PrefabManager::GetPrefabNames() const
{
    std::vector<std::string> names;
    for (const auto& pair : m_Prefabs) {
        names.push_back(pair.first);
    }
    return names;
}

int PrefabManager::SaveAllPrefabs(SceneSerializer::SerializationFormat format)
{
    int savedCount = 0;
    for (auto& pair : m_Prefabs) {
        if (SavePrefab(pair.second, pair.first, format)) {
            savedCount++;
        }
    }
    return savedCount;
}

int PrefabManager::LoadAllPrefabs()
{
    int loadedCount = 0;
    try {
        if (!std::filesystem::exists(m_PrefabDirectory)) {
            return 0;
        }

        for (const auto& entry : std::filesystem::directory_iterator(m_PrefabDirectory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (ends_with(filename, ".json") || ends_with(filename, ".bin")) {
                    auto prefab = LoadPrefab(entry.path().string());
                    if (prefab) {
                        std::string prefabName = filename.substr(0, filename.find_last_of('.'));
                        RegisterPrefab(prefab, prefabName);
                        loadedCount++;
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        SetError(std::string("Error loading prefabs: ") + e.what());
    }
    return loadedCount;
}

void PrefabManager::Clear()
{
    m_Prefabs.clear();
}

std::vector<std::string> PrefabManager::SearchByName(const std::string& nameFilter) const
{
    std::vector<std::string> results;
    std::string lowerFilter = nameFilter;
    std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);

    for (const auto& pair : m_Prefabs) {
        std::string lowerName = pair.first;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        if (lowerName.find(lowerFilter) != std::string::npos) {
            results.push_back(pair.first);
        }
    }
    return results;
}

std::vector<std::string> PrefabManager::SearchByTag(const std::string& tag) const
{
    std::vector<std::string> results;
    for (const auto& pair : m_Prefabs) {
        const auto& tags = pair.second->GetMetadata().tags;
        if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
            results.push_back(pair.first);
        }
    }
    return results;
}

void PrefabManager::SetError(const std::string& error)
{
    m_LastError = error;
    std::cerr << "[PrefabManager] " << error << std::endl;
}

void PrefabManager::SetPhysXBackend(PhysXBackend* backend) {
    m_Serializer.SetPhysXBackend(backend);
}

std::string PrefabManager::BuildPrefabPath(const std::string& filename, const std::string& ext)
{
    std::string path = m_PrefabDirectory + "/" + filename;
    if (!ends_with(filename, ext)) {
        path += ext;
    }
    return path;
}

bool PrefabManager::DirectoryExists(const std::string& directory) const
{
    try {
        return std::filesystem::exists(directory) && std::filesystem::is_directory(directory);
    }
    catch (...) {
        return false;
    }
}

void PrefabManager::CreateDirectory(const std::string& directory) const
{
    std::filesystem::create_directories(directory);
}
