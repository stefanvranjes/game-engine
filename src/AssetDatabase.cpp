#include "AssetDatabase.h"
#include <fstream>
#include <iostream>

nlohmann::json AssetDatabase::AssetEntry::ToJson() const {
    nlohmann::json j;
    j["path"] = path;
    j["type"] = type;
    j["sourceHash"] = sourceHash.sha256;
    j["processedHash"] = processedHash.sha256;
    j["processedPath"] = processedPath;
    j["dependencies"] = dependencies;
    j["lastProcessedTime"] = lastProcessedTime;
    j["isDirty"] = isDirty;
    j["metadata"] = metadata;
    return j;
}

AssetDatabase::AssetEntry AssetDatabase::AssetEntry::FromJson(const nlohmann::json& j) {
    AssetEntry entry;
    entry.path = j.value("path", "");
    entry.type = j.value("type", "");
    entry.sourceHash.sha256 = j.value("sourceHash", "");
    entry.processedHash.sha256 = j.value("processedHash", "");
    entry.processedPath = j.value("processedPath", "");
    entry.dependencies = j.value("dependencies", std::vector<std::string>{});
    entry.lastProcessedTime = j.value("lastProcessedTime", "");
    entry.isDirty = j.value("isDirty", true);
    entry.metadata = j.value("metadata", std::map<std::string, std::string>{});
    return entry;
}

bool AssetDatabase::Initialize(const std::string& dbPath) {
    m_DatabasePath = dbPath;

    if (std::filesystem::exists(dbPath)) {
        std::ifstream file(dbPath);
        try {
            file >> m_RootJson;
            
            if (m_RootJson.contains("assets")) {
                for (const auto& [key, value] : m_RootJson["assets"].items()) {
                    m_Entries[key] = AssetEntry::FromJson(value);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load asset database: " << e.what() << std::endl;
            return false;
        }
    } else {
        // Create new database
        m_RootJson["version"] = "1.0";
        m_RootJson["assets"] = nlohmann::json::object();
    }

    return true;
}

bool AssetDatabase::Save() {
    m_RootJson["assets"] = nlohmann::json::object();
    for (const auto& [path, entry] : m_Entries) {
        m_RootJson["assets"][path] = entry.ToJson();
    }

    std::ofstream file(m_DatabasePath);
    if (!file.is_open()) {
        std::cerr << "Failed to save asset database: " << m_DatabasePath << std::endl;
        return false;
    }

    file << m_RootJson.dump(2);
    file.close();
    return true;
}

void AssetDatabase::SetAssetEntry(const AssetEntry& entry) {
    m_Entries[entry.path] = entry;
}

const AssetDatabase::AssetEntry* AssetDatabase::GetAssetEntry(const std::string& assetPath) const {
    auto it = m_Entries.find(assetPath);
    if (it != m_Entries.end()) {
        return &it->second;
    }
    return nullptr;
}

bool AssetDatabase::IsAssetDirty(const std::string& assetPath) {
    auto it = m_Entries.find(assetPath);
    if (it == m_Entries.end()) {
        return true;  // Not in database = needs processing
    }
    return it->second.isDirty;
}

void AssetDatabase::MarkAssetClean(const std::string& assetPath) {
    auto it = m_Entries.find(assetPath);
    if (it != m_Entries.end()) {
        it->second.isDirty = false;
    }
}

void AssetDatabase::MarkAssetDirty(const std::string& assetPath) {
    auto it = m_Entries.find(assetPath);
    if (it != m_Entries.end()) {
        it->second.isDirty = true;
    }
}

std::vector<std::string> AssetDatabase::GetDirtyAssets(const std::string& assetType) {
    std::vector<std::string> dirtyAssets;
    for (const auto& [path, entry] : m_Entries) {
        if (entry.isDirty && (assetType.empty() || entry.type == assetType)) {
            dirtyAssets.push_back(path);
        }
    }
    return dirtyAssets;
}

std::vector<std::string> AssetDatabase::GetAssetsByType(const std::string& assetType) {
    std::vector<std::string> assets;
    for (const auto& [path, entry] : m_Entries) {
        if (entry.type == assetType) {
            assets.push_back(path);
        }
    }
    return assets;
}

std::vector<std::string> AssetDatabase::FindDependents(const std::string& assetPath) {
    std::vector<std::string> dependents;
    for (const auto& [path, entry] : m_Entries) {
        auto it = std::find(entry.dependencies.begin(), entry.dependencies.end(), assetPath);
        if (it != entry.dependencies.end()) {
            dependents.push_back(path);
        }
    }
    return dependents;
}

void AssetDatabase::AddDependency(const std::string& assetPath, const std::string& dependencyPath) {
    auto it = m_Entries.find(assetPath);
    if (it != m_Entries.end()) {
        auto depIt = std::find(it->second.dependencies.begin(), it->second.dependencies.end(), dependencyPath);
        if (depIt == it->second.dependencies.end()) {
            it->second.dependencies.push_back(dependencyPath);
        }
    }
}

std::vector<std::string> AssetDatabase::VerifyIntegrity() {
    std::vector<std::string> failedAssets;
    for (auto& [path, entry] : m_Entries) {
        std::string fullPath = std::filesystem::absolute(path).string();
        if (!AssetHash::VerifyIntegrity(fullPath, entry.sourceHash)) {
            failedAssets.push_back(path);
            entry.isDirty = true;
        }
    }
    return failedAssets;
}

void AssetDatabase::Clear() {
    m_Entries.clear();
    m_RootJson["assets"] = nlohmann::json::object();
}

size_t AssetDatabase::GetDirtyAssetCount() const {
    size_t count = 0;
    for (const auto& [path, entry] : m_Entries) {
        if (entry.isDirty) {
            count++;
        }
    }
    return count;
}
