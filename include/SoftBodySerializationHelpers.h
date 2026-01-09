#pragma once

#include "Math/Vec3.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

/**
 * @brief Helper functions for soft body serialization
 */
namespace SoftBodySerializationHelpers {
    
    /**
     * @brief Convert Vec3 to JSON array
     */
    inline nlohmann::json Vec3ToJson(const Vec3& v) {
        return nlohmann::json::array({v.x, v.y, v.z});
    }
    
    /**
     * @brief Convert JSON array to Vec3
     */
    inline Vec3 Vec3FromJson(const nlohmann::json& j) {
        if (!j.is_array() || j.size() != 3) {
            return Vec3(0, 0, 0);
        }
        return Vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    }
    
    /**
     * @brief Serialize array of Vec3
     */
    inline nlohmann::json SerializeVec3Array(const std::vector<Vec3>& arr) {
        nlohmann::json result = nlohmann::json::array();
        for (const auto& v : arr) {
            result.push_back(Vec3ToJson(v));
        }
        return result;
    }
    
    /**
     * @brief Deserialize array of Vec3
     */
    inline std::vector<Vec3> DeserializeVec3Array(const nlohmann::json& j) {
        std::vector<Vec3> result;
        if (!j.is_array()) return result;
        
        result.reserve(j.size());
        for (const auto& item : j) {
            result.push_back(Vec3FromJson(item));
        }
        return result;
    }
    
    /**
     * @brief Serialize array of integers
     */
    inline nlohmann::json SerializeIntArray(const std::vector<int>& arr) {
        return nlohmann::json(arr);
    }
    
    /**
     * @brief Deserialize array of integers
     */
    inline std::vector<int> DeserializeIntArray(const nlohmann::json& j) {
        if (!j.is_array()) return std::vector<int>();
        return j.get<std::vector<int>>();
    }
    
    /**
     * @brief Serialize array of floats
     */
    inline nlohmann::json SerializeFloatArray(const std::vector<float>& arr) {
        return nlohmann::json(arr);
    }
    
    /**
     * @brief Deserialize array of floats
     */
    inline std::vector<float> DeserializeFloatArray(const nlohmann::json& j) {
        if (!j.is_array()) return std::vector<float>();
        return j.get<std::vector<float>>();
    }
    
    /**
     * @brief Validate serialization version
     * @param version Version string from JSON
     * @param expectedMajor Expected major version
     * @return True if compatible
     */
    inline bool ValidateVersion(const std::string& version, int expectedMajor) {
        // Parse version string "major.minor"
        size_t dotPos = version.find('.');
        if (dotPos == std::string::npos) return false;
        
        try {
            int major = std::stoi(version.substr(0, dotPos));
            return major == expectedMajor;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Get current serialization version
     */
    inline std::string GetCurrentVersion() {
        return "1.0";
    }
}
