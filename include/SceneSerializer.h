#pragma once

#include "GameObject.h"
#include "Light.h"
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @class SceneSerializer
 * @brief Handles serialization and deserialization of game scenes in both JSON and binary formats.
 * 
 * Supports:
 * - Complete scene hierarchy (GameObjects with parent-child relationships)
 * - Transforms (position, rotation, scale)
 * - Materials and textures
 * - Lights (directional, point, spot)
 * - Animation and skeletal data
 * - Physics components (RigidBody, KinematicController)
 * - LOD levels and impostors
 * - Custom properties via JSON extensions
 * 
 * JSON format is human-readable and suitable for editing.
 * Binary format is compact and efficient for production.
 */
class SceneSerializer {
public:
    enum class SerializationFormat {
        JSON,     // Human-readable text format (.scene.json)
        BINARY    // Compact binary format (.scene.bin)
    };

    struct SerializeOptions {
        SerializationFormat format = SerializationFormat::JSON;
        bool includeChildren = true;
        bool includeLights = true;
        bool includeAnimations = true;
        bool includePhysics = true;
        bool includeMaterials = true;
        bool prettyPrintJSON = true;  // Only for JSON format
    };

    SceneSerializer() : m_PhysXBackend(nullptr), m_PrefabManager(nullptr) {}
    ~SceneSerializer() = default;

    void SetPhysXBackend(class PhysXBackend* backend) { m_PhysXBackend = backend; }
    void SetPrefabManager(class PrefabManager* manager) { m_PrefabManager = manager; }

    // ===== Scene Serialization =====
    /**
     * @brief Serialize entire scene to file
     * @param root Root GameObject of the scene
     * @param filename Output file path
     * @param lights List of lights in the scene
     * @param options Serialization options
     * @return true if successful
     */
    bool SerializeScene(
        std::shared_ptr<GameObject> root,
        const std::string& filename,
        const std::vector<Light>& lights,
        const SerializeOptions& options = SerializeOptions()
    );

    /**
     * @brief Deserialize scene from file
     * @param filename Input file path
     * @param outRoot Pointer to receive root GameObject
     * @param outLights Pointer to receive lights list
     * @return true if successful
     */
    bool DeserializeScene(
        const std::string& filename,
        std::shared_ptr<GameObject>& outRoot,
        std::vector<Light>& outLights
    );

    // ===== Single GameObject Serialization =====
    /**
     * @brief Serialize single GameObject to JSON
     * @param obj GameObject to serialize
     * @param includeChildren Whether to include child GameObjects
     * @return JSON object representing the GameObject
     */
    json SerializeGameObjectToJson(
        std::shared_ptr<GameObject> obj,
        bool includeChildren = true
    );

    /**
     * @brief Deserialize GameObject from JSON
     * @param data JSON object
     * @return Reconstructed GameObject
     */
    std::shared_ptr<GameObject> DeserializeGameObjectFromJson(const json& data, std::shared_ptr<GameObject> parent = nullptr);

    /**
     * @brief Serialize single GameObject to binary
     * @param obj GameObject to serialize
     * @param includeChildren Whether to include child GameObjects
     * @return Binary data as vector<uint8_t>
     */
    std::vector<uint8_t> SerializeGameObjectToBinary(
        std::shared_ptr<GameObject> obj,
        bool includeChildren = true
    );

    /**
     * @brief Deserialize GameObject from binary
     * @param data Binary data
     * @return Reconstructed GameObject
     */
    std::shared_ptr<GameObject> DeserializeGameObjectFromBinary(const std::vector<uint8_t>& data);

    // ===== Light Serialization =====
    /**
     * @brief Serialize lights to JSON
     * @param lights Vector of lights to serialize
     * @return JSON array of lights
     */
    json SerializeLightsToJson(const std::vector<Light>& lights);

    /**
     * @brief Deserialize lights from JSON
     * @param data JSON array of lights
     * @return Vector of deserialized lights
     */
    std::vector<Light> DeserializeLightsFromJson(const json& data);

    // ===== Utility Functions =====
    /**
     * @brief Check if a file is in the expected format
     * @param filename Path to check
     * @return true if file exists and is readable
     */
    static bool IsValidSceneFile(const std::string& filename);

    /**
     * @brief Get format from file extension
     * @param filename File path
     * @return Detected format
     */
    static SerializationFormat DetectFormat(const std::string& filename);

    /**
     * @brief Convert binary serialization to JSON (for inspection)
     * @param binaryData Binary scene data
     * @return JSON representation
     */
    json ConvertBinaryToJson(const std::vector<uint8_t>& binaryData);

    /**
     * @brief Get human-readable error message from last operation
     * @return Error string
     */
    const std::string& GetLastError() const { return m_LastError; }

private:
    // ===== JSON Serialization Helpers =====
    json SerializeTransform(const Transform& transform);
    Transform DeserializeTransform(const json& data);

    json SerializeMaterial(std::shared_ptr<Material> material);
    std::shared_ptr<Material> DeserializeMaterial(const json& data);

    json SerializeLight(const Light& light);
    Light DeserializeLight(const json& data);

    json SerializeAnimator(std::shared_ptr<class Animator> animator);
    std::shared_ptr<class Animator> DeserializeAnimator(const json& data);

    json SerializeRigidBody(std::shared_ptr<class RigidBody> rigidbody);
    std::shared_ptr<class RigidBody> DeserializeRigidBody(const json& data);

    json SerializeKinematicController(std::shared_ptr<class KinematicController> controller);
    std::shared_ptr<class KinematicController> DeserializeKinematicController(const json& data);

    json SerializeLODLevels(const std::shared_ptr<GameObject>& obj);
    void DeserializeLODLevels(std::shared_ptr<GameObject>& obj, const json& data);

    json SerializePhysXRigidBody(std::shared_ptr<class IPhysicsRigidBody> rigidbody);
    std::shared_ptr<class IPhysicsRigidBody> DeserializePhysXRigidBody(const json& data);

    json SerializeDestructible(std::shared_ptr<class PhysXDestructible> destructible);
    std::shared_ptr<class PhysXDestructible> DeserializeDestructible(const json& data);

    json SerializeArticulation(std::shared_ptr<class PhysXArticulation> articulation);
    std::shared_ptr<class PhysXArticulation> DeserializeArticulation(const json& data);

    json SerializeArticulationLink(std::shared_ptr<class PhysXArticulationLink> link);
    std::shared_ptr<class PhysXArticulationLink> DeserializeArticulationLink(const json& data, std::shared_ptr<GameObject> parent, std::shared_ptr<GameObject> current);

    json SerializeAggregate(std::shared_ptr<class PhysXAggregate> aggregate);
    std::shared_ptr<class PhysXAggregate> DeserializeAggregate(const json& data);

    // ===== Binary Serialization Helpers =====
    void WriteBinaryString(std::vector<uint8_t>& buffer, const std::string& str);
    std::string ReadBinaryString(const std::vector<uint8_t>& buffer, size_t& offset);

    void WriteBinaryVec3(std::vector<uint8_t>& buffer, const Vec3& v);
    Vec3 ReadBinaryVec3(const std::vector<uint8_t>& buffer, size_t& offset);

    void WriteBinaryVec2(std::vector<uint8_t>& buffer, const Vec2& v);
    Vec2 ReadBinaryVec2(const std::vector<uint8_t>& buffer, size_t& offset);

    void WriteBinaryGameObject(
        std::vector<uint8_t>& buffer,
        std::shared_ptr<GameObject> obj,
        bool includeChildren
    );
    std::shared_ptr<GameObject> ReadBinaryGameObject(
        const std::vector<uint8_t>& buffer,
        size_t& offset
    );

    // ===== Error Handling =====
    void SetError(const std::string& error);
    std::string m_LastError;

    // ===== Constants =====
    static constexpr uint32_t BINARY_FORMAT_VERSION = 1;
    static constexpr uint32_t BINARY_MAGIC = 0x53434E45;  // "SCNE" in hex

    class PhysXBackend* m_PhysXBackend;
    class PrefabManager* m_PrefabManager;
};
