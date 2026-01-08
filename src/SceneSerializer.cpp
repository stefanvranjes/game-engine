#include "SceneSerializer.h"
#include "MaterialNew.h"
#include "Animator.h"
#include "RigidBody.h"
#include "KinematicController.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>

bool SceneSerializer::SerializeScene(
    std::shared_ptr<GameObject> root,
    const std::string& filename,
    const std::vector<Light>& lights,
    const SerializeOptions& options)
{
    try {
        if (!root) {
            SetError("Root GameObject is null");
            return false;
        }

        if (options.format == SerializationFormat::JSON) {
            json sceneJson;
            sceneJson["version"] = BINARY_FORMAT_VERSION;
            sceneJson["format"] = "json";
            sceneJson["rootObject"] = SerializeGameObjectToJson(root, options.includeChildren);
            
            if (options.includeLights) {
                sceneJson["lights"] = SerializeLightsToJson(lights);
            }

            std::ofstream file(filename);
            if (!file.is_open()) {
                SetError("Failed to open file for writing: " + filename);
                return false;
            }

            if (options.prettyPrintJSON) {
                file << sceneJson.dump(4);
            } else {
                file << sceneJson.dump();
            }
            file.close();

            std::cout << "Scene saved (JSON): " << filename << std::endl;
            return true;
        }
        else if (options.format == SerializationFormat::BINARY) {
            std::vector<uint8_t> buffer;

            // Write magic and version
            uint32_t magic = BINARY_MAGIC;
            buffer.insert(buffer.end(), (uint8_t*)&magic, (uint8_t*)&magic + 4);
            uint32_t version = BINARY_FORMAT_VERSION;
            buffer.insert(buffer.end(), (uint8_t*)&version, (uint8_t*)&version + 4);

            // Write root object
            WriteBinaryGameObject(buffer, root, options.includeChildren);

            // Write lights
            if (options.includeLights) {
                uint32_t lightCount = static_cast<uint32_t>(lights.size());
                buffer.insert(buffer.end(), (uint8_t*)&lightCount, (uint8_t*)&lightCount + 4);
                for (const auto& light : lights) {
                    json lightJson = SerializeLight(light);
                    WriteBinaryString(buffer, lightJson.dump());
                }
            } else {
                uint32_t lightCount = 0;
                buffer.insert(buffer.end(), (uint8_t*)&lightCount, (uint8_t*)&lightCount + 4);
            }

            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                SetError("Failed to open file for writing: " + filename);
                return false;
            }

            file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            file.close();

            std::cout << "Scene saved (BINARY): " << filename << " (" << buffer.size() << " bytes)" << std::endl;
            return true;
        }

        SetError("Unknown serialization format");
        return false;
    }
    catch (const std::exception& e) {
        SetError(std::string("Exception during serialization: ") + e.what());
        return false;
    }
}

bool SceneSerializer::DeserializeScene(
    const std::string& filename,
    std::shared_ptr<GameObject>& outRoot,
    std::vector<Light>& outLights)
{
    try {
        if (!IsValidSceneFile(filename)) {
            SetError("Invalid or non-existent scene file: " + filename);
            return false;
        }

        SerializationFormat format = DetectFormat(filename);

        if (format == SerializationFormat::JSON) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                SetError("Failed to open file for reading: " + filename);
                return false;
            }

            json sceneJson;
            file >> sceneJson;
            file.close();

            // Validate version
            if (sceneJson.contains("version")) {
                uint32_t version = sceneJson["version"];
                if (version != BINARY_FORMAT_VERSION) {
                    std::cerr << "Warning: Version mismatch. Expected " << BINARY_FORMAT_VERSION
                              << ", got " << version << std::endl;
                }
            }

            outRoot = DeserializeGameObjectFromJson(sceneJson["rootObject"]);
            if (!outRoot) {
                SetError("Failed to deserialize root GameObject");
                return false;
            }

            if (sceneJson.contains("lights")) {
                outLights = DeserializeLightsFromJson(sceneJson["lights"]);
            }

            std::cout << "Scene loaded (JSON): " << filename << std::endl;
            return true;
        }
        else if (format == SerializationFormat::BINARY) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                SetError("Failed to open file for reading: " + filename);
                return false;
            }

            std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
            file.close();

            size_t offset = 0;

            // Read and validate magic
            if (buffer.size() < 8) {
                SetError("Invalid binary format (too small)");
                return false;
            }

            uint32_t magic = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
            offset += 4;
            if (magic != BINARY_MAGIC) {
                SetError("Invalid magic number");
                return false;
            }

            uint32_t version = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
            offset += 4;
            if (version != BINARY_FORMAT_VERSION) {
                std::cerr << "Warning: Version mismatch. Expected " << BINARY_FORMAT_VERSION
                          << ", got " << version << std::endl;
            }

            // Read root object
            outRoot = ReadBinaryGameObject(buffer, offset);
            if (!outRoot) {
                SetError("Failed to deserialize root GameObject");
                return false;
            }

            // Read lights
            if (offset + 4 <= buffer.size()) {
                uint32_t lightCount = *reinterpret_cast<uint32_t*>(buffer.data() + offset);
                offset += 4;
                for (uint32_t i = 0; i < lightCount && offset < buffer.size(); ++i) {
                    std::string lightJson = ReadBinaryString(buffer, offset);
                    try {
                        json lightData = json::parse(lightJson);
                        outLights.push_back(DeserializeLight(lightData));
                    }
                    catch (...) {
                        std::cerr << "Warning: Failed to deserialize light " << i << std::endl;
                    }
                }
            }

            std::cout << "Scene loaded (BINARY): " << filename << " (" << buffer.size() << " bytes)" << std::endl;
            return true;
        }

        SetError("Unknown file format");
        return false;
    }
    catch (const std::exception& e) {
        SetError(std::string("Exception during deserialization: ") + e.what());
        return false;
    }
}

json SceneSerializer::SerializeGameObjectToJson(
    std::shared_ptr<GameObject> obj,
    bool includeChildren)
{
    json objJson;

    if (!obj) return objJson;

    objJson["name"] = obj->GetName();
    objJson["transform"] = SerializeTransform(obj->GetTransform());
    objJson["uvOffset"] = {obj->GetUVOffset().x, obj->GetUVOffset().y};
    objJson["uvScale"] = {obj->GetUVScale().x, obj->GetUVScale().y};
    objJson["visible"] = obj->IsVisible();

    // Serialize material
    if (auto mat = obj->GetMaterial()) {
        objJson["material"] = SerializeMaterial(mat);
    }

    // Serialize animator
    if (auto animator = obj->GetAnimator()) {
        objJson["animator"] = SerializeAnimator(animator);
    }

    // Serialize physics
    if (auto rb = obj->GetRigidBody()) {
        objJson["rigidBody"] = SerializeRigidBody(rb);
    }
    if (auto kc = obj->GetKinematicController()) {
        objJson["kinematicController"] = SerializeKinematicController(kc);
    }

    // Serialize LOD levels
    objJson["lodLevels"] = SerializeLODLevels(obj);

    // Serialize children recursively
    if (includeChildren && !obj->GetChildren().empty()) {
        json childrenJson = json::array();
        for (auto& child : obj->GetChildren()) {
            childrenJson.push_back(SerializeGameObjectToJson(child, true));
        }
        objJson["children"] = childrenJson;
    }

    return objJson;
}

std::shared_ptr<GameObject> SceneSerializer::DeserializeGameObjectFromJson(const json& data)
{
    if (!data.contains("name")) {
        SetError("Missing 'name' field in GameObject data");
        return nullptr;
    }

    auto obj = std::make_shared<GameObject>(data["name"].get<std::string>());

    // Deserialize transform
    if (data.contains("transform")) {
        obj->GetTransform() = DeserializeTransform(data["transform"]);
    }

    // Deserialize UV settings
    if (data.contains("uvOffset")) {
        obj->SetUVOffset(Vec2(data["uvOffset"][0], data["uvOffset"][1]));
    }
    if (data.contains("uvScale")) {
        obj->SetUVScale(Vec2(data["uvScale"][0], data["uvScale"][1]));
    }

    // Deserialize visibility
    if (data.contains("visible")) {
        obj->SetVisible(data["visible"].get<bool>());
    }

    // Deserialize material
    if (data.contains("material")) {
        if (auto mat = DeserializeMaterial(data["material"])) {
            obj->SetMaterial(mat);
        }
    }

    // Deserialize animator
    if (data.contains("animator")) {
        if (auto animator = DeserializeAnimator(data["animator"])) {
            obj->SetAnimator(animator);
        }
    }

    // Deserialize physics
    if (data.contains("rigidBody")) {
        if (auto rb = DeserializeRigidBody(data["rigidBody"])) {
            obj->SetRigidBody(rb);
        }
    }
    if (data.contains("kinematicController")) {
        if (auto kc = DeserializeKinematicController(data["kinematicController"])) {
            obj->SetKinematicController(kc);
        }
    }

    // Deserialize LOD levels
    if (data.contains("lodLevels")) {
        DeserializeLODLevels(obj, data["lodLevels"]);
    }

    // Deserialize children recursively
    if (data.contains("children")) {
        for (const auto& childData : data["children"]) {
            if (auto child = DeserializeGameObjectFromJson(childData)) {
                obj->AddChild(child);
            }
        }
    }

    return obj;
}

json SceneSerializer::SerializeTransform(const Transform& transform)
{
    json tJson;
    tJson["position"] = {transform.position.x, transform.position.y, transform.position.z};
    tJson["rotation"] = {transform.rotation.x, transform.rotation.y, transform.rotation.z};
    tJson["scale"] = {transform.scale.x, transform.scale.y, transform.scale.z};
    return tJson;
}

Transform SceneSerializer::DeserializeTransform(const json& data)
{
    Vec3 pos(0), rot(0), scale(1);

    if (data.contains("position")) {
        pos = Vec3(data["position"][0], data["position"][1], data["position"][2]);
    }
    if (data.contains("rotation")) {
        rot = Vec3(data["rotation"][0], data["rotation"][1], data["rotation"][2]);
    }
    if (data.contains("scale")) {
        scale = Vec3(data["scale"][0], data["scale"][1], data["scale"][2]);
    }

    return Transform(pos, rot, scale);
}

json SceneSerializer::SerializeMaterial(std::shared_ptr<Material> material)
{
    json matJson;
    // Store material properties - extend as needed based on Material class
    matJson["name"] = "Material"; // You might want to add name to Material class
    // Additional material properties can be added here
    return matJson;
}

std::shared_ptr<Material> SceneSerializer::DeserializeMaterial(const json& data)
{
    auto mat = std::make_shared<Material>();
    // Deserialize material properties - extend as needed
    return mat;
}

json SceneSerializer::SerializeLight(const Light& light)
{
    json lightJson;
    lightJson["type"] = static_cast<int>(light.type);
    lightJson["position"] = {light.position.x, light.position.y, light.position.z};
    lightJson["direction"] = {light.direction.x, light.direction.y, light.direction.z};
    lightJson["color"] = {light.color.x, light.color.y, light.color.z};
    lightJson["intensity"] = light.intensity;
    lightJson["range"] = light.range;
    lightJson["cutOff"] = light.cutOff;
    lightJson["outerCutOff"] = light.outerCutOff;
    return lightJson;
}

Light SceneSerializer::DeserializeLight(const json& data)
{
    Light light;
    if (data.contains("type")) light.type = static_cast<LightType>(data["type"].get<int>());
    if (data.contains("position")) light.position = Vec3(data["position"][0], data["position"][1], data["position"][2]);
    if (data.contains("direction")) light.direction = Vec3(data["direction"][0], data["direction"][1], data["direction"][2]);
    if (data.contains("color")) light.color = Vec3(data["color"][0], data["color"][1], data["color"][2]);
    if (data.contains("intensity")) light.intensity = data["intensity"];
    if (data.contains("range")) light.range = data["range"];
    if (data.contains("cutOff")) light.cutOff = data["cutOff"];
    if (data.contains("outerCutOff")) light.outerCutOff = data["outerCutOff"];
    return light;
}

json SceneSerializer::SerializeLightsToJson(const std::vector<Light>& lights)
{
    json lightsJson = json::array();
    for (const auto& light : lights) {
        lightsJson.push_back(SerializeLight(light));
    }
    return lightsJson;
}

std::vector<Light> SceneSerializer::DeserializeLightsFromJson(const json& data)
{
    std::vector<Light> lights;
    if (data.is_array()) {
        for (const auto& lightData : data) {
            lights.push_back(DeserializeLight(lightData));
        }
    }
    return lights;
}

json SceneSerializer::SerializeAnimator(std::shared_ptr<class Animator> animator)
{
    json animJson;
    // Implement based on Animator class structure
    // Placeholder for now
    return animJson;
}

std::shared_ptr<class Animator> SceneSerializer::DeserializeAnimator(const json& data)
{
    // Implement based on Animator class structure
    // Placeholder for now
    return nullptr;
}

json SceneSerializer::SerializeRigidBody(std::shared_ptr<class RigidBody> rigidbody)
{
    json rbJson;
    // Implement based on RigidBody class structure
    return rbJson;
}

std::shared_ptr<class RigidBody> SceneSerializer::DeserializeRigidBody(const json& data)
{
    // Implement based on RigidBody class structure
    return nullptr;
}

json SceneSerializer::SerializeKinematicController(std::shared_ptr<class KinematicController> controller)
{
    json kcJson;
    // Implement based on KinematicController class structure
    return kcJson;
}

std::shared_ptr<class KinematicController> SceneSerializer::DeserializeKinematicController(const json& data)
{
    // Implement based on KinematicController class structure
    return nullptr;
}

json SceneSerializer::SerializeLODLevels(const std::shared_ptr<GameObject>& obj)
{
    json lodJson = json::array();
    // Extract and serialize LOD levels from object
    // This would need public access to m_LODs in GameObject
    return lodJson;
}

void SceneSerializer::DeserializeLODLevels(std::shared_ptr<GameObject>& obj, const json& data)
{
    // Deserialize and add LOD levels to object
}

void SceneSerializer::WriteBinaryString(std::vector<uint8_t>& buffer, const std::string& str)
{
    uint32_t length = str.length();
    buffer.insert(buffer.end(), (uint8_t*)&length, (uint8_t*)&length + 4);
    buffer.insert(buffer.end(), str.begin(), str.end());
}

std::string SceneSerializer::ReadBinaryString(const std::vector<uint8_t>& buffer, size_t& offset)
{
    if (offset + 4 > buffer.size()) return "";
    
    uint32_t length = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += 4;

    if (offset + length > buffer.size()) return "";
    
    std::string result(buffer.begin() + offset, buffer.begin() + offset + length);
    offset += length;
    return result;
}

void SceneSerializer::WriteBinaryVec3(std::vector<uint8_t>& buffer, const Vec3& v)
{
    buffer.insert(buffer.end(), (uint8_t*)&v.x, (uint8_t*)&v.x + sizeof(float));
    buffer.insert(buffer.end(), (uint8_t*)&v.y, (uint8_t*)&v.y + sizeof(float));
    buffer.insert(buffer.end(), (uint8_t*)&v.z, (uint8_t*)&v.z + sizeof(float));
}

Vec3 SceneSerializer::ReadBinaryVec3(const std::vector<uint8_t>& buffer, size_t& offset)
{
    if (offset + 12 > buffer.size()) return Vec3(0);
    Vec3 result;
    result.x = *reinterpret_cast<const float*>(buffer.data() + offset);
    offset += 4;
    result.y = *reinterpret_cast<const float*>(buffer.data() + offset);
    offset += 4;
    result.z = *reinterpret_cast<const float*>(buffer.data() + offset);
    offset += 4;
    return result;
}

void SceneSerializer::WriteBinaryVec2(std::vector<uint8_t>& buffer, const Vec2& v)
{
    buffer.insert(buffer.end(), (uint8_t*)&v.x, (uint8_t*)&v.x + sizeof(float));
    buffer.insert(buffer.end(), (uint8_t*)&v.y, (uint8_t*)&v.y + sizeof(float));
}

Vec2 SceneSerializer::ReadBinaryVec2(const std::vector<uint8_t>& buffer, size_t& offset)
{
    if (offset + 8 > buffer.size()) return Vec2(0, 0);
    Vec2 result;
    result.x = *reinterpret_cast<const float*>(buffer.data() + offset);
    offset += 4;
    result.y = *reinterpret_cast<const float*>(buffer.data() + offset);
    offset += 4;
    return result;
}

void SceneSerializer::WriteBinaryGameObject(
    std::vector<uint8_t>& buffer,
    std::shared_ptr<GameObject> obj,
    bool includeChildren)
{
    if (!obj) return;

    // Serialize as JSON for now, write to binary
    json objJson = SerializeGameObjectToJson(obj, includeChildren);
    WriteBinaryString(buffer, objJson.dump());
}

std::shared_ptr<GameObject> SceneSerializer::ReadBinaryGameObject(
    const std::vector<uint8_t>& buffer,
    size_t& offset)
{
    std::string objJsonStr = ReadBinaryString(buffer, offset);
    if (objJsonStr.empty()) return nullptr;

    try {
        json objJson = json::parse(objJsonStr);
        return DeserializeGameObjectFromJson(objJson);
    }
    catch (const std::exception& e) {
        SetError(std::string("Failed to parse binary GameObject: ") + e.what());
        return nullptr;
    }
}

std::vector<uint8_t> SceneSerializer::SerializeGameObjectToBinary(
    std::shared_ptr<GameObject> obj,
    bool includeChildren)
{
    std::vector<uint8_t> buffer;
    WriteBinaryGameObject(buffer, obj, includeChildren);
    return buffer;
}

std::shared_ptr<GameObject> SceneSerializer::DeserializeGameObjectFromBinary(const std::vector<uint8_t>& data)
{
    size_t offset = 0;
    return ReadBinaryGameObject(data, offset);
}

bool SceneSerializer::IsValidSceneFile(const std::string& filename)
{
    std::ifstream file(filename);
    return file.good();
}

SceneSerializer::SerializationFormat SceneSerializer::DetectFormat(const std::string& filename)
{
    if (filename.length() >= 5) {
        std::string ext = filename.substr(filename.length() - 5);
        if (ext == ".json") return SerializationFormat::JSON;
    }
    if (filename.length() >= 4) {
        std::string ext = filename.substr(filename.length() - 4);
        if (ext == ".bin") return SerializationFormat::BINARY;
    }
    return SerializationFormat::JSON; // Default to JSON
}

json SceneSerializer::ConvertBinaryToJson(const std::vector<uint8_t>& binaryData)
{
    try {
        size_t offset = 0;
        if (binaryData.size() < 8) {
            SetError("Binary data too small");
            return json();
        }

        uint32_t magic = *reinterpret_cast<const uint32_t*>(binaryData.data());
        if (magic != BINARY_MAGIC) {
            SetError("Invalid magic number");
            return json();
        }
        offset += 4;

        uint32_t version = *reinterpret_cast<const uint32_t*>(binaryData.data() + offset);
        offset += 4;

        std::shared_ptr<GameObject> root = ReadBinaryGameObject(binaryData, offset);
        if (!root) {
            SetError("Failed to deserialize root object");
            return json();
        }

        json result;
        result["version"] = version;
        result["rootObject"] = SerializeGameObjectToJson(root, true);
        return result;
    }
    catch (const std::exception& e) {
        SetError(std::string("Exception during binary conversion: ") + e.what());
        return json();
    }
}

void SceneSerializer::SetError(const std::string& error)
{
    m_LastError = error;
    std::cerr << "[SceneSerializer] " << error << std::endl;
}
