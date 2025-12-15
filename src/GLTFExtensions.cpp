#include "GLTFExtensions.h"
#include <iostream>
#include <algorithm>

/**
 * @brief Parse extensions from JSON
 */
std::vector<GLTFExtensions::Extension> GLTFExtensions::ParseExtensions(const nlohmann::json& json) {
    std::vector<Extension> extensions;

    if (!json.contains("extensions") || !json["extensions"].is_object()) {
        return extensions;
    }

    const auto& exts = json["extensions"];
    
    for (const auto& [key, value] : exts.items()) {
        if (key == "KHR_materials_pbrSpecularGlossiness") {
            extensions.push_back(Extension::KHR_Materials_PBRSpecularGlossiness);
        }
        else if (key == "KHR_materials_unlit") {
            extensions.push_back(Extension::KHR_Materials_Unlit);
        }
        else if (key == "KHR_texture_transform") {
            extensions.push_back(Extension::KHR_Texture_Transform);
        }
        else if (key == "KHR_lights_punctual") {
            extensions.push_back(Extension::KHR_Lights_Punctual);
        }
        else if (key == "KHR_materials_clearcoat") {
            extensions.push_back(Extension::KHR_Materials_ClearCoat);
        }
        else if (key == "KHR_materials_sheen") {
            extensions.push_back(Extension::KHR_Materials_Sheen);
        }
        else if (key == "KHR_materials_transmission") {
            extensions.push_back(Extension::KHR_Materials_Transmission);
        }
        else if (key == "KHR_materials_ior") {
            extensions.push_back(Extension::KHR_Materials_IOR);
        }
        else if (key == "KHR_materials_volume") {
            extensions.push_back(Extension::KHR_Materials_Volume);
        }
        else if (key == "KHR_materials_emissive_strength") {
            extensions.push_back(Extension::KHR_Materials_Emissive_Strength);
        }
        else if (key == "KHR_draco_mesh_compression") {
            extensions.push_back(Extension::KHR_Draco_Mesh_Compression);
        }
        else if (key == "KHR_mesh_quantization") {
            extensions.push_back(Extension::KHR_Mesh_Quantization);
        }
        else if (key == "KHR_materials_variants") {
            extensions.push_back(Extension::KHR_Materials_Variants);
        }
        else if (key == "KHR_animation_pointer") {
            extensions.push_back(Extension::KHR_Animation_Pointer);
        }
        else if (key == "EXT_meshopt_compression") {
            extensions.push_back(Extension::EXT_Meshopt_Compression);
        }
        else if (key == "EXT_texture_webp") {
            extensions.push_back(Extension::EXT_Texture_WebP);
        }
        else {
            extensions.push_back(Extension::Unknown);
        }
    }

    return extensions;
}

/**
 * @brief Check if extension is supported
 */
bool GLTFExtensions::IsExtensionSupported(Extension ext) {
    switch (ext) {
        // Fully supported extensions
        case Extension::KHR_Materials_Unlit:
        case Extension::KHR_Texture_Transform:
        case Extension::KHR_Lights_Punctual:
        case Extension::KHR_Mesh_Quantization:
        case Extension::KHR_Animation_Pointer:
            return true;
        
        // Partially supported (fallback)
        case Extension::KHR_Materials_PBRSpecularGlossiness:
        case Extension::KHR_Materials_ClearCoat:
        case Extension::KHR_Materials_Sheen:
        case Extension::KHR_Materials_Transmission:
        case Extension::KHR_Materials_IOR:
        case Extension::KHR_Materials_Volume:
        case Extension::KHR_Materials_Emissive_Strength:
            return true;
        
        // Not yet supported but can be implemented
        case Extension::KHR_Draco_Mesh_Compression:
        case Extension::EXT_Meshopt_Compression:
            return false;  // Requires additional libraries
        
        case Extension::KHR_Materials_Variants:
        case Extension::EXT_Texture_WebP:
            return false;  // Planned for future
        
        default:
            return false;
    }
}

/**
 * @brief Get extension name
 */
std::string GLTFExtensions::GetExtensionName(Extension ext) {
    switch (ext) {
        case Extension::KHR_Materials_PBRSpecularGlossiness:
            return "KHR_materials_pbrSpecularGlossiness";
        case Extension::KHR_Materials_Unlit:
            return "KHR_materials_unlit";
        case Extension::KHR_Texture_Transform:
            return "KHR_texture_transform";
        case Extension::KHR_Lights_Punctual:
            return "KHR_lights_punctual";
        case Extension::KHR_Materials_ClearCoat:
            return "KHR_materials_clearcoat";
        case Extension::KHR_Materials_Sheen:
            return "KHR_materials_sheen";
        case Extension::KHR_Materials_Transmission:
            return "KHR_materials_transmission";
        case Extension::KHR_Materials_IOR:
            return "KHR_materials_ior";
        case Extension::KHR_Materials_Volume:
            return "KHR_materials_volume";
        case Extension::KHR_Materials_Emissive_Strength:
            return "KHR_materials_emissive_strength";
        case Extension::KHR_Draco_Mesh_Compression:
            return "KHR_draco_mesh_compression";
        case Extension::KHR_Mesh_Quantization:
            return "KHR_mesh_quantization";
        case Extension::KHR_Materials_Variants:
            return "KHR_materials_variants";
        case Extension::KHR_Animation_Pointer:
            return "KHR_animation_pointer";
        case Extension::EXT_Meshopt_Compression:
            return "EXT_meshopt_compression";
        case Extension::EXT_Texture_WebP:
            return "EXT_texture_webp";
        default:
            return "Unknown";
    }
}

/**
 * @brief Parse texture transform extension
 */
GLTFExtensions::TextureTransform GLTFExtensions::ParseTextureTransform(const nlohmann::json& json) {
    TextureTransform transform;

    if (!json.is_object()) {
        return transform;
    }

    if (json.contains("offset") && json["offset"].is_array() && json["offset"].size() >= 2) {
        transform.offsetU = json["offset"][0].get<float>();
        transform.offsetV = json["offset"][1].get<float>();
    }

    if (json.contains("rotation")) {
        transform.rotationZ = json["rotation"].get<float>();
    }

    if (json.contains("scale") && json["scale"].is_array() && json["scale"].size() >= 2) {
        transform.scaleU = json["scale"][0].get<float>();
        transform.scaleV = json["scale"][1].get<float>();
    }

    return transform;
}

/**
 * @brief Parse punctual light extension
 */
GLTFExtensions::PunctualLight GLTFExtensions::ParsePunctualLight(const nlohmann::json& json) {
    PunctualLight light;
    light.type = PunctualLight::Type::Point;

    if (!json.is_object()) {
        return light;
    }

    if (json.contains("intensity")) {
        light.intensity = json["intensity"].get<float>();
    }

    if (json.contains("range")) {
        light.range = json["range"].get<float>();
    }

    if (json.contains("color") && json["color"].is_array() && json["color"].size() >= 3) {
        light.colorR = json["color"][0].get<float>();
        light.colorG = json["color"][1].get<float>();
        light.colorB = json["color"][2].get<float>();
    }

    if (json.contains("type")) {
        std::string typeStr = json["type"].get<std::string>();
        if (typeStr == "directional") {
            light.type = PunctualLight::Type::Directional;
        } else if (typeStr == "point") {
            light.type = PunctualLight::Type::Point;
        } else if (typeStr == "spot") {
            light.type = PunctualLight::Type::Spot;
            
            if (json.contains("spot")) {
                const auto& spot = json["spot"];
                if (spot.contains("innerConeAngle")) {
                    light.innerConeAngle = spot["innerConeAngle"].get<float>();
                }
                if (spot.contains("outerConeAngle")) {
                    light.outerConeAngle = spot["outerConeAngle"].get<float>();
                }
            }
        }
    }

    return light;
}

/**
 * @brief Parse material variants extension
 */
std::vector<GLTFExtensions::MaterialVariant> GLTFExtensions::ParseMaterialVariants(const nlohmann::json& json) {
    std::vector<MaterialVariant> variants;

    if (!json.contains("variants") || !json["variants"].is_array()) {
        return variants;
    }

    for (size_t i = 0; i < json["variants"].size(); ++i) {
        const auto& variantJson = json["variants"][i];
        
        MaterialVariant variant;
        variant.index = i;
        
        if (variantJson.contains("name")) {
            variant.name = variantJson["name"].get<std::string>();
        }

        variants.push_back(variant);
    }

    return variants;
}

/**
 * @brief Check if material uses Specular-Glossiness workflow
 */
bool GLTFExtensions::IsSpecularGlossiness(const nlohmann::json& materialJson) {
    if (!materialJson.contains("extensions")) {
        return false;
    }

    const auto& exts = materialJson["extensions"];
    return exts.contains("KHR_materials_pbrSpecularGlossiness");
}

/**
 * @brief Check if material is unlit
 */
bool GLTFExtensions::IsUnlit(const nlohmann::json& materialJson) {
    if (!materialJson.contains("extensions")) {
        return false;
    }

    const auto& exts = materialJson["extensions"];
    return exts.contains("KHR_materials_unlit");
}

/**
 * @brief Check if material has clear coat
 */
bool GLTFExtensions::HasClearCoat(const nlohmann::json& materialJson) {
    if (!materialJson.contains("extensions")) {
        return false;
    }

    const auto& exts = materialJson["extensions"];
    return exts.contains("KHR_materials_clearcoat");
}

/**
 * @brief Check if geometry is Draco compressed
 */
bool GLTFExtensions::IsDracoCompressed(const nlohmann::json& primitiveJson) {
    if (!primitiveJson.contains("extensions")) {
        return false;
    }

    const auto& exts = primitiveJson["extensions"];
    return exts.contains("KHR_draco_mesh_compression");
}

/**
 * @brief Check if geometry is quantized
 */
bool GLTFExtensions::IsQuantized(const nlohmann::json& primitiveJson) {
    if (!primitiveJson.contains("extensions")) {
        return false;
    }

    const auto& exts = primitiveJson["extensions"];
    return exts.contains("KHR_mesh_quantization");
}

/**
 * @brief Get Draco decompression info
 */
nlohmann::json GLTFExtensions::GetDracoDecompressionInfo(const nlohmann::json& json) {
    if (!json.contains("KHR_draco_mesh_compression")) {
        return nlohmann::json::object();
    }

    return json["KHR_draco_mesh_compression"];
}

/**
 * @brief Validate extension
 */
bool GLTFExtensions::ValidateExtension(const std::string& extensionName,
                                       const std::vector<std::string>& requiredExtensions) {
    // Check if this extension is required
    auto it = std::find(requiredExtensions.begin(), requiredExtensions.end(), extensionName);
    
    if (it != requiredExtensions.end()) {
        // Extension is required, check if it's supported
        if (extensionName == "KHR_materials_unlit" ||
            extensionName == "KHR_texture_transform" ||
            extensionName == "KHR_mesh_quantization") {
            return true;  // Supported
        }
        return false;  // Required but not supported
    }

    return true;  // Not required
}
