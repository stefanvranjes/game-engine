#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

/**
 * @brief Support for glTF 2.0 extensions and additional features
 * 
 * This module adds support for common glTF extensions like:
 * - KHR_materials_pbrSpecularGlossiness (legacy PBR)
 * - KHR_materials_unlit (unlit materials)
 * - KHR_texture_transform (texture coordinate transforms)
 * - KHR_draco_mesh_compression (geometry compression)
 * - KHR_mesh_quantization (vertex position quantization)
 * - KHR_lights_punctual (advanced lighting)
 * - KHR_materials_variants (material variants/LOD)
 */

class GLTFExtensions {
public:
    /**
     * @brief Available glTF extensions
     */
    enum class Extension {
        // Core extensions
        KHR_Materials_PBRSpecularGlossiness,  // Legacy PBR workflow
        KHR_Materials_Unlit,                   // Unlit (emissive-only) materials
        KHR_Texture_Transform,                 // UV coordinate transforms
        KHR_Lights_Punctual,                   // Advanced punctual lights
        KHR_Materials_ClearCoat,              // Clear coat layer
        KHR_Materials_Sheen,                  // Fabric/cloth sheen
        KHR_Materials_Transmission,           // Glass/transparent materials
        KHR_Materials_IOR,                    // Index of refraction
        KHR_Materials_Volume,                 // Volume rendering
        KHR_Materials_Emissive_Strength,      // Emissive intensity control
        
        // Optimization extensions
        KHR_Draco_Mesh_Compression,           // Geometry compression
        KHR_Mesh_Quantization,                // Vertex quantization
        
        // Variant/LOD extensions
        KHR_Materials_Variants,               // Material variants (LOD)
        
        // Animation extensions
        KHR_Animation_Pointer,                // Advanced animation targeting
        
        // Vendor/custom extensions
        EXT_Meshopt_Compression,              // Alternative mesh compression
        EXT_Texture_WebP,                     // WebP texture support
        
        Unknown
    };

    /**
     * @brief Information about a texture transform
     */
    struct TextureTransform {
        float offsetU = 0.0f;
        float offsetV = 0.0f;
        float rotationZ = 0.0f;
        float scaleU = 1.0f;
        float scaleV = 1.0f;
    };

    /**
     * @brief Information about a punctual light
     */
    struct PunctualLight {
        enum class Type { Directional, Point, Spot } type;
        float intensity = 1.0f;
        float range = 0.0f;  // 0 = infinite
        float innerConeAngle = 0.0f;  // Spot only
        float outerConeAngle = 0.0f;  // Spot only
        float colorR = 1.0f;
        float colorG = 1.0f;
        float colorB = 1.0f;
    };

    /**
     * @brief Information about material variants/LOD
     */
    struct MaterialVariant {
        std::string name;
        int index;
        float transitionDistance = 0.0f;  // For distance-based LOD
    };

    /**
     * @brief Parse glTF JSON extras and extensions
     * 
     * @param json JSON object to parse
     * @return List of extensions found in the object
     */
    static std::vector<Extension> ParseExtensions(const nlohmann::json& json);

    /**
     * @brief Check if extension is supported
     * 
     * @param ext Extension to check
     * @return true if the extension can be handled
     */
    static bool IsExtensionSupported(Extension ext);

    /**
     * @brief Get human-readable extension name
     * 
     * @param ext Extension enum
     * @return Extension name string
     */
    static std::string GetExtensionName(Extension ext);

    /**
     * @brief Extract texture transform from glTF extension
     * 
     * @param json KHR_texture_transform JSON object
     * @return TextureTransform with parsed values
     */
    static TextureTransform ParseTextureTransform(const nlohmann::json& json);

    /**
     * @brief Extract punctual light info from glTF extension
     * 
     * @param json KHR_lights_punctual JSON object
     * @return PunctualLight with parsed values
     */
    static PunctualLight ParsePunctualLight(const nlohmann::json& json);

    /**
     * @brief Extract material variants from glTF extension
     * 
     * @param json KHR_materials_variants JSON object
     * @return Vector of material variants
     */
    static std::vector<MaterialVariant> ParseMaterialVariants(const nlohmann::json& json);

    /**
     * @brief Check if material uses Specular-Glossiness workflow
     * 
     * @param materialJson Material JSON object
     * @return true if using legacy PBR workflow
     */
    static bool IsSpecularGlossiness(const nlohmann::json& materialJson);

    /**
     * @brief Check if material is unlit
     * 
     * @param materialJson Material JSON object
     * @return true if material is unlit (emissive only)
     */
    static bool IsUnlit(const nlohmann::json& materialJson);

    /**
     * @brief Check if material has clear coat layer
     * 
     * @param materialJson Material JSON object
     * @return true if clear coat extension is present
     */
    static bool HasClearCoat(const nlohmann::json& materialJson);

    /**
     * @brief Check if geometry uses Draco compression
     * 
     * @param primitiveJson Primitive JSON object
     * @return true if Draco compression is used
     */
    static bool IsDracoCompressed(const nlohmann::json& primitiveJson);

    /**
     * @brief Check if geometry is quantized
     * 
     * @param primitiveJson Primitive JSON object
     * @return true if mesh quantization is used
     */
    static bool IsQuantized(const nlohmann::json& primitiveJson);

    /**
     * @brief Get decompression info for Draco-compressed geometry
     * 
     * @param json Draco extension JSON
     * @return Decompression parameters as JSON
     */
    static nlohmann::json GetDracoDecompressionInfo(const nlohmann::json& json);

    /**
     * @brief Validate extension compatibility
     * 
     * @param extensionName Extension name string
     * @param requiredExtensions List of required extensions
     * @return true if extension is required and supported
     */
    static bool ValidateExtension(const std::string& extensionName,
                                 const std::vector<std::string>& requiredExtensions);
};

/**
 * @brief Settings for glTF loader behavior with extensions
 */
struct GLTFExtensionSettings {
    bool supportDracoCompression = true;
    bool supportTextureTransforms = true;
    bool supportPunctualLights = true;
    bool supportMaterialVariants = true;
    bool supportClearCoat = true;
    bool supportUnlit = true;
    bool supportEmissiveStrength = true;
    bool fallbackOnUnsupported = true;  // Fall back to basic rendering if extension unsupported
    bool verbose = false;
};
