#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <filesystem>
#include "AssetHash.h"

class Texture;
class Mesh;
class Material;

/**
 * @brief Asset format conversion and optimization
 * 
 * Converts assets between formats and optimizes them for runtime use.
 * Supports textures, models, materials, and other asset types with
 * various compression and optimization options.
 */
class AssetConverter {
public:
    /**
     * @brief Conversion options for textures
     */
    struct TextureConversionOptions {
        bool compress = true;              // Use BC compression (DXT)
        int maxMipLevels = 12;             // Maximum mipmap levels
        bool generateNormalMap = false;    // Generate from height map
        bool sRGB = true;                  // sRGB color space
        int maxWidth = 4096;               // Maximum texture width
        int maxHeight = 4096;              // Maximum texture height
        bool flipVertical = false;         // Flip texture vertically
        std::string targetFormat = "png";  // Output format (png, jpg, dds, ktx2)
        float compressionQuality = 0.95f;  // Compression quality (0.0-1.0)
    };

    /**
     * @brief Conversion options for meshes
     */
    struct MeshConversionOptions {
        bool optimize = true;              // Optimize vertex data
        bool mergeSubmeshes = false;       // Merge submeshes where possible
        bool removeDuplicates = true;      // Remove duplicate vertices
        bool generateNormals = false;      // Generate missing normals
        bool generateTangents = true;      // Generate tangent/bitangent
        int quantizationBits = 16;         // Vertex position quantization
        bool stripUnusedData = true;       // Remove unused vertex attributes
        std::string targetFormat = "glb";  // Output format (glb, gltf, fbx)
    };

    /**
     * @brief Conversion options for materials
     */
    struct MaterialConversionOptions {
        bool optimizeTextures = true;      // Optimize referenced textures
        bool bakeTextures = false;         // Bake material to texture
        std::string targetFormat = "json"; // Output format (json, yaml)
    };

    /**
     * @brief Conversion result with metadata
     */
    struct ConversionResult {
        bool success = false;
        std::string outputPath;
        std::string errorMessage;
        size_t inputSize = 0;
        size_t outputSize = 0;
        float compressionRatio = 1.0f;
        double conversionTimeMs = 0.0;
        
        // Quality metrics
        bool qualityReduced = false;
        std::string qualityWarning;
    };

    /**
     * @brief Convert texture to optimized format
     * @param sourcePath Input texture file
     * @param outputPath Output texture file
     * @param options Conversion options
     * @return Conversion result
     */
    static ConversionResult ConvertTexture(
        const std::string& sourcePath,
        const std::string& outputPath,
        const TextureConversionOptions& options = TextureConversionOptions());

    /**
     * @brief Convert mesh/model to optimized format
     * @param sourcePath Input model file
     * @param outputPath Output model file
     * @param options Conversion options
     * @return Conversion result
     */
    static ConversionResult ConvertMesh(
        const std::string& sourcePath,
        const std::string& outputPath,
        const MeshConversionOptions& options = MeshConversionOptions());

    /**
     * @brief Convert material to optimized format
     * @param sourcePath Input material file
     * @param outputPath Output material file
     * @param options Conversion options
     * @return Conversion result
     */
    static ConversionResult ConvertMaterial(
        const std::string& sourcePath,
        const std::string& outputPath,
        const MaterialConversionOptions& options = MaterialConversionOptions());

    /**
     * @brief Detect asset type from file extension
     * @param filepath Asset file path
     * @return Asset type string (texture, mesh, material, shader, etc.)
     */
    static std::string DetectAssetType(const std::string& filepath);

    /**
     * @brief Check if asset type is supported by converter
     * @param assetType Asset type to check
     * @return true if supported
     */
    static bool IsAssetTypeSupported(const std::string& assetType);

    /**
     * @brief Get list of supported input formats for asset type
     * @param assetType Asset type
     * @return Vector of supported extensions (e.g., {"png", "jpg", "tga"})
     */
    static std::vector<std::string> GetSupportedInputFormats(const std::string& assetType);

    /**
     * @brief Get list of supported output formats for asset type
     * @param assetType Asset type
     * @return Vector of supported extensions
     */
    static std::vector<std::string> GetSupportedOutputFormats(const std::string& assetType);

    /**
     * @brief Optimize texture by resizing and mipmapping
     * @param texturePath Path to texture
     * @param maxWidth Maximum width
     * @param maxHeight Maximum height
     * @return true if successful
     */
    static bool OptimizeTexture(const std::string& texturePath, int maxWidth, int maxHeight);

    /**
     * @brief Generate texture mipmaps
     * @param texturePath Path to texture
     * @param maxLevels Maximum mipmap levels
     * @return true if successful
     */
    static bool GenerateMipmaps(const std::string& texturePath, int maxLevels = 12);

    /**
     * @brief Validate asset file integrity
     * @param filepath Path to asset file
     * @return true if file is valid and uncorrupted
     */
    static bool ValidateAsset(const std::string& filepath);

    /**
     * @brief Get estimated memory size when loaded
     * @param filepath Asset file path
     * @return Estimated memory in bytes
     */
    static size_t EstimateMemorySize(const std::string& filepath);

    /**
     * @brief Enable progress reporting for conversions
     * @param callback Function called with (progress 0.0-1.0, description)
     */
    static void SetProgressCallback(std::function<void(float, const std::string&)> callback);

private:
    static std::function<void(float, const std::string&)> m_ProgressCallback;
};
