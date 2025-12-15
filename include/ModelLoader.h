#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "GameObject.h"
#include "TextureManager.h"

/**
 * @brief Unified model loader supporting multiple formats: OBJ, glTF/GLB, and Assimp formats.
 * 
 * Handles automatic format detection and provides consistent interface for loading
 * 3D models with support for:
 * - Skeletal animation and rigging
 * - PBR materials and textures
 * - Multi-material meshes
 * - Custom extensions (glTF)
 * - Error handling and validation
 * - Async loading (with future support)
 */
class ModelLoader {
public:
    /**
     * @brief Supported file formats
     */
    enum class Format {
        Unknown = 0,
        OBJ,           // Wavefront OBJ (.obj)
        GLTF,          // glTF 2.0 text format (.gltf)
        GLB,           // glTF binary format (.glb)
        FBX,           // Autodesk FBX (Assimp)
        DAE,           // COLLADA (.dae, Assimp)
        BLEND,         // Blender (Assimp)
        USD,           // USD format (Assimp)
        STL,           // STL format (Assimp)
        IQM,           // Inter-Quake Model (Assimp)
        MD5,           // Doom 3 MD5 (Assimp)
    };

    /**
     * @brief Loading configuration options
     */
    struct LoadOptions {
        bool loadAnimations = true;
        bool loadMaterials = true;
        bool loadTextures = true;
        bool generateNormalsIfMissing = true;
        bool generateTangents = false;
        bool optimizeMeshes = false;
        bool mergeVertexBones = true;
        float meshOptimizationThreshold = 0.001f;
        bool verbose = false;
    };

    /**
     * @brief Loading result with status and error information
     */
    struct LoadResult {
        std::shared_ptr<GameObject> root;
        bool success = false;
        std::string errorMessage;
        std::vector<std::string> warnings;
        Format detectedFormat = Format::Unknown;
        int meshCount = 0;
        int materialCount = 0;
        int animationCount = 0;
        int textureCount = 0;

        explicit operator bool() const { return success; }
    };

    /**
     * @brief Load a 3D model from file with automatic format detection
     * 
     * @param path File path to the model
     * @param texManager Texture manager for loading material textures
     * @param options Loading configuration options
     * @return LoadResult with root GameObject and status information
     */
    static LoadResult Load(const std::string& path, 
                          TextureManager* texManager = nullptr, 
                          const LoadOptions& options = LoadOptions());

    /**
     * @brief Load a 3D model with explicit format specification
     * 
     * @param path File path to the model
     * @param format Explicit format to use (avoids auto-detection)
     * @param texManager Texture manager for loading material textures
     * @param options Loading configuration options
     * @return LoadResult with root GameObject and status information
     */
    static LoadResult LoadAs(const std::string& path,
                            Format format,
                            TextureManager* texManager = nullptr,
                            const LoadOptions& options = LoadOptions());

    /**
     * @brief Load model from memory buffer with explicit format
     * 
     * @param data Raw buffer data
     * @param size Size of buffer in bytes
     * @param format Format of the data
     * @param texManager Texture manager for loading material textures
     * @param options Loading configuration options
     * @return LoadResult with root GameObject and status information
     */
    static LoadResult LoadFromMemory(const uint8_t* data,
                                    size_t size,
                                    Format format,
                                    TextureManager* texManager = nullptr,
                                    const LoadOptions& options = LoadOptions());

    /**
     * @brief Detect file format from path/magic bytes
     * 
     * @param path File path to analyze
     * @return Detected format, or Format::Unknown if not recognized
     */
    static Format DetectFormat(const std::string& path);

    /**
     * @brief Get human-readable format name
     * 
     * @param format Format to describe
     * @return Format name string
     */
    static std::string GetFormatName(Format format);

    /**
     * @brief Check if format is supported by this loader
     * 
     * @param format Format to check
     * @return true if supported
     */
    static bool IsFormatSupported(Format format);

    /**
     * @brief Get list of supported file extensions
     * 
     * @return Vector of supported extensions (e.g., ".obj", ".glb")
     */
    static std::vector<std::string> GetSupportedExtensions();

    /**
     * @brief Validate a model file for integrity
     * 
     * @param path Path to model file
     * @return true if file appears valid for its format
     */
    static bool ValidateFile(const std::string& path);

    /**
     * @brief Get version information for loaders
     * 
     * @return Version string with loader details
     */
    static std::string GetVersionInfo();

private:
    // Private implementation details
    ModelLoader() = default;
    ~ModelLoader() = default;
    
    // Format-specific loaders
    static LoadResult LoadOBJ(const std::string& path, 
                             TextureManager* texManager,
                             const LoadOptions& options);
    
    static LoadResult LoadGLTF(const std::string& path,
                              TextureManager* texManager,
                              const LoadOptions& options);
    
    static LoadResult LoadWithAssimp(const std::string& path,
                                     TextureManager* texManager,
                                     const LoadOptions& options);
};
