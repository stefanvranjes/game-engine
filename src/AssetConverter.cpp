#include "AssetConverter.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>

std::function<void(float, const std::string&)> AssetConverter::m_ProgressCallback;

AssetConverter::ConversionResult AssetConverter::ConvertTexture(
    const std::string& sourcePath,
    const std::string& outputPath,
    const TextureConversionOptions& options) {

    ConversionResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Check source exists
    if (!std::filesystem::exists(sourcePath)) {
        result.success = false;
        result.errorMessage = "Source file not found: " + sourcePath;
        return result;
    }

    try {
        result.inputSize = std::filesystem::file_size(sourcePath);

        // Create output directory
        std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

        // For now, implement basic texture validation and copy
        // In production, use external libraries like:
        // - STB Image for basic conversion
        // - libSquish for DXT compression
        // - ispc_texcomp for ASTC compression
        // - KTX2 tools for KTX2 format

        std::ifstream src(sourcePath, std::ios::binary);
        std::ofstream dst(outputPath, std::ios::binary);

        if (!src.is_open() || !dst.is_open()) {
            result.success = false;
            result.errorMessage = "Failed to open files for conversion";
            return result;
        }

        // Copy file content
        dst << src.rdbuf();
        src.close();
        dst.close();

        result.outputPath = outputPath;
        result.outputSize = std::filesystem::file_size(outputPath);
        result.compressionRatio = result.inputSize > 0 ? (float)result.outputSize / result.inputSize : 1.0f;
        result.success = true;

        // Warn if no actual compression happened
        if (options.compress && result.compressionRatio > 0.99f) {
            result.qualityReduced = false;
            result.qualityWarning = "Note: Compression was requested but file was copied as-is. "
                                    "Consider using external compression tools for production builds.";
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Texture conversion error: ") + e.what();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.conversionTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (m_ProgressCallback) {
        m_ProgressCallback(1.0f, "Converted texture: " + sourcePath);
    }

    return result;
}

AssetConverter::ConversionResult AssetConverter::ConvertMesh(
    const std::string& sourcePath,
    const std::string& outputPath,
    const MeshConversionOptions& options) {

    ConversionResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Check source exists
    if (!std::filesystem::exists(sourcePath)) {
        result.success = false;
        result.errorMessage = "Source file not found: " + sourcePath;
        return result;
    }

    try {
        result.inputSize = std::filesystem::file_size(sourcePath);

        // Create output directory
        std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

        // For now, implement basic mesh copy
        // In production, use:
        // - tinygltf for glTF/GLB conversion
        // - assimp for multi-format support
        // - meshoptimizer for mesh optimization

        std::ifstream src(sourcePath, std::ios::binary);
        std::ofstream dst(outputPath, std::ios::binary);

        if (!src.is_open() || !dst.is_open()) {
            result.success = false;
            result.errorMessage = "Failed to open files for conversion";
            return result;
        }

        // Copy file content
        dst << src.rdbuf();
        src.close();
        dst.close();

        result.outputPath = outputPath;
        result.outputSize = std::filesystem::file_size(outputPath);
        result.compressionRatio = result.inputSize > 0 ? (float)result.outputSize / result.inputSize : 1.0f;
        result.success = true;

        if (options.optimize && result.compressionRatio > 0.95f) {
            result.qualityWarning = "Note: Optimization was requested but mesh was copied as-is. "
                                    "Consider using meshoptimizer or assimp for production builds.";
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Mesh conversion error: ") + e.what();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.conversionTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    if (m_ProgressCallback) {
        m_ProgressCallback(1.0f, "Converted mesh: " + sourcePath);
    }

    return result;
}

AssetConverter::ConversionResult AssetConverter::ConvertMaterial(
    const std::string& sourcePath,
    const std::string& outputPath,
    const MaterialConversionOptions& options) {

    ConversionResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!std::filesystem::exists(sourcePath)) {
        result.success = false;
        result.errorMessage = "Source file not found: " + sourcePath;
        return result;
    }

    try {
        result.inputSize = std::filesystem::file_size(sourcePath);
        std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

        // Copy material definition
        std::ifstream src(sourcePath, std::ios::binary);
        std::ofstream dst(outputPath, std::ios::binary);

        if (!src.is_open() || !dst.is_open()) {
            result.success = false;
            result.errorMessage = "Failed to open files for conversion";
            return result;
        }

        dst << src.rdbuf();
        src.close();
        dst.close();

        result.outputPath = outputPath;
        result.outputSize = std::filesystem::file_size(outputPath);
        result.compressionRatio = 1.0f;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Material conversion error: ") + e.what();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.conversionTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    return result;
}

std::string AssetConverter::DetectAssetType(const std::string& filepath) {
    std::string ext = std::filesystem::path(filepath).extension().string();

    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Image formats
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" ||
        ext == ".bmp" || ext == ".hdr" || ext == ".exr" || ext == ".dds" || ext == ".ktx2") {
        return "texture";
    }

    // Model formats
    if (ext == ".obj" || ext == ".fbx" || ext == ".gltf" || ext == ".glb" ||
        ext == ".dae" || ext == ".usdz" || ext == ".blend") {
        return "model";
    }

    // Shader formats
    if (ext == ".glsl" || ext == ".vert" || ext == ".frag" || ext == ".geom" ||
        ext == ".comp" || ext == ".tese" || ext == ".tesc" || ext == ".spv") {
        return "shader";
    }

    // Material/Config formats
    if (ext == ".json" || ext == ".yaml" || ext == ".yml" || ext == ".mat") {
        return "material";
    }

    // Audio formats
    if (ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".ogg") {
        return "audio";
    }

    // Text/Scene formats
    if (ext == ".txt" || ext == ".scene" || ext == ".prefab") {
        return "scene";
    }

    return "unknown";
}

bool AssetConverter::IsAssetTypeSupported(const std::string& assetType) {
    return assetType == "texture" || assetType == "model" || assetType == "shader" ||
           assetType == "material" || assetType == "audio" || assetType == "scene";
}

std::vector<std::string> AssetConverter::GetSupportedInputFormats(const std::string& assetType) {
    if (assetType == "texture") {
        return {"png", "jpg", "jpeg", "tga", "bmp", "hdr", "exr", "dds", "ktx2"};
    } else if (assetType == "model") {
        return {"obj", "fbx", "gltf", "glb", "dae", "usdz", "blend"};
    } else if (assetType == "shader") {
        return {"glsl", "vert", "frag", "geom", "comp", "tese", "tesc", "spv"};
    } else if (assetType == "material") {
        return {"json", "yaml", "yml", "mat"};
    } else if (assetType == "audio") {
        return {"wav", "mp3", "flac", "ogg"};
    }
    return {};
}

std::vector<std::string> AssetConverter::GetSupportedOutputFormats(const std::string& assetType) {
    if (assetType == "texture") {
        return {"png", "dds", "ktx2", "jpg"};
    } else if (assetType == "model") {
        return {"glb", "gltf", "obj"};
    } else if (assetType == "shader") {
        return {"spv", "glsl"};
    } else if (assetType == "material") {
        return {"json", "yaml"};
    } else if (assetType == "audio") {
        return {"wav", "ogg"};
    }
    return {};
}

bool AssetConverter::OptimizeTexture(const std::string& texturePath, int maxWidth, int maxHeight) {
    // TODO: Implement texture optimization using external libraries
    // - Load with stb_image
    // - Resize if necessary
    // - Save with optimization
    return true;
}

bool AssetConverter::GenerateMipmaps(const std::string& texturePath, int maxLevels) {
    // TODO: Implement mipmap generation
    return true;
}

bool AssetConverter::ValidateAsset(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }

    size_t fileSize = std::filesystem::file_size(filepath);
    if (fileSize == 0) {
        return false;
    }

    // Try to open and read first few bytes
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    unsigned char header[4] = {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    file.close();

    // Basic magic number checks
    std::string ext = std::filesystem::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".png") {
        return header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47;
    } else if (ext == ".jpg" || ext == ".jpeg") {
        return header[0] == 0xFF && header[1] == 0xD8;
    } else if (ext == ".glb") {
        return header[0] == 'g' && header[1] == 'l' && header[2] == 'T' && header[3] == 'F';
    }

    return true;  // Default to valid for unknown formats
}

size_t AssetConverter::EstimateMemorySize(const std::string& filepath) {
    std::string assetType = DetectAssetType(filepath);

    if (assetType == "texture") {
        // Rough estimate: most textures are RGBA = 4 bytes per pixel
        // PNG/JPG files are compressed, so decompress memory = width*height*4
        // Plus mipmaps = ~1.33x
        size_t fileSize = std::filesystem::file_size(filepath);
        // Assume compression ratio of 4:1 for typical textures
        return fileSize * 4 * 1.33f;
    } else if (assetType == "model") {
        // Estimate based on file size
        size_t fileSize = std::filesystem::file_size(filepath);
        // GLB files are somewhat compressed, estimate 2-3x expansion
        return fileSize * 2.5f;
    }

    return std::filesystem::file_size(filepath);
}

void AssetConverter::SetProgressCallback(std::function<void(float, const std::string&)> callback) {
    m_ProgressCallback = callback;
}
