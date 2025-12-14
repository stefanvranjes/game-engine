#include "AssetImporter.h"
#include <iostream>
#include <filesystem>

namespace Tools {

std::shared_ptr<Texture> AssetImporter::ImportTexture(const std::string& filepath, bool isHDR) {
    if (!ValidateAsset(filepath)) {
        std::cerr << "Failed to validate asset: " << filepath << std::endl;
        return nullptr;
    }
    
    // TODO: Implement texture loading based on file extension
    // Support: PNG, JPG, TGA, HDR, EXR
    auto texture = std::make_shared<Texture>();
    
    std::cout << "Imported texture: " << filepath << std::endl;
    return texture;
}

std::shared_ptr<Texture> AssetImporter::ImportTextureAtlas(const std::string& filepath, int rows, int cols) {
    auto baseTexture = ImportTexture(filepath);
    if (!baseTexture) return nullptr;
    
    std::cout << "Created texture atlas: " << rows << "x" << cols << std::endl;
    return baseTexture;
}

std::shared_ptr<Mesh> AssetImporter::ImportMesh(const std::string& filepath) {
    if (!ValidateAsset(filepath)) {
        std::cerr << "Failed to validate mesh: " << filepath << std::endl;
        return nullptr;
    }
    
    // TODO: Implement mesh loading based on file extension
    // Support: FBX, OBJ, GLTF, DAE
    auto mesh = std::make_shared<Mesh>();
    
    std::cout << "Imported mesh: " << filepath << std::endl;
    return mesh;
}

std::shared_ptr<Material> AssetImporter::ImportMaterial(const std::string& materialPath) {
    if (!ValidateAsset(materialPath)) {
        std::cerr << "Failed to validate material: " << materialPath << std::endl;
        return nullptr;
    }
    
    // TODO: Load material from JSON/YAML config
    auto material = std::make_shared<Material>();
    
    std::cout << "Imported material: " << materialPath << std::endl;
    return material;
}

void AssetImporter::BatchImportTextures(const std::string& sourceDir, const std::string& outputDir) {
    std::cout << "Batch importing textures from: " << sourceDir << std::endl;
    
    // TODO: Iterate through directory and import all textures
    // Copy processed textures to outputDir
}

void AssetImporter::GenerateORM(const std::string& textureDir) {
    std::cout << "Generating ORM textures in: " << textureDir << std::endl;
    
    // TODO: Generate Occlusion, Roughness, Metallic maps from base textures
}

bool AssetImporter::ValidateAsset(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string AssetImporter::GetAssetMetadata(const std::string& filepath) {
    // TODO: Extract metadata from asset file
    return "";
}

} // namespace Tools