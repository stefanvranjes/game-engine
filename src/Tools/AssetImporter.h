#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../Rendering/Texture.h"
#include "../Rendering/Mesh.h"
#include "../Rendering/Material.h"

namespace Tools {

class AssetImporter {
public:
    // Texture import with format conversion
    static std::shared_ptr<Texture> ImportTexture(const std::string& filepath, bool isHDR = false);
    static std::shared_ptr<Texture> ImportTextureAtlas(const std::string& filepath, int rows, int cols);
    
    // Mesh/Model import
    static std::shared_ptr<Mesh> ImportMesh(const std::string& filepath);
    
    // Material import
    static std::shared_ptr<Material> ImportMaterial(const std::string& materialPath);
    
    // Batch processing
    static void BatchImportTextures(const std::string& sourceDir, const std::string& outputDir);
    static void GenerateORM(const std::string& textureDir); // Occlusion, Roughness, Metallic

private:
    static bool ValidateAsset(const std::string& filepath);
    static std::string GetAssetMetadata(const std::string& filepath);
};

} // namespace Tools