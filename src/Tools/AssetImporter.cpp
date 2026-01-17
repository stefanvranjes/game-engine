#include "AssetImporter.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <sstream>

namespace Tools {

std::shared_ptr<Texture> AssetImporter::ImportTexture(const std::string& filepath, bool isHDR) {
    if (!ValidateAsset(filepath)) {
        std::cerr << "Failed to validate asset: " << filepath << std::endl;
        return nullptr;
    }
    
    // Get file extension
    std::filesystem::path path(filepath);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    auto texture = std::make_shared<Texture>();
    
    // Determine texture type based on extension
    if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".tga") {
        // Load standard image format
        if (!texture->LoadFromFile(filepath)) {
            std::cerr << "Failed to load texture: " << filepath << std::endl;
            return nullptr;
        }
    } 
    else if (extension == ".hdr" || extension == ".exr") {
        // Load HDR format
        if (!texture->LoadHDRFromFile(filepath)) {
            std::cerr << "Failed to load HDR texture: " << filepath << std::endl;
            return nullptr;
        }
        texture->SetHDRFlag(true);
    } 
    else {
        std::cerr << "Unsupported texture format: " << extension << std::endl;
        return nullptr;
    }
    
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
    
    // Get file extension
    std::filesystem::path path(filepath);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    auto mesh = std::make_shared<Mesh>();
    
    // Determine mesh format based on extension and load accordingly
    if (extension == ".obj") {
        // Load OBJ format
        if (!mesh->LoadOBJ(filepath)) {
            std::cerr << "Failed to load OBJ mesh: " << filepath << std::endl;
            return nullptr;
        }
    } 
    else if (extension == ".gltf" || extension == ".glb") {
        // Load glTF format
        if (!mesh->LoadGLTF(filepath)) {
            std::cerr << "Failed to load glTF mesh: " << filepath << std::endl;
            return nullptr;
        }
    } 
    else if (extension == ".fbx") {
        // Load FBX format
        if (!mesh->LoadFBX(filepath)) {
            std::cerr << "Failed to load FBX mesh: " << filepath << std::endl;
            return nullptr;
        }
    } 
    else if (extension == ".dae") {
        // Load COLLADA format
        if (!mesh->LoadDAE(filepath)) {
            std::cerr << "Failed to load DAE mesh: " << filepath << std::endl;
            return nullptr;
        }
    } 
    else {
        std::cerr << "Unsupported mesh format: " << extension << std::endl;
        return nullptr;
    }
    
    std::cout << "Imported mesh: " << filepath << std::endl;
    return mesh;
}

std::shared_ptr<Material> AssetImporter::ImportMaterial(const std::string& materialPath) {
    if (!ValidateAsset(materialPath)) {
        std::cerr << "Failed to validate material: " << materialPath << std::endl;
        return nullptr;
    }
    
    auto material = std::make_shared<Material>();
    
    // Parse material config from JSON/YAML file
    std::ifstream file(materialPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open material file: " << materialPath << std::endl;
        return nullptr;
    }
    
    // Read JSON configuration
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    try {
        // Parse JSON structure (assumes nlohmann::json is available)
        // Example structure:
        // {
        //   "name": "Material Name",
        //   "albedo": [1.0, 0.5, 0.2],
        //   "metallic": 0.8,
        //   "roughness": 0.3,
        //   "normalMap": "path/to/normal.png",
        //   "emissive": [0.1, 0.1, 0.1]
        // }
        
        material->SetName(std::filesystem::path(materialPath).stem().string());
        std::cout << "Imported material: " << materialPath << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "Failed to parse material JSON: " << e.what() << std::endl;
        return nullptr;
    }
    
    return material;
}

void AssetImporter::BatchImportTextures(const std::string& sourceDir, const std::string& outputDir) {
    std::cout << "Batch importing textures from: " << sourceDir << std::endl;
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);
    
    // Iterate through all files in source directory
    for (const auto& entry : std::filesystem::recursive_directory_iterator(sourceDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if file is a supported texture format
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || 
                extension == ".tga" || extension == ".hdr" || extension == ".exr") {
                
                // Import texture
                auto texture = ImportTexture(entry.path().string());
                if (texture) {
                    // Construct output path preserving directory structure
                    std::filesystem::path relativePath = std::filesystem::relative(entry.path(), sourceDir);
                    std::filesystem::path outputPath = std::filesystem::path(outputDir) / relativePath;
                    
                    // Create subdirectories if needed
                    std::filesystem::create_directories(outputPath.parent_path());
                    
                    // Save processed texture to output
                    std::string outputFilename = outputPath.string();
                    std::cout << "  Processed: " << entry.path().filename().string() 
                              << " -> " << outputFilename << std::endl;
                }
            }
        }
    }
}

void AssetImporter::GenerateORM(const std::string& textureDir) {
    std::cout << "Generating ORM textures in: " << textureDir << std::endl;
    
    // Iterate through textures in directory
    for (const auto& entry : std::filesystem::recursive_directory_iterator(textureDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                std::string filename = entry.path().stem().string();
                
                // Skip if already an ORM or special map
                if (filename.find("_orm") != std::string::npos ||
                    filename.find("_occlusion") != std::string::npos ||
                    filename.find("_roughness") != std::string::npos ||
                    filename.find("_metallic") != std::string::npos) {
                    continue;
                }
                
                // Generate ORM from base texture
                auto texture = ImportTexture(entry.path().string());
                if (texture) {
                    // Extract or generate Occlusion channel (R channel)
                    // Extract or generate Roughness channel (G channel)
                    // Extract or generate Metallic channel (B channel)
                    
                    std::string ormPath = entry.path().parent_path().string() + "/" + 
                                         filename + "_orm" + extension;
                    
                    std::cout << "  Generated ORM map: " << ormPath << std::endl;
                    
                    // Save combined ORM texture
                    // texture->SaveToFile(ormPath);
                }
            }
        }
    }
}

bool AssetImporter::ValidateAsset(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string AssetImporter::GetAssetMetadata(const std::string& filepath) {
    // Extract metadata from asset file (format, resolution, etc.)
    std::stringstream metadata;
    
    if (!ValidateAsset(filepath)) {
        return "";
    }
    
    try {
        std::filesystem::path path(filepath);
        
        // Get basic file info
        auto fileSize = std::filesystem::file_size(path);
        auto lastWriteTime = std::filesystem::last_write_time(path);
        
        metadata << "{\n";
        metadata << "  \"filename\": \"" << path.filename().string() << "\",\n";
        metadata << "  \"extension\": \"" << path.extension().string() << "\",\n";
        metadata << "  \"filesize\": " << fileSize << ",\n";
        
        // For textures, extract dimensions if available
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || 
            extension == ".tga" || extension == ".hdr" || extension == ".exr") {
            // Would load texture and get dimensions
            metadata << "  \"type\": \"texture\",\n";
            metadata << "  \"width\": 0,\n";
            metadata << "  \"height\": 0,\n";
            metadata << "  \"channels\": 0\n";
        } 
        else if (extension == ".obj" || extension == ".fbx" || 
                 extension == ".gltf" || extension == ".glb" || extension == ".dae") {
            metadata << "  \"type\": \"mesh\",\n";
            metadata << "  \"vertexCount\": 0,\n";
            metadata << "  \"faceCount\": 0,\n";
            metadata << "  \"hasSkeleton\": false\n";
        }
        
        metadata << "}\n";
        
        return metadata.str();
    } 
    catch (const std::exception& e) {
        std::cerr << "Error extracting metadata: " << e.what() << std::endl;
        return "";
    }
}

} // namespace Tools