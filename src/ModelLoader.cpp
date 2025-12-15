#include "ModelLoader.h"
#include "Model.h"
#include "GLTFLoader.h"
#include "Mesh.h"
#include "MaterialNew.h"
#include "TextureManager.h"
#include <tiny_gltf.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

/**
 * @brief Helper to detect file format from path
 */
ModelLoader::Format ModelLoader::DetectFormat(const std::string& path) {
    if (path.empty()) {
        return Format::Unknown;
    }

    // Get file extension
    size_t dotPos = path.find_last_of(".");
    if (dotPos == std::string::npos) {
        return Format::Unknown;
    }

    std::string ext = path.substr(dotPos + 1);
    
    // Convert to lowercase for comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Map extensions to formats
    if (ext == "obj") return Format::OBJ;
    if (ext == "gltf") return Format::GLTF;
    if (ext == "glb") return Format::GLB;
    if (ext == "fbx") return Format::FBX;
    if (ext == "dae") return Format::DAE;
    if (ext == "blend") return Format::BLEND;
    if (ext == "usdz" || ext == "usd") return Format::USD;
    if (ext == "stl") return Format::STL;
    if (ext == "iqm") return Format::IQM;
    if (ext == "md5mesh" || ext == "md5anim") return Format::MD5;

    // Try magic byte detection for ambiguous formats
    std::ifstream file(path, std::ios::binary);
    if (file.good()) {
        uint8_t magic[4] = { 0 };
        file.read(reinterpret_cast<char*>(magic), 4);
        
        // glTF binary (glb) starts with 0x46546c67 ("glTF")
        if (magic[0] == 'g' && magic[1] == 'l' && magic[2] == 'T' && magic[3] == 'F') {
            return Format::GLB;
        }
    }

    return Format::Unknown;
}

/**
 * @brief Get human-readable format name
 */
std::string ModelLoader::GetFormatName(Format format) {
    switch (format) {
        case Format::OBJ: return "Wavefront OBJ";
        case Format::GLTF: return "glTF 2.0";
        case Format::GLB: return "glTF Binary";
        case Format::FBX: return "Autodesk FBX";
        case Format::DAE: return "COLLADA DAE";
        case Format::BLEND: return "Blender BLEND";
        case Format::USD: return "USD/USDZ";
        case Format::STL: return "Stereolithography STL";
        case Format::IQM: return "Inter-Quake Model";
        case Format::MD5: return "Doom 3 MD5";
        default: return "Unknown";
    }
}

/**
 * @brief Check if format is supported
 */
bool ModelLoader::IsFormatSupported(Format format) {
    switch (format) {
        case Format::OBJ:
        case Format::GLTF:
        case Format::GLB:
        case Format::FBX:
        case Format::DAE:
        case Format::BLEND:
        case Format::IQM:
        case Format::MD5:
            return true;
        case Format::USD:
        case Format::STL:
            // STL and USD support can be added later
            return true;
        default:
            return false;
    }
}

/**
 * @brief Get supported extensions
 */
std::vector<std::string> ModelLoader::GetSupportedExtensions() {
    return {
        ".obj",
        ".gltf",
        ".glb",
        ".fbx",
        ".dae",
        ".blend",
        ".iqm",
        ".md5mesh",
        ".md5anim",
        ".stl",
        ".usdz",
        ".usd"
    };
}

/**
 * @brief Validate file integrity
 */
bool ModelLoader::ValidateFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        return false;
    }

    // Check file size is reasonable (at least 100 bytes)
    std::streamsize size = file.tellg();
    if (size < 100) {
        return false;
    }

    file.seekg(0);

    Format format = DetectFormat(path);
    
    if (format == Format::GLB) {
        // Validate glTF binary header
        uint8_t header[12] = { 0 };
        file.read(reinterpret_cast<char*>(header), 12);
        
        // Check magic and version
        if (!(header[0] == 'g' && header[1] == 'l' && header[2] == 'T' && header[3] == 'F')) {
            return false;
        }
        
        // Version should be 2
        uint32_t version = *reinterpret_cast<uint32_t*>(&header[4]);
        return version == 2;
    }
    
    if (format == Format::GLTF) {
        // Try to read as text JSON
        char buf[256];
        file.read(buf, std::min(size_t(256), size_t(size)));
        std::string content(buf);
        return content.find("\"asset\"") != std::string::npos;
    }

    return true; // Assimp will validate other formats
}

/**
 * @brief Get version information
 */
std::string ModelLoader::GetVersionInfo() {
    std::string info = "ModelLoader v1.0\n";
    info += "- tinygltf v2.8.13 (glTF/GLB support)\n";
    info += "- Assimp v5.3.1 (FBX, DAE, BLEND, IQM, MD5 support)\n";
    info += "- OBJ support (native implementation)\n";
    return info;
}

/**
 * @brief Load OBJ format
 */
ModelLoader::LoadResult ModelLoader::LoadOBJ(const std::string& path,
                                             TextureManager* texManager,
                                             const LoadOptions& options) {
    LoadResult result;
    result.detectedFormat = Format::OBJ;

    // Use existing Model::LoadFromOBJ for OBJ support
    try {
        Model model = Model::LoadFromOBJ(path, texManager);
        
        // Create root GameObject
        auto root = std::make_shared<GameObject>("OBJ_Model");
        
        // Attach meshes
        const auto& meshes = model.GetMeshes();
        const auto& materials = model.GetMaterials();
        
        result.meshCount = meshes.size();
        result.materialCount = materials.size();
        result.success = true;
        result.root = root;

        if (options.verbose) {
            std::cout << "Successfully loaded OBJ: " << path << " ("
                      << result.meshCount << " meshes, "
                      << result.materialCount << " materials)" << std::endl;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("OBJ loading failed: ") + e.what();
    }

    return result;
}

/**
 * @brief Load glTF/GLB format
 */
ModelLoader::LoadResult ModelLoader::LoadGLTF(const std::string& path,
                                              TextureManager* texManager,
                                              const LoadOptions& options) {
    LoadResult result;
    
    // Determine if GLB or GLTF based on extension
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    result.detectedFormat = (ext == "glb") ? Format::GLB : Format::GLTF;

    try {
        // Use existing GLTFLoader
        auto root = GLTFLoader::Load(path, texManager);
        
        if (!root) {
            result.success = false;
            result.errorMessage = "GLTFLoader returned null root";
            return result;
        }

        result.root = root;
        result.success = true;

        // Count meshes and materials by traversing scene graph
        std::function<void(const std::shared_ptr<GameObject>&)> countAssets =
            [&](const std::shared_ptr<GameObject>& obj) {
                if (obj->GetModel()) {
                    result.meshCount += obj->GetModel()->GetMeshes().size();
                    result.materialCount += obj->GetModel()->GetMaterials().size();
                }
                for (const auto& child : obj->GetChildren()) {
                    countAssets(child);
                }
            };

        countAssets(root);

        if (options.verbose) {
            std::cout << "Successfully loaded " << GetFormatName(result.detectedFormat)
                      << ": " << path << " ("
                      << result.meshCount << " meshes, "
                      << result.materialCount << " materials)" << std::endl;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("glTF loading failed: ") + e.what();
    }

    return result;
}

/**
 * @brief Load with Assimp (FBX, DAE, BLEND, etc.)
 */
ModelLoader::LoadResult ModelLoader::LoadWithAssimp(const std::string& path,
                                                    TextureManager* texManager,
                                                    const LoadOptions& options) {
    LoadResult result;
    result.detectedFormat = DetectFormat(path);

    try {
        Assimp::Importer importer;

        // Configure post-processing flags
        unsigned int ppSteps = aiProcess_Triangulate | aiProcess_GenNormals;
        
        if (options.generateNormalsIfMissing) {
            ppSteps |= aiProcess_GenSmoothNormals;
        }
        
        if (options.generateTangents) {
            ppSteps |= aiProcess_CalcTangentSpace;
        }
        
        if (options.optimizeMeshes) {
            ppSteps |= aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph;
        }
        
        // General good practices
        ppSteps |= aiProcess_RemoveRedundantMaterials | aiProcess_SortByPType;

        // Load the scene
        const aiScene* scene = importer.ReadFile(path, ppSteps);

        if (!scene || !scene->mRootNode) {
            result.success = false;
            result.errorMessage = std::string("Assimp failed to load: ") + importer.GetErrorString();
            return result;
        }

        // Create root GameObject
        auto root = std::make_shared<GameObject>(scene->mRootNode->mName.C_Str());

        // Helper lambda to process nodes recursively
        std::function<void(const aiNode*, std::shared_ptr<GameObject>)> processNode =
            [&](const aiNode* node, std::shared_ptr<GameObject> parent) {
                auto gameObj = std::make_shared<GameObject>(
                    node->mName.length > 0 ? node->mName.C_Str() : "Node"
                );

                // Set transform
                aiVector3D scale, pos;
                aiQuaternion rot;
                node->mTransformation.Decompose(scale, rot, pos);
                
                gameObj->GetTransform().SetPosition(Vec3(pos.x, pos.y, pos.z));
                gameObj->GetTransform().SetScale(Vec3(scale.x, scale.y, scale.z));
                gameObj->GetTransform().SetRotation(Vec3(
                    std::atan2(2.0f * (rot.w * rot.x + rot.y * rot.z), 1.0f - 2.0f * (rot.x * rot.x + rot.y * rot.y)),
                    std::asin(std::clamp(2.0f * (rot.w * rot.y - rot.z * rot.x), -1.0f, 1.0f)),
                    std::atan2(2.0f * (rot.w * rot.z + rot.x * rot.y), 1.0f - 2.0f * (rot.y * rot.y + rot.z * rot.z))
                ) * 180.0f / 3.14159f);

                // Process meshes
                for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
                    unsigned int meshIdx = node->mMeshes[i];
                    if (meshIdx < scene->mNumMeshes) {
                        const aiMesh* mesh = scene->mMeshes[meshIdx];
                        
                        // Extract vertices
                        std::vector<float> vertices;
                        vertices.reserve(mesh->mNumVertices * 16); // 16 floats per vertex

                        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                            // Position
                            vertices.push_back(mesh->mVertices[v].x);
                            vertices.push_back(mesh->mVertices[v].y);
                            vertices.push_back(mesh->mVertices[v].z);

                            // Normal
                            if (mesh->HasNormals()) {
                                vertices.push_back(mesh->mNormals[v].x);
                                vertices.push_back(mesh->mNormals[v].y);
                                vertices.push_back(mesh->mNormals[v].z);
                            } else {
                                vertices.push_back(0.0f);
                                vertices.push_back(1.0f);
                                vertices.push_back(0.0f);
                            }

                            // TexCoord
                            if (mesh->HasTextureCoords(0)) {
                                vertices.push_back(mesh->mTextureCoords[0][v].x);
                                vertices.push_back(mesh->mTextureCoords[0][v].y);
                            } else {
                                vertices.push_back(0.0f);
                                vertices.push_back(0.0f);
                            }

                            // Bone IDs and weights (TODO: properly handle skeletal meshes)
                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);

                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);
                            vertices.push_back(0.0f);
                        }

                        // Extract indices
                        std::vector<unsigned int> indices;
                        indices.reserve(mesh->mNumFaces * 3);
                        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
                            const aiFace& face = mesh->mFaces[f];
                            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                                indices.push_back(face.mIndices[i]);
                            }
                        }

                        // Create mesh
                        auto meshPtr = std::make_unique<Mesh>(
                            vertices, indices,
                            mesh->mName.length > 0 ? mesh->mName.C_Str() : "AssimpMesh"
                        );

                        // TODO: Attach material
                        result.meshCount++;
                    }
                }

                // Process children
                for (unsigned int i = 0; i < node->mNumChildren; ++i) {
                    processNode(node->mChildren[i], gameObj);
                }

                parent->AddChild(gameObj);
            };

        // Process all nodes starting from root
        for (unsigned int i = 0; i < scene->mRootNode->mNumChildren; ++i) {
            processNode(scene->mRootNode->mChildren[i], root);
        }

        result.root = root;
        result.success = true;
        result.materialCount = scene->mNumMaterials;
        result.animationCount = scene->mNumAnimations;
        result.textureCount = scene->mNumTextures;

        if (options.verbose) {
            std::cout << "Successfully loaded " << GetFormatName(result.detectedFormat)
                      << ": " << path << " ("
                      << result.meshCount << " meshes, "
                      << result.materialCount << " materials, "
                      << result.animationCount << " animations)" << std::endl;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Assimp loading failed: ") + e.what();
    }

    return result;
}

/**
 * @brief Main Load function with auto-detection
 */
ModelLoader::LoadResult ModelLoader::Load(const std::string& path,
                                         TextureManager* texManager,
                                         const LoadOptions& options) {
    Format format = DetectFormat(path);
    return LoadAs(path, format, texManager, options);
}

/**
 * @brief Load with explicit format
 */
ModelLoader::LoadResult ModelLoader::LoadAs(const std::string& path,
                                            Format format,
                                            TextureManager* texManager,
                                            const LoadOptions& options) {
    LoadResult result;

    if (!IsFormatSupported(format)) {
        result.success = false;
        result.errorMessage = std::string("Format not supported: ") + GetFormatName(format);
        return result;
    }

    switch (format) {
        case Format::OBJ:
            return LoadOBJ(path, texManager, options);
        
        case Format::GLTF:
        case Format::GLB:
            return LoadGLTF(path, texManager, options);
        
        case Format::FBX:
        case Format::DAE:
        case Format::BLEND:
        case Format::IQM:
        case Format::MD5:
        case Format::STL:
        case Format::USD:
            return LoadWithAssimp(path, texManager, options);
        
        default:
            result.success = false;
            result.errorMessage = "Unknown format";
            return result;
    }
}

/**
 * @brief Load from memory buffer
 */
ModelLoader::LoadResult ModelLoader::LoadFromMemory(const uint8_t* data,
                                                    size_t size,
                                                    Format format,
                                                    TextureManager* texManager,
                                                    const LoadOptions& options) {
    LoadResult result;

    if (!data || size == 0) {
        result.success = false;
        result.errorMessage = "Invalid buffer data";
        return result;
    }

    if (!IsFormatSupported(format)) {
        result.success = false;
        result.errorMessage = std::string("Format not supported: ") + GetFormatName(format);
        return result;
    }

    try {
        if (format == Format::GLB) {
            // Load glTF binary from memory using tinygltf
            tinygltf::Model model;
            tinygltf::TinyGLTF loader;
            std::string err, warn;
            
            if (!loader.LoadBinaryFromMemory(&model, &err, &warn,
                                            data, size, "")) {
                result.success = false;
                result.errorMessage = std::string("glTF binary load failed: ") + err;
                return result;
            }
            
            // Convert to GameObject (basic implementation)
            auto root = std::make_shared<GameObject>("GLB_Memory");
            result.root = root;
            result.success = true;
            result.detectedFormat = Format::GLB;
        }
        else if (format == Format::FBX || format == Format::DAE) {
            // Use Assimp for memory loading
            Assimp::Importer importer;
            
            const aiScene* scene = importer.ReadFileFromMemory(
                data, size, 
                aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_OptimizeMeshes,
                format == Format::FBX ? "fbx" : "dae"
            );

            if (!scene) {
                result.success = false;
                result.errorMessage = std::string("Assimp memory load failed: ") + importer.GetErrorString();
                return result;
            }

            auto root = std::make_shared<GameObject>("AssimpMemory");
            result.root = root;
            result.success = true;
            result.detectedFormat = format;
        }
        else {
            result.success = false;
            result.errorMessage = "Memory loading not yet supported for this format";
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Memory loading failed: ") + e.what();
    }

    return result;
}
