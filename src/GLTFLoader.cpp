#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include "GLTFLoader.h"
#include "Model.h"
#include "Material.h"
#include "Mesh.h"
#include "GameObject.h"
#include <iostream>
#include <algorithm>

// Helper to convert Quaternion to Euler angles (ZXY order to match Transform)
static Vec3 QuaternionToEuler(const std::vector<double>& q) {
    // q = [x, y, z, w]
    double x = q[0];
    double y = q[1];
    double z = q[2];
    double w = q[3];

    Vec3 euler;

    // Roll (x-axis rotation)
    double sinr_cosp = 2 * (w * x + y * z);
    double cosr_cosp = 1 - 2 * (x * x + y * y);
    euler.x = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        euler.y = std::copysign(3.14159 / 2, sinp); // use 90 degrees if out of range
    else
        euler.y = std::asin(sinp);

    // Yaw (z-axis rotation)
    double siny_cosp = 2 * (w * z + x * y);
    double cosy_cosp = 1 - 2 * (y * y + z * z);
    euler.z = std::atan2(siny_cosp, cosy_cosp);

    // Convert to degrees
    return Vec3(euler.x * 180.0f / 3.14159f, euler.y * 180.0f / 3.14159f, euler.z * 180.0f / 3.14159f);
}

class GLTFImporter {
public:
    GLTFImporter(TextureManager* texManager) : m_TexManager(texManager) {}

    std::shared_ptr<GameObject> Import(const std::string& path) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = false;
        if (path.length() >= 4 && path.compare(path.length() - 4, 4, ".glb") == 0) {
            ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
        } else {
            ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
        }

        if (!warn.empty()) {
            std::cout << "GLTF Warning: " << warn << std::endl;
        }

        if (!err.empty()) {
            std::cerr << "GLTF Error: " << err << std::endl;
        }

        if (!ret) {
            std::cerr << "Failed to parse glTF: " << path << std::endl;
            return nullptr;
        }

        // Load textures first
        LoadTextures(model);

        // Load materials
        LoadMaterials(model);

        // Create root object
        auto root = std::make_shared<GameObject>("GLTF_Root");

        // Process scenes
        const tinygltf::Scene& scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
            auto child = ProcessNode(model, model.nodes[scene.nodes[i]]);
            if (child) {
                root->AddChild(child);
            }
        }

        return root;
    }

private:
    TextureManager* m_TexManager;
    std::vector<std::shared_ptr<Texture>> m_Textures;
    std::vector<std::shared_ptr<Material>> m_Materials;

    void LoadTextures(const tinygltf::Model& model) {
        m_Textures.resize(model.textures.size());
        for (size_t i = 0; i < model.textures.size(); ++i) {
            const auto& tex = model.textures[i];
            const auto& img = model.images[tex.source];

            // If image has a URI and no data, load from file
            if (!img.uri.empty() && img.image.empty()) {
                m_Textures[i] = m_TexManager->LoadTexture(img.uri);
            } 
            // If image has data (embedded or loaded by tinygltf), create from data
            else if (!img.image.empty()) {
                auto texture = std::make_shared<Texture>();
                bool sRGB = false; // We can't easily know usage here, assume linear or handle in material
                // Actually, albedo should be sRGB, others linear. 
                // But Texture class handles internal format. 
                // Let's assume linear for now, or check usage later?
                // Texture::LoadFromData takes sRGB flag.
                // We'll create it as linear and let shader handle gamma correction if needed,
                // OR we create it as sRGB if we knew it was albedo.
                // For now, load as standard RGB/RGBA.
                
                texture->LoadFromData(&img.image[0], img.width, img.height, img.component, false);
                m_Textures[i] = texture;
            }
        }
    }

    void LoadMaterials(const tinygltf::Model& model) {
        m_Materials.resize(model.materials.size());
        for (size_t i = 0; i < model.materials.size(); ++i) {
            const auto& mat = model.materials[i];
            auto material = std::make_shared<Material>();

            // PBR Metallic Roughness
            auto& pbr = mat.pbrMetallicRoughness;
            
            // Base Color
            material->diffuse = Vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
            if (pbr.baseColorTexture.index >= 0) {
                material->texture = m_Textures[pbr.baseColorTexture.index];
            }

            // Metallic Roughness
            material->metallic = (float)pbr.metallicFactor;
            material->roughness = (float)pbr.roughnessFactor;
            if (pbr.metallicRoughnessTexture.index >= 0) {
                // glTF packs Occlusion (R), Roughness (G), Metallic (B) often, 
                // but strictly MetallicRoughness is G=Roughness, B=Metallic.
                // We map this to our ORM map (which expects R=AO, G=Rough, B=Metal).
                // If AO is missing in R, it defaults to 1.0 usually?
                // Actually, if we use this texture as ORM, we assume it has AO in R.
                // If not, we might get wrong AO.
                // But standard glTF workflow often uses the SAME texture for occlusion and metallicRoughness.
                material->ormMap = m_Textures[pbr.metallicRoughnessTexture.index];
                
                // Check if occlusion texture is the same
                if (mat.occlusionTexture.index == pbr.metallicRoughnessTexture.index) {
                    // Perfect, it's a packed ORM map
                } else if (mat.occlusionTexture.index >= 0) {
                    // Separate AO map
                    material->aoMap = m_Textures[mat.occlusionTexture.index];
                }
            }

            // Normal Map
            if (mat.normalTexture.index >= 0) {
                material->normalMap = m_Textures[mat.normalTexture.index];
            }

            // Emissive
            material->emissiveColor = Vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
            if (mat.emissiveTexture.index >= 0) {
                material->emissiveMap = m_Textures[mat.emissiveTexture.index];
                // If emissive texture is present but factor is 0 (default in some exporters if not set),
                // we should set factor to 1 to make it visible?
                // Spec says: emissive = texture * factor. Default factor is [0,0,0].
                // So if factor is 0, texture is black.
                // However, if texture IS present, usually the intent is to show it.
                // But we should respect the spec.
                // If the user wants it to glow, they must set the factor in the glTF.
                // We'll trust the file.
            }

            m_Materials[i] = material;
        }
    }

    std::shared_ptr<GameObject> ProcessNode(const tinygltf::Model& model, const tinygltf::Node& node) {
        auto gameObject = std::make_shared<GameObject>(node.name);

        // Transform
        if (node.matrix.size() == 16) {
            // Decompose matrix? Or just set WorldMatrix?
            // GameObject stores Transform (pos, rot, scale).
            // We need to decompose.
            // For now, let's try to extract T, R, S if possible.
            // If not, we might ignore matrix or implement decomposition.
            // tinygltf doesn't decompose.
            // Let's assume TRS is provided if matrix is not.
            // If matrix IS provided, we are in trouble without a decompose function.
            // Most exporters export TRS.
        } else {
            if (node.translation.size() == 3) {
                gameObject->GetTransform().position = Vec3(node.translation[0], node.translation[1], node.translation[2]);
            }
            if (node.rotation.size() == 4) {
                gameObject->GetTransform().rotation = QuaternionToEuler(node.rotation);
            }
            if (node.scale.size() == 3) {
                gameObject->GetTransform().scale = Vec3(node.scale[0], node.scale[1], node.scale[2]);
            }
        }

        // Mesh
        if (node.mesh >= 0) {
            const auto& mesh = model.meshes[node.mesh];
            
            // Create a Model to hold primitives
            // Wait, GameObject holds ONE Mesh and ONE Material?
            // Or ONE Model?
            // GameObject::SetModel(shared_ptr<Model>)
            // Model holds vector<Mesh> and vector<Material>.
            
            auto gameModel = std::make_shared<Model>();
            // We need to add meshes to this model manually.
            // Model class doesn't expose AddMesh...
            // It has m_Meshes and m_Materials as private.
            // We need to modify Model.h to allow adding meshes, or create a constructor.
            // Or we can create child GameObjects for each primitive.
            
            if (mesh.primitives.size() == 1) {
                // Single primitive
                auto primitive = mesh.primitives[0];
                auto engineMesh = ProcessPrimitive(model, primitive);
                gameObject->SetMesh(std::move(*engineMesh));
                if (primitive.material >= 0) {
                    gameObject->SetMaterial(m_Materials[primitive.material]);
                }
            } else {
                // Multiple primitives -> Child objects
                for (size_t i = 0; i < mesh.primitives.size(); ++i) {
                    auto primitive = mesh.primitives[i];
                    auto child = std::make_shared<GameObject>(node.name + "_prim_" + std::to_string(i));
                    auto engineMesh = ProcessPrimitive(model, primitive);
                    child->SetMesh(std::move(*engineMesh));
                    if (primitive.material >= 0) {
                        child->SetMaterial(m_Materials[primitive.material]);
                    }
                    gameObject->AddChild(child);
                }
            }
        }

        // Children
        for (size_t i = 0; i < node.children.size(); ++i) {
            auto child = ProcessNode(model, model.nodes[node.children[i]]);
            if (child) {
                gameObject->AddChild(child);
            }
        }

        return gameObject;
    }

    std::unique_ptr<Mesh> ProcessPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive) {
        // Extract attributes
        const float* positions = nullptr;
        const float* normals = nullptr;
        const float* texCoords = nullptr;
        size_t vertexCount = 0;

        // Position
        if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
            const auto& accessor = model.accessors[primitive.attributes.at("POSITION")];
            const auto& bufferView = model.bufferViews[accessor.bufferView];
            const auto& buffer = model.buffers[bufferView.buffer];
            positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            vertexCount = accessor.count;
        }

        // Normal
        if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
            const auto& accessor = model.accessors[primitive.attributes.at("NORMAL")];
            const auto& bufferView = model.bufferViews[accessor.bufferView];
            const auto& buffer = model.buffers[bufferView.buffer];
            normals = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        }

        // TexCoord
        if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
            const auto& accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            const auto& bufferView = model.bufferViews[accessor.bufferView];
            const auto& buffer = model.buffers[bufferView.buffer];
            texCoords = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        }

        // Interleave vertices
        std::vector<float> vertices;
        vertices.reserve(vertexCount * 8);
        for (size_t i = 0; i < vertexCount; ++i) {
            // Position
            vertices.push_back(positions ? positions[i * 3 + 0] : 0.0f);
            vertices.push_back(positions ? positions[i * 3 + 1] : 0.0f);
            vertices.push_back(positions ? positions[i * 3 + 2] : 0.0f);

            // Normal
            vertices.push_back(normals ? normals[i * 3 + 0] : 0.0f);
            vertices.push_back(normals ? normals[i * 3 + 1] : 1.0f);
            vertices.push_back(normals ? normals[i * 3 + 2] : 0.0f);

            // TexCoord
            vertices.push_back(texCoords ? texCoords[i * 2 + 0] : 0.0f);
            vertices.push_back(texCoords ? texCoords[i * 2 + 1] : 0.0f);
        }

        // Indices
        std::vector<unsigned int> indices;
        if (primitive.indices >= 0) {
            const auto& accessor = model.accessors[primitive.indices];
            const auto& bufferView = model.bufferViews[accessor.bufferView];
            const auto& buffer = model.buffers[bufferView.buffer];
            
            indices.reserve(accessor.count);
            
            if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const unsigned short* buf = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices.push_back(buf[i]);
                }
            } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const unsigned int* buf = reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices.push_back(buf[i]);
                }
            } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                const unsigned char* buf = reinterpret_cast<const unsigned char*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices.push_back(buf[i]);
                }
            }
        }

        return std::make_unique<Mesh>(vertices, indices, "gltf_mesh");
    }
};

std::shared_ptr<GameObject> GLTFLoader::Load(const std::string& path, TextureManager* texManager) {
    GLTFImporter importer(texManager);
    return importer.Import(path);
}
