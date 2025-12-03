#include "Model.h"
#include "Vertex.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

Model Model::LoadFromOBJ(const std::string& path, TextureManager* texManager) {
    Model model;
    
    // Extract directory from path
    size_t lastSlash = path.find_last_of("/\\");
    model.m_Directory = (lastSlash != std::string::npos) ? path.substr(0, lastSlash + 1) : "";

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << path << std::endl;
        return model;
    }

    // Temporary storage for OBJ data
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> texCoords;
    
    // Material library
    std::map<std::string, std::shared_ptr<Material>> materials;
    std::string currentMaterial;
    
    // Per-material vertex data
    std::map<std::string, std::vector<Vertex>> materialVertices;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            // Vertex position
            Vec3 pos;
            iss >> pos.x >> pos.y >> pos.z;
            positions.push_back(pos);
        }
        else if (prefix == "vn") {
            // Vertex normal
            Vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (prefix == "vt") {
            // Texture coordinate
            Vec2 texCoord;
            iss >> texCoord.x >> texCoord.y;
            texCoords.push_back(texCoord);
        }
        else if (prefix == "mtllib") {
            // Material library file
            std::string mtlFile;
            iss >> mtlFile;
            model.LoadMTL(model.m_Directory + mtlFile, materials, texManager);
        }
        else if (prefix == "usemtl") {
            // Use material
            iss >> currentMaterial;
        }
        else if (prefix == "f") {
            // Face
            std::string vertexStr;
            std::vector<Vertex> faceVertices;

            while (iss >> vertexStr) {
                Vertex vertex;
                
                // Parse vertex indices (format: v/vt/vn or v//vn or v/vt or v)
                size_t pos1 = vertexStr.find('/');
                size_t pos2 = vertexStr.find('/', pos1 + 1);

                int posIdx = std::stoi(vertexStr.substr(0, pos1)) - 1;
                vertex.position = positions[posIdx];

                if (pos1 != std::string::npos) {
                    if (pos2 != std::string::npos) {
                        // v/vt/vn
                        if (pos2 > pos1 + 1) {
                            int texIdx = std::stoi(vertexStr.substr(pos1 + 1, pos2 - pos1 - 1)) - 1;
                            vertex.texCoord = texCoords[texIdx];
                        }
                        int normIdx = std::stoi(vertexStr.substr(pos2 + 1)) - 1;
                        vertex.normal = normals[normIdx];
                    }
                    else {
                        // v/vt
                        int texIdx = std::stoi(vertexStr.substr(pos1 + 1)) - 1;
                        vertex.texCoord = texCoords[texIdx];
                    }
                }

                faceVertices.push_back(vertex);
            }

            // Triangulate face (assuming convex polygons)
            for (size_t i = 1; i + 1 < faceVertices.size(); ++i) {
                materialVertices[currentMaterial].push_back(faceVertices[0]);
                materialVertices[currentMaterial].push_back(faceVertices[i]);
                materialVertices[currentMaterial].push_back(faceVertices[i + 1]);
            }
        }
    }

    // Create meshes from material groups
    for (const auto& pair : materialVertices) {
        const std::string& matName = pair.first;
        const std::vector<Vertex>& vertices = pair.second;

        if (!vertices.empty()) {
            // Convert Vertex data to flat arrays
            std::vector<float> vertexData;
            std::vector<unsigned int> indices;
            
            for (size_t i = 0; i < vertices.size(); ++i) {
                const Vertex& v = vertices[i];
                // Position
                vertexData.push_back(v.position.x);
                vertexData.push_back(v.position.y);
                vertexData.push_back(v.position.z);
                // Normal
                vertexData.push_back(v.normal.x);
                vertexData.push_back(v.normal.y);
                vertexData.push_back(v.normal.z);
                // TexCoord
                vertexData.push_back(v.texCoord.x);
                vertexData.push_back(v.texCoord.y);
                
                indices.push_back(static_cast<unsigned int>(i));
            }
            
            auto mesh = std::make_shared<Mesh>(vertexData, indices, path);
            model.m_Meshes.push_back(mesh);

            // Assign material
            if (materials.find(matName) != materials.end()) {
                model.m_Materials.push_back(materials[matName]);
            } else {
                // Default material
                model.m_Materials.push_back(std::make_shared<Material>());
            }
        }
    }

    std::cout << "Loaded model: " << path << " with " << model.m_Meshes.size() << " meshes" << std::endl;
    return model;
}

void Model::LoadMTL(const std::string& path, std::map<std::string, std::shared_ptr<Material>>& materials, TextureManager* texManager) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open MTL file: " << path << std::endl;
        return;
    }

    std::shared_ptr<Material> currentMaterial = nullptr;
    std::string currentName;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "newmtl") {
            // New material
            iss >> currentName;
            currentMaterial = std::make_shared<Material>();
            materials[currentName] = currentMaterial;
        }
        else if (currentMaterial) {
            if (prefix == "Ka") {
                // Ambient color
                Vec3 ambient;
                iss >> ambient.x >> ambient.y >> ambient.z;
                currentMaterial->SetAmbient(ambient);
            }
            else if (prefix == "Kd") {
                // Diffuse color
                Vec3 diffuse;
                iss >> diffuse.x >> diffuse.y >> diffuse.z;
                currentMaterial->SetDiffuse(diffuse);
            }
            else if (prefix == "Ks") {
                // Specular color
                Vec3 specular;
                iss >> specular.x >> specular.y >> specular.z;
                currentMaterial->SetSpecular(specular);
            }
            else if (prefix == "Ns") {
                // Shininess
                float shininess;
                iss >> shininess;
                currentMaterial->SetShininess(shininess);
            }
            else if (prefix == "map_Kd") {
                // Diffuse texture
                std::string texPath;
                iss >> texPath;
                // Extract directory from MTL path
                size_t lastSlash = path.find_last_of("/\\");
                std::string dir = (lastSlash != std::string::npos) ? path.substr(0, lastSlash + 1) : "";
                currentMaterial->SetTexture(texManager->LoadTexture(dir + texPath));
            }
            else if (prefix == "map_Ks") {
                // Specular texture
                std::string texPath;
                iss >> texPath;
                // Extract directory from MTL path
                size_t lastSlash = path.find_last_of("/\\");
                std::string dir = (lastSlash != std::string::npos) ? path.substr(0, lastSlash + 1) : "";
                currentMaterial->SetSpecularMap(texManager->LoadTexture(dir + texPath));
            }
            else if (prefix == "map_Bump" || prefix == "bump") {
                // Normal map
                std::string texPath;
                iss >> texPath;
                // Extract directory from MTL path
                size_t lastSlash = path.find_last_of("/\\");
                std::string dir = (lastSlash != std::string::npos) ? path.substr(0, lastSlash + 1) : "";
                currentMaterial->SetNormalMap(texManager->LoadTexture(dir + texPath));
            }
        }
    }

    std::cout << "Loaded MTL: " << path << " with " << materials.size() << " materials" << std::endl;
}

void Model::Draw(Shader* shader) {
    for (size_t i = 0; i < m_Meshes.size(); ++i) {
        if (i < m_Materials.size() && m_Materials[i]) {
            m_Materials[i]->Bind(shader);
        }
        m_Meshes[i]->Draw();
    }
}
