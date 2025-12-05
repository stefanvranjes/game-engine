#pragma once

#include "Mesh.h"
#include "MaterialNew.h"
#include "Shader.h"
#include "TextureManager.h"
#include <vector>
#include <memory>
#include <string>

class Model {
public:
    Model() = default;
    ~Model() = default;

    // Load model from OBJ file with MTL support
    static Model LoadFromOBJ(const std::string& path, TextureManager* texManager);

    // Draw all meshes with their materials
    void Draw(Shader* shader);

    // Getters
    const std::vector<std::shared_ptr<Mesh>>& GetMeshes() const { return m_Meshes; }
    const std::vector<std::shared_ptr<Material>>& GetMaterials() const { return m_Materials; }


private:
    void LoadMTL(const std::string& path, std::map<std::string, std::shared_ptr<Material>>& materials, TextureManager* texManager);
    
    std::vector<std::shared_ptr<Mesh>> m_Meshes;
    std::vector<std::shared_ptr<Material>> m_Materials;
    std::string m_Directory;
};
