#pragma once

#include "Transform.h"
#include "Mesh.h"
#include "Material.h"
#include "Model.h"
#include "Math/Mat4.h"
#include <vector>
#include <memory>
#include <string>

class GameObject : public std::enable_shared_from_this<GameObject> {
public:
    GameObject(const std::string& name = "GameObject");
    ~GameObject();

    void Update(const Mat4& parentMatrix);
    void Draw(Shader* shader, const Mat4& view, const Mat4& projection, class Frustum* frustum = nullptr);

    void AddChild(std::shared_ptr<GameObject> child);
    void RemoveChild(std::shared_ptr<GameObject> child);
    
    // Getters/Setters
    void SetMesh(Mesh&& mesh) { m_Mesh = std::make_shared<Mesh>(std::move(mesh)); }
    void SetMaterial(std::shared_ptr<Material> material) { m_Material = material; }
    void SetModel(std::shared_ptr<Model> model) { m_Model = model; }
    
    Transform& GetTransform() { return m_Transform; }
    const Mat4& GetWorldMatrix() const { return m_WorldMatrix; }
    const std::string& GetName() const { return m_Name; }
    std::shared_ptr<Material> GetMaterial() { return m_Material; }
    
    std::vector<std::shared_ptr<GameObject>>& GetChildren() { return m_Children; }
    
    // Helper to check collision recursively
    bool CheckCollision(const AABB& bounds);

private:
    std::string m_Name;
    Transform m_Transform;
    Mat4 m_WorldMatrix;
    
    std::shared_ptr<Mesh> m_Mesh;
    std::shared_ptr<Material> m_Material;
    std::shared_ptr<Model> m_Model;
    
    std::vector<std::shared_ptr<GameObject>> m_Children;
    std::weak_ptr<GameObject> m_Parent;
};
