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
    void Draw(Shader* shader, const Mat4& view, const Mat4& projection, class Frustum* frustum = nullptr, bool forceRender = false);

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
    std::shared_ptr<Mesh> GetActiveMesh(const Mat4& view) const;
    
    std::vector<std::shared_ptr<GameObject>>& GetChildren() { return m_Children; }
    
    // Helper to check collision recursively
    bool CheckCollision(const AABB& bounds);
    
    // Get World Space AABB
    AABB GetWorldAABB() const;

    // LOD System
    struct LODLevel {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<Model> model;
        float minDistance; // Switch to this LOD when distance >= minDistance
    };

    void AddLOD(std::shared_ptr<Mesh> mesh, float minDistance);
    void AddLOD(std::shared_ptr<Model> model, float minDistance);

    // Occlusion Culling
    void InitQuery();
    void RenderBoundingBox(Shader* shader, const Mat4& view, const Mat4& projection);
    unsigned int GetQueryID() const { return m_QueryID; }
    bool IsVisible() const { return m_Visible; }
    void SetVisible(bool visible) { m_Visible = visible; }
    bool IsQueryIssued() const { return m_QueryIssued; }
    void SetQueryIssued(bool issued) { m_QueryIssued = issued; }
    
    // Adaptive query frequency
    int GetQueryInterval() const { return m_QueryFrameInterval; }
    void UpdateQueryInterval(); // Update based on visibility stability
    bool ShouldIssueQuery() const; // Check if query should be issued this frame
    void ResetQueryFrameCounter() { m_FramesSinceLastQuery = 0; }
    void IncrementQueryFrameCounter() { m_FramesSinceLastQuery++; }

private:
    std::string m_Name;
    Transform m_Transform;
    Mat4 m_WorldMatrix;
    
    std::shared_ptr<Mesh> m_Mesh;
    std::shared_ptr<Material> m_Material;
    std::shared_ptr<Model> m_Model;
    
    std::vector<LODLevel> m_LODs; // Sorted by distance descending

    // Occlusion Culling Data
    unsigned int m_QueryID;
    bool m_Visible;
    bool m_QueryIssued;
    
    // Adaptive query frequency
    bool m_PreviousVisible;           // Visibility from last frame
    int m_VisibilityStableFrames;     // Consecutive frames with same visibility
    int m_QueryFrameInterval;         // How often to issue queries (1 = every frame)
    int m_FramesSinceLastQuery;       // Counter for skipping frames

    std::vector<std::shared_ptr<GameObject>> m_Children;
    std::weak_ptr<GameObject> m_Parent;
};
