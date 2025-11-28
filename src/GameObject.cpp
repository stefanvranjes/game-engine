#include "GameObject.h"
#include <algorithm>
#include <iostream>

GameObject::GameObject(const std::string& name) 
    : m_Name(name)
    , m_WorldMatrix(Mat4::Identity())
{
}

GameObject::~GameObject() {
}

void GameObject::Update(const Mat4& parentMatrix) {
    // Calculate world matrix based on parent and local transform
    m_WorldMatrix = parentMatrix * m_Transform.GetModelMatrix();
    
    // Update children
    for (auto& child : m_Children) {
        child->Update(m_WorldMatrix);
    }
}

void GameObject::Draw(Shader* shader, const Mat4& view, const Mat4& projection) {
    if (m_Model) {
        // Draw model (handles its own materials)
        Mat4 mvp = projection * view * m_WorldMatrix;
        shader->SetMat4("u_MVP", mvp.m);
        shader->SetMat4("u_Model", m_WorldMatrix.m);
        m_Model->Draw(shader);
    }
    else if (m_Mesh && m_Material) {
        // Draw single mesh with material
        Mat4 mvp = projection * view * m_WorldMatrix;
        shader->SetMat4("u_MVP", mvp.m);
        shader->SetMat4("u_Model", m_WorldMatrix.m);
        m_Material->Bind(shader);
        m_Mesh->Draw();
    }
    
    // Draw children
    for (auto& child : m_Children) {
        child->Draw(shader, view, projection);
    }
}

void GameObject::AddChild(std::shared_ptr<GameObject> child) {
    child->m_Parent = shared_from_this();
    m_Children.push_back(child);
}

void GameObject::RemoveChild(std::shared_ptr<GameObject> child) {
    auto it = std::remove(m_Children.begin(), m_Children.end(), child);
    if (it != m_Children.end()) {
        m_Children.erase(it, m_Children.end());
        child->m_Parent.reset();
    }
}

bool GameObject::CheckCollision(const AABB& bounds) {
    if (m_Mesh) {
        const AABB& localBounds = m_Mesh->GetBounds();
        
        // Transform all 8 corners of the AABB
        Vec3 corners[8];
        corners[0] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.min.y, localBounds.min.z);
        corners[1] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.min.y, localBounds.min.z);
        corners[2] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.max.y, localBounds.min.z);
        corners[3] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.min.y, localBounds.max.z);
        corners[4] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.max.y, localBounds.min.z);
        corners[5] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.min.y, localBounds.max.z);
        corners[6] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.max.y, localBounds.max.z);
        corners[7] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.max.y, localBounds.max.z);

        // Calculate new world AABB
        Vec3 minBounds = corners[0];
        Vec3 maxBounds = corners[0];

        for (int j = 1; j < 8; ++j) {
            if (corners[j].x < minBounds.x) minBounds.x = corners[j].x;
            if (corners[j].y < minBounds.y) minBounds.y = corners[j].y;
            if (corners[j].z < minBounds.z) minBounds.z = corners[j].z;

            if (corners[j].x > maxBounds.x) maxBounds.x = corners[j].x;
            if (corners[j].y > maxBounds.y) maxBounds.y = corners[j].y;
            if (corners[j].z > maxBounds.z) maxBounds.z = corners[j].z;
        }

        AABB worldBounds(minBounds, maxBounds);
        if (bounds.Intersects(worldBounds)) {
            return true;
        }
    }
    
    // Check children
    for (auto& child : m_Children) {
        if (child->CheckCollision(bounds)) {
            return true;
        }
    }
    
    return false;
}
