#include "GameObject.h"
#include "Frustum.h"
#include "GLExtensions.h"
#include <algorithm>
#include <iostream>

static std::shared_ptr<Mesh> s_UnitCube = nullptr;

// Adaptive query frequency parameters
const int STABLE_THRESHOLD = 10;      // Frames before considered stable
const int MAX_QUERY_INTERVAL = 4;     // Max frames between queries
const int MIN_QUERY_INTERVAL = 1;     // Min frames (always test unstable objects)

GameObject::GameObject(const std::string& name) 
    : m_Name(name)
    , m_WorldMatrix(Mat4::Identity())
    , m_QueryID(0)
    , m_Visible(true)
    , m_QueryIssued(false)
    , m_PreviousVisible(true)
    , m_VisibilityStableFrames(0)
    , m_QueryFrameInterval(1)  // Start with every frame
    , m_FramesSinceLastQuery(0)
    , m_UVOffset(0.0f, 0.0f)
    , m_UVScale(1.0f, 1.0f)
{
}

GameObject::~GameObject() {
    if (m_QueryID != 0) {
        glDeleteQueries(1, &m_QueryID);
    }
}

void GameObject::Update(const Mat4& parentMatrix) {
    // Calculate world matrix based on parent and local transform
    m_WorldMatrix = parentMatrix * m_Transform.GetModelMatrix();
    
    // Update children
    for (auto& child : m_Children) {
        child->Update(m_WorldMatrix);
    }
}

void GameObject::Draw(Shader* shader, const Mat4& view, const Mat4& projection, Frustum* frustum, bool forceRender) {
    // Frustum Culling
    if (frustum && m_Mesh) {
        const AABB& localBounds = m_Mesh->GetBounds();
        
        // Transform AABB to world space
        Vec3 corners[8];
        corners[0] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.min.y, localBounds.min.z);
        corners[1] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.min.y, localBounds.min.z);
        corners[2] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.max.y, localBounds.min.z);
        corners[3] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.min.y, localBounds.max.z);
        corners[4] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.max.y, localBounds.min.z);
        corners[5] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.min.y, localBounds.max.z);
        corners[6] = m_WorldMatrix * Vec3(localBounds.min.x, localBounds.max.y, localBounds.max.z);
        corners[7] = m_WorldMatrix * Vec3(localBounds.max.x, localBounds.max.y, localBounds.max.z);

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
        
        // If object is outside frustum, skip rendering
        if (!frustum->ContainsAABB(worldBounds)) {
            // Still need to check children
            for (auto& child : m_Children) {
                child->Draw(shader, view, projection, frustum, forceRender);
            }
            return;
        }
    }
    
    // Occlusion Culling Check (skip if forceRender is true)
    if (!forceRender && !m_Visible) {
        // If not visible from last frame's query, skip drawing this object
        // But still draw children
        for (auto& child : m_Children) {
            child->Draw(shader, view, projection, frustum, forceRender);
        }
        return;
    }
    
    // LOD Selection
    std::shared_ptr<Mesh> meshToDraw = m_Mesh;
    std::shared_ptr<Model> modelToDraw = m_Model;
    
    if (!m_LODs.empty()) {
        // Calculate distance to camera
        Vec3 viewPos = view * m_WorldMatrix.GetTranslation();
        float dist = viewPos.Length();
        
        for (const auto& lod : m_LODs) {
            if (dist >= lod.minDistance) {
                if (lod.mesh) {
                    meshToDraw = lod.mesh;
                    modelToDraw = nullptr;
                } else if (lod.model) {
                    modelToDraw = lod.model;
                    meshToDraw = nullptr;
                }
                break;
            }
        }
    }

    if (modelToDraw) {
        // Draw model (handles its own materials)
        Mat4 mvp = projection * view * m_WorldMatrix;
        shader->SetMat4("u_MVP", mvp.m);
        shader->SetMat4("u_Model", m_WorldMatrix.m);
        modelToDraw->Draw(shader);
    }
    else if (meshToDraw && m_Material) {
        // Draw single mesh with material
        Mat4 mvp = projection * view * m_WorldMatrix;
        shader->SetMat4("u_MVP", mvp.m);
        shader->SetMat4("u_Model", m_WorldMatrix.m);
        
        // Set UV offset and scale for sprite atlases
        shader->SetVec2("u_UVOffset", m_UVOffset.x, m_UVOffset.y);
        shader->SetVec2("u_UVScale", m_UVScale.x, m_UVScale.y);
        
        m_Material->Bind(shader);
        meshToDraw->Draw();
    }
    
    // Draw children
    for (auto& child : m_Children) {
        child->Draw(shader, view, projection, frustum, forceRender);
    }
}

void GameObject::AddLOD(std::shared_ptr<Mesh> mesh, float minDistance) {
    LODLevel lod;
    lod.mesh = mesh;
    lod.minDistance = minDistance;
    m_LODs.push_back(lod);
    
    // Sort by distance descending (farthest first)
    std::sort(m_LODs.begin(), m_LODs.end(), [](const LODLevel& a, const LODLevel& b) {
        return a.minDistance > b.minDistance;
    });
}

void GameObject::AddLOD(std::shared_ptr<Model> model, float minDistance) {
    LODLevel lod;
    lod.model = model;
    lod.minDistance = minDistance;
    m_LODs.push_back(lod);
    
    // Sort by distance descending
    std::sort(m_LODs.begin(), m_LODs.end(), [](const LODLevel& a, const LODLevel& b) {
        return a.minDistance > b.minDistance;
    });
}

void GameObject::InitQuery() {
    if (m_QueryID == 0) {
        glGenQueries(1, &m_QueryID);
    }
}

void GameObject::RenderBoundingBox(Shader* shader, const Mat4& view, const Mat4& projection) {
    if (!m_Mesh && !m_Model) return;

    if (!s_UnitCube) {
        s_UnitCube = std::make_shared<Mesh>(Mesh::CreateCube());
    }

    AABB totalBounds;
    bool boundsInitialized = false;

    if (m_Mesh) {
        totalBounds = m_Mesh->GetBounds();
        boundsInitialized = true;
    } else if (m_Model) {
        for (const auto& mesh : m_Model->GetMeshes()) {
            if (!boundsInitialized) {
                totalBounds = mesh->GetBounds();
                boundsInitialized = true;
            } else {
                // Union bounds
                const AABB& b = mesh->GetBounds();
                totalBounds.min.x = std::min(totalBounds.min.x, b.min.x);
                totalBounds.min.y = std::min(totalBounds.min.y, b.min.y);
                totalBounds.min.z = std::min(totalBounds.min.z, b.min.z);
                totalBounds.max.x = std::max(totalBounds.max.x, b.max.x);
                totalBounds.max.y = std::max(totalBounds.max.y, b.max.y);
                totalBounds.max.z = std::max(totalBounds.max.z, b.max.z);
            }
        }
    }

    if (!boundsInitialized) return;

    // Calculate transform
    Vec3 size = totalBounds.max - totalBounds.min;
    Vec3 center = (totalBounds.min + totalBounds.max) * 0.5f;

    Mat4 scaleMatrix = Mat4::Scale(size);
    Mat4 translationMatrix = Mat4::Translate(center);
    
    // Model matrix for the bounding box (in local space of the object)
    Mat4 localBoxMatrix = translationMatrix * scaleMatrix;
    
    // Combine with object's world matrix
    Mat4 worldBoxMatrix = m_WorldMatrix * localBoxMatrix;
    
    Mat4 mvp = projection * view * worldBoxMatrix;
    
    shader->SetMat4("u_MVP", mvp.m);
    shader->SetMat4("u_Model", worldBoxMatrix.m);
    
    s_UnitCube->Draw();
}

void GameObject::UpdateQueryInterval() {
    // Track visibility stability
    if (m_Visible == m_PreviousVisible) {
        m_VisibilityStableFrames++;
    } else {
        m_VisibilityStableFrames = 0;
        m_QueryFrameInterval = MIN_QUERY_INTERVAL; // Reset to every frame when visibility changes
    }
    
    // Update previous visibility for next frame
    m_PreviousVisible = m_Visible;
    
    // Increase interval if stable
    if (m_VisibilityStableFrames >= STABLE_THRESHOLD) {
        m_QueryFrameInterval = std::min(m_QueryFrameInterval + 1, MAX_QUERY_INTERVAL);
    }
}

bool GameObject::ShouldIssueQuery() const {
    // Always issue query if we haven't issued one yet
    if (m_QueryID == 0) {
        return true;
    }
    
    // Check if enough frames have passed
    return m_FramesSinceLastQuery >= m_QueryFrameInterval;
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

std::shared_ptr<Mesh> GameObject::GetActiveMesh(const Mat4& view) const {
    // If we have a Model, we can't return a single mesh
    // Models handle their own rendering with multiple meshes
    if (m_Model) {
        return nullptr;
    }
    
    // Start with default mesh
    std::shared_ptr<Mesh> meshToReturn = m_Mesh;
    
    // Check LOD levels if any exist
    if (!m_LODs.empty()) {
        // Calculate distance to camera
        // Extract camera position from view matrix (inverse of view)
        Vec3 viewPos = view * m_WorldMatrix.GetTranslation();
        float dist = viewPos.Length();
        
        // Find appropriate LOD level
        for (const auto& lod : m_LODs) {
            if (dist >= lod.minDistance) {
                if (lod.mesh) {
                    meshToReturn = lod.mesh;
                }
                // If LOD has a model instead, we can't return a single mesh
                else if (lod.model) {
                    return nullptr;
                }
                break;
            }
        }
    }
    
    return meshToReturn;
}

AABB GameObject::GetWorldAABB() const {
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

        return AABB(minBounds, maxBounds);
    }
    
    // Fallback for no mesh: return point AABB at position
    Vec3 pos = m_WorldMatrix.GetTranslation();
    return AABB(pos, pos);
}
