#include "GameObject.h"
#include "Frustum.h"
#include "GLExtensions.h"
#include "Animator.h"
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

void GameObject::Update(const Mat4& parentMatrix, float deltaTime) {
    // Calculate world matrix based on parent and local transform
    m_WorldMatrix = parentMatrix * m_Transform.GetModelMatrix();
    
    // Velocity Calculation
    Vec3 currentPos = m_WorldMatrix.GetTranslation();
    
    // Calculate velocity (units per second)
    if (deltaTime > 0.0001f) {
        m_Velocity = (currentPos - m_LastPosition) / deltaTime;
    } else {
        m_Velocity = Vec3(0, 0, 0);
    }
    
    m_LastPosition = currentPos;

    // Update AudioSource position and velocity
    for (auto& source : m_AudioSources) {
        if (source) {
            source->SetPosition(currentPos);
            source->SetVelocity(m_Velocity);
        }
    }
    
    // Update AudioListener if present
    if (m_AudioListener) {
        // Assume forward is Z for now, or extract from matrix
        // m_WorldMatrix has Forward in 3rd column (usually)
        Vec3 forward = Vec3(m_WorldMatrix.m[8], m_WorldMatrix.m[9], m_WorldMatrix.m[10]).Normalized();
        // Or cleaner: (m_WorldMatrix * Vec4(0,0,1,0)).xyz
        
        m_AudioListener->UpdateState(currentPos, forward, m_Velocity);
    }
    
    // Update children
    for (auto& child : m_Children) {
        child->Update(m_WorldMatrix, deltaTime);
    }
    
    // Auto-update LOD state
    if (!m_LODs.empty()) {
        // We need camera position. But Update doesn't have camera info.
        // We usually do this in Draw or a separate Cull/LOD pass.
        // Doing it in Draw is tricky because Draw might not be called if parent is culled?
        // Actually, Draw is recursive.
        // However, updating state in Draw is generally discouraged (side effects).
        // Let's postpone LOD selection to Draw or pre-Draw traversal.
        // But we need to update transition timer here.
        
        if (m_IsLODTransitioning) {
            m_LODTransitionProgress += deltaTime / LOD_TRANSITION_DURATION;
            if (m_LODTransitionProgress >= 1.0f) {
                m_LODTransitionProgress = 1.0f;
                m_IsLODTransitioning = false;
                m_CurrentLODIndex = m_TargetLODIndex;
            }
        }
    }
}

void GameObject::AddAudioSource(std::shared_ptr<AudioSource> source) {
    if (source) {
        m_AudioSources.push_back(source);
    }
}

void GameObject::RemoveAudioSource(std::shared_ptr<AudioSource> source) {
    m_AudioSources.erase(
        std::remove(m_AudioSources.begin(), m_AudioSources.end(), source),
        m_AudioSources.end()
    );
}

void GameObject::RemoveAudioSource(int index) {
    if (index >= 0 && index < m_AudioSources.size()) {
        m_AudioSources.erase(m_AudioSources.begin() + index);
    }
}

void GameObject::SetAudioSource(std::shared_ptr<AudioSource> source) {
    // Compatibility: Clear and add
    m_AudioSources.clear();
    if (source) {
        AddAudioSource(source);
    }
}

std::shared_ptr<AudioSource> GameObject::GetAudioSource() const {
    if (!m_AudioSources.empty()) {
        return m_AudioSources[0];
    }
    return nullptr;
}

void GameObject::UpdateAnimator(float deltaTime) {
    // Update animator if present
    if (m_Animator) {
        m_Animator->Update(deltaTime);
    }
    
    // Update children animators
    for (auto& child : m_Children) {
        child->UpdateAnimator(deltaTime);
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
    
    // LOD Selection & Transition
    int desiredLOD = -1; // Base
    
    if (!m_LODs.empty()) {
        Vec3 viewPos = view * m_WorldMatrix.GetTranslation();
        float dist = viewPos.Length();
        
        for (size_t i = 0; i < m_LODs.size(); ++i) {
            if (dist >= m_LODs[i].minDistance) {
                desiredLOD = static_cast<int>(i);
                break;
            }
        }
        
        // Check for state change
        if (desiredLOD != m_TargetLODIndex && desiredLOD != m_CurrentLODIndex) {
            // Start transition
            m_TargetLODIndex = desiredLOD;
            
            // If we were already transitioning, we blend from current visual state?
            // Simplification: Just reset blend to 0 and go towards new target from current snapshot?
            // Or simpler: always blend from m_CurrentLODIndex to m_TargetLODIndex.
            m_IsLODTransitioning = true;
            m_LODTransitionProgress = 0.0f;
        }
        // If desired == current, and we are transitioning, we might want to reverse? 
        // Logic: if desired == m_CurrentLODIndex && transitioning, we are moving back.
        // That effectively means Target becomes Current, and Current becomes Target.
        // Let's implement simpler: Only change target if different from current target.
        if (desiredLOD != m_TargetLODIndex) {
             // Retarget
             // If we were transitioning A->B, and now want A, revert?
             // Not strictly handled here for simplicity.
        }
    }

    // Helper to draw a specific mesh/model
    auto DrawLOD = [&](int lodIndex, float alpha) {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<Model> model;
        bool isBillboard = false;
        
        if (lodIndex == -1) {
            mesh = m_Mesh;
            model = m_Model;
        } else if (lodIndex >= 0 && lodIndex < m_LODs.size()) {
            mesh = m_LODs[lodIndex].mesh;
            model = m_LODs[lodIndex].model;
            isBillboard = m_LODs[lodIndex].isBillboard;
        }
        
        Shader* shaderToUse = shader;
        // Basic shader switching logic (Hack since we don't have a Shader Manager)
        static std::unique_ptr<Shader> billboardShader;
        if (isBillboard) {
             if (!billboardShader) {
                 billboardShader = std::make_unique<Shader>();
                 billboardShader->LoadFromFiles("shaders/billboard.vert", "shaders/billboard.frag");
             }
             shaderToUse = billboardShader.get();
             shaderToUse->Use();
             // Set Billboard Uniforms
             // Billboard shader needs: u_View, u_Projection, u_CenterPos, u_Size
             shaderToUse = billboardShader.get();
             shaderToUse->Use();
             // Set Billboard Uniforms
             shaderToUse->SetMat4("u_View", view.m);
             shaderToUse->SetMat4("u_Projection", projection.m);
             shaderToUse->SetVec3("u_CenterPos", m_WorldMatrix.GetTranslation().x, m_WorldMatrix.GetTranslation().y, m_WorldMatrix.GetTranslation().z);
             shaderToUse->SetVec2("u_Size", m_WorldMatrix.GetScale().x, m_WorldMatrix.GetScale().y); // Use scale for size
             shaderToUse->SetInt("u_Rows", 4);
             shaderToUse->SetInt("u_Cols", 4);
             
             // Extract Camera Pos from View Matrix
             // GLSL View is Column-Major:
             // [ Rx Ux Fx 0 ]
             // [ Ry Uy Fy 0 ]
             // [ Rz Uz Fz 0 ]
             // [ Tx Ty Tz 1 ]
             // Wait, LookAt implementation usually:
             // [ Rx Ry Rz -dot(R,P) ]
             // [ Ux Uy Uz -dot(U,P) ]
             // [ Fx Fy Fz -dot(F,P) ]
             // [ 0  0  0  1         ]
             // In Column Major:
             // m[0] = Rx, m[1] = Ux, m[2] = Fx, m[3] = 0
             // m[4] = Ry, m[5] = Uy, m[6] = Fy, m[7] = 0
             // m[8] = Rz, m[9] = Uz, m[10]= Fz, m[11]= 0
             // m[12]=Tx,  m[13]=Ty,  m[14]=Tz,  m[15]= 1
             
             // Rotation Transpose (Inverse Rotation)
             // R^T = [ Rx Ux Fx ] -> [ m0 m1 m2 ]
             //       [ Ry Uy Fy ]    [ m4 m5 m6 ]
             //       [ Rz Uz Fz ]    [ m8 m9 m10]
             
             // CameraPos = -R^T * T_vec
             // T_vec = [ m12 m13 m14 ]
             
             float m0 = view.m[0][0], m4 = view.m[1][0], m8 = view.m[2][0];
             float m1 = view.m[0][1], m5 = view.m[1][1], m9 = view.m[2][1];
             float m2 = view.m[0][2], m6 = view.m[1][2], m10= view.m[2][2];
             float m12= view.m[3][0], m13= view.m[3][1], m14= view.m[3][2];
             
             // Check matrix access. Mat4 is usually m[col][row] or m[row][col]? 
             // Math/Mat4.h usually column-major internal array if OpenGL compatible.
             // If Mat4.m is float[4][4], then m[col][row].
             // let's verify Mat4 definition later. Assuming standard OpenGL layout.
             // R^T row 0: (m0, m4, m8). (Actually m0=xx, m4=xy .. wait.)
             // If column major:
             // Col 0 (Right axis): m00, m01, m02 -> R.x, R.y, R.z (No! Right vector is usually the ROW in lookat? No.
             // LookAt:
             // Row 0: R.x, R.y, R.z
             // Row 1: U.x, U.y, U.z
             // Row 2: F.x, F.y, F.z
             // OpenGL stores Column-Major. So Row 0 is stored as: m[0], m[4], m[8].
             // So m00, m10, m20.
             // Access m[col][row]?
             // Let's assume m[col][row].
             // Row 0: m[0][0], m[1][0], m[2][0]
             
             // Camera Pos = - (Row0*Tx + Row1*Ty + Row2*Tz) ? No.
             // P = -R^T * T
             // R^T = [ col0 col1 col2 ] (Because R is orthogonal, inv(R) = R^T)
             // T = [ m[3][0], m[3][1], m[3][2] ]
             
             // C.x = -(R.x*Tx + U.x*Ty + F.x*Tz) 
             // C.x = -(m[0][0]*m[3][0] + m[0][1]*m[3][1] + m[0][2]*m[3][2])
             // C.y = -(m[1][0]*m[3][0] + m[1][1]*m[3][1] + m[1][2]*m[3][2])
             // C.z = -(m[2][0]*m[3][0] + m[2][1]*m[3][1] + m[2][2]*m[3][2])
            
             float tx = view.m[3][0];
             float ty = view.m[3][1];
             float tz = view.m[3][2];
             
             float cx = -(view.m[0][0]*tx + view.m[0][1]*ty + view.m[0][2]*tz);
             float cy = -(view.m[1][0]*tx + view.m[1][1]*ty + view.m[1][2]*tz);
             float cz = -(view.m[2][0]*tx + view.m[2][1]*ty + view.m[2][2]*tz);
             
             shaderToUse->SetVec3("u_CameraPos", cx, cy, cz);

        if (model) {
            // Model drawing doesn't support alpha override easily yet without traversing materials.
            // For now, Models pop.
            Mat4 mvp = projection * view * m_WorldMatrix;
            shaderToUse->SetMat4("u_MVP", mvp.m); // Use shaderToUse here
            shaderToUse->SetMat4("u_Model", m_WorldMatrix.m); // Use shaderToUse here
            model->Draw(shaderToUse); // Pass shaderToUse to model
        } else if (mesh && m_Material) {
            if (!isBillboard) {
                Mat4 mvp = projection * view * m_WorldMatrix;
                shaderToUse->SetMat4("u_MVP", mvp.m);
                shaderToUse->SetMat4("u_Model", m_WorldMatrix.m);
                shaderToUse->SetVec2("u_UVOffset", m_UVOffset.x, m_UVOffset.y);
                shaderToUse->SetVec2("u_UVScale", m_UVScale.x, m_UVScale.y);
                
                // Set Dithering Uniforms
                shaderToUse->SetFloat("u_DitherThreshold", alpha); 
            }
            
            m_Material->Bind(shaderToUse); // Bind material to the active shader
            mesh->Draw();
            
            // Restore shader if changed
            if (isBillboard) {
                shader->Use();
            }
        }
    };

    if (m_IsLODTransitioning) {
        // Draw Current (fading out)
        DrawLOD(m_CurrentLODIndex, 1.0f - m_LODTransitionProgress);
        
        // Draw Target (fading in)
        // Only if different
        if (m_CurrentLODIndex != m_TargetLODIndex) {
            DrawLOD(m_TargetLODIndex, m_LODTransitionProgress);
        }
    } else {
        // Draw Current steady state
        DrawLOD(m_CurrentLODIndex, 1.0f);
    }
    
    // Draw children
    for (auto& child : m_Children) {
        child->Draw(shader, view, projection, frustum, forceRender);
    }
}

void GameObject::AddLOD(std::shared_ptr<Mesh> mesh, float minDistance, bool isBillboard) {
    LODLevel lod;
    lod.mesh = mesh;
    lod.minDistance = minDistance;
    lod.isBillboard = isBillboard;
    m_LODs.push_back(lod);
    
    // Maintain sort order (descending distance)
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
