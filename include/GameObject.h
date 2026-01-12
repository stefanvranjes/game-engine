#pragma once

#include "Transform.h"
#include "Mesh.h"
#include "MaterialNew.h"
#include "Model.h"
#include "Math/Mat4.h"
#include "Math/Vec2.h"
#include <vector>
#include <memory>
#include <string>

// Forward declarations
class RigidBody;
class KinematicController;
class Decal;
class Water;
class Terrain;
class ScriptComponent;
class IPhysicsCloth;
class IPhysicsSoftBody;
#ifdef USE_PHYSX
class PhysXArticulation;
class PhysXArticulationLink;
class PhysXArticulation;
class PhysXArticulationLink;
class PhysXAggregate;
class PhysXVehicle;
#endif
class IPhysicsCharacterController;

class GameObject : public std::enable_shared_from_this<GameObject> {
public:
    GameObject(const std::string& name = "GameObject");
    virtual ~GameObject();

    std::shared_ptr<GameObject> Clone();

    void SetActive(bool active) { m_IsActive = active; }
    bool IsActive() const { return m_IsActive; }

    virtual void Update(const Mat4& parentMatrix, float deltaTime);
    void UpdateAnimator(float deltaTime);  // Update animation state
    void Draw(Shader* shader, const Mat4& view, const Mat4& projection, class Frustum* frustum = nullptr, bool forceRender = false);

    void AddChild(std::shared_ptr<GameObject> child);
    void RemoveChild(std::shared_ptr<GameObject> child);
    std::shared_ptr<GameObject> GetParent() const { return m_Parent.lock(); }
    
    // Getters/Setters
    void SetMesh(Mesh&& mesh) { m_Mesh = std::make_shared<Mesh>(std::move(mesh)); }
    std::shared_ptr<Mesh> GetMesh() const { return m_Mesh; }
    void SetMaterial(std::shared_ptr<Material> material) { m_Material = material; }
    void SetModel(std::shared_ptr<Model> model) { m_Model = model; }
    std::shared_ptr<Model> GetModel() const { return m_Model; }
    
    Transform& GetTransform() { return m_Transform; }
    const Mat4& GetWorldMatrix() const { return m_WorldMatrix; }
    const std::string& GetName() const { return m_Name; }
    std::shared_ptr<Material> GetMaterial() { return m_Material; }
    std::shared_ptr<Mesh> GetActiveMesh(const Mat4& view) const;
    
    std::vector<std::shared_ptr<GameObject>>& GetChildren() { return m_Children; }
    
    // UV manipulation for sprite atlases
    void SetUVOffset(const Vec2& offset) { m_UVOffset = offset; }
    void SetUVScale(const Vec2& scale) { m_UVScale = scale; }
    Vec2 GetUVOffset() const { return m_UVOffset; }
    Vec2 GetUVScale() const { return m_UVScale; }
    
    // Helper to check collision recursively
    bool CheckCollision(const AABB& bounds);
    
    // Get World Space AABB
    AABB GetWorldAABB() const;
    
    // Animation
    void SetAnimator(std::shared_ptr<class Animator> animator) { m_Animator = animator; }
    std::shared_ptr<class Animator> GetAnimator() const { return m_Animator; }
    
    // Physics - Rigid Bodies (Bullet)
    void SetRigidBody(std::shared_ptr<RigidBody> rigidBody) { m_RigidBody = rigidBody; }
    std::shared_ptr<RigidBody> GetRigidBody() const { return m_RigidBody; }

    // Physics - Rigid Bodies (PhysX Interface)
    void SetPhysicsRigidBody(std::shared_ptr<IPhysicsRigidBody> body) { 
        m_PhysicsRigidBody = body; 
        if (m_PhysicsRigidBody) {
            m_PhysicsRigidBody->SetUserData(this);
        }
    }
    std::shared_ptr<IPhysicsRigidBody> GetPhysicsRigidBody() const { return m_PhysicsRigidBody; }
    
    // Physics - Kinematic Controllers
    void SetKinematicController(std::shared_ptr<KinematicController> controller) { m_KinematicController = controller; }
    std::shared_ptr<KinematicController> GetKinematicController() const { return m_KinematicController; }

    // Decal
    void SetDecal(std::shared_ptr<class Decal> decal) { m_Decal = decal; }
    std::shared_ptr<class Decal> GetDecal() const { return m_Decal; }
    
    // Water
    void SetWater(std::shared_ptr<class Water> water) { m_Water = water; }
    std::shared_ptr<class Water> GetWater() const { return m_Water; }
    
    // Terrain
    void SetTerrain(std::shared_ptr<class Terrain> terrain) { m_Terrain = terrain; }
    std::shared_ptr<class Terrain> GetTerrain() const { return m_Terrain; }

    // Cloth
    void SetCloth(std::shared_ptr<IPhysicsCloth> cloth) { m_Cloth = cloth; }
    std::shared_ptr<IPhysicsCloth> GetCloth() const { return m_Cloth; }

    // Soft Body
    void SetSoftBody(std::shared_ptr<IPhysicsSoftBody> softBody) { m_SoftBody = softBody; }
    std::shared_ptr<IPhysicsSoftBody> GetSoftBody() const { return m_SoftBody; }
    
#ifdef USE_PHYSX
    // Destructible
    void SetDestructible(std::shared_ptr<class PhysXDestructible> destructible) { m_Destructible = destructible; }
    std::shared_ptr<class PhysXDestructible> GetDestructible() const { return m_Destructible; }

    // Articulation
    void SetArticulation(std::shared_ptr<class PhysXArticulation> articulation) { m_Articulation = articulation; }
    std::shared_ptr<class PhysXArticulation> GetArticulation() const { return m_Articulation; }
    
    void SetArticulationLink(std::shared_ptr<class PhysXArticulationLink> link) { m_ArticulationLink = link; }
    void SetArticulationLink(std::shared_ptr<class PhysXArticulationLink> link) { m_ArticulationLink = link; }
    std::shared_ptr<class PhysXArticulationLink> GetArticulationLink() const { return m_ArticulationLink; }

    // Aggregate
    void SetPhysXAggregate(std::shared_ptr<class PhysXAggregate> aggregate) { m_PhysXAggregate = aggregate; }
    std::shared_ptr<class PhysXAggregate> GetPhysXAggregate() const { return m_PhysXAggregate; }

    // PhysX Character Controller
    void SetPhysicsCharacterController(std::shared_ptr<IPhysicsCharacterController> controller) { m_PhysicsCharacterController = controller; }
    std::shared_ptr<IPhysicsCharacterController> GetPhysicsCharacterController() const { return m_PhysicsCharacterController; }

    // PhysX Vehicle
    void SetPhysXVehicle(std::shared_ptr<PhysXVehicle> vehicle) { m_PhysXVehicle = vehicle; }
    std::shared_ptr<PhysXVehicle> GetPhysXVehicle() const { return m_PhysXVehicle; }
#endif

    // Scripting
    void SetScriptComponent(std::shared_ptr<class ScriptComponent> script) { m_ScriptComponent = script; }
    std::shared_ptr<class ScriptComponent> GetScriptComponent() const { return m_ScriptComponent; }
    
    // Auto-LOD Grouping
    void ProcessLODGroups();
    // LOD System
    struct LODLevel {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<Model> model;
        float minDistance; // Switch to this LOD when distance >= minDistance
        bool isBillboard = false; // Flag to enable billboard shader
    };

    void AddLOD(std::shared_ptr<Mesh> mesh, float minDistance, bool isBillboard = false);
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

    // Audio
    void AddAudioSource(std::shared_ptr<class AudioSource> source);
    void RemoveAudioSource(std::shared_ptr<class AudioSource> source);
    void RemoveAudioSource(int index);
    void SetAudioSource(std::shared_ptr<class AudioSource> audioSource);
    std::shared_ptr<class AudioSource> GetAudioSource() const;

    void SetAudioListener(std::shared_ptr<class AudioListener> listener) { m_AudioListener = listener; }
    std::shared_ptr<class AudioListener> GetAudioListener() const { return m_AudioListener; }

private:

    std::string m_Name;
    bool m_IsActive = true;
    Transform m_Transform;
    Mat4 m_WorldMatrix;
    
    std::shared_ptr<Mesh> m_Mesh;
    std::shared_ptr<Material> m_Material;
    std::shared_ptr<Model> m_Model;
    
    Vec2 m_UVOffset;  // UV offset for sprite atlases
    Vec2 m_UVScale;   // UV scale for sprite atlases
    
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

    // LOD Transition
    int m_CurrentLODIndex = -1;       // -1 = Base Mesh
    int m_TargetLODIndex = -1;
    float m_LODTransitionProgress = 1.0f; // 0.0 to 1.0
    bool m_IsLODTransitioning = false;
    const float LOD_TRANSITION_DURATION = 1.0f; // Seconds to fade

    // Physics/Audio
    Vec3 m_Velocity;
    Vec3 m_LastPosition;

    // Physics Interpolation
    Vec3 m_PhysicsPosition;
    Quat m_PhysicsRotation;
    Vec3 m_PreviousPhysicsPosition;
    Quat m_PreviousPhysicsRotation;

public:
    void UpdatePhysicsState(const Vec3& pos, const Quat& rot);
    void InterpolatePhysicsState(float alpha);
private:

#ifdef USE_PHYSX
    // Articulation (Must be declared before m_Children to ensure correct destruction order)
    std::shared_ptr<class PhysXArticulation> m_Articulation;
    std::shared_ptr<class PhysXArticulationLink> m_ArticulationLink;
    std::shared_ptr<class PhysXAggregate> m_PhysXAggregate;
#endif

    std::vector<std::shared_ptr<GameObject>> m_Children;
    std::weak_ptr<GameObject> m_Parent;
    
    // Animation
    std::shared_ptr<class Animator> m_Animator;

    // Physics components
    std::shared_ptr<RigidBody> m_RigidBody;
    std::shared_ptr<IPhysicsRigidBody> m_PhysicsRigidBody;
    std::shared_ptr<KinematicController> m_KinematicController;
#ifdef USE_PHYSX
    std::shared_ptr<IPhysicsCharacterController> m_PhysicsCharacterController;
    std::shared_ptr<PhysXVehicle> m_PhysXVehicle;
#endif

    // Audio
    std::vector<std::shared_ptr<class AudioSource>> m_AudioSources;
    std::shared_ptr<class AudioListener> m_AudioListener;
    
    // Decal
    std::shared_ptr<class Decal> m_Decal;
    
    // Water
    std::shared_ptr<class Water> m_Water;
    
    // Terrain
    std::shared_ptr<class Terrain> m_Terrain;

    // Cloth
    std::shared_ptr<IPhysicsCloth> m_Cloth;

    // Soft Body
    std::shared_ptr<IPhysicsSoftBody> m_SoftBody;

    // Destructible
#ifdef USE_PHYSX
    std::shared_ptr<class PhysXDestructible> m_Destructible;
#endif

    // Scripting
    std::shared_ptr<class ScriptComponent> m_ScriptComponent;
    

};
