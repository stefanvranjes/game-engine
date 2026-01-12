#include "PhysXRagdoll.h"
#include "PhysXBackend.h"
#include <iostream>

#ifdef USE_PHYSX
#include <PxPhysicsAPI.h>
using namespace physx;
#endif

PhysXRagdoll::PhysXRagdoll(PhysXBackend* backend, std::shared_ptr<Skeleton> skeleton)
    : m_Backend(backend)
    , m_Skeleton(skeleton)
    , m_Articulation(nullptr)
    , m_State(RagdollState::Kinematic)
{
}

PhysXRagdoll::~PhysXRagdoll() {
    if (m_Articulation) {
        delete m_Articulation;
        m_Articulation = nullptr;
    }
}

void PhysXRagdoll::Initialize(const Vec3& rootPosition, const Quat& rootRotation) {
    if (!m_Backend) return;

    m_Articulation = new PhysXArticulation(m_Backend);
    // Articulations are floating by default, not fixed base, unless we clamp the root
    m_Articulation->Initialize(false); 
    
    // Add to scene immediately? Or wait? Usually better to wait until configured.
    m_Backend->GetScene()->addArticulation(*static_cast<PxArticulationReducedCoordinate*>(m_Articulation->GetNativeArticulation()));
}

PhysXArticulationLink* PhysXRagdoll::AddBone(const RagdollBoneConfig& config, std::shared_ptr<IPhysicsShape> shape) {
    if (!m_Articulation || !m_Skeleton) return nullptr;

    int boneIndex = m_Skeleton->FindBoneIndex(config.boneName);
    if (boneIndex < 0) {
        std::cerr << "Ragdoll bone not found in skeleton: " << config.boneName << std::endl;
        return nullptr;
    }

    // Find parent link
    const Bone& bone = m_Skeleton->GetBone(boneIndex);
    PhysXArticulationLink* parentLink = nullptr;
    
    if (bone.parentIndex >= 0) {
        // Find parent bone name
        const Bone& parentBone = m_Skeleton->GetBone(bone.parentIndex);
        auto it = m_Bones.find(parentBone.name);
        if (it != m_Bones.end()) {
            parentLink = it->second.link;
        } else {
            // If parent isn't in ragdoll, this must be a root or attached to non-simulated bone?
            // PhysX Articulations require a single root tree. 
            // If we skip bones, we must attach to the nearest ragdolled ancestor or be a new root (which isn't allowed for one articulation).
            // For now assume strictly hierarchical setup.
        }
    }

    // Positions are normally handled by Initial Pose, but here we just create links.
    // PhysX Articulation setup is relative. 
    // HOWEVER, `AddLink` wrapper might take world pose.
    // Let's assume we initialize at identity/origin and `MatchAnimation` moves it.
    
    PhysXArticulationLink* link = m_Articulation->AddLink(parentLink, Vec3(0,0,0), Quat(0,0,0,1));
    if (!link) return nullptr;

    link->Initialize(BodyType::Dynamic, config.mass, shape);
    link->SetFriction(config.friction);
    link->SetRestitution(config.restitution);
    link->SetLinearDamping(config.linearDamping);
    link->SetAngularDamping(config.angularDamping);

    BoneEntry entry;
    entry.boneIndex = boneIndex;
    entry.link = link;
    entry.config = config;
    entry.joint = link->GetInboundJoint();
    entry.bindPoseInv = bone.inverseBindMatrix; // Actually inverse bind matrix transforms FROM world(mesh) TO bone local.

    // Config Joint
    if (entry.joint) {
        // Set limits
        entry.joint->SetMotion(Axis::X, Motion::Limited); // Twist
        entry.joint->SetMotion(Axis::Y, Motion::Limited); // Swing1
        entry.joint->SetMotion(Axis::Z, Motion::Limited); // Swing2
        
        entry.joint->SetTwistLimit(config.twistLimitLow * SM_DEG_TO_RAD, config.twistLimitHigh * SM_DEG_TO_RAD);
        entry.joint->SetSwingLimit(config.swing1Limit * SM_DEG_TO_RAD, config.swing2Limit * SM_DEG_TO_RAD);
        
        // Config Drives (initially disabled or stiff based on state, managed in SetState/DriveToAnimation)
    }

    m_Bones[config.boneName] = entry;
    m_OrderedBones.push_back(&m_Bones[config.boneName]); // Store pointer to map value
    
    return link;
}

void PhysXRagdoll::SetState(RagdollState state) {
    if (m_State == state) return;
    m_State = state;
    
    if (!m_Articulation) return;

    if (m_State == RagdollState::Kinematic) {
        // Disable gravity, maybe high damping or kinematic flags if supported by articulation links (they aren't really).
        // ReducedCoordinate articulations are always dynamic. To make them "Kinematic", we usually:
        // 1. Fix the base
        // 2. Set extremely high stiffness drives to target pose (Puppet)
        // OR
        // 3. Use `m_Articulation->putToSleep()` if it shouldn't move.
        // But true Kinematic that follows animation requires teleporting/setting global pose every frame.
        // We will handle "Kinematic" by calling `MatchAnimation` every frame and potentially sleeping functionality.
        m_Articulation->SetGravityEnabled(false);
    } else {
        m_Articulation->SetGravityEnabled(true);
        m_Articulation->WakeUp();
    }
    
    // Config drives based on state
    for (auto* entry : m_OrderedBones) {
        if (!entry->joint) continue;
        
        if (m_State == RagdollState::Active) {
            // Enable Drives
            entry->joint->SetDrive(Axis::X, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            entry->joint->SetDrive(Axis::Y, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            entry->joint->SetDrive(Axis::Z, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            
            // For SLERP/Quaternion drive:
            entry->joint->SetDrive(Axis::RotX, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            entry->joint->SetDrive(Axis::RotY, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            entry->joint->SetDrive(Axis::RotZ, entry->config.driveStiffness, entry->config.driveDamping, entry->config.driveForceLimit, DriveType::Target);
            
        } else if (m_State == RagdollState::Dynamic) {
             // Disable Drives for free fall
            entry->joint->SetDrive(Axis::X, 0, 0, 0, DriveType::None);
            entry->joint->SetDrive(Axis::Y, 0, 0, 0, DriveType::None);
            entry->joint->SetDrive(Axis::Z, 0, 0, 0, DriveType::None);
             // Also rot axes
            entry->joint->SetDrive(Axis::RotX, 0, 0, 0, DriveType::None);
            entry->joint->SetDrive(Axis::RotY, 0, 0, 0, DriveType::None);
            entry->joint->SetDrive(Axis::RotZ, 0, 0, 0, DriveType::None);
        }
    }
}

void PhysXRagdoll::Update(float deltaTime) {
    // If Kinematic, user should be calling MatchAnimation manually or we do it here if we had ref to Animator
}

void PhysXRagdoll::MatchAnimation(Animator* animator) {
    if (!animator || !m_Articulation) return;

    // Hard sync: Set global pose of every link to match animation
    const std::vector<Mat4>& globalTransforms = animator->GetGlobalTransforms();
    
    for (auto* entry : m_OrderedBones) {
        if (entry->boneIndex >= globalTransforms.size()) continue;
        
        Mat4 transform = globalTransforms[entry->boneIndex];
        
        Vec3 pos; Quat rot; Vec3 scale;
        DecomposeMatrix(transform, pos, rot, scale); // Defined in Animation.cpp, need to expose or reimplement
        
        // We can expose DecomposeMatrix in Animation.h or util
        // For now assuming we have access or use direct math
        
        entry->link->SyncTransformToPhysics(pos, rot);
        
        // Reset velocities to zero to prevent explosions when switching back to dynamic
        entry->link->SetLinearVelocity(Vec3(0));
        entry->link->SetAngularVelocity(Vec3(0));
    }
    
    // Also reset articulation internal velocities/accelerations if possible
    m_Articulation->PutToSleep(); // Or clear cache
}

void PhysXRagdoll::DriveToAnimation(Animator* animator) {
    if (!animator || !m_Articulation) return;
    if (m_State != RagdollState::Active) return;

    const std::vector<Mat4>& globalTransforms = animator->GetGlobalTransforms();
    const std::vector<Mat4>& localTransforms = animator->GetLocalTransforms(); // Drive Targets are usually local?

    // Articulation Drives drive the JOINT, so they need the RELATIVE transform (Local Pose)
    // BUT PhysX Reduced Coordinate Articulation drives are a bit specific. 
    // They drive the joint coordinates (DoF).
    // For spherical joints, the target is a Quaternion relative to parent frame.
    
    for (auto* entry : m_OrderedBones) {
        if (!entry->joint) continue; // Skip root
        if (entry->boneIndex >= localTransforms.size()) continue;

        Mat4 localTransform = localTransforms[entry->boneIndex];
        
        Vec3 pos; Quat rot; Vec3 scale;
        DecomposeMatrix(localTransform, pos, rot, scale);
        
        // PhysX Articulation Drive Target is relative to parent frame by default for spherical joints
        // This rot from localTransform is the rotation relative to parent bone.
        // This matches exactly what the joint drive expects!
        
        entry->joint->SetDriveTarget(rot);
    }
}

void PhysXRagdoll::ApplyPoseToSkeleton(std::vector<Mat4>& outGlobalTransforms) {
    // Read back physics poses to skeleton global transforms
    if (!m_Articulation) return;
    
    for (auto* entry : m_OrderedBones) {
        if (entry->boneIndex >= outGlobalTransforms.size()) continue;
        
        Vec3 pos; Quat rot;
        entry->link->SyncTransformFromPhysics(pos, rot);
        
        outGlobalTransforms[entry->boneIndex] = ComposeMatrix(pos, rot, Vec3(1,1,1));
    }
}

PhysXArticulationLink* PhysXRagdoll::GetLink(const std::string& boneName) const {
    auto it = m_Bones.find(boneName);
    if (it != m_Bones.end()) {
        return it->second.link;
    }
    return nullptr;
}
