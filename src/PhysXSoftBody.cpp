#include "PhysXSoftBody.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "PhysXRigidBody.h"
#include "SoftBodyTearSystem.h"
#include "TetrahedralMeshSplitter.h"
#include "TearResistanceMap.h"
#include "FractureLine.h"
#include "SoftBodyTearPattern.h"
#include "StraightTearPattern.h"
#include "CurvedTearPattern.h"
#include "RadialTearPattern.h"
#include "PhysXManager.h"
#include "GLExtensions.h" 
#include <PxPhysicsAPI.h>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace physx;

PhysXSoftBody::PhysXSoftBody(PhysXBackend* backend)
    : m_Backend(backend)
    , m_SoftBody(nullptr)
    , m_TetraMesh(nullptr)
    , m_SoftBodyMesh(nullptr)
    , m_Enabled(true)
    , m_Tearable(false)
    , m_VolumeStiffness(0.5f)
    , m_ShapeStiffness(0.5f)
    , m_DeformationStiffness(0.5f)
    , m_MaxStretch(1.5f)
    , m_MaxCompress(0.5f)
    , m_LinearDamping(0.01f)
    , m_AngularDamping(0.01f)
    , m_CollisionMargin(0.01f)
    , m_SceneCollisionEnabled(true)
    , m_SelfCollisionEnabled(false)
    , m_VertexCount(0)
    , m_TetrahedronCount(0)
    , m_TearThreshold(2.0f)
    , m_CheckTearing(false)
    , m_LastTearCheckTime(0.0f)
    , m_TearCheckInterval(0.1f)
    , m_ClothCollisionEnabled(false)
    , m_CollisionSphereRadius(0.1f)
    , m_CollisionSpheresNeedUpdate(true)
    , m_UseAdaptiveSphereCount(true)
    , m_MinCollisionSpheres(4)
    , m_MaxCollisionSpheres(32)
    , m_VerticesPerSphere(50)
    , m_AdaptiveVertexWeight(0.5f)
    , m_AdaptiveAreaWeight(0.5f)
    , m_AreaPerSphere(1.0f)
    , m_CachedSurfaceArea(0.0f)
    , m_SurfaceAreaNeedsUpdate(true)
    , m_SurfaceAreaMode(SurfaceAreaMode::BoundingBox)
    , m_HullAlgorithm(ConvexHullAlgorithm::QuickHull)
{
    m_TearSystem = std::make_unique<SoftBodyTearSystem>();
}

PhysXSoftBody::~PhysXSoftBody() {
    if (m_SoftBody) {
        PxScene* scene = m_Backend->GetScene();
        if (scene) {
            scene->removeActor(*m_SoftBody);
        }
        m_SoftBody->release();
        m_SoftBody = nullptr;
    }
    
    if (m_TetraMesh) {
        m_TetraMesh->release();
        m_TetraMesh = nullptr;
    }
    
    if (m_SoftBodyMesh) {
        m_SoftBodyMesh->release();
        m_SoftBodyMesh = nullptr;
    }
    
    // Clean up collision shapes
    for (auto& shape : m_CollisionShapes) {
        if (shape.shape) {
            shape.shape->release();
        }
    }
    m_CollisionShapes.clear();
}

void PhysXSoftBody::Initialize(const SoftBodyDesc& desc) {
    if (!m_Backend || !m_Backend->GetPhysics() || !m_Backend->GetScene()) {
        std::cerr << "PhysXSoftBody: Invalid backend or scene!" << std::endl;
        return;
    }

    m_VertexCount = desc.vertexCount;
    
    // Store initial positions
    m_InitialPositions.resize(m_VertexCount);
    for (int i = 0; i < m_VertexCount; ++i) {
        m_InitialPositions[i] = desc.vertexPositions[i];
    }
    
    // Store configuration
    m_VolumeStiffness = desc.volumeStiffness;
    m_ShapeStiffness = desc.shapeStiffness;
    m_DeformationStiffness = desc.deformationStiffness;
    m_MaxStretch = desc.maxStretch;
    m_MaxCompress = desc.maxCompress;
    m_LinearDamping = desc.linearDamping;
    m_AngularDamping = desc.angularDamping;
    m_SceneCollisionEnabled = desc.enableSceneCollision;
    m_SelfCollisionEnabled = desc.enableSelfCollision;
    m_CollisionMargin = desc.collisionMargin;
    
    // Create tetrahedral mesh
    CreateTetrahedralMesh(desc);
    
    if (!m_TetraMesh) {
        std::cerr << "PhysXSoftBody: Failed to create tetrahedral mesh!" << std::endl;
        return;
    }
    
    // Create soft body actor
    PxPhysics* physics = m_Backend->GetPhysics();
    PxScene* scene = m_Backend->GetScene();
    
    // Create soft body material
    PxPBDMaterial* material = physics->createPBDMaterial(
        m_VolumeStiffness,      // Volume stiffness
        m_ShapeStiffness,       // Shape stiffness
        m_DeformationStiffness  // Deformation stiffness
    );
    
    if (!material) {
        std::cerr << "PhysXSoftBody: Failed to create PBD material!" << std::endl;
        return;
    }
    
    // Create soft body from tetrahedral mesh
    PxTransform transform(PxVec3(0, 0, 0));
    m_SoftBody = physics->createSoftBody(
        *m_TetraMesh,
        transform,
        *material,
        desc.solverIterations
    );
    
    if (!m_SoftBody) {
        std::cerr << "PhysXSoftBody: Failed to create soft body!" << std::endl;
        material->release();
        return;
    }
    
    // Configure soft body properties
    m_SoftBody->setDamping(m_LinearDamping);
    
    // Set gravity
    m_SoftBody->setExternalAcceleration(PxVec3(desc.gravity.x, desc.gravity.y, desc.gravity.z));
    
    // Configure collision
    if (m_SceneCollisionEnabled) {
        m_SoftBody->setSoftBodyFlag(PxSoftBodyFlag::eDISABLE_SELF_COLLISION, !m_SelfCollisionEnabled);
    }
    
    // Set mass
    if (desc.useDensity) {
        m_SoftBody->setDensity(desc.density);
    } else {
        // Calculate density from total mass and volume
        float volume = GetVolume();
        if (volume > 0.0f) {
            float density = desc.totalMass / volume;
            m_SoftBody->setDensity(density);
        }
    }
    
    // Add to scene
    scene->addActor(*m_SoftBody);
    
    // Initialize resistance map
    m_ResistanceMap.Initialize(m_TetrahedronCount, 1.0f);
    
    std::cout << "PhysXSoftBody initialized with " << m_VertexCount << " vertices and "
              << m_TetrahedronCount << " tetrahedra" << std::endl;
}

void PhysXSoftBody::CreateTetrahedralMesh(const SoftBodyDesc& desc) {
    PxPhysics* physics = m_Backend->GetPhysics();
    
    // If tetrahedral mesh is provided, use it
    if (desc.tetrahedronVertices && desc.tetrahedronIndices) {
        m_TetrahedronCount = desc.tetrahedronCount;
        
        // Create tetrahedral mesh descriptor
        PxTetrahedronMeshDesc meshDesc;
        meshDesc.points.count = desc.tetrahedronVertexCount;
        meshDesc.points.stride = sizeof(Vec3);
        meshDesc.points.data = desc.tetrahedronVertices;
        
        meshDesc.tetrahedrons.count = desc.tetrahedronCount;
        meshDesc.tetrahedrons.stride = sizeof(int) * 4;
        meshDesc.tetrahedrons.data = desc.tetrahedronIndices;
        
        // Cook tetrahedral mesh
        PxCooking* cooking = PxCreateCooking(PX_PHYSICS_VERSION, physics->getFoundation(), PxCookingParams(PxTolerancesScale()));
        if (cooking) {
            m_TetraMesh = cooking->createTetrahedronMesh(meshDesc, physics->getPhysicsInsertionCallback());
            cooking->release();
        }
    } else {
        // Auto-generate tetrahedral mesh from surface mesh
        // For now, use a simple voxel-based approach
        std::cerr << "PhysXSoftBody: Auto-generation of tetrahedral mesh not yet implemented!" << std::endl;
        std::cerr << "Please provide tetrahedral mesh data in SoftBodyDesc." << std::endl;
        
        // TODO: Implement voxel-based or Delaunay tetrahedralization
        // This would require:
        // 1. Voxelize the surface mesh
        // 2. Create tetrahedra from voxel grid
        // 3. Map surface vertices to tetrahedral vertices
    }
}

void PhysXSoftBody::Update(float deltaTime) {
    if (!m_SoftBody || !m_Enabled) {
        return;
    }
    
    // Mark collision spheres and surface area as needing update (soft body has deformed)
    if (m_ClothCollisionEnabled) {
        m_CollisionSpheresNeedUpdate = true;
        m_SurfaceAreaNeedsUpdate = true;
    }
    
    // Update collision shapes if needed
    UpdateCollisionShapes();
    
    // Update healing
    if (m_TearSystem) {
        m_TearSystem->UpdateHealing(deltaTime, m_ResistanceMap);
    }
    
    // Update plasticity
    if (m_TearSystem && !m_RestPositions.empty()) {
        std::vector<Vec3> currentPositions(m_VertexCount);
        GetVertexPositions(currentPositions.data());
        
        m_TearSystem->UpdatePlasticity(
            currentPositions.data(),
            m_RestPositions.data(),
            m_TetrahedronIndices.data(),
            m_TetrahedronCount,
            m_TearThreshold
        );
    }
    
    // PhysX soft body simulation is handled automatically by the scene
    // We just need to read back the results when needed
}

void PhysXSoftBody::UpdateCollisionShapes() {
    if (!m_SoftBody || m_CollisionShapes.empty()) {
        return;
    }
    
    // Update collision shape transforms if they're attached to moving objects
    // This is handled automatically by PhysX if shapes are attached to actors
}

void PhysXSoftBody::SetEnabled(bool enabled) {
    m_Enabled = enabled;
    if (m_SoftBody) {
        m_SoftBody->setActorFlag(PxActorFlag::eDISABLE_SIMULATION, !enabled);
    }
}

bool PhysXSoftBody::IsEnabled() const {
    return m_Enabled;
}

int PhysXSoftBody::GetVertexCount() const {
    return m_VertexCount;
}

void PhysXSoftBody::GetVertexPositions(Vec3* positions) const {
    if (!m_SoftBody || !positions) {
        return;
    }
    
    // Read vertex positions from soft body
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        const PxVec4* vertexPositions = data->getPositionInvMass();
        for (int i = 0; i < m_VertexCount; ++i) {
            positions[i].x = vertexPositions[i].x;
            positions[i].y = vertexPositions[i].y;
            positions[i].z = vertexPositions[i].z;
        }
    }
}

void PhysXSoftBody::SetVertexPositions(const Vec3* positions) {
    if (!m_SoftBody || !positions) {
        return;
    }
    
    // Set vertex positions
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* vertexPositions = data->getPositionInvMass();
        for (int i = 0; i < m_VertexCount; ++i) {
            vertexPositions[i].x = positions[i].x;
            vertexPositions[i].y = positions[i].y;
            vertexPositions[i].z = positions[i].z;
            // Keep existing inverse mass (w component)
        }
    }
}

void PhysXSoftBody::GetVertexNormals(Vec3* normals) const {
    if (!normals) {
        return;
    }
    
    // Calculate normals from deformed mesh
    // This is a simplified version - for production, use the surface mesh
    RecalculateNormals(normals);
}

void PhysXSoftBody::RecalculateNormals(Vec3* normals) const {
    // Reset normals
    std::memset(normals, 0, m_VertexCount * sizeof(Vec3));
    
    // Get current positions
    std::vector<Vec3> positions(m_VertexCount);
    GetVertexPositions(positions.data());
    
    // Calculate face normals and accumulate
    // Note: This requires surface triangle data, which we don't have in the current implementation
    // For now, just set default normals
    for (int i = 0; i < m_VertexCount; ++i) {
        normals[i] = Vec3(0, 1, 0); // Default up normal
    }
}

void PhysXSoftBody::GetVertexVelocities(Vec3* velocities) const {
    if (!m_SoftBody || !velocities) {
        return;
    }
    
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        const PxVec4* vertexVelocities = data->getVelocity();
        for (int i = 0; i < m_VertexCount; ++i) {
            velocities[i].x = vertexVelocities[i].x;
            velocities[i].y = vertexVelocities[i].y;
            velocities[i].z = vertexVelocities[i].z;
        }
    }
}

void PhysXSoftBody::SetVolumeStiffness(float stiffness) {
    m_VolumeStiffness = stiffness;
    if (m_SoftBody) {
        // Update material properties
        PxPBDMaterial* material = m_SoftBody->getMaterial();
        if (material) {
            material->setVolumeStiffness(stiffness);
        }
    }
}

float PhysXSoftBody::GetVolumeStiffness() const {
    return m_VolumeStiffness;
}

void PhysXSoftBody::SetShapeStiffness(float stiffness) {
    m_ShapeStiffness = stiffness;
    if (m_SoftBody) {
        PxPBDMaterial* material = m_SoftBody->getMaterial();
        if (material) {
            material->setShapeMatchingStiffness(stiffness);
        }
    }
}

float PhysXSoftBody::GetShapeStiffness() const {
    return m_ShapeStiffness;
}

void PhysXSoftBody::SetDeformationStiffness(float stiffness) {
    m_DeformationStiffness = stiffness;
    if (m_SoftBody) {
        PxPBDMaterial* material = m_SoftBody->getMaterial();
        if (material) {
            material->setDeformationStiffness(stiffness);
        }
    }
}

float PhysXSoftBody::GetDeformationStiffness() const {
    return m_DeformationStiffness;
}

void PhysXSoftBody::SetMaxStretch(float maxStretch) {
    m_MaxStretch = maxStretch;
    // PhysX handles this through material properties
}

void PhysXSoftBody::SetMaxCompress(float maxCompress) {
    m_MaxCompress = maxCompress;
    // PhysX handles this through material properties
}

void PhysXSoftBody::SetDamping(float linear, float angular) {
    m_LinearDamping = linear;
    m_AngularDamping = angular;
    if (m_SoftBody) {
        m_SoftBody->setDamping(linear);
    }
}

void PhysXSoftBody::AddForce(const Vec3& force) {
    if (!m_SoftBody) {
        return;
    }
    
    // Apply force to all vertices
    PxVec3 pxForce(force.x, force.y, force.z);
    m_SoftBody->addForce(pxForce);
}

void PhysXSoftBody::AddForceAtVertex(int vertexIndex, const Vec3& force) {
    if (!m_SoftBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    // Apply force to specific vertex
    PxVec3 pxForce(force.x, force.y, force.z);
    m_SoftBody->addForce(pxForce, vertexIndex);
}

void PhysXSoftBody::AddImpulse(const Vec3& impulse) {
    if (!m_SoftBody) {
        return;
    }
    
    // Apply impulse to all vertices
    PxVec3 pxImpulse(impulse.x, impulse.y, impulse.z);
    
    // Get vertex data and apply impulse manually
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* velocities = data->getVelocity();
        const PxVec4* masses = data->getPositionInvMass();
        
        for (int i = 0; i < m_VertexCount; ++i) {
            float invMass = masses[i].w;
            if (invMass > 0.0f) {
                float mass = 1.0f / invMass;
                PxVec3 deltaV = pxImpulse * invMass;
                velocities[i].x += deltaV.x;
                velocities[i].y += deltaV.y;
                velocities[i].z += deltaV.z;
            }
        }
    }
}

void PhysXSoftBody::AddImpulseAtVertex(int vertexIndex, const Vec3& impulse) {
    if (!m_SoftBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* velocities = data->getVelocity();
        const PxVec4* masses = data->getPositionInvMass();
        
        float invMass = masses[vertexIndex].w;
        if (invMass > 0.0f) {
            PxVec3 pxImpulse(impulse.x, impulse.y, impulse.z);
            PxVec3 deltaV = pxImpulse * invMass;
            velocities[vertexIndex].x += deltaV.x;
            velocities[vertexIndex].y += deltaV.y;
            velocities[vertexIndex].z += deltaV.z;
        }
    }
}

void PhysXSoftBody::AttachVertexToRigidBody(int vertexIndex, IPhysicsRigidBody* rigidBody, const Vec3& localPos) {
    if (!m_SoftBody || !rigidBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    // Get PhysX rigid body actor
    PhysXRigidBody* physxRigidBody = static_cast<PhysXRigidBody*>(rigidBody);
    PxRigidActor* actor = static_cast<PxRigidActor*>(physxRigidBody->GetNativeBody());
    
    if (actor) {
        // Create attachment
        PxVec3 pxLocalPos(localPos.x, localPos.y, localPos.z);
        m_SoftBody->attachShape(*actor, pxLocalPos, vertexIndex);
    }
}

void PhysXSoftBody::DetachVertex(int vertexIndex) {
    if (!m_SoftBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    // Detach vertex by setting it to free (restore normal mass)
    UnfixVertex(vertexIndex);
}

void PhysXSoftBody::FixVertex(int vertexIndex, const Vec3& worldPos) {
    if (!m_SoftBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    // Set vertex to infinite mass (fixed)
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* positions = data->getPositionInvMass();
        positions[vertexIndex].x = worldPos.x;
        positions[vertexIndex].y = worldPos.y;
        positions[vertexIndex].z = worldPos.z;
        positions[vertexIndex].w = 0.0f; // Infinite mass
        
        m_FixedVertices.push_back(vertexIndex);
    }
}

void PhysXSoftBody::UnfixVertex(int vertexIndex) {
    if (!m_SoftBody || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return;
    }
    
    // Restore normal mass
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* positions = data->getPositionInvMass();
        positions[vertexIndex].w = 1.0f; // Normal mass (inverse mass = 1.0)
        
        // Remove from fixed vertices list
        auto it = std::find(m_FixedVertices.begin(), m_FixedVertices.end(), vertexIndex);
        if (it != m_FixedVertices.end()) {
            m_FixedVertices.erase(it);
        }
    }
}

void PhysXSoftBody::SetSceneCollision(bool enabled) {
    m_SceneCollisionEnabled = enabled;
    if (m_SoftBody) {
        m_SoftBody->setSoftBodyFlag(PxSoftBodyFlag::eDISABLE_SELF_COLLISION, !enabled);
    }
}

void PhysXSoftBody::SetSelfCollision(bool enabled) {
    m_SelfCollisionEnabled = enabled;
    if (m_SoftBody) {
        m_SoftBody->setSoftBodyFlag(PxSoftBodyFlag::eDISABLE_SELF_COLLISION, !enabled);
    }
}

void PhysXSoftBody::SetCollisionMargin(float margin) {
    m_CollisionMargin = margin;
    // PhysX handles collision margin internally
}

void PhysXSoftBody::AddCollisionSphere(const Vec3& center, float radius) {
    if (!m_SoftBody) {
        return;
    }
    
    PxPhysics* physics = m_Backend->GetPhysics();
    PxMaterial* material = m_Backend->GetDefaultMaterial();
    
    // Create sphere shape
    PxShape* shape = physics->createShape(
        PxSphereGeometry(radius),
        *material,
        true // exclusive
    );
    
    if (shape) {
        // Create static actor for the collision shape
        PxTransform transform(PxVec3(center.x, center.y, center.z));
        PxRigidStatic* staticActor = physics->createRigidStatic(transform);
        staticActor->attachShape(*shape);
        
        m_Backend->GetScene()->addActor(*staticActor);
        
        CollisionShape collisionShape;
        collisionShape.type = 0; // Sphere
        collisionShape.pos0 = center;
        collisionShape.radius = radius;
        collisionShape.shape = shape;
        
        m_CollisionShapes.push_back(collisionShape);
    }
}

void PhysXSoftBody::AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) {
    if (!m_SoftBody) {
        return;
    }
    
    PxPhysics* physics = m_Backend->GetPhysics();
    PxMaterial* material = m_Backend->GetDefaultMaterial();
    
    // Calculate capsule parameters
    Vec3 axis = p1 - p0;
    float halfHeight = axis.Length() * 0.5f;
    
    // Create capsule shape
    PxShape* shape = physics->createShape(
        PxCapsuleGeometry(radius, halfHeight),
        *material,
        true // exclusive
    );
    
    if (shape) {
        // Calculate transform
        Vec3 center = (p0 + p1) * 0.5f;
        PxTransform transform(PxVec3(center.x, center.y, center.z));
        
        PxRigidStatic* staticActor = physics->createRigidStatic(transform);
        staticActor->attachShape(*shape);
        
        m_Backend->GetScene()->addActor(*staticActor);
        
        CollisionShape collisionShape;
        collisionShape.type = 1; // Capsule
        collisionShape.pos0 = p0;
        collisionShape.pos1 = p1;
        collisionShape.radius = radius;
        collisionShape.shape = shape;
        
        m_CollisionShapes.push_back(collisionShape);
    }
}

void PhysXSoftBody::SetTearable(bool tearable) {
    m_Tearable = tearable;
    // Tearing implementation would require mesh splitting logic
}

bool PhysXSoftBody::TearAtVertex(int vertexIndex) {
    if (!m_Tearable || vertexIndex < 0 || vertexIndex >= m_VertexCount) {
        return false;
    }
    
    // Tearing implementation would require:
    // 1. Duplicate vertex
    // 2. Split connected tetrahedra
    // 3. Update topology
    // This is complex and would be implemented in a future enhancement
    
    std::cerr << "PhysXSoftBody: Tearing not yet implemented!" << std::endl;
    return false;
}

float PhysXSoftBody::GetTotalMass() const {
    if (!m_SoftBody) {
        return 0.0f;
    }
    
    // Calculate total mass from vertices
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        const PxVec4* masses = data->getPositionInvMass();
        float totalMass = 0.0f;
        
        for (int i = 0; i < m_VertexCount; ++i) {
            float invMass = masses[i].w;
            if (invMass > 0.0f) {
                totalMass += 1.0f / invMass;
            }
        }
        
        return totalMass;
    }
    
    return 0.0f;
}

float PhysXSoftBody::GetVolume() const {
    // Calculate volume from tetrahedral mesh
    // This is a simplified calculation
    if (m_TetrahedronCount == 0) {
        return 1.0f; // Default volume
    }
    
    // For accurate volume calculation, we would need to sum the volumes
    // of all tetrahedra in the mesh
    // Volume of tetrahedron = |det(v1-v0, v2-v0, v3-v0)| / 6
    
    return static_cast<float>(m_TetrahedronCount) * 0.001f; // Approximate
}

Vec3 PhysXSoftBody::GetCenterOfMass() const {
    if (!m_SoftBody) {
        return Vec3(0, 0, 0);
    }
    
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        const PxVec4* positions = data->getPositionInvMass();
        const PxVec4* masses = data->getPositionInvMass();
        
        Vec3 centerOfMass(0, 0, 0);
        float totalMass = 0.0f;
        
        for (int i = 0; i < m_VertexCount; ++i) {
            float invMass = masses[i].w;
            if (invMass > 0.0f) {
                float mass = 1.0f / invMass;
                centerOfMass.x += positions[i].x * mass;
                centerOfMass.y += positions[i].y * mass;
                centerOfMass.z += positions[i].z * mass;
                totalMass += mass;
            }
        }
        
        if (totalMass > 0.0f) {
            centerOfMass = centerOfMass * (1.0f / totalMass);
        }
        
        return centerOfMass;
    }
    
    return Vec3(0, 0, 0);
}

bool PhysXSoftBody::IsActive() const {
    if (m_SoftBody) {
        return !m_SoftBody->isSleeping();
    }
    return false;
}

void PhysXSoftBody::SetActive(bool active) {
    if (m_SoftBody) {
        if (active) {
            m_SoftBody->wakeUp();
        } else {
            m_SoftBody->putToSleep();
        }
    }
}

void* PhysXSoftBody::GetNativeSoftBody() {
    return m_SoftBody;
}

void PhysXSoftBody::SetTearThreshold(float threshold) {
    m_TearThreshold = threshold;
    m_CheckTearing = (threshold > 0.0f);
}

float PhysXSoftBody::GetTearThreshold() const {
    return m_TearThreshold;
}

void PhysXSoftBody::SetTearCallback(std::function<void(int, float)> callback) {
    m_TearCallback = callback;
}

void PhysXSoftBody::SetPieceCreatedCallback(std::function<void(std::shared_ptr<PhysXSoftBody>)> callback) {
    m_PieceCreatedCallback = callback;
}

void PhysXSoftBody::DetectAndProcessTears() {
    if (!m_CheckTearing || !m_TearSystem || m_RestPositions.empty()) {
        return;
    }
    
    // Get current positions
    std::vector<Vec3> currentPositions(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    
    // Detect tears
    std::vector<SoftBodyTearSystem::TearInfo> tears;
    m_TearSystem->DetectStress(
        currentPositions.data(),
        m_RestPositions.data(),
        m_TetrahedronIndices.data(),
        m_TetrahedronCount,
        m_TearThreshold,
        &m_ResistanceMap,  // Pass resistance map
        tears
    );
    
    if (!tears.empty()) {
        std::cout << "Detected " << tears.size() << " tears" << std::endl;
        
        // Notify callback
        if (m_TearCallback) {
            for (const auto& tear : tears) {
                m_TearCallback(tear.tetrahedronIndex, tear.stress);
            }
        }
        
        // Collect torn tetrahedra
        std::vector<int> tornTets;
        for (const auto& tear : tears) {
            tornTets.push_back(tear.tetrahedronIndex);
        }
        
        // Split soft body
        SplitSoftBody(tornTets);
    }
}

void PhysXSoftBody::SplitSoftBody(const std::vector<int>& tornTetrahedra) {
    std::cout << "Splitting soft body along " << tornTetrahedra.size() << " torn tetrahedra" << std::endl;
    
    // For now, just log the tear - full implementation would:
    // 1. Use TetrahedralMeshSplitter to split the mesh
    // 2. Create new PhysXSoftBody for the torn piece
    // 3. Update current soft body with remaining mesh
    // 4. Call piece created callback
    
    // This is a complex operation that requires:
    // - Mesh topology analysis
    // - Vertex duplication
    // - PhysX actor recreation
    // - State transfer
    
    std::cerr << "Full mesh splitting not yet implemented - tears detected but not applied" << std::endl;
}

void PhysXSoftBody::RecreateWithMesh(const std::vector<Vec3>& vertices, const std::vector<int>& tetrahedra) {
    // This would recreate the PhysX soft body with new mesh topology
    // Complex operation requiring:
    // 1. Store current state (velocities, forces)
    // 2. Release current PhysX actor
    // 3. Create new tetrahedral mesh
    // 4. Create new soft body actor
    // 5. Restore state
    
    std::cerr << "Mesh recreation not yet implemented" << std::endl;
}

void PhysXSoftBody::TearAlongPattern(
    const SoftBodyTearPattern& pattern,
    const Vec3& startPoint,
    const Vec3& endPoint)
{
    if (!m_SoftBody || m_RestPositions.empty()) {
        std::cerr << "Cannot tear: soft body not initialized" << std::endl;
        return;
    }
    
    // Get current vertex positions
    std::vector<Vec3> currentPositions(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    
    // Select tetrahedra along pattern
    std::vector<int> selectedTets = pattern.SelectTetrahedra(
        currentPositions.data(),
        m_VertexCount,
        m_TetrahedronIndices.data(),
        m_TetrahedronCount,
        startPoint,
        endPoint
    );
    
    if (selectedTets.empty()) {
        std::cout << "No tetrahedra selected by pattern" << std::endl;
        return;
    }
    
    std::cout << "Tearing along pattern: " << selectedTets.size() << " tetrahedra" << std::endl;
    
    // Notify tear callback for each selected tet
    if (m_TearCallback) {
        for (int tetIdx : selectedTets) {
            m_TearCallback(tetIdx, m_TearThreshold);
        }
    }
    
    // Register tears for healing
    if (m_TearSystem) {
        for (int tetIdx : selectedTets) {
            float originalResistance = m_ResistanceMap.GetResistance(tetIdx);
            m_TearSystem->RegisterTearForHealing(tetIdx, originalResistance);
        }
    }
    
    // Split soft body
    SplitSoftBody(selectedTets);
}

void PhysXSoftBody::TearStraightLine(const Vec3& start, const Vec3& end, float width)
{
    StraightTearPattern pattern(width);
    TearAlongPattern(pattern, start, end);
}

void PhysXSoftBody::TearCurvedPath(const Vec3& start, const Vec3& end, float curvature)
{
    CurvedTearPattern pattern(0.1f, curvature);
    TearAlongPattern(pattern, start, end);
}

void PhysXSoftBody::TearRadialBurst(const Vec3& center, int rayCount, float radius)
{
    RadialTearPattern pattern(rayCount, 0.05f);
    pattern.SetRadius(radius);
    
    // Use up vector as direction
    Vec3 upVector(0, 1, 0);
    TearAlongPattern(pattern, center, upVector);
}

void PhysXSoftBody::SetRegionResistance(const Vec3& center, float radius, float resistance)
{
    if (!m_ResistanceMap.IsInitialized()) {
        std::cerr << "Resistance map not initialized" << std::endl;
        return;
    }
    
    // Calculate tetrahedron centers
    std::vector<Vec3> tetCenters(m_TetrahedronCount);
    for (int i = 0; i < m_TetrahedronCount; ++i) {
        int v0 = m_TetrahedronIndices[i * 4 + 0];
        int v1 = m_TetrahedronIndices[i * 4 + 1];
        int v2 = m_TetrahedronIndices[i * 4 + 2];
        int v3 = m_TetrahedronIndices[i * 4 + 3];
        
        tetCenters[i] = (m_RestPositions[v0] + m_RestPositions[v1] + 
                        m_RestPositions[v2] + m_RestPositions[v3]) * 0.25f;
    }
    
    m_ResistanceMap.SetSphereResistance(
        tetCenters.data(),
        m_TetrahedronCount,
        center,
        radius,
        resistance
    );
}

void PhysXSoftBody::SetResistanceGradient(
    const Vec3& start, const Vec3& end,
    float startResistance, float endResistance)
{
    if (!m_ResistanceMap.IsInitialized()) {
        std::cerr << "Resistance map not initialized" << std::endl;
        return;
    }
    
    // Calculate tetrahedron centers
    std::vector<Vec3> tetCenters(m_TetrahedronCount);
    for (int i = 0; i < m_TetrahedronCount; ++i) {
        int v0 = m_TetrahedronIndices[i * 4 + 0];
        int v1 = m_TetrahedronIndices[i * 4 + 1];
        int v2 = m_TetrahedronIndices[i * 4 + 2];
        int v3 = m_TetrahedronIndices[i * 4 + 3];
        
        tetCenters[i] = (m_RestPositions[v0] + m_RestPositions[v1] + 
                        m_RestPositions[v2] + m_RestPositions[v3]) * 0.25f;
    }
    
    m_ResistanceMap.SetGradient(
        tetCenters.data(),
        m_TetrahedronCount,
        start,
        end,
        startResistance,
        endResistance
    );
}

void PhysXSoftBody::SetHealingEnabled(bool enabled) {
    if (m_TearSystem) {
        m_TearSystem->SetHealingEnabled(enabled);
        std::cout << "Healing " << (enabled ? "enabled" : "disabled") << std::endl;
    }
}

void PhysXSoftBody::SetHealingRate(float progressPerSecond) {
    if (m_TearSystem) {
        m_TearSystem->SetHealingRate(progressPerSecond);
        std::cout << "Healing rate set to " << progressPerSecond << "/sec" << std::endl;
    }
}

void PhysXSoftBody::SetHealingDelay(float seconds) {
    if (m_TearSystem) {
        m_TearSystem->SetHealingDelay(seconds);
        std::cout << "Healing delay set to " << seconds << " seconds" << std::endl;
    }
}

int PhysXSoftBody::GetHealingTearCount() const {
    if (m_TearSystem) {
        return static_cast<int>(m_TearSystem->GetHealingTears().size());
    }
    return 0;
}

void PhysXSoftBody::SetPlasticityEnabled(bool enabled) {
    if (m_TearSystem) {
        m_TearSystem->SetPlasticityEnabled(enabled);
        std::cout << "Plasticity " << (enabled ? "enabled" : "disabled") << std::endl;
    }
}

void PhysXSoftBody::SetPlasticThreshold(float threshold) {
    if (m_TearSystem) {
        m_TearSystem->SetPlasticThreshold(threshold);
        std::cout << "Plastic threshold set to " << threshold << std::endl;
    }
}

void PhysXSoftBody::SetPlasticityRate(float rate) {
    if (m_TearSystem) {
        m_TearSystem->SetPlasticityRate(rate);
        std::cout << "Plasticity rate set to " << rate << std::endl;
    }
}

void PhysXSoftBody::ResetRestShape() {
    if (m_InitialPositions.empty()) {
        std::cerr << "No initial positions to reset to" << std::endl;
        return;
    }
    
    // Reset rest positions to initial positions
    m_RestPositions = m_InitialPositions;
    std::cout << "Rest shape reset to original" << std::endl;
}

void PhysXSoftBody::AddFractureLine(const FractureLine& fractureLine) {
    if (!m_ResistanceMap.IsInitialized()) {
        std::cerr << "Resistance map not initialized" << std::endl;
        return;
    }
    
    // Calculate tetrahedron centers
    std::vector<Vec3> tetCenters(m_TetrahedronCount);
    for (int i = 0; i < m_TetrahedronCount; ++i) {
        int v0 = m_TetrahedronIndices[i * 4 + 0];
        int v1 = m_TetrahedronIndices[i * 4 + 1];
        int v2 = m_TetrahedronIndices[i * 4 + 2];
        int v3 = m_TetrahedronIndices[i * 4 + 3];
        
        tetCenters[i] = (m_RestPositions[v0] + m_RestPositions[v1] + 
                        m_RestPositions[v2] + m_RestPositions[v3]) * 0.25f;
    }
    
    // Apply fracture line to resistance map
    fractureLine.ApplyToResistanceMap(m_ResistanceMap, tetCenters.data(), m_TetrahedronCount);
}

void PhysXSoftBody::ClearFractureLines() {
    // Reset resistance map to default
    if (m_ResistanceMap.IsInitialized()) {
        m_ResistanceMap.Reset();
        std::cout << "Fracture lines cleared" << std::endl;
    }
}

// Cloth Collision Implementation

void PhysXSoftBody::SetClothCollisionEnabled(bool enabled) {
    m_ClothCollisionEnabled = enabled;
    if (enabled) {
        m_CollisionSpheresNeedUpdate = true;
    }
}

int PhysXSoftBody::GetCollisionSpheres(std::vector<Vec3>& positions, std::vector<float>& radii, int maxSpheres) const {
    if (!m_SoftBody || !m_ClothCollisionEnabled || m_VertexCount == 0) {
        positions.clear();
        radii.clear();
        return 0;
    }
    
    // Determine target sphere count
    int targetSphereCount = maxSpheres;
    if (m_UseAdaptiveSphereCount) {
        targetSphereCount = std::min(CalculateOptimalSphereCount(), maxSpheres);
    }
    
    // Update cached collision spheres if needed
    if (m_CollisionSpheresNeedUpdate) {
        m_CachedCollisionSpherePositions.clear();
        m_CachedCollisionSphereRadii.clear();
        
        // Get current vertex positions
        std::vector<Vec3> currentPositions(m_VertexCount);
        GetVertexPositions(currentPositions.data());
        
        // Strategy: Create collision spheres by clustering vertices
        // We'll use a simple grid-based clustering approach
        
        // Calculate bounding box
        Vec3 minBounds = currentPositions[0];
        Vec3 maxBounds = currentPositions[0];
        for (int i = 1; i < m_VertexCount; ++i) {
            minBounds.x = std::min(minBounds.x, currentPositions[i].x);
            minBounds.y = std::min(minBounds.y, currentPositions[i].y);
            minBounds.z = std::min(minBounds.z, currentPositions[i].z);
            maxBounds.x = std::max(maxBounds.x, currentPositions[i].x);
            maxBounds.y = std::max(maxBounds.y, currentPositions[i].y);
            maxBounds.z = std::max(maxBounds.z, currentPositions[i].z);
        }
        
        Vec3 size = maxBounds - minBounds;
        float maxDim = std::max(std::max(size.x, size.y), size.z);
        
        // Determine grid resolution based on target sphere count
        int gridRes = static_cast<int>(std::cbrt(static_cast<float>(targetSphereCount))) + 1;
        gridRes = std::max(2, std::min(gridRes, 5)); // Clamp between 2 and 5
        
        float cellSize = maxDim / static_cast<float>(gridRes);
        
        // Create grid cells
        struct GridCell {
            std::vector<int> vertexIndices;
            Vec3 center;
            float radius;
        };
        
        std::vector<GridCell> cells;
        cells.reserve(gridRes * gridRes * gridRes);
        
        // Assign vertices to grid cells
        for (int i = 0; i < m_VertexCount; ++i) {
            Vec3 localPos = currentPositions[i] - minBounds;
            int ix = std::min(static_cast<int>(localPos.x / cellSize), gridRes - 1);
            int iy = std::min(static_cast<int>(localPos.y / cellSize), gridRes - 1);
            int iz = std::min(static_cast<int>(localPos.z / cellSize), gridRes - 1);
            int cellIndex = ix + iy * gridRes + iz * gridRes * gridRes;
            
            // Ensure cell exists
            while (static_cast<int>(cells.size()) <= cellIndex) {
                cells.push_back(GridCell());
            }
            
            cells[cellIndex].vertexIndices.push_back(i);
        }
        
        // Create spheres from non-empty cells
        for (auto& cell : cells) {
            if (cell.vertexIndices.empty()) {
                continue;
            }
            
            // Calculate center of vertices in this cell
            Vec3 center(0, 0, 0);
            for (int idx : cell.vertexIndices) {
                center = center + currentPositions[idx];
            }
            center = center * (1.0f / static_cast<float>(cell.vertexIndices.size()));
            
            // Calculate radius as max distance from center
            float maxDist = 0.0f;
            for (int idx : cell.vertexIndices) {
                float dist = (currentPositions[idx] - center).Length();
                maxDist = std::max(maxDist, dist);
            }
            
            // Add padding to radius
            float radius = maxDist + m_CollisionSphereRadius;
            
            m_CachedCollisionSpherePositions.push_back(center);
            m_CachedCollisionSphereRadii.push_back(radius);
            
            // Stop if we've reached target sphere count
            if (static_cast<int>(m_CachedCollisionSpherePositions.size()) >= targetSphereCount) {
                break;
            }
        }
        
        m_CollisionSpheresNeedUpdate = false;
    }
    
    // Copy cached spheres to output
    positions = m_CachedCollisionSpherePositions;
    radii = m_CachedCollisionSphereRadii;
    
    return static_cast<int>(positions.size());
}

// Adaptive Sphere Count Implementation

int PhysXSoftBody::CalculateOptimalSphereCount() const {
    if (m_VertexCount == 0) {
        return m_MinCollisionSpheres;
    }
    
    // Vertex-based component
    float vertexSpheres = static_cast<float>(m_VertexCount) / static_cast<float>(m_VerticesPerSphere);
    
    // Surface area-based component
    float surfaceArea = CalculateSurfaceArea();
    float areaSpheres = surfaceArea / m_AreaPerSphere;
    
    // Weighted combination
    float totalSpheres = static_cast<float>(m_MinCollisionSpheres) + 
                         (vertexSpheres * m_AdaptiveVertexWeight) + 
                         (areaSpheres * m_AdaptiveAreaWeight);
    
    // Clamp to min/max range
    int sphereCount = static_cast<int>(totalSpheres + 0.5f); // Round to nearest
    sphereCount = std::max(m_MinCollisionSpheres, std::min(sphereCount, m_MaxCollisionSpheres));
    
    return sphereCount;
}

void PhysXSoftBody::SetAdaptiveSphereParams(int minSpheres, int maxSpheres, int verticesPerSphere) {
    m_MinCollisionSpheres = std::max(1, minSpheres);
    m_MaxCollisionSpheres = std::max(m_MinCollisionSpheres, std::min(maxSpheres, 32));
    m_VerticesPerSphere = std::max(1, verticesPerSphere);
    
    // Mark spheres as needing update with new parameters
    m_CollisionSpheresNeedUpdate = true;
}

void PhysXSoftBody::GetAdaptiveSphereParams(int& minSpheres, int& maxSpheres, int& verticesPerSphere) const {
    minSpheres = m_MinCollisionSpheres;
    maxSpheres = m_MaxCollisionSpheres;
    verticesPerSphere = m_VerticesPerSphere;
}

// Surface Area Calculation Implementation

float PhysXSoftBody::CalculateSurfaceArea() const {
    // Update cached surface area if needed
    if (m_SurfaceAreaNeedsUpdate) {
        if (m_SurfaceAreaMode == SurfaceAreaMode::BoundingBox) {
            m_CachedSurfaceArea = CalculateBoundingBoxArea();
        } else if (m_SurfaceAreaMode == SurfaceAreaMode::Triangles) {
            m_CachedSurfaceArea = CalculateTriangleArea();
        } else { // ConvexHull
            m_CachedSurfaceArea = CalculateConvexHullArea();
        }
        m_SurfaceAreaNeedsUpdate = false;
    }
    
    return m_CachedSurfaceArea;
}

float PhysXSoftBody::CalculateBoundingBoxArea() const {
    if (m_VertexCount == 0) {
        return 0.0f;
    }
    
    // Get current vertex positions
    std::vector<Vec3> currentPositions(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    
    // Calculate bounding box
    Vec3 minBounds = currentPositions[0];
    Vec3 maxBounds = currentPositions[0];
    for (int i = 1; i < m_VertexCount; ++i) {
        minBounds.x = std::min(minBounds.x, currentPositions[i].x);
        minBounds.y = std::min(minBounds.y, currentPositions[i].y);
        minBounds.z = std::min(minBounds.z, currentPositions[i].z);
        maxBounds.x = std::max(maxBounds.x, currentPositions[i].x);
        maxBounds.y = std::max(maxBounds.y, currentPositions[i].y);
        maxBounds.z = std::max(maxBounds.z, currentPositions[i].z);
    }
    
    // Calculate surface area using bounding box approximation
    Vec3 size = maxBounds - minBounds;
    return 2.0f * (size.x * size.y + size.y * size.z + size.z * size.x);
}

float PhysXSoftBody::CalculateTriangleArea() const {
    if (m_TetrahedronCount == 0 || m_VertexCount == 0) {
        return 0.0f;
    }
    
    // Get current vertex positions
    std::vector<Vec3> currentPositions(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    
    // Extract surface faces from tetrahedral mesh
    // A face is on the surface if it's referenced by only one tetrahedron
    
    // Use a map to count face references
    // Face key: sorted triple of vertex indices
    struct Face {
        int v0, v1, v2;
        
        Face(int a, int b, int c) {
            // Sort vertices to create canonical representation
            if (a > b) std::swap(a, b);
            if (b > c) std::swap(b, c);
            if (a > b) std::swap(a, b);
            v0 = a; v1 = b; v2 = c;
        }
        
        bool operator==(const Face& other) const {
            return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
        }
    };
    
    struct FaceHash {
        size_t operator()(const Face& f) const {
            return std::hash<int>()(f.v0) ^ (std::hash<int>()(f.v1) << 1) ^ (std::hash<int>()(f.v2) << 2);
        }
    };
    
    std::unordered_map<Face, int, FaceHash> faceCount;
    
    // Iterate through all tetrahedra and count face occurrences
    for (int i = 0; i < m_TetrahedronCount; ++i) {
        int v0 = m_TetrahedronIndices[i * 4 + 0];
        int v1 = m_TetrahedronIndices[i * 4 + 1];
        int v2 = m_TetrahedronIndices[i * 4 + 2];
        int v3 = m_TetrahedronIndices[i * 4 + 3];
        
        // Four faces of the tetrahedron
        faceCount[Face(v0, v1, v2)]++;
        faceCount[Face(v0, v1, v3)]++;
        faceCount[Face(v0, v2, v3)]++;
        faceCount[Face(v1, v2, v3)]++;
    }
    
    // Calculate total surface area from faces with count == 1
    float totalArea = 0.0f;
    for (const auto& pair : faceCount) {
        if (pair.second == 1) {  // Surface face
            const Face& face = pair.first;
            
            // Get triangle vertices
            const Vec3& p0 = currentPositions[face.v0];
            const Vec3& p1 = currentPositions[face.v1];
            const Vec3& p2 = currentPositions[face.v2];
            
            // Calculate triangle area using cross product
            Vec3 edge1 = p1 - p0;
            Vec3 edge2 = p2 - p0;
            Vec3 cross = edge1.Cross(edge2);
            float area = 0.5f * cross.Length();
            
            totalArea += area;
        }
    }
    
    return totalArea;
}

void PhysXSoftBody::SetSurfaceAreaMode(SurfaceAreaMode mode) {
    if (m_SurfaceAreaMode != mode) {
        m_SurfaceAreaMode = mode;
        m_SurfaceAreaNeedsUpdate = true;
        m_CollisionSpheresNeedUpdate = true;  // Sphere count may change
    }
}

void PhysXSoftBody::SetAdaptiveWeights(float vertexWeight, float areaWeight, float areaPerSphere) {
    // Normalize weights if they don't sum to 1.0
    float totalWeight = vertexWeight + areaWeight;
    if (totalWeight > 0.0f) {
        m_AdaptiveVertexWeight = vertexWeight / totalWeight;
        m_AdaptiveAreaWeight = areaWeight / totalWeight;
    } else {
        // Default to equal weights
        m_AdaptiveVertexWeight = 0.5f;
        m_AdaptiveAreaWeight = 0.5f;
    }
    
    m_AreaPerSphere = std::max(0.1f, areaPerSphere);
    
    // Mark spheres as needing update with new weights
    m_CollisionSpheresNeedUpdate = true;
}

void PhysXSoftBody::GetAdaptiveWeights(float& vertexWeight, float& areaWeight, float& areaPerSphere) const {
    vertexWeight = m_AdaptiveVertexWeight;
    areaWeight = m_AdaptiveAreaWeight;
    areaPerSphere = m_AreaPerSphere;
}

#include "QuickHull.h"

// ... existing includes ...

// In PhysXSoftBody.cpp

#include "GiftWrapping.h"
#include "IncrementalHull.h"
#include "DivideAndConquerHull.h"

void PhysXSoftBody::SetConvexHullAlgorithm(ConvexHullAlgorithm algo) {
    m_HullAlgorithm = algo;
}

float PhysXSoftBody::CalculateConvexHullArea() const {
    if (m_VertexCount < 4) {
        return 0.0f; 
    }
    
    // Get current vertex positions
    std::vector<Vec3> points(m_VertexCount);
    GetVertexPositions(points.data());
    
    ConvexHull hull;
    
    if (m_HullAlgorithm == ConvexHullAlgorithm::GiftWrapping) {
        // Use robust O(nh) Gift Wrapping
        GiftWrapping giftWrapping;
        giftWrapping.SetEpsilon(1e-5f);
        hull = giftWrapping.ComputeHull(points.data(), m_VertexCount);
    } else if (m_HullAlgorithm == ConvexHullAlgorithm::Incremental) {
        // Use Randomized Incremental Construction
        IncrementalHull incrementalHull;
        incrementalHull.SetEpsilon(1e-5f);
        hull = incrementalHull.ComputeHull(points.data(), m_VertexCount);
    } else if (m_HullAlgorithm == ConvexHullAlgorithm::DivideAndConquer) {
        // Use Divide and Conquer
        DivideAndConquerHull dcHull;
        dcHull.SetEpsilon(1e-5f);
        hull = dcHull.ComputeHull(points.data(), m_VertexCount);
    } else {
        // Use efficient O(n log n) QuickHull (default)
        QuickHull quickHull;
        quickHull.SetEpsilon(1e-5f);
        quickHull.SetParallel(true);
        hull = quickHull.ComputeHull(points.data(), m_VertexCount);
    }
    
    return hull.surfaceArea;
}


// -------------------------------------------------------------
// Debug Rendering
// -------------------------------------------------------------

void PhysXSoftBody::CreateDebugResources() {
    if (m_DebugResourcesInitialized) return;
    
    glGenVertexArrays(1, &m_DebugVAO);
    glGenBuffers(1, &m_DebugVBO);
    
    m_DebugResourcesInitialized = true;
}

void PhysXSoftBody::UpdateDebugBuffers() {
    if (m_VertexCount < 4) return;
    
    // Compute hull using selected algorithm
    std::vector<Vec3> points(m_VertexCount);
    GetVertexPositions(points.data());
    
    ConvexHull hull;
    
    if (m_HullAlgorithm == ConvexHullAlgorithm::GiftWrapping) {
        GiftWrapping giftWrapping;
        giftWrapping.SetEpsilon(1e-5f);
        hull = giftWrapping.ComputeHull(points.data(), m_VertexCount);
    } else if (m_HullAlgorithm == ConvexHullAlgorithm::Incremental) {
        IncrementalHull incrementalHull;
        incrementalHull.SetEpsilon(1e-5f);
        hull = incrementalHull.ComputeHull(points.data(), m_VertexCount);
    } else if (m_HullAlgorithm == ConvexHullAlgorithm::DivideAndConquer) {
        DivideAndConquerHull dcHull;
        dcHull.SetEpsilon(1e-5f);
        hull = dcHull.ComputeHull(points.data(), m_VertexCount);
    } else {
        QuickHull quickHull;
        quickHull.SetEpsilon(1e-5f);
        hull = quickHull.ComputeHull(points.data(), m_VertexCount);
    }
    
    // Generate lines from hull faces
    std::vector<Vec3> lines;
    lines.reserve(hull.indices.size() * 2); // Overestimate
    
    for (size_t i = 0; i < hull.indices.size(); i += 3) {
        Vec3 v0 = hull.vertices[hull.indices[i]];
        Vec3 v1 = hull.vertices[hull.indices[i+1]];
        Vec3 v2 = hull.vertices[hull.indices[i+2]];
        
        lines.push_back(v0); lines.push_back(v1);
        lines.push_back(v1); lines.push_back(v2);
        lines.push_back(v2); lines.push_back(v0);
    }
    
    m_DebugHullVertices = lines;
    
    // Upload to GPU
    glBindVertexArray(m_DebugVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_DebugVBO);
    glBufferData(GL_ARRAY_BUFFER, m_DebugHullVertices.size() * sizeof(Vec3), m_DebugHullVertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    
    m_DebugHullDirty = false;
}

void PhysXSoftBody::DebugRender(Shader* shader) {
    if (!m_DebugDrawHull) return;
    
    if (!m_DebugResourcesInitialized) {
        CreateDebugResources();
    }
    
    // Update every frame for soft body
    UpdateDebugBuffers();
    
    if (m_DebugHullVertices.empty()) return;
    
    // Draw
    Mat4 identity = Mat4::Identity();
    shader->SetMat4("model", identity.m);
    shader->SetVec3("color", 1.0f, 0.0f, 1.0f); // Magenta for hull
    
    glBindVertexArray(m_DebugVAO);
    glDrawArrays(GL_LINES, 0, (int)m_DebugHullVertices.size());
    glBindVertexArray(0);
}

#endif // USE_PHYSX
