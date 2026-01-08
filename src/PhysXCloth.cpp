#include "PhysXCloth.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "ClothMeshSplitter.h"
#include "Mesh.h"
#include "AsyncClothFactory.h"
#include "Profiler.h"
#include "ClothTearPattern.h"
#include "ClothTearPatternLibrary.h"
#include <PxPhysicsAPI.h>
#include <extensions/PxClothFabricCooker.h>
#include <iostream>
#include <cstring>

using namespace physx;

PhysXCloth::PhysXCloth(PhysXBackend* backend)
    : m_Backend(backend)
    , m_Cloth(nullptr)
    , m_Fabric(nullptr)
    , m_ParticleCount(0)
    , m_TriangleCount(0)
    , m_Enabled(true)
    , m_Tearable(false)
    , m_MaxStretchRatio(1.5f)
    , m_WindVelocity(0, 0, 0)
    , m_StretchStiffness(0.8f)
    , m_BendStiffness(0.5f)
    , m_ShearStiffness(0.6f)
    , m_Damping(0.2f)
    , m_TearCallback(nullptr)
    , m_CurrentLOD(0)
    , m_IsFrozen(false)
    , m_EnableSceneCollision(true)
    , m_EnableSelfCollision(false)
    , m_SelfCollisionDistance(0.01f)
    , m_SelfCollisionStiffness(1.0f)
    , m_EnableTwoWayCoupling(false)
    , m_CollisionMassScale(1.0f)
    , m_CurrentLODLevel(0)
    , m_UpdateCounter(0)
    , m_UpdateFrequency(1)
{
    m_SpatialGrid = std::make_unique<SpatialGrid<int>>(2.0f); // 2 meter grid cells
}

PhysXCloth::~PhysXCloth() {
    if (m_Cloth) {
        PxScene* scene = m_Backend->GetScene();
        if (scene) {
            scene->removeActor(*m_Cloth);
        }
        m_Cloth->release();
        m_Cloth = nullptr;
    }
    
    if (m_Fabric) {
        m_Fabric->release();
        m_Fabric = nullptr;
    }
}

void PhysXCloth::Initialize(const ClothDesc& desc) {
    if (!m_Backend || !m_Backend->GetPhysics() || !m_Backend->GetScene()) {
        std::cerr << "PhysXCloth: Invalid backend or scene!" << std::endl;
        return;
    }

    m_ParticleCount = desc.particleCount;
    m_TriangleCount = desc.triangleCount;

    // Store mesh data
    m_ParticlePositions.resize(m_ParticleCount);
    m_ParticleNormals.resize(m_ParticleCount);
    m_TriangleIndices.resize(m_TriangleCount * 3);

    for (int i = 0; i < m_ParticleCount; ++i) {
        m_ParticlePositions[i] = desc.particlePositions[i];
    }

    for (int i = 0; i < m_TriangleCount * 3; ++i) {
        m_TriangleIndices[i] = desc.triangleIndices[i];
    }

    // Store collision settings
    m_EnableSceneCollision = desc.enableSceneCollision;
    m_EnableSelfCollision = desc.enableSelfCollision;
    m_SelfCollisionDistance = desc.selfCollisionDistance;
    m_SelfCollisionStiffness = desc.selfCollisionStiffness;
    m_EnableTwoWayCoupling = desc.enableTwoWayCoupling;
    m_CollisionMassScale = desc.collisionMassScale;

    // Create PhysX cloth
    CreateClothActor(desc);

    if (!m_Fabric) {
        std::cerr << "PhysXCloth: Failed to create cloth fabric!" << std::endl;
        return;
    }

    // Create cloth particles
    std::vector<PxClothParticle> particles(m_ParticleCount);
    for (int i = 0; i < m_ParticleCount; ++i) {
        particles[i].pos = PxVec3(
            desc.particlePositions[i].x,
            desc.particlePositions[i].y,
            desc.particlePositions[i].z
        );
        particles[i].invWeight = 1.0f / desc.particleMass; // Inverse mass
    }

    // Create cloth
    PxTransform clothTransform(PxVec3(0, 0, 0));
    m_Cloth = m_Backend->GetPhysics()->createCloth(
        clothTransform,
        *m_Fabric,
        particles.data(),
        PxClothFlags()
    );

    if (!m_Cloth) {
        std::cerr << "PhysXCloth: Failed to create cloth!" << std::endl;
        return;
    }

    // Add to scene
    m_Backend->GetScene()->addActor(*m_Cloth);

    // Setup constraints and parameters
    SetupConstraints();

    // Set gravity
    m_Cloth->setExternalAcceleration(PxVec3(desc.gravity.x, desc.gravity.y, desc.gravity.z));

    // Calculate initial normals
    RecalculateNormals();
    
    // Generate LOD meshes if LOD config is set
    if (m_LODConfig.GetLODCount() == 0) {
        // Create default LOD config
        m_LODConfig = ClothLODConfig::CreateDefault(m_ParticleCount, m_TriangleCount);
    }
    
    // Generate simplified meshes for all LOD levels
    m_LODConfig.GenerateLODMeshes(m_ParticlePositions, m_TriangleIndices);

    std::cout << "PhysXCloth initialized with " << m_ParticleCount << " particles and "
              << m_LODConfig.GetLODCount() << " LOD levels" << std::endl;
}

int PhysXCloth::InitializeAsync(
    PhysXBackend* backend,
    const ClothDesc& desc,
    AsyncCallback onComplete,
    ErrorCallback onError)
{
    return AsyncClothFactory::GetInstance().CreateClothAsync(
        backend,
        desc,
        onComplete,
        onError
    );
}

void PhysXCloth::CreateClothFabric(const ClothDesc& desc) {
    PxPhysics* physics = m_Backend->GetPhysics();
    
    // Prepare mesh descriptor
    PxClothMeshDesc meshDesc;
    meshDesc.setToDefault();
    
    // Set vertices
    std::vector<PxVec3> vertices(desc.particleCount);
    for (int i = 0; i < desc.particleCount; ++i) {
        vertices[i] = PxVec3(
            desc.particlePositions[i].x,
            desc.particlePositions[i].y,
            desc.particlePositions[i].z
        );
    }
    meshDesc.points.data = vertices.data();
    meshDesc.points.count = desc.particleCount;
    meshDesc.points.stride = sizeof(PxVec3);
    
    // Set triangles
    meshDesc.triangles.data = desc.triangleIndices;
    meshDesc.triangles.count = desc.triangleCount;
    meshDesc.triangles.stride = sizeof(int) * 3;
    
    // Cook fabric
    PxClothFabricCooker cooker(meshDesc, PxVec3(0, -1, 0)); // Gravity direction for cooking
    m_Fabric = cooker.getClothFabric(*physics);
}

void PhysXCloth::SetupConstraints() {
    if (!m_Cloth) return;

    // Set solver frequency (iterations per frame)
    m_Cloth->setSolverFrequency(60.0f);

    // Configure stiffness for different constraint types
    SetStretchStiffness(m_StretchStiffness);
    SetBendStiffness(m_BendStiffness);
    SetShearStiffness(m_ShearStiffness);
    SetDamping(m_Damping);

    // Apply collision settings
    if (m_EnableSceneCollision) {
        m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, true);
    }

    if (m_EnableSelfCollision) {
        m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
        m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
    }
    
    // Two-way coupling logic
    // In PhysX 3.4/4/5, two-way coupling is generally implicit if scene collision is on
    // and the cloth has mass.
    // However, we can control the impulse scale.
    // Note: setCollisionMassScale is used to set the mass of the particles for collision purposes.
    if (m_EnableTwoWayCoupling) {
        m_Cloth->setCollisionMassScale(m_CollisionMassScale);
    } else {
        m_Cloth->setCollisionMassScale(0.0f); // 0 means cloth reacts but dynamic bodies don't (one-way)
    }

    // Set solver frequency
    m_Cloth->setSolverFrequency(m_SolverFrequency);
}

void PhysXCloth::Update(float deltaTime) {
    SCOPED_PROFILE("PhysXCloth::Update");
    // Skip update if frozen (LOD optimization)
    if (m_IsFrozen) {
        return;
    }
    
    // Frame skipping for LOD
    m_UpdateCounter++;
    if (m_UpdateCounter % m_UpdateFrequency != 0) {
        return;
    }
    
    if (!m_Cloth || !m_Enabled) {
        return;
    }

    // Apply wind force
    if (m_WindVelocity.Length() > 0.001f) {
        PxVec3 wind(m_WindVelocity.x, m_WindVelocity.y, m_WindVelocity.z);
        m_Cloth->setWindVelocity(wind);
        m_Cloth->setDragCoefficient(0.5f); // Air resistance
        m_Cloth->setLiftCoefficient(0.3f); // Lift force
    }

    // Update collision shapes (culling)
    UpdateCollisionShapes();

    // Update particle data from simulation
    UpdateParticleData();
    
    // Update progressive tears
    UpdateProgressiveTears(deltaTime);
}

void PhysXCloth::GetWorldBounds(Vec3& outMin, Vec3& outMax) const {
    if (m_ParticleCount == 0) {
        outMin = Vec3(0, 0, 0);
        outMax = Vec3(0, 0, 0);
        return;
    }

    outMin = Vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    outMax = Vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    // Iterate particles (using local cache or PhysX read data? Local cache is updated every frame in UpdateParticleData, 
    // but GetWorldBounds is called BEFORE UpdateParticleData in this flow? 
    // Actually, UpdateCollisionShapes is called before UpdateParticleData, so m_ParticlePositions depends on LAST frame.
    // That is fine for broadphase culling.
    for (int i = 0; i < m_ParticleCount; ++i) {
        const Vec3& p = m_ParticlePositions[i];
        if (p.x < outMin.x) outMin.x = p.x;
        if (p.y < outMin.y) outMin.y = p.y;
        if (p.z < outMin.z) outMin.z = p.z;
        
        if (p.x > outMax.x) outMax.x = p.x;
        if (p.y > outMax.y) outMax.y = p.y;
        if (p.z > outMax.z) outMax.z = p.z;
    }
}

void PhysXCloth::UpdateCollisionShapes() {
    SCOPED_PROFILE("PhysXCloth::UpdateCollisionShapes");
    if (!m_Cloth || m_CollisionShapesList.empty()) return;

    // Calculate cloth bounds
    Vec3 minBounds, maxBounds;
    GetWorldBounds(minBounds, maxBounds);
    
    // Add margin (e.g. 1 meter)
    minBounds = minBounds - Vec3(1.0f, 1.0f, 1.0f);
    maxBounds = maxBounds + Vec3(1.0f, 1.0f, 1.0f);
    
    SpatialGrid<int>::AABB queryBounds;
    queryBounds.min = minBounds;
    queryBounds.max = maxBounds;

    std::vector<int> nearbyShapeIndices;
    m_SpatialGrid->Query(queryBounds, nearbyShapeIndices);
    
    // PhysX Limit is 32 spheres
    if (nearbyShapeIndices.size() > 32) {
        // Simple priority: just take first 32, or sort by distance to bounds center?
        // Let's sort by distance to cloth center
        Vec3 center = (minBounds + maxBounds) * 0.5f;
        std::sort(nearbyShapeIndices.begin(), nearbyShapeIndices.end(), 
            [&](int a, int b) {
                const auto& sa = m_CollisionShapesList[a];
                const auto& sb = m_CollisionShapesList[b];
                float da = (sa.pos0 - center).LengthSquared(); // rough dist
                float db = (sb.pos0 - center).LengthSquared();
                return da < db;
            });
        nearbyShapeIndices.resize(32);
    }

    // Convert to PhysX spheres/capsules
    // PhysX manages spheres and capsules together. Capsules reference indices in the sphere array.
    // We need to rebuild the frame-local spheres list.
    
    m_Cloth->clearCollisionSpheres(); // Not exposed in standard API, usually we just setCollisionSpheres
    // API: setCollisionSpheres(const PxClothCollisionSphere* spheres, PxU32 sphereCount)
    //      setCollisionCapsules(const PxU32* capsules, PxU32 capsuleCount)
    
    // We need to flatten our selected shapes into spheres and indices
    std::vector<PxClothCollisionSphere> pxSpheres;
    std::vector<PxU32> pxCapsules; // pairs of indices

    for (int idx : nearbyShapeIndices) {
        const auto& shape = m_CollisionShapesList[idx];
        
        if (shape.type == 0) { // Sphere
            PxClothCollisionSphere s;
            s.pos = PxVec3(shape.pos0.x, shape.pos0.y, shape.pos0.z);
            s.radius = shape.radius;
            pxSpheres.push_back(s);
        } else if (shape.type == 1) { // Capsule
            // Capsule needs two spheres
            PxClothCollisionSphere s0, s1;
            s0.pos = PxVec3(shape.pos0.x, shape.pos0.y, shape.pos0.z);
            s0.radius = shape.radius;
            s1.pos = PxVec3(shape.pos1.x, shape.pos1.y, shape.pos1.z);
            s1.radius = shape.radius;
            
            // Check if we have space (Max 32 spheres)
            if (pxSpheres.size() + 2 > 32) break;

            uint32_t idx0 = static_cast<uint32_t>(pxSpheres.size());
            pxSpheres.push_back(s0);
            uint32_t idx1 = static_cast<uint32_t>(pxSpheres.size());
            pxSpheres.push_back(s1);
            
            pxCapsules.push_back(idx0);
            pxCapsules.push_back(idx1);
        }
    }
    
    m_Cloth->setCollisionSpheres(pxSpheres.data(), static_cast<PxU32>(pxSpheres.size()));
    if (!pxCapsules.empty()) {
        m_Cloth->setCollisionCapsules(pxCapsules.data(), static_cast<PxU32>(pxCapsules.size() / 2));
    }
}

void PhysXCloth::UpdateParticleData() {
    SCOPED_PROFILE("PhysXCloth::UpdateParticleData");
    if (!m_Cloth) return;

    // Get particle positions from PhysX
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (readData) {
        const PxClothParticle* particles = readData->particles;
        // Optimization: Unroll or rely on compiler vectorization
        // PxClothParticle is 16 bytes (pos + invWeight), Vec3 is 12 bytes
        // Compilers usually handle this loop well, but we can hint
        for (int i = 0; i < m_ParticleCount; ++i) {
            const auto& p = particles[i];
            m_ParticlePositions[i].x = p.pos.x;
            m_ParticlePositions[i].y = p.pos.y;
            m_ParticlePositions[i].z = p.pos.z;
        }
        readData->unlock();
    }

    // Recalculate normals for rendering
    RecalculateNormals();
}

// Helper to avoid allocations
void PhysXCloth::RecalculateNormals() {
    SCOPED_PROFILE("PhysXCloth::RecalculateNormals");
    
    // Reset normals efficiently
    std::memset(m_ParticleNormals.data(), 0, m_ParticleCount * sizeof(Vec3));

    // Calculate face normals and accumulate
    // Access raw pointers for speed
    Vec3* normals = m_ParticleNormals.data();
    const Vec3* positions = m_ParticlePositions.data();
    const int* indices = m_TriangleIndices.data();
    
    for (int i = 0; i < m_TriangleCount; ++i) {
        int i0 = indices[i * 3 + 0];
        int i1 = indices[i * 3 + 1];
        int i2 = indices[i * 3 + 2];

        const Vec3& v0 = positions[i0];
        const Vec3& v1 = positions[i1];
        const Vec3& v2 = positions[i2];

        // Inline cross product for better optimization chane
        float ax = v1.x - v0.x;
        float ay = v1.y - v0.y;
        float az = v1.z - v0.z;
        
        float bx = v2.x - v0.x;
        float by = v2.y - v0.y;
        float bz = v2.z - v0.z;
        
        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;

        normals[i0].x += nx; normals[i0].y += ny; normals[i0].z += nz;
        normals[i1].x += nx; normals[i1].y += ny; normals[i1].z += nz;
        normals[i2].x += nx; normals[i2].y += ny; normals[i2].z += nz;
    }

    // Normalize
    const float minLengthSq = 0.000001f;
    for (int i = 0; i < m_ParticleCount; ++i) {
        float x = normals[i].x;
        float y = normals[i].y;
        float z = normals[i].z;
        float lenSq = x*x + y*y + z*z;
        
        if (lenSq > minLengthSq) {
            float invLen = 1.0f / std::sqrt(lenSq);
            normals[i].x *= invLen;
            normals[i].y *= invLen;
            normals[i].z *= invLen;
        }
    }
}

void PhysXCloth::SetEnabled(bool enabled) {
    m_Enabled = enabled;
}

bool PhysXCloth::IsEnabled() const {
    return m_Enabled;
}

int PhysXCloth::GetParticleCount() const {
    return m_ParticleCount;
}

void PhysXCloth::GetParticlePositions(Vec3* positions) const {
    for (int i = 0; i < m_ParticleCount; ++i) {
        positions[i] = m_ParticlePositions[i];
    }
}

void PhysXCloth::SetParticlePositions(const Vec3* positions) {
    if (!m_Cloth) return;

    std::vector<PxClothParticle> particles(m_ParticleCount);
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (readData) {
        const PxClothParticle* currentParticles = readData->particles;
        for (int i = 0; i < m_ParticleCount; ++i) {
            particles[i].pos = PxVec3(positions[i].x, positions[i].y, positions[i].z);
            particles[i].invWeight = currentParticles[i].invWeight;
        }
        readData->unlock();
    }

    m_Cloth->setParticles(particles.data(), nullptr);
}

void PhysXCloth::GetParticleNormals(Vec3* normals) const {
    for (int i = 0; i < m_ParticleCount; ++i) {
        normals[i] = m_ParticleNormals[i];
    }
}

void PhysXCloth::SetStretchStiffness(float stiffness) {
    m_StretchStiffness = stiffness;
    if (m_Cloth) {
        m_Cloth->setStretchConfig(PxClothFabricPhaseType::eVERTICAL, PxClothStretchConfig(stiffness));
        m_Cloth->setStretchConfig(PxClothFabricPhaseType::eHORIZONTAL, PxClothStretchConfig(stiffness));
    }
}

void PhysXCloth::SetBendStiffness(float stiffness) {
    m_BendStiffness = stiffness;
    if (m_Cloth) {
        m_Cloth->setStretchConfig(PxClothFabricPhaseType::eBENDING, PxClothStretchConfig(stiffness));
    }
}

void PhysXCloth::SetShearStiffness(float stiffness) {
    m_ShearStiffness = stiffness;
    if (m_Cloth) {
        m_Cloth->setStretchConfig(PxClothFabricPhaseType::eSHEARING, PxClothStretchConfig(stiffness));
    }
}

void PhysXCloth::AddForce(const Vec3& force) {
    if (m_Cloth) {
        PxVec3 pxForce(force.x, force.y, force.z);
        m_Cloth->setExternalAcceleration(pxForce);
    }
}

void PhysXCloth::SetWindVelocity(const Vec3& velocity) {
    m_WindVelocity = velocity;
}

void PhysXCloth::SetDamping(float damping) {
    m_Damping = damping;
    if (m_Cloth) {
        m_Cloth->setDampingCoefficient(PxVec3(damping, damping, damping));
    }
}

void PhysXCloth::AttachParticleToActor(int particleIndex, void* actor, const Vec3& localPos) {
    if (!m_Cloth || particleIndex < 0 || particleIndex >= m_ParticleCount) {
        return;
    }

    // Set particle inverse weight to 0 (infinite mass = fixed)
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (readData) {
        std::vector<PxClothParticle> particles(m_ParticleCount);
        const PxClothParticle* currentParticles = readData->particles;
        
        for (int i = 0; i < m_ParticleCount; ++i) {
            particles[i] = currentParticles[i];
        }
        
        particles[particleIndex].invWeight = 0.0f; // Fixed particle
        readData->unlock();
        
        m_Cloth->setParticles(particles.data(), nullptr);
    }
}

void PhysXCloth::FreeParticle(int particleIndex) {
    if (!m_Cloth || particleIndex < 0 || particleIndex >= m_ParticleCount) {
        return;
    }

    // Restore particle inverse weight
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (readData) {
        std::vector<PxClothParticle> particles(m_ParticleCount);
        const PxClothParticle* currentParticles = readData->particles;
        
        for (int i = 0; i < m_ParticleCount; ++i) {
            particles[i] = currentParticles[i];
        }
        
        particles[particleIndex].invWeight = 1.0f; // Normal mass
        readData->unlock();
        
        m_Cloth->setParticles(particles.data(), nullptr);
    }
}

void PhysXCloth::AddCollisionSphere(const Vec3& center, float radius) {
    if (m_Cloth) {
        InternalCollisionShape shape;
        shape.type = 0;
        shape.pos0 = center;
        shape.radius = radius;
        shape.id = static_cast<int>(m_CollisionShapesList.size());
        
        m_CollisionShapesList.push_back(shape);
        m_SpatialGrid->Insert(shape.id, center, radius);
        
        // We defer PxCloth update to Update() loop
    }
}

void PhysXCloth::AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) {
    if (m_Cloth) {
        InternalCollisionShape shape;
        shape.type = 1;
        shape.pos0 = p0;
        shape.pos1 = p1;
        shape.radius = radius;
        shape.id = static_cast<int>(m_CollisionShapesList.size());
        
        m_CollisionShapesList.push_back(shape);
        
        // Calculate capsule bounds for grid
        Vec3 minPos = Vec3(std::min(p0.x, p1.x) - radius, std::min(p0.y, p1.y) - radius, std::min(p0.z, p1.z) - radius);
        Vec3 maxPos = Vec3(std::max(p0.x, p1.x) + radius, std::max(p0.y, p1.y) + radius, std::max(p0.z, p1.z) + radius);
        
        SpatialGrid<int>::AABB bounds;
        bounds.min = minPos;
        bounds.max = maxPos;
        
        m_SpatialGrid->Insert(shape.id, bounds);
    }
}

void PhysXCloth::SetTearable(bool tearable) {
    m_Tearable = tearable;
    if (tearable) {
        m_MaxParticles = m_ParticleCount * 2; // Allow 2x growth
        m_TearCount = 0;
        m_LastTearTime = 0.0f;
    }
}

void PhysXCloth::SetMaxStretchRatio(float ratio) {
    m_MaxStretchRatio = ratio;
}

bool PhysXCloth::TearAtParticle(int particleIndex) {
    if (!CanTear() || particleIndex < 0 || particleIndex >= m_ParticleCount) {
        return false;
    }
    
    TearInfo tear;
    tear.particleIndex = particleIndex;
    tear.stretchRatio = m_MaxStretchRatio + 0.1f; // Exceed threshold
    tear.position = m_ParticlePositions[particleIndex];
    
    ProcessTear(tear);
    return true;
}

int PhysXCloth::TearAlongLine(const Vec3& start, const Vec3& end) {
    if (!CanTear()) {
        return 0;
    }
    
    int tearsCreated = 0;
    Vec3 lineDir = end - start;
    float lineLength = lineDir.Length();
    
    if (lineLength < 0.001f) {
        return 0;
    }
    
    lineDir = lineDir * (1.0f / lineLength);
    
    // Find particles along the line
    for (int i = 0; i < m_ParticleCount; ++i) {
        Vec3 toParticle = m_ParticlePositions[i] - start;
        float projection = toParticle.Dot(lineDir);
        
        if (projection >= 0.0f && projection <= lineLength) {
            Vec3 closestPoint = start + lineDir * projection;
            float distance = (m_ParticlePositions[i] - closestPoint).Length();
            
            // If particle is close to line, tear it
            if (distance < 0.1f) { // 10cm threshold
                if (TearAtParticle(i)) {
                    tearsCreated++;
                    
                    // Limit tears per call
                    if (tearsCreated >= 10) {
                        break;
                    }
                }
            }
        }
    }
    
    std::cout << "Tore " << tearsCreated << " particles along line" << std::endl;
    return tearsCreated;
}

int PhysXCloth::GetTearCount() const {
    return m_TearCount;
}

void PhysXCloth::ResetTears() {
    m_TearCount = 0;
    m_TornParticles.clear();
    m_TearCandidates.clear();
    m_LastTearTime = 0.0f;
    
    std::cout << "Cloth tears reset" << std::endl;
}

void* PhysXCloth::GetNativeCloth() {
    return m_Cloth;
}

void PhysXCloth::SetSceneCollision(bool enabled) {
    m_EnableSceneCollision = enabled;
    if (m_Cloth) {
        m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, enabled);
    }
}

void PhysXCloth::SetSelfCollision(bool enabled) {
    m_EnableSelfCollision = enabled;
    if (m_Cloth && enabled) {
        // Ensure parameters are set when enabling
        m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
        m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
    }
    // Note: PhysX 5 doesn't have a flag for self collision, it's enabled by setting distance > 0
    // But we might need to check if we can disable it by setting distance to 0, or if there is a config
    // Actually typically one sets the rest position checks.
    // However, looking at PxClothFlag, there isn't a generic eSELF_COLLISION flag in recent versions,
    // it interacts via virtual particles or rest positions. 
    // Wait, let's verify if we need to set PxClothFlag::eSWEPT_CONTACT?
    // Start with simple param setting.
    if (m_Cloth && !enabled) {
        m_Cloth->setSelfCollisionDistance(0.0f);
    }
}

void PhysXCloth::SetSelfCollisionDistance(float distance) {
    m_SelfCollisionDistance = distance;
    if (m_Cloth && m_EnableSelfCollision) {
        m_Cloth->setSelfCollisionDistance(distance);
    }
}

void PhysXCloth::SetSelfCollisionStiffness(float stiffness) {
    m_SelfCollisionStiffness = stiffness;
    if (m_Cloth && m_EnableSelfCollision) {
        m_Cloth->setSelfCollisionStiffness(stiffness);
    }
}

void PhysXCloth::SetTwoWayCoupling(bool enabled) {
    m_EnableTwoWayCoupling = enabled;
    if (m_Cloth) {
        if (enabled) {
            m_Cloth->setCollisionMassScale(m_CollisionMassScale);
        } else {
            m_Cloth->setCollisionMassScale(0.0f);
        }
    }
}

void PhysXCloth::SetCollisionMassScale(float scale) {
    m_CollisionMassScale = scale;
    if (m_Cloth && m_EnableTwoWayCoupling) {
        m_Cloth->setCollisionMassScale(scale);
    }
}

void PhysXCloth::SetSolverFrequency(float frequency) {
    m_SolverFrequency = frequency;
    if (m_Cloth) {
        m_Cloth->setSolverFrequency(frequency);
    }
}

// Tearing helper methods

void PhysXCloth::DetectTears(float deltaTime) {
    SCOPED_PROFILE("PhysXCloth::DetectTears");
    if (!m_Cloth || !m_Tearable || !CanTear()) {
        return;
    }
    
    // Rate limiting
    m_LastTearTime += deltaTime;
    if (m_LastTearTime < 0.1f) { // Min 100ms between tear checks
        return;
    }
    
    m_TearCandidates.clear();
    
    // Check each triangle for overstretching
    for (int i = 0; i < m_TriangleCount; ++i) {
        int i0 = m_TriangleIndices[i * 3 + 0];
        int i1 = m_TriangleIndices[i * 3 + 1];
        int i2 = m_TriangleIndices[i * 3 + 2];
        
        // Calculate edge lengths
        Vec3 p0 = m_ParticlePositions[i0];
        Vec3 p1 = m_ParticlePositions[i1];
        Vec3 p2 = m_ParticlePositions[i2];
        
        float edge01 = (p1 - p0).Length();
        float edge12 = (p2 - p1).Length();
        float edge20 = (p0 - p2).Length();
        
        // Estimate rest lengths (simplified - should store original lengths)
        float avgEdge = (edge01 + edge12 + edge20) / 3.0f;
        
        // Check if any edge is overstretched
        if (edge01 / avgEdge > m_MaxStretchRatio) {
            TearInfo tear;
            tear.particleIndex = i0;
            tear.stretchRatio = edge01 / avgEdge;
            tear.position = (p0 + p1) * 0.5f;
            m_TearCandidates.push_back(tear);
        }
    }
    
    // Process tears (limit to prevent cascading)
    int maxTearsPerFrame = 2;
    int tearsProcessed = 0;
    
    for (const auto& tear : m_TearCandidates) {
        if (tearsProcessed >= maxTearsPerFrame) break;
        
        ProcessTear(tear);
        tearsProcessed++;
    }
    
    if (tearsProcessed > 0) {
        m_LastTearTime = 0.0f;
    }
}

void PhysXCloth::ProcessTear(const TearInfo& tear) {
    // Mark particle as torn
    m_TornParticles.push_back(tear.particleIndex);
    m_TearCount++;
    
    // Visual feedback - free the particle temporarily
    FreeParticle(tear.particleIndex);
    
    std::cout << "Cloth tear #" << m_TearCount 
              << " at particle " << tear.particleIndex 
              << " (stretch: " << tear.stretchRatio << "x)" << std::endl;
    
    // Note: Full tearing would require:
    // 1. Duplicate particle at tear point
    // 2. Split triangles using torn particle
    // 3. Update cloth fabric with new topology
    // 4. Recreate PhysX cloth object
    // This is a simplified visual-only tear
}

bool PhysXCloth::CanTear() const {
    return m_Tearable && 
           m_TearCount < 100 && 
           m_ParticleCount < m_MaxParticles;
}

int PhysXCloth::FindNearestParticle(const Vec3& position) const {
    int nearestIndex = -1;
    float nearestDist = FLT_MAX;
    
    for (int i = 0; i < m_ParticleCount; ++i) {
        float dist = (m_ParticlePositions[i] - position).Length();
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestIndex = i;
        }
    }
    
    return nearestIndex;
}

// Rendering support

void PhysXCloth::UpdateMeshData(Mesh* mesh) {
    SCOPED_PROFILE("PhysXCloth::UpdateMeshData");
    if (!mesh || !m_Cloth) {
        return;
    }

    // Update particle data from simulation
    UpdateParticleData();

    // Handle Proxy Mesh (LODs)
    if (m_CurrentLODLevel > 0) {
        // Find current level data
        const ClothLODLevel* level = m_LODConfig.GetLODLevel(m_CurrentLODLevel);
        if (level && !level->particleMapping.empty()) {
            UpdateProxyMesh(mesh, level->particleMapping);
            return;
        }
    }

    // High fidelity mode (LOD 0) - Original logic
    // Get mesh vertices and normals
    auto& vertices = mesh->GetVertices();
    
    // Resize if needed (only for LOD 0 where 1:1 mapping is expected)
    if (static_cast<int>(vertices.size()) != m_ParticleCount) {
        vertices.resize(m_ParticleCount);
    }

    // Copy positions and normals to mesh
    for (int i = 0; i < m_ParticleCount; ++i) {
        vertices[i].Position = m_ParticlePositions[i];
        vertices[i].Normal = m_ParticleNormals[i];
        // Keep existing UV coordinates
    }

    // Update mesh buffers
    mesh->UpdateVertices();
}

void PhysXCloth::UpdateProxyMesh(Mesh* mesh, const std::vector<int>& mapping) {
    SCOPED_PROFILE("PhysXCloth::UpdateProxyMesh");
    if (!mesh) return;

    auto& vertices = mesh->GetVertices();
    int vertexCount = static_cast<int>(vertices.size());
    
    // Safety check - make sure mapping is sufficient for the mesh
    // Note: The render mesh is likely higher res than the physics mesh, so mapping.size() 
    // should match vertexCount (each render vertex maps to a physics particle)
    if (static_cast<int>(mapping.size()) < vertexCount) {
        // Fallback or warning? For now just handle what we can
        vertexCount = static_cast<int>(mapping.size());
    }

    // Update render vertices based on mapped physics particles
    for (int i = 0; i < vertexCount; ++i) {
        int physicsIndex = mapping[i];
        
        // Safety check for physics index
        if (physicsIndex >= 0 && physicsIndex < m_ParticleCount) {
            vertices[i].Position = m_ParticlePositions[physicsIndex];
            vertices[i].Normal = m_ParticleNormals[physicsIndex];
        }
    }
    
    // Ideally we would recalculate smooth normals here for better appearance
    // mesh->RecalculateNormals(); // Expensive, maybe optional?
    
    mesh->UpdateVertices();
}

// Mesh splitting methods

bool PhysXCloth::SplitAtParticle(
    int tearParticle,
    std::shared_ptr<PhysXCloth>& outPiece1,
    std::shared_ptr<PhysXCloth>& outPiece2)
{
    SCOPED_PROFILE("PhysXCloth::SplitAtParticle");
    if (!m_Cloth || tearParticle < 0 || tearParticle >= m_ParticleCount) {
        std::cerr << "Invalid tear particle for splitting" << std::endl;
        return false;
    }
    
    std::cout << "Splitting cloth at particle " << tearParticle << "..." << std::endl;
    
    // Use ClothMeshSplitter to split the mesh
    auto result = ClothMeshSplitter::SplitAtParticle(
        m_ParticlePositions,
        m_TriangleIndices,
        tearParticle
    );
    
    if (!result.success) {
        std::cerr << "Mesh splitting failed" << std::endl;
        return false;
    }
    
    // Create two new cloth pieces
    outPiece1 = CreateFromSplit(result.piece1Positions, result.piece1Indices);
    outPiece2 = CreateFromSplit(result.piece2Positions, result.piece2Indices);
    
    if (!outPiece1 || !outPiece2) {
        std::cerr << "Failed to create cloth pieces" << std::endl;
        return false;
    }
    
    // Copy properties to new pieces
    outPiece1->SetStretchStiffness(m_StretchStiffness);
    outPiece1->SetBendStiffness(m_BendStiffness);
    outPiece1->SetShearStiffness(m_ShearStiffness);
    outPiece1->SetDamping(m_Damping);
    outPiece1->SetWindVelocity(m_WindVelocity);
    outPiece1->SetTearable(m_Tearable);
    outPiece1->SetMaxStretchRatio(m_MaxStretchRatio);
    // Copy LOD state
    outPiece1->m_UpdateFrequency = m_UpdateFrequency;
    // Copy collision settings
    outPiece1->m_EnableSceneCollision = m_EnableSceneCollision;
    outPiece1->m_EnableSelfCollision = m_EnableSelfCollision;
    outPiece1->m_SelfCollisionDistance = m_SelfCollisionDistance;
    outPiece1->m_SelfCollisionStiffness = m_SelfCollisionStiffness;
    outPiece1->m_EnableTwoWayCoupling = m_EnableTwoWayCoupling;
    outPiece1->m_CollisionMassScale = m_CollisionMassScale;
    // Apply collision settings to new piece
    if (outPiece1->m_Cloth) {
        if (m_EnableSceneCollision) outPiece1->m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, true);
        if (m_EnableSelfCollision) {
            outPiece1->m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
            outPiece1->m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
        }
        if (m_EnableTwoWayCoupling) {
             outPiece1->m_Cloth->setCollisionMassScale(m_CollisionMassScale);
        } else {
             outPiece1->m_Cloth->setCollisionMassScale(0.0f);
        }
    }
    
    outPiece2->SetStretchStiffness(m_StretchStiffness);
    outPiece2->SetBendStiffness(m_BendStiffness);
    outPiece2->SetShearStiffness(m_ShearStiffness);
    outPiece2->SetDamping(m_Damping);
    outPiece2->SetWindVelocity(m_WindVelocity);
    outPiece2->SetTearable(m_Tearable);
    outPiece2->SetMaxStretchRatio(m_MaxStretchRatio);
    // Copy LOD state
    outPiece2->m_UpdateFrequency = m_UpdateFrequency;
    // Copy collision settings
    outPiece2->m_EnableSceneCollision = m_EnableSceneCollision;
    outPiece2->m_EnableSelfCollision = m_EnableSelfCollision;
    outPiece2->m_SelfCollisionDistance = m_SelfCollisionDistance;
    outPiece2->m_SelfCollisionStiffness = m_SelfCollisionStiffness;
    outPiece2->m_EnableTwoWayCoupling = m_EnableTwoWayCoupling;
    outPiece2->m_CollisionMassScale = m_CollisionMassScale;
    // Apply collision settings to new piece
    if (outPiece2->m_Cloth) {
        if (m_EnableSceneCollision) outPiece2->m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, true);
        if (m_EnableSelfCollision) {
            outPiece2->m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
            outPiece2->m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
        }
        if (m_EnableTwoWayCoupling) {
             outPiece2->m_Cloth->setCollisionMassScale(m_CollisionMassScale);
        } else {
             outPiece2->m_Cloth->setCollisionMassScale(0.0f);
        }
    }
    
    std::cout << "Successfully split cloth into 2 pieces" << std::endl;
    
    // Invoke tear callback if set
    if (m_TearCallback) {
        m_TearCallback(outPiece1, outPiece2);
    }
    
    return true;
}

bool PhysXCloth::SplitAlongLine(
    const Vec3& start,
    const Vec3& end,
    std::shared_ptr<PhysXCloth>& outPiece1,
    std::shared_ptr<PhysXCloth>& outPiece2)
{
    SCOPED_PROFILE("PhysXCloth::SplitAlongLine");
    if (!m_Cloth) {
        std::cerr << "Invalid cloth for splitting" << std::endl;
        return false;
    }
    
    std::cout << "Splitting cloth along line..." << std::endl;
    
    // Use ClothMeshSplitter to split along line
    auto result = ClothMeshSplitter::SplitAlongLine(
        m_ParticlePositions,
        m_TriangleIndices,
        start,
        end
    );
    
    if (!result.success) {
        std::cerr << "Line splitting failed" << std::endl;
        return false;
    }
    
    // Create two new cloth pieces
    outPiece1 = CreateFromSplit(result.piece1Positions, result.piece1Indices);
    outPiece2 = CreateFromSplit(result.piece2Positions, result.piece2Indices);
    
    if (!outPiece1 || !outPiece2) {
        std::cerr << "Failed to create cloth pieces" << std::endl;
        return false;
    }
    
    // Copy properties
    outPiece1->SetStretchStiffness(m_StretchStiffness);
    outPiece1->SetBendStiffness(m_BendStiffness);
    outPiece1->SetShearStiffness(m_ShearStiffness);
    outPiece1->SetDamping(m_Damping);
    outPiece1->SetWindVelocity(m_WindVelocity);
    outPiece1->SetTearable(m_Tearable);
    outPiece1->SetMaxStretchRatio(m_MaxStretchRatio);
    // Copy LOD state
    outPiece1->m_UpdateFrequency = m_UpdateFrequency;
    // Copy collision settings
    outPiece1->m_EnableSceneCollision = m_EnableSceneCollision;
    outPiece1->m_EnableSelfCollision = m_EnableSelfCollision;
    outPiece1->m_SelfCollisionDistance = m_SelfCollisionDistance;
    outPiece1->m_SelfCollisionStiffness = m_SelfCollisionStiffness;
    outPiece1->m_EnableTwoWayCoupling = m_EnableTwoWayCoupling;
    outPiece1->m_CollisionMassScale = m_CollisionMassScale;
    // Apply collision settings to new piece
    if (outPiece1->m_Cloth) {
        if (m_EnableSceneCollision) outPiece1->m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, true);
        if (m_EnableSelfCollision) {
            outPiece1->m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
            outPiece1->m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
        }
        if (m_EnableTwoWayCoupling) {
             outPiece1->m_Cloth->setCollisionMassScale(m_CollisionMassScale);
        } else {
             outPiece1->m_Cloth->setCollisionMassScale(0.0f);
        }
    }
    
    outPiece2->SetStretchStiffness(m_StretchStiffness);
    outPiece2->SetBendStiffness(m_BendStiffness);
    outPiece2->SetShearStiffness(m_ShearStiffness);
    outPiece2->SetDamping(m_Damping);
    outPiece2->SetWindVelocity(m_WindVelocity);
    outPiece2->SetTearable(m_Tearable);
    outPiece2->SetMaxStretchRatio(m_MaxStretchRatio);
    // Copy LOD state
    outPiece2->m_UpdateFrequency = m_UpdateFrequency;
    // Copy collision settings
    outPiece2->m_EnableSceneCollision = m_EnableSceneCollision;
    outPiece2->m_EnableSelfCollision = m_EnableSelfCollision;
    outPiece2->m_SelfCollisionDistance = m_SelfCollisionDistance;
    outPiece2->m_SelfCollisionStiffness = m_SelfCollisionStiffness;
    outPiece2->m_EnableTwoWayCoupling = m_EnableTwoWayCoupling;
    outPiece2->m_CollisionMassScale = m_CollisionMassScale;
    // Apply collision settings to new piece
    if (outPiece2->m_Cloth) {
        if (m_EnableSceneCollision) outPiece2->m_Cloth->setClothFlag(PxClothFlag::eSCENE_COLLISION, true);
        if (m_EnableSelfCollision) {
            outPiece2->m_Cloth->setSelfCollisionDistance(m_SelfCollisionDistance);
            outPiece2->m_Cloth->setSelfCollisionStiffness(m_SelfCollisionStiffness);
        }
        if (m_EnableTwoWayCoupling) {
             outPiece2->m_Cloth->setCollisionMassScale(m_CollisionMassScale);
        } else {
             outPiece2->m_Cloth->setCollisionMassScale(0.0f);
        }
    }
    
    std::cout << "Successfully split cloth along line" << std::endl;
    
    // Invoke tear callback if set
    if (m_TearCallback) {
        m_TearCallback(outPiece1, outPiece2);
    }
    
    return true;
}

std::shared_ptr<PhysXCloth> PhysXCloth::CreateFromSplit(
    const std::vector<Vec3>& positions,
    const std::vector<int>& indices)
{
    if (positions.empty() || indices.empty()) {
        std::cerr << "Cannot create cloth from empty mesh" << std::endl;
        return nullptr;
    }
    
    // Create cloth descriptor
    ClothDesc desc;
    desc.particleCount = static_cast<int>(positions.size());
    desc.triangleCount = static_cast<int>(indices.size()) / 3;
    desc.particlePositions = new Vec3[desc.particleCount];
    desc.triangleIndices = new int[indices.size()];
    
    for (int i = 0; i < desc.particleCount; ++i) {
        desc.particlePositions[i] = positions[i];
    }
    
    for (size_t i = 0; i < indices.size(); ++i) {
        desc.triangleIndices[i] = indices[i];
    }
    
    desc.particleMass = 0.1f;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    // Create new cloth
    auto newCloth = std::make_shared<PhysXCloth>(m_Backend);
    newCloth->Initialize(desc);
    
    // Clean up
    delete[] desc.particlePositions;
    delete[] desc.triangleIndices;
    
    std::cout << "Created cloth piece with " << desc.particleCount << " particles" << std::endl;
    
    return newCloth;
}

void PhysXCloth::SetLOD(int lodLevel) {
    if (lodLevel == m_CurrentLOD) {
        return;  // Already at this LOD
    }
    
    const ClothLODLevel* level = m_LODConfig.GetLODLevel(lodLevel);
    if (!level) {
        std::cerr << "PhysXCloth: Invalid LOD level " << lodLevel << std::endl;
        return;
    }
    
    std::cout << "PhysXCloth: Transitioning from LOD " << m_CurrentLOD 
              << " to LOD " << lodLevel << std::endl;
    
    // Handle frozen state
    if (level->isFrozen) {
        Freeze();
        m_CurrentLOD = lodLevel;
        return;
    } else if (m_IsFrozen) {
        Unfreeze();
    }
    
    // If LOD level has mesh data, recreate cloth with simplified mesh
    if (level->hasMeshData && level->particleCount > 0) {
        if (RecreateClothWithLOD(level)) {
            m_CurrentLOD = lodLevel;
            std::cout << "PhysXCloth: Successfully transitioned to LOD " << lodLevel 
                      << " (" << m_ParticleCount << " particles)" << std::endl;
        } else {
            std::cerr << "PhysXCloth: Failed to recreate cloth for LOD " << lodLevel << std::endl;
        }
    } else {
        // Fallback: just update solver iterations
        if (m_Cloth) {
            m_Cloth->setSolverFrequency(static_cast<float>(level->solverIterations));
        }
        m_CurrentLOD = lodLevel;
    }
    
    // Apply LOD level settings
    m_SolverFrequency = (float)level->substeps * 60.0f; // Approximate
    if (m_Cloth) {
        m_Cloth->setSolverFrequency(m_SolverFrequency);
    }
    
    // Set update frequency
    m_UpdateFrequency = level->updateFrequency;
    if (m_UpdateFrequency < 1) m_UpdateFrequency = 1;
    
    // Check if frozen
    m_IsFrozen = level->isFrozen;
}

void PhysXCloth::Freeze() {
    if (m_IsFrozen) {
        return;  // Already frozen
    }
    
    std::cout << "PhysXCloth: Freezing cloth simulation" << std::endl;
    
    // Save current state
    m_FrozenPositions = m_ParticlePositions;
    m_FrozenNormals = m_ParticleNormals;
    
    m_IsFrozen = true;
}

void PhysXCloth::Unfreeze() {
    if (!m_IsFrozen) {
        return;  // Already active
    }
    
    std::cout << "PhysXCloth: Unfreezing cloth simulation" << std::endl;
    
    // Restore state if we have frozen data
    if (!m_FrozenPositions.empty() && m_Cloth) {
        // Restore particle positions
        PxClothParticleData* data = m_Cloth->lockParticleData();
        if (data) {
            for (int i = 0; i < m_ParticleCount && i < static_cast<int>(m_FrozenPositions.size()); ++i) {
                data->particles[i].pos = PxVec3(
                    m_FrozenPositions[i].x,
                    m_FrozenPositions[i].y,
                    m_FrozenPositions[i].z
                );
            }
            data->unlock();
        }
    }
    
    m_IsFrozen = false;
}

// LOD mesh recreation methods

bool PhysXCloth::RecreateClothWithLOD(const ClothLODLevel* level) {
    if (!level || !m_Cloth || !m_Backend) {
        return false;
    }
    
    std::cout << "PhysXCloth: Recreating cloth with LOD " << level->lodIndex 
              << " mesh (" << level->particleCount << " particles)..." << std::endl;
    
    // Prepare new particle data with state transfer
    std::vector<PxClothParticle> newParticles;
    TransferParticleState(level, newParticles);
    
    // Remove old cloth from scene
    m_Backend->GetScene()->removeActor(*m_Cloth);
    m_Cloth->release();
    m_Cloth = nullptr;
    
    if (m_Fabric) {
        m_Fabric->release();
        m_Fabric = nullptr;
    }
    
    // Create new cloth descriptor
    ClothDesc desc;
    desc.particleCount = level->particleCount;
    desc.triangleCount = level->triangleCount;
    desc.particlePositions = const_cast<Vec3*>(level->particlePositions.data());
    desc.triangleIndices = const_cast<int*>(level->triangleIndices.data());
    desc.particleMass = 0.1f;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    // Create new fabric
    CreateClothFabric(desc);
    if (!m_Fabric) {
        std::cerr << "PhysXCloth: Failed to create fabric for LOD" << std::endl;
        return false;
    }
    
    // Create new cloth
    PxTransform transform(PxVec3(0, 0, 0));
    m_Cloth = m_Backend->GetPhysics()->createCloth(
        transform,
        *m_Fabric,
        newParticles.data(),
        PxClothFlags()
    );
    
    if (!m_Cloth) {
        std::cerr << "PhysXCloth: Failed to create cloth for LOD" << std::endl;
        return false;
    }
    
    // Add to scene
    m_Backend->GetScene()->addActor(*m_Cloth);
    
    // Update internal state
    m_ParticleCount = level->particleCount;
    m_TriangleCount = level->triangleCount;
    m_ParticlePositions = level->particlePositions;
    m_TriangleIndices = level->triangleIndices;
    m_ParticleNormals.resize(m_ParticleCount);
    
    // Restore constraints
    SetupConstraints();
    
    std::cout << "PhysXCloth: Successfully recreated cloth with " << m_ParticleCount 
              << " particles" << std::endl;
    
    return true;
}

void PhysXCloth::TransferParticleState(
    const ClothLODLevel* level,
    std::vector<PxClothParticle>& outParticles)
{
    outParticles.resize(level->particleCount);
    
    // Get current particle data
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (!readData) {
        // Fallback: use rest positions
        for (int i = 0; i < level->particleCount; ++i) {
            outParticles[i].pos = PxVec3(
                level->particlePositions[i].x,
                level->particlePositions[i].y,
                level->particlePositions[i].z
            );
            outParticles[i].invWeight = 10.0f;  // Default mass
        }
        return;
    }
    
    const PxClothParticle* currentParticles = readData->particles;
    
    // Transfer state using vertex mapping
    // For each LOD vertex, find the closest original vertex it maps to
    for (int i = 0; i < level->particleCount; ++i) {
        // Find which original vertex(es) map to this LOD vertex
        // The mapping is: original vertex index -> LOD vertex index
        // We need to find original vertices that map to LOD vertex i
        
        int closestOriginal = -1;
        float closestDist = std::numeric_limits<float>::max();
        
        // Search through mapping to find original vertices that map to this LOD vertex
        for (size_t j = 0; j < level->particleMapping.size() && j < static_cast<size_t>(m_ParticleCount); ++j) {
            if (level->particleMapping[j] == i) {
                // Original vertex j maps to LOD vertex i
                // Use this as the source
                closestOriginal = static_cast<int>(j);
                break;
            }
        }
        
        if (closestOriginal >= 0 && closestOriginal < m_ParticleCount) {
            // Copy state from original particle
            outParticles[i] = currentParticles[closestOriginal];
        } else {
            // Fallback: find closest particle by position
            Vec3 lodPos = level->particlePositions[i];
            
            for (int j = 0; j < m_ParticleCount; ++j) {
                Vec3 diff = m_ParticlePositions[j] - lodPos;
                float dist = diff.Length();
                
                if (dist < closestDist) {
                    closestDist = dist;
                    closestOriginal = j;
                }
            }
            
            if (closestOriginal >= 0) {
                outParticles[i] = currentParticles[closestOriginal];
            } else {
                // Ultimate fallback: use rest position
                outParticles[i].pos = PxVec3(
                    level->particlePositions[i].x,
                    level->particlePositions[i].y,
                    level->particlePositions[i].z
                );
                outParticles[i].invWeight = 10.0f;
            }
        }
    }
    
    readData-^>unlock();
}

// ============================================================================
// Pattern-Based Tearing System
// ============================================================================

ClothTearPatternLibrary& PhysXCloth::GetPatternLibrary() {
    return ClothTearPatternLibrary::GetInstance();
}

bool PhysXCloth::ApplyTearPattern(
    const std::string& patternName,
    const Vec3& position,
    const Vec3& direction,
    float scale
) {
    auto pattern = GetPatternLibrary().GetPattern(patternName);
    if (!pattern) {
        std::cerr << "PhysXCloth: Pattern '" << patternName << "' not found in library" << std::endl;
        return false;
    }
    
    return ApplyTearPattern(pattern, position, direction, scale);
}

bool PhysXCloth::ApplyTearPattern(
    std::shared_ptr<ClothTearPattern> pattern,
    const Vec3& position,
    const Vec3& direction,
    float scale
) {
    if (!pattern || !m_Tearable || !CanTear()) {
        return false;
    }
    
    // Get affected particles from pattern
    auto affectedParticles = pattern->GetAffectedParticles(
        m_ParticlePositions,
        position,
        direction,
        scale
    );
    
    if (affectedParticles.empty()) {
        std::cout << "PhysXCloth: Pattern '" << pattern->GetName() 
                  << "' did not affect any particles" << std::endl;
        return false;
    }
    
    // Apply tears to affected particles
    int tearsApplied = 0;
    for (int particleIndex : affectedParticles) {
        if (TearAtParticle(particleIndex)) {
            tearsApplied++;
        }
    }
    
    std::cout << "PhysXCloth: Applied pattern '" << pattern->GetName() 
              << "' affecting " << affectedParticles.size() << " particles, "
              << tearsApplied << " tears created" << std::endl;
    
    return tearsApplied > 0;
}

void PhysXCloth::StartProgressiveTear(
    std::shared_ptr<ClothTearPattern> pattern,
    const Vec3& position,
    const Vec3& direction,
    float duration,
    float scale
) {
    if (!pattern || !m_Tearable) {
        return;
    }
    
    ProgressiveTear tear;
    tear.pattern = pattern;
    tear.position = position;
    tear.direction = direction;
    tear.scale = scale;
    tear.progress = 0.0f;
    tear.duration = duration;
    tear.elapsed = 0.0f;
    
    m_ProgressiveTears.push_back(tear);
    
    std::cout << "PhysXCloth: Started progressive tear with pattern '" 
              << pattern->GetName() << "' over " << duration << " seconds" << std::endl;
}

void PhysXCloth::StartProgressiveTear(
    const std::string& patternName,
    const Vec3& position,
    const Vec3& direction,
    float duration,
    float scale
) {
    auto pattern = GetPatternLibrary().GetPattern(patternName);
    if (!pattern) {
        std::cerr << "PhysXCloth: Pattern '" << patternName << "' not found in library" << std::endl;
        return;
    }
    
    StartProgressiveTear(pattern, position, direction, duration, scale);
}

void PhysXCloth::UpdateProgressiveTears(float deltaTime) {
    if (m_ProgressiveTears.empty()) {
        return;
    }
    
    SCOPED_PROFILE("PhysXCloth::UpdateProgressiveTears");
    
    // Update each progressive tear
    for (auto it = m_ProgressiveTears.begin(); it != m_ProgressiveTears.end(); ) {
        ProgressiveTear& tear = *it;
        
        tear.elapsed += deltaTime;
        tear.progress = std::min(1.0f, tear.elapsed / tear.duration);
        
        // Get affected particles at current progress
        auto affectedParticles = tear.pattern->GetAffectedParticles(
            m_ParticlePositions,
            tear.position,
            tear.direction,
            tear.scale * tear.progress  // Scale pattern by progress
        );
        
        // Apply tears to newly affected particles
        for (int particleIndex : affectedParticles) {
            // Check if particle hasn't been torn yet
            if (std::find(m_TornParticles.begin(), m_TornParticles.end(), particleIndex) 
                == m_TornParticles.end()) {
                TearAtParticle(particleIndex);
            }
        }
        
        // Remove completed tears
        if (tear.progress >= 1.0f) {
            std::cout << "PhysXCloth: Completed progressive tear with pattern '" 
                      << tear.pattern->GetName() << "'" << std::endl;
            it = m_ProgressiveTears.erase(it);
        } else {
            ++it;
        }
    }
}

#endif // USE_PHYSX
