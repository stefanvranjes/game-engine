#include "PhysXCloth.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "ClothMeshSplitter.h"
#include "Mesh.h"
#include "AsyncClothFactory.h"
#include <PxPhysicsAPI.h>
#include <extensions/PxClothFabricCooker.h>
#include <iostream>

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
{
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

    // Create cloth fabric
    CreateClothFabric(desc);

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

    std::cout << "PhysXCloth initialized with " << m_ParticleCount << " particles" << std::endl;
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
}

void PhysXCloth::Update(float deltaTime) {
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

    // Update particle data from simulation
    UpdateParticleData();
}

void PhysXCloth::UpdateParticleData() {
    if (!m_Cloth) return;

    // Get particle positions from PhysX
    PxClothReadData* readData = m_Cloth->lockClothReadData();
    if (readData) {
        const PxClothParticle* particles = readData->particles;
        for (int i = 0; i < m_ParticleCount; ++i) {
            m_ParticlePositions[i] = Vec3(
                particles[i].pos.x,
                particles[i].pos.y,
                particles[i].pos.z
            );
        }
        readData->unlock();
    }

    // Recalculate normals for rendering
    RecalculateNormals();
}

void PhysXCloth::RecalculateNormals() {
    // Reset normals
    for (int i = 0; i < m_ParticleCount; ++i) {
        m_ParticleNormals[i] = Vec3(0, 0, 0);
    }

    // Calculate face normals and accumulate
    for (int i = 0; i < m_TriangleCount; ++i) {
        int i0 = m_TriangleIndices[i * 3 + 0];
        int i1 = m_TriangleIndices[i * 3 + 1];
        int i2 = m_TriangleIndices[i * 3 + 2];

        Vec3 v0 = m_ParticlePositions[i0];
        Vec3 v1 = m_ParticlePositions[i1];
        Vec3 v2 = m_ParticlePositions[i2];

        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 normal = edge1.Cross(edge2);

        m_ParticleNormals[i0] = m_ParticleNormals[i0] + normal;
        m_ParticleNormals[i1] = m_ParticleNormals[i1] + normal;
        m_ParticleNormals[i2] = m_ParticleNormals[i2] + normal;
    }

    // Normalize
    for (int i = 0; i < m_ParticleCount; ++i) {
        float length = m_ParticleNormals[i].Length();
        if (length > 0.001f) {
            m_ParticleNormals[i] = m_ParticleNormals[i] * (1.0f / length);
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
        PxClothCollisionSphere sphere;
        sphere.pos = PxVec3(center.x, center.y, center.z);
        sphere.radius = radius;
        m_Cloth->addCollisionSphere(sphere);
    }
}

void PhysXCloth::AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) {
    if (m_Cloth) {
        PxClothCollisionSphere sphere0, sphere1;
        sphere0.pos = PxVec3(p0.x, p0.y, p0.z);
        sphere0.radius = radius;
        sphere1.pos = PxVec3(p1.x, p1.y, p1.z);
        sphere1.radius = radius;
        
        m_Cloth->addCollisionSphere(sphere0);
        m_Cloth->addCollisionSphere(sphere1);
        m_Cloth->addCollisionCapsule(0, 1); // Connect last two spheres
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

// Tearing helper methods

void PhysXCloth::DetectTears(float deltaTime) {
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
    if (!mesh || !m_Cloth) {
        return;
    }

    // Update particle data from simulation
    UpdateParticleData();

    // Get mesh vertices and normals
    auto& vertices = mesh->GetVertices();
    
    // Resize if needed
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

// Mesh splitting methods

bool PhysXCloth::SplitAtParticle(
    int tearParticle,
    std::shared_ptr<PhysXCloth>& outPiece1,
    std::shared_ptr<PhysXCloth>& outPiece2)
{
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
    
    outPiece2->SetStretchStiffness(m_StretchStiffness);
    outPiece2->SetBendStiffness(m_BendStiffness);
    outPiece2->SetShearStiffness(m_ShearStiffness);
    outPiece2->SetDamping(m_Damping);
    outPiece2->SetWindVelocity(m_WindVelocity);
    outPiece2->SetTearable(m_Tearable);
    outPiece2->SetMaxStretchRatio(m_MaxStretchRatio);
    
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
    
    outPiece2->SetStretchStiffness(m_StretchStiffness);
    outPiece2->SetBendStiffness(m_BendStiffness);
    outPiece2->SetShearStiffness(m_ShearStiffness);
    outPiece2->SetDamping(m_Damping);
    outPiece2->SetWindVelocity(m_WindVelocity);
    outPiece2->SetTearable(m_Tearable);
    outPiece2->SetMaxStretchRatio(m_MaxStretchRatio);
    
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

#endif // USE_PHYSX
