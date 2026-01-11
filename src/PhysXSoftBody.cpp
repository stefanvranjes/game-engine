#include "PhysXSoftBody.h"

#ifdef USE_PHYSX

#include "PhysXBackend.h"
#include "PhysXRigidBody.h"
#include "SoftBodyTearSystem.h"
#include "PartialTearSystem.h"
#include "CrackRenderer.h"
#include "TetrahedralMeshSplitter.h"
#include "TearResistanceMap.h"
#include "FractureLine.h"
#include "SoftBodyTearPattern.h"
#include "StraightTearPattern.h"
#include "CurvedTearPattern.h"
#include "RadialTearPattern.h"
#include "SoftBodyLOD.h"
#include "SoftBodyLODManager.h"
#include "SoftBodySerializer.h"
#include "SoftBodyDeformationRecorder.h"
#include "PhysXManager.h"
#include "GLExtensions.h"
#include "GpuProfiler.h"
#include <PxPhysicsAPI.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <execution>

using namespace physx;

namespace {
    // 3x3 Symmetric Matrix Eigendecomposition using Jacobi algorithm
    struct Mat3Sym {
        float m[3][3];

        void SetZero() {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    m[i][j] = 0.0f;
        }

        // Add outer product of v to matrix: M += v * v^T
        void AddOuterProduct(const Vec3& v) {
            m[0][0] += v.x * v.x; m[0][1] += v.x * v.y; m[0][2] += v.x * v.z;
            m[1][0] += v.y * v.x; m[1][1] += v.y * v.y; m[1][2] += v.y * v.z;
            m[2][0] += v.z * v.x; m[2][1] += v.z * v.y; m[2][2] += v.z * v.z;
        }

        void Scale(float s) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    m[i][j] *= s;
        }
    };

    void ComputeEigenDecomposition(const Mat3Sym& A, Vec3& eigenValues, Vec3 eigenVectors[3]) {
        // Initialize identity matrix for eigenvectors
        eigenVectors[0] = Vec3(1, 0, 0);
        eigenVectors[1] = Vec3(0, 1, 0);
        eigenVectors[2] = Vec3(0, 0, 1);

        // Copy matrix
        float D[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                D[i][j] = A.m[i][j];

        const int MAX_ITER = 50;
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            // Find largest off-diagonal element
            int p = 0, q = 1;
            float maxOff = std::abs(D[0][1]);
            if (std::abs(D[0][2]) > maxOff) { p = 0; q = 2; maxOff = std::abs(D[0][2]); }
            if (std::abs(D[1][2]) > maxOff) { p = 1; q = 2; maxOff = std::abs(D[1][2]); }

            if (maxOff < 1e-6f) break; // Converged

            // Jacobi rotation
            float phi;
            float diff = D[q][q] - D[p][p];
            if (std::abs(diff) < 1e-6f) {
                phi = 3.14159265f / 4.0f;
            } else {
                phi = 0.5f * std::atan2(2.0f * D[p][q], diff);
            }

            float c = std::cos(phi);
            float s = std::sin(phi);

            // Update diagonal elements
             float temp = D[p][p];
            D[p][p] = c * c * temp - 2.0f * s * c * D[p][q] + s * s * D[q][q];
            D[q][q] = s * s * temp + 2.0f * s * c * D[p][q] + c * c * D[q][q];
            D[p][q] = 0.0f;
            D[q][p] = 0.0f;

            // Update off-diagonal elements
            for (int r = 0; r < 3; ++r) {
                if (r != p && r != q) {
                    float Arp = D[r][p];
                    float Arq = D[r][q];
                    D[r][p] = c * Arp - s * Arq;
                    D[p][r] = D[r][p];
                    D[r][q] = s * Arp + c * Arq;
                    D[q][r] = D[r][q];
                }
            }

            // Update eigenvectors
            auto RotateComp = [&](float& x, float& y, float c, float s) {
                float tx = x; float ty = y;
                x = c * tx - s * ty;
                y = s * tx + c * ty;
            };

            RotateComp(eigenVectors[p].x, eigenVectors[q].x, c, s);
            RotateComp(eigenVectors[p].y, eigenVectors[q].y, c, s);
            RotateComp(eigenVectors[p].z, eigenVectors[q].z, c, s);
        }

        eigenValues.x = D[0][0];
        eigenValues.y = D[1][1];
        eigenValues.z = D[2][2];
    }
}


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
    , m_SphereGenerationScale(1.0f, 1.0f, 1.0f)
    , m_ElongationThreshold(3.0f)
    , m_UseAnisotropicMaterial(false)
    , m_TopologyDirty(true)
    , m_DebugDrawHull(false)
    , m_DebugDrawSurface(false)
    , m_DebugDrawCollisionSpheres(false)
    , m_DebugResourcesInitialized(false)
    , m_DebugVAO(0)
    , m_DebugVBO(0)
    , m_DebugSurfaceVAO(0)
    , m_DebugSurfaceVBO(0)
    , m_DebugSpheresVAO(0)
    , m_DebugSpheresVBO(0)
    , m_DebugHullDirty(true)
    , m_AutoTuningEnabled(true)
    , m_TargetFrameTimeMs(2.0) // 2ms budget per soft body
    , m_OriginalMaxCollisionSpheres(32)
    , m_OriginalTearCheckInterval(0.1f)
    , m_LODEnabled(false)
    , m_CameraPosition(0, 0, 0)
    , m_UseDirectGpuApi(false)
    , m_GpuMemoryUsage(0)
    , m_GpuSimulationTimeMs(0.0f)
    , m_CpuGpuTransferTimeMs(0.0f)
    , m_IsPlayingBack(false)
    , m_RecordingStartTime(0.0f)
    , m_PartialTearingEnabled(false)
    , m_CrackThreshold(1.2f)
    , m_CrackProgressionRate(0.1f)
    , m_SimulationTime(0.0f)
    , m_CrackVisualizationEnabled(false)
{
    m_TearSystem = std::make_unique<SoftBodyTearSystem>();
    m_PartialTearSystem = std::make_unique<PartialTearSystem>();
    m_CrackRenderer = std::make_unique<CrackRenderer>();
    m_LODManager = std::make_unique<SoftBodyLODManager>();
    m_Recorder = std::make_unique<SoftBodyDeformationRecorder>();
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
    
    // Apply detailed settings
    const auto& details = desc.detailSettings;
    m_SphereGenerationScale = details.sphereGenerationScale;
    
    // Adaptive sphere count settings
    m_UseAdaptiveSphereCount = details.useAdaptiveSphereCount;
    m_MinCollisionSpheres = details.minCollisionSpheres;
    m_MaxCollisionSpheres = details.maxCollisionSpheres;
    m_VerticesPerSphere = details.verticesPerSphere;
    m_AdaptiveVertexWeight = details.adaptiveVertexWeight;
    m_AdaptiveAreaWeight = details.adaptiveAreaWeight;
    m_AreaPerSphere = details.areaPerSphere;
    
    // Algorithms
    m_SurfaceAreaMode = details.surfaceAreaMode;
    m_HullAlgorithm = details.hullAlgorithm;
    
    // Cloth collision
    m_ClothCollisionEnabled = details.enableClothCollision;
    m_CollisionSphereRadius = details.collisionSphereRadius;
    
    // Store original values for auto-tuning
    m_OriginalMaxCollisionSpheres = m_MaxCollisionSpheres;
    m_OriginalTearCheckInterval = m_TearCheckInterval;
    
    // Create tetrahedral mesh
    CreateTetrahedralMesh(desc);
    
    if (!m_TetraMesh) {
        std::cerr << "PhysXSoftBody: Failed to create tetrahedral mesh!" << std::endl;
        return;
    }
    
    // Validate GPU support (PhysX 5.x soft bodies require GPU)
    if (!m_Backend->IsGpuSoftBodySupported()) {
        std::cerr << "PhysXSoftBody: GPU soft body support not available!" << std::endl;
        std::cerr << "  PhysX 5.x soft bodies require a CUDA-capable NVIDIA GPU." << std::endl;
        std::cerr << "  Please ensure:" << std::endl;
        std::cerr << "    - NVIDIA GPU with compute capability >= 3.0" << std::endl;
        std::cerr << "    - At least 512MB free GPU memory" << std::endl;
        std::cerr << "    - Latest NVIDIA drivers installed" << std::endl;
        
        // Clean up tetrahedral mesh
        if (m_TetraMesh) {
            m_TetraMesh->release();
            m_TetraMesh = nullptr;
        }
        return;
    }
    
    // Enable direct GPU API by default if GPU is available
    m_UseDirectGpuApi = m_Backend->IsGpuEnabled();
    
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
    
    // Calculate GPU memory usage
    // Estimate: positions (4 floats), velocities (4 floats), rest positions, etc.
    size_t vertexDataSize = m_VertexCount * sizeof(float) * 4 * 3; // pos, vel, rest pos
    size_t tetraDataSize = m_TetrahedronCount * sizeof(int) * 4;
    m_GpuMemoryUsage = vertexDataSize + tetraDataSize;
    
    std::cout << "PhysXSoftBody initialized with " << m_VertexCount << " vertices and "
              << m_TetrahedronCount << " tetrahedra" << std::endl;
    std::cout << "  GPU Memory Usage: " << (m_GpuMemoryUsage / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Direct GPU API: " << (m_UseDirectGpuApi ? "Enabled" : "Disabled") << std::endl;
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
    GPU_PROFILE_SCOPE_COLOR("PhysXSoftBody::Update", GpuProfiler::COLOR_SOFT_BODY);
    
    auto startTotal = std::chrono::high_resolution_clock::now();
    
    if (!m_SoftBody || !m_Enabled) {
        return;
    }
    
    // Track simulation time for crack system
    m_SimulationTime += deltaTime;
    
    // Handle playback mode
    if (m_IsPlayingBack && m_Recorder) {
        std::vector<Vec3> playbackPositions(m_VertexCount);
        bool stillPlaying = m_Recorder->UpdatePlayback(deltaTime, playbackPositions.data());
        
        if (stillPlaying) {
            // Apply playback positions
            SetVertexPositions(playbackPositions.data());
            
            // Skip physics simulation during playback
            return;
        } else {
            // Playback finished
            m_IsPlayingBack = false;
        }
    }
    
    // Update LOD system if enabled
    {
        GPU_PROFILE_SCOPE_COLOR("LOD Update", GpuProfiler::COLOR_LOD);
        
        if (m_LODEnabled && m_LODManager) {
            bool lodChanged = m_LODManager->UpdateLOD(this, m_CameraPosition, deltaTime);
            
            // Check if we should skip update based on LOD frequency
            if (!m_LODManager->ShouldUpdateThisFrame()) {
                return;  // Skip this frame's update
            }
            
            // If LOD changed, the mesh may have been recreated
            // State transfer is handled by the LOD manager
            if (lodChanged) {
                // Invalidate cached data
                m_CollisionSpheresNeedUpdate = true;
                m_SurfaceAreaNeedsUpdate = true;
                m_TopologyDirty = true;
            }
        }
    }
    
    // Mark collision spheres and surface area as needing update (soft body has deformed)
    if (m_ClothCollisionEnabled) {
        m_CollisionSpheresNeedUpdate = true;
        m_SurfaceAreaNeedsUpdate = true;
    }
    
    // Update collision shapes
    {
        GPU_PROFILE_SCOPE_COLOR("Collision Generation", GpuProfiler::COLOR_COLLISION);
        
        auto startCollision = std::chrono::high_resolution_clock::now();
        UpdateCollisionShapes();
        auto endCollision = std::chrono::high_resolution_clock::now();
        m_Stats.collisionGenTimeMs = std::chrono::duration<double, std::milli>(endCollision - startCollision).count();
    }
    
    // Update healing and plasticity
    m_LastTearCheckTime += deltaTime;
    
    auto startTear = std::chrono::high_resolution_clock::now();
    bool ranTearCheck = false;
    
    if (m_TearSystem) {
        // Healing runs every frame (usually cheap)
        m_TearSystem->UpdateHealing(deltaTime, m_ResistanceMap);
        
        // Plasticity/Tearing check is expensive, so throttle it
        if (m_LastTearCheckTime >= m_TearCheckInterval) {
            ranTearCheck = true;
            m_LastTearCheckTime = 0.0f;
            
            // Update plasticity
            if (!m_RestPositions.empty()) {
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
        }
    }
    auto endTear = std::chrono::high_resolution_clock::now();
    
    if (ranTearCheck) {
        m_Stats.tearCheckTimeMs = std::chrono::duration<double, std::milli>(endTear - startTear).count();
    }
    
    // PhysX soft body simulation is handled automatically by the scene
    
    // Handle recording mode
    if (m_Recorder && m_Recorder->IsRecording()) {
        m_RecordingStartTime += deltaTime;
        
        std::vector<Vec3> positions(m_VertexCount);
        std::vector<Vec3> velocities(m_VertexCount);
        
        GetVertexPositions(positions.data());
        GetVertexVelocities(velocities.data());
        
        m_Recorder->RecordFrame(
            positions.data(),
            m_Recorder->HasVelocities() ? velocities.data() : nullptr,
            m_RecordingStartTime
        );
    }
    
    auto endTotal = std::chrono::high_resolution_clock::now();
    m_Stats.updateTimeMs = std::chrono::duration<double, std::milli>(endTotal - startTotal).count();
    
    if (m_AutoTuningEnabled) {
        AutoTuneParameters();
    }
}

void PhysXSoftBody::AutoTuneParameters() {
    // If total update time exceeds budget, reduce quality
    if (m_Stats.updateTimeMs > m_TargetFrameTimeMs) {
        // Reduce collision spheres if they are expensive
        if (m_ClothCollisionEnabled && m_MaxCollisionSpheres > m_MinCollisionSpheres) {
            m_MaxCollisionSpheres = std::max(m_MinCollisionSpheres, m_MaxCollisionSpheres - 1);
        }
        
        // Increase tear check interval if tearing is expensive
        // (Note: Currently we aren't using the interval in the Update loop above, but we should)
        if (m_Stats.tearCheckTimeMs > 0.5) { // If tearing takes > 0.5ms
             m_TearCheckInterval *= 1.1f; // Increase interval by 10%
        }
    } else if (m_Stats.updateTimeMs < m_TargetFrameTimeMs * 0.5) {
        // If we represent a small fraction of the budget, slowly restore quality
        
        // Restore collision spheres
        if (m_MaxCollisionSpheres < m_OriginalMaxCollisionSpheres) {
            // Only increase occasionally to avoid oscillation
            if (rand() % 100 < 5) { 
                m_MaxCollisionSpheres++;
            }
        }
        
        // Restore tear check interval
        if (m_TearCheckInterval > m_OriginalTearCheckInterval) {
            m_TearCheckInterval *= 0.95f;
            if (m_TearCheckInterval < m_OriginalTearCheckInterval) {
                m_TearCheckInterval = m_OriginalTearCheckInterval;
            }
        }
    }
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

void PhysXSoftBody::SetVertexVelocities(const Vec3* velocities, int count) {
    if (!m_SoftBody || !velocities || count != m_VertexCount) {
        return;
    }
    
    PxSoftBodyData* data = m_SoftBody->getSoftBodyData();
    if (data) {
        PxVec4* vertexVelocities = data->getVelocity();
        for (int i = 0; i < count; ++i) {
            vertexVelocities[i].x = velocities[i].x;
            vertexVelocities[i].y = velocities[i].y;
            vertexVelocities[i].z = velocities[i].z;
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
    
    // Get stress data
    const auto& stressData = m_TearSystem->GetStressData();
    
    // 1. Handle partial tearing (cracks)
    if (m_PartialTearingEnabled && m_PartialTearSystem) {
        // Detect new cracks
        m_PartialTearSystem->DetectCracks(
            currentPositions.data(),
            m_RestPositions.data(),
            m_TetrahedronIndices.data(),
            m_TetrahedronCount,
            stressData.data(),
            m_CrackThreshold,
            m_TearThreshold,
            m_SimulationTime
        );
        
        // Progress existing cracks
        m_PartialTearSystem->ProgressCracks(
            0.016f,  // Approximate deltaTime (will be improved)
            stressData.data(),
            m_CrackProgressionRate,
            m_SimulationTime
        );
        
        // Check for fully damaged cracks that should convert to tears
        auto fullyDamagedTets = m_PartialTearSystem->GetFullyDamagedTets();
        if (!fullyDamagedTets.empty()) {
            std::cout << "Converting " << fullyDamagedTets.size() << " fully damaged cracks to tears" << std::endl;
            
            // Remove cracks before tearing
            m_PartialTearSystem->RemoveCracks(fullyDamagedTets);
            
            // Split along fully damaged cracks
            SplitSoftBody(fullyDamagedTets);
            return;  // Don't process regular tears this frame
        }
    }
    
    // 2. Detect regular tears (stress > tear threshold)
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
    
    if (tornTetrahedra.empty() || m_RestPositions.empty()) {
        std::cerr << "Cannot split: no tears or mesh not initialized" << std::endl;
        return;
    }
    
    // Get current positions and velocities
    std::vector<Vec3> currentPositions(m_VertexCount);
    std::vector<Vec3> currentVelocities(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    GetVertexVelocities(currentVelocities.data());
    
    // Build tear info from torn tetrahedra
    std::vector<SoftBodyTearSystem::TearInfo> tears;
    for (int tetIdx : tornTetrahedra) {
        if (tetIdx >= 0 && tetIdx < m_TetrahedronCount) {
            const auto& stressData = m_TearSystem->GetStressData();
            if (tetIdx < static_cast<int>(stressData.size())) {
                SoftBodyTearSystem::TearInfo tear;
                tear.tetrahedronIndex = tetIdx;
                tear.stress = 0.0f;
                
                // Find the most stressed edge
                int maxEdgeIdx = 0;
                for (int i = 1; i < 6; ++i) {
                    if (stressData[tetIdx].edgeStress[i] > stressData[tetIdx].edgeStress[maxEdgeIdx]) {
                        maxEdgeIdx = i;
                    }
                }
                
                // Get edge vertices (simplified - using first edge)
                const int* tet = &m_TetrahedronIndices[tetIdx * 4];
                tear.edgeVertices[0] = tet[0];
                tear.edgeVertices[1] = tet[1];
                tear.stress = stressData[tetIdx].edgeStress[maxEdgeIdx];
                tear.tearPosition = (currentPositions[tear.edgeVertices[0]] + 
                                    currentPositions[tear.edgeVertices[1]]) * 0.5f;
                tear.tearNormal = Vec3(0, 1, 0);  // Placeholder
                tear.timestamp = 0.0f;
                
                tears.push_back(tear);
            }
        }
    }
    
    if (tears.empty()) {
        std::cerr << "No valid tears to process" << std::endl;
        return;
    }
    
    // Perform mesh split with state transfer
    std::vector<Vec3> velocities1, velocities2;
    auto result = TetrahedralMeshSplitter::SplitWithStateTransfer(
        currentPositions.data(),
        m_VertexCount,
        m_TetrahedronIndices.data(),
        m_TetrahedronCount,
        tears,
        currentVelocities.data(),
        velocities1,
        velocities2
    );
    
    if (!result.splitSuccessful) {
        std::cerr << "Mesh split failed" << std::endl;
        return;
    }
    
    std::cout << "Split successful: piece1=" << result.piece1VertexCount 
             << " verts, piece2=" << result.piece2VertexCount << " verts" << std::endl;
    
    // Check if piece 2 is large enough to create
    float minPieceSizeRatio = 0.05f;  // At least 5% of original
    float piece2Ratio = static_cast<float>(result.piece2VertexCount) / m_VertexCount;
    
    if (piece2Ratio < minPieceSizeRatio) {
        std::cout << "Piece 2 too small (" << (piece2Ratio * 100.0f) << "%), not creating" << std::endl;
        return;
    }
    
    // Create new soft body for piece 2
    auto newPiece = std::make_shared<PhysXSoftBody>(m_Backend);
    
    // Initialize piece 2 with split mesh
    SoftBodyDesc desc2;
    desc2.tetrahedronVertices = result.vertices2.data();
    desc2.tetrahedronVertexCount = result.piece2VertexCount;
    desc2.tetrahedronIndices = result.tetrahedra2.data();
    desc2.tetrahedronCount = static_cast<int>(result.tetrahedra2.size() / 4);
    
    // Copy material properties from original
    desc2.volumeStiffness = m_VolumeStiffness;
    desc2.shapeStiffness = m_ShapeStiffness;
    desc2.deformationStiffness = m_DeformationStiffness;
    desc2.totalMass = m_TotalMass * piece2Ratio;
    desc2.useDensity = false;
    desc2.enableSceneCollision = m_SceneCollisionEnabled;
    desc2.gravity = Vec3(0, -9.81f, 0);
    
    if (newPiece->Initialize(desc2)) {
        // Set velocities for piece 2
        newPiece->SetVertexVelocities(velocities2.data(), static_cast<int>(velocities2.size()));
        
        // Notify callback
        if (m_PieceCreatedCallback) {
            m_PieceCreatedCallback(newPiece);
        }
        
        std::cout << "Created new piece with " << desc2.tetrahedronCount << " tetrahedra" << std::endl;
    } else {
        std::cerr << "Failed to initialize new piece" << std::endl;
    }
    
    // Update current soft body with piece 1
    RecreateWithMesh(result.vertices1, result.tetrahedra1, velocities1);
}

void PhysXSoftBody::RecreateWithMesh(const std::vector<Vec3>& vertices, const std::vector<int>& tetrahedra) {
    std::vector<Vec3> velocities(vertices.size(), Vec3(0, 0, 0));
    RecreateWithMesh(vertices, tetrahedra, velocities);
}

void PhysXSoftBody::RecreateWithMesh(const std::vector<Vec3>& vertices, const std::vector<int>& tetrahedra, const std::vector<Vec3>& velocities) {
    std::cout << "Recreating soft body with new mesh: " << vertices.size() 
             << " vertices, " << (tetrahedra.size() / 4) << " tetrahedra" << std::endl;
    
    if (vertices.empty() || tetrahedra.empty()) {
        std::cerr << "Cannot recreate: invalid mesh data" << std::endl;
        return;
    }
    
    // Store current state
    std::vector<int> fixedVertices = m_FixedVertices;
    float volumeStiffness = m_VolumeStiffness;
    float shapeStiffness = m_ShapeStiffness;
    float deformationStiffness = m_DeformationStiffness;
    float totalMass = m_TotalMass;
    bool sceneCollision = m_SceneCollisionEnabled;
    bool selfCollision = m_SelfCollisionEnabled;
    
    // Release current PhysX actor
    if (m_SoftBody) {
        PxScene* scene = m_Backend->GetScene();
        if (scene) {
            scene->removeActor(*m_SoftBody);
        }
        m_SoftBody->release();
        m_SoftBody = nullptr;
    }
    
    // Update internal mesh data
    m_VertexCount = static_cast<int>(vertices.size());
    m_TetrahedronCount = static_cast<int>(tetrahedra.size() / 4);
    m_TetrahedronIndices = tetrahedra;
    m_RestPositions = vertices;
    m_InitialPositions = vertices;
    
    // Create new soft body descriptor
    SoftBodyDesc desc;
    desc.tetrahedronVertices = vertices.data();
    desc.tetrahedronVertexCount = m_VertexCount;
    desc.tetrahedronIndices = tetrahedra.data();
    desc.tetrahedronCount = m_TetrahedronCount;
    desc.volumeStiffness = volumeStiffness;
    desc.shapeStiffness = shapeStiffness;
    desc.deformationStiffness = deformationStiffness;
    desc.totalMass = totalMass;
    desc.useDensity = false;
    desc.enableSceneCollision = sceneCollision;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    // Reinitialize PhysX actor
    if (!Initialize(desc)) {
        std::cerr << "Failed to reinitialize soft body" << std::endl;
        return;
    }
    
    // Restore velocities
    if (!velocities.empty() && velocities.size() == vertices.size()) {
        SetVertexVelocities(velocities.data(), static_cast<int>(velocities.size()));
    }
    
    // Restore fixed vertices (note: indices may have changed, so this is approximate)
    // In a production system, you'd need to track vertex mappings
    for (int vertexIdx : fixedVertices) {
        if (vertexIdx < m_VertexCount) {
            FixVertex(vertexIdx, vertices[vertexIdx]);
        }
    }
    
    // Reinitialize tear system
    if (m_TearSystem) {
        m_TearSystem->ClearStressData();
    }
    
    // Reinitialize resistance map
    if (m_ResistanceMap.IsInitialized()) {
        m_ResistanceMap.Reset();
    }
    
    std::cout << "Soft body recreated successfully" << std::endl;
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
    
    // Store fracture line
    m_FractureLines.push_back(fractureLine);
}

void PhysXSoftBody::ClearFractureLines() {
    // Clear stored fracture lines
    m_FractureLines.clear();
    
    // Reset resistance map to default
    if (m_ResistanceMap.IsInitialized()) {
        m_ResistanceMap.Reset();
        std::cout << "Fracture lines cleared" << std::endl;
    }
}

const std::vector<FractureLine>& PhysXSoftBody::GetFractureLines() const {
    return m_FractureLines;
}

FractureLine* PhysXSoftBody::GetFractureLine(int index) {
    if (index >= 0 && index < static_cast<int>(m_FractureLines.size())) {
        return &m_FractureLines[index];
    }
    return nullptr;
}

void PhysXSoftBody::RemoveFractureLine(int index) {
    if (index >= 0 && index < static_cast<int>(m_FractureLines.size())) {
        m_FractureLines.erase(m_FractureLines.begin() + index);
        
        // Reapply all remaining fracture lines
        if (m_ResistanceMap.IsInitialized()) {
            m_ResistanceMap.Reset();
            
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
            
            // Reapply all fracture lines
            for (const auto& line : m_FractureLines) {
                line.ApplyToResistanceMap(m_ResistanceMap, tetCenters.data(), m_TetrahedronCount);
            }
        }
    }
}

void PhysXSoftBody::UpdateFractureLine(int index) {
    if (index < 0 || index >= static_cast<int>(m_FractureLines.size())) {
        return;
    }
    
    if (!m_ResistanceMap.IsInitialized()) {
        std::cerr << "Resistance map not initialized" << std::endl;
        return;
    }
    
    // Reset resistance map and reapply all fracture lines
    m_ResistanceMap.Reset();
    
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
    
    // Reapply all fracture lines
    for (const auto& line : m_FractureLines) {
        line.ApplyToResistanceMap(m_ResistanceMap, tetCenters.data(), m_TetrahedronCount);
    }
}


// Anisotropic Material Implementation

void PhysXSoftBody::EnableAnisotropy(bool enable) {
    m_UseAnisotropicMaterial = enable;
    
    if (enable && !m_AnisotropicMaterial) {
        // Create and initialize anisotropic material
        m_AnisotropicMaterial = std::make_unique<AnisotropicMaterial>();
        m_AnisotropicMaterial->Initialize(m_TetrahedronCount);
    }
}

void PhysXSoftBody::SetFiberDirection(int tetIndex, const Vec3& direction, float longStiff, float transStiff) {
    if (!m_AnisotropicMaterial) {
        EnableAnisotropy(true);
    }
    
    m_AnisotropicMaterial->SetFiberDirection(tetIndex, direction, longStiff, transStiff);
}

void PhysXSoftBody::SetUniformFiberDirection(const Vec3& direction, float longStiff, float transStiff) {
    if (!m_AnisotropicMaterial) {
        EnableAnisotropy(true);
    }
    
    m_AnisotropicMaterial->SetUniformFiberDirection(direction, longStiff, transStiff);
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
        
        // Strategy: Use PCA to determine Principal Axes and align grid to soft body shape
        // This provides much better packing for elongated or flat objects
        
        // 1. Calculate Mean and Covariance
        Vec3 mean(0, 0, 0);
        for (const Vec3& p : currentPositions) {
            mean = mean + p;
        }
        mean = mean * (1.0f / m_VertexCount);

        Mat3Sym covariance;
        covariance.SetZero();
        for (const Vec3& p : currentPositions) {
            Vec3 diff = p - mean;
            covariance.AddOuterProduct(diff);
        }
        covariance.Scale(1.0f / m_VertexCount);

        // 2. Eigen Decomposition
        Vec3 eigenValues;
        Vec3 eigenVectors[3];
        ComputeEigenDecomposition(covariance, eigenValues, eigenVectors);

        // Ensure non-zero eigenvalues for extent calculation (sqrt of variance)
        eigenValues.x = std::max(eigenValues.x, 1e-6f);
        eigenValues.y = std::max(eigenValues.y, 1e-6f);
        eigenValues.z = std::max(eigenValues.z, 1e-6f);

        Vec3 extents(std::sqrt(eigenValues.x), std::sqrt(eigenValues.y), std::sqrt(eigenValues.z));

        // Apply per-axis scale scaling to extents for grid resolution calculation
        Vec3 scaledExtents = extents;
        scaledExtents.x *= m_SphereGenerationScale.x;
        scaledExtents.y *= m_SphereGenerationScale.y;
        scaledExtents.z *= m_SphereGenerationScale.z;

        // 3. Determine Grid Dimensions (Nx, Ny, Nz)
        // We want Nx * Ny * Nz approx targetSphereCount
        // And Nx:Ny:Nz approx extents.x:extents.y:extents.z
        
        float product = scaledExtents.x * scaledExtents.y * scaledExtents.z;
        float scaleFactor = std::pow(static_cast<float>(targetSphereCount) / product, 1.0f / 3.0f);
        
        int gridCounts[3];
        gridCounts[0] = std::max(1, static_cast<int>(scaledExtents.x * scaleFactor + 0.5f));
        gridCounts[1] = std::max(1, static_cast<int>(scaledExtents.y * scaleFactor + 0.5f));
        gridCounts[2] = std::max(1, static_cast<int>(scaledExtents.z * scaleFactor + 0.5f));
        
        // Adjust if total is too far off (simple iterative check) or just clamp
        while (gridCounts[0] * gridCounts[1] * gridCounts[2] > targetSphereCount * 2) {
            // Find max dim and reduce
            int maxAxis = 0;
            if (gridCounts[1] > gridCounts[maxAxis]) maxAxis = 1;
            if (gridCounts[2] > gridCounts[maxAxis]) maxAxis = 2;
            if (gridCounts[maxAxis] > 1) gridCounts[maxAxis]--;
            else break;
        }

        // 4. Cluster Vertices in Principal Frame
        // Project vertices into PCA space: p' = V^T * (p - mean)
        
        struct GridCell {
            std::vector<int> vertexIndices;
        };
        
        std::map<int, GridCell> grid;
        
        // Calculate bounds in PCA space to normalize grid coords
        float minPCA[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
        float maxPCA[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

        // We need to store projected points to avoid recomputing
        std::vector<Vec3> projectedPoints(m_VertexCount);

        for (int i = 0; i < m_VertexCount; ++i) {
            Vec3 diff = currentPositions[i] - mean;
            // Dot product with eigenvectors
            projectedPoints[i].x = diff.Dot(eigenVectors[0]);
            projectedPoints[i].y = diff.Dot(eigenVectors[1]);
            projectedPoints[i].z = diff.Dot(eigenVectors[2]);

            for (int k = 0; k < 3; k++) {
                 float val = (k == 0) ? projectedPoints[i].x : ((k == 1) ? projectedPoints[i].y : projectedPoints[i].z);
                 if (val < minPCA[k]) minPCA[k] = val;
                 if (val > maxPCA[k]) maxPCA[k] = val;
            }
        }
        
        // Add small epsilon to max to include boundary points
        for(int k=0; k<3; k++) maxPCA[k] += 1e-4f;

        // Bucket points
        for (int i = 0; i < m_VertexCount; ++i) {
            int idx[3];
            for (int k = 0; k < 3; k++) {
                float val = (k == 0) ? projectedPoints[i].x : ((k == 1) ? projectedPoints[i].y : projectedPoints[i].z);
                float range = maxPCA[k] - minPCA[k];
                if (range < 1e-9f) {
                    idx[k] = 0;
                } else {
                    float t = (val - minPCA[k]) / range;
                    idx[k] = static_cast<int>(t * gridCounts[k]);
                    if (idx[k] >= gridCounts[k]) idx[k] = gridCounts[k] - 1;
                }
            }
            
            // Flatten index
            int flatIdx = idx[0] + gridCounts[0] * (idx[1] + gridCounts[1] * idx[2]);
            grid[flatIdx].vertexIndices.push_back(i);
        }

        // 5. Generate Spheres from Clusters
        for (const auto& pair : grid) {
            const auto& cell = pair.second;
            if (cell.vertexIndices.empty()) continue;

            // Compute center in PCA space
            Vec3 centerPCA(0, 0, 0);
            for (int vIdx : cell.vertexIndices) {
                centerPCA = centerPCA + projectedPoints[vIdx];
            }
            centerPCA = centerPCA * (1.0f / cell.vertexIndices.size());

            // Compute radius as max distance from center in 3D
            // (Distance is invariant under rotation, so we can use PCA space distance)
            float maxDistSq = 0.0f;
            for (int vIdx : cell.vertexIndices) {
                Vec3 diff = projectedPoints[vIdx] - centerPCA;
                float dSq = diff.LengthSquared();
                if (dSq > maxDistSq) maxDistSq = dSq;
            }
            float radius = std::sqrt(maxDistSq) + m_CollisionSphereRadius;

            // Transform center back to World Space
            // centerWorld = mean + V * centerPCA
            Vec3 centerWorld = mean;
            centerWorld = centerWorld + eigenVectors[0] * centerPCA.x;
            centerWorld = centerWorld + eigenVectors[1] * centerPCA.y;
            centerWorld = centerWorld + eigenVectors[2] * centerPCA.z;

            m_CachedCollisionSpherePositions.push_back(centerWorld);
            m_CachedCollisionSphereRadii.push_back(radius);

            if (static_cast<int>(m_CachedCollisionSpherePositions.size()) >= targetSphereCount) break;
        }

        m_CollisionSpheresNeedUpdate = false;
    }
    
    // Copy cached spheres to output
    positions = m_CachedCollisionSpherePositions;
    radii = m_CachedCollisionSphereRadii;
    
    return static_cast<int>(positions.size());
}

int PhysXSoftBody::GetCollisionCapsules(std::vector<CollisionCapsule>& capsules, int maxCapsules) const {
    capsules.clear();
    
    if (!m_SoftBody || m_VertexCount < 4) {
        return 0;
    }
    
    // Get current vertex positions
    std::vector<Vec3> currentPositions(m_VertexCount);
    GetVertexPositions(currentPositions.data());
    
    // 1. Calculate Mean and Covariance
    Vec3 mean(0, 0, 0);
    for (const Vec3& p : currentPositions) {
        mean = mean + p;
    }
    mean = mean * (1.0f / m_VertexCount);

    Mat3Sym covariance;
    covariance.SetZero();
    for (const Vec3& p : currentPositions) {
        Vec3 diff = p - mean;
        covariance.AddOuterProduct(diff);
    }
    covariance.Scale(1.0f / m_VertexCount);

    // 2. Eigen Decomposition
    Vec3 eigenValues;
    Vec3 eigenVectors[3];
    ComputeEigenDecomposition(covariance, eigenValues, eigenVectors);

    // Ensure non-zero eigenvalues
    eigenValues.x = std::max(eigenValues.x, 1e-6f);
    eigenValues.y = std::max(eigenValues.y, 1e-6f);
    eigenValues.z = std::max(eigenValues.z, 1e-6f);

    // 3. Check for elongation (largest eigenvalue >> others)
    float maxEigenvalue = std::max(std::max(eigenValues.x, eigenValues.y), eigenValues.z);
    float minEigenvalue = std::min(std::min(eigenValues.x, eigenValues.y), eigenValues.z);
    
    float elongationRatio = maxEigenvalue / minEigenvalue;
    
    if (elongationRatio < m_ElongationThreshold) {
        // Not elongated enough, return 0 (caller should use spheres)
        return 0;
    }
    
    // 4. Determine principal axis (eigenvector with largest eigenvalue)
    int principalAxis = 0;
    if (eigenValues.y > eigenValues.x) principalAxis = 1;
    if (eigenValues.z > eigenValues[principalAxis]) principalAxis = 2;
    
    Vec3 principalDir = eigenVectors[principalAxis];
    
    // 5. Project all vertices onto principal axis to find extent
    float minProj = FLT_MAX;
    float maxProj = -FLT_MAX;
    
    for (const Vec3& p : currentPositions) {
        Vec3 diff = p - mean;
        float proj = diff.Dot(principalDir);
        minProj = std::min(minProj, proj);
        maxProj = std::max(maxProj, proj);
    }
    
    // 6. Calculate perpendicular radius (average of other two eigenvalues)
    float radius = 0.0f;
    int count = 0;
    for (int i = 0; i < 3; ++i) {
        if (i != principalAxis) {
            radius += std::sqrt(eigenValues[i]);
            count++;
        }
    }
    radius /= count;
    radius *= m_CollisionSphereRadius; // Apply user-defined scale
    
    // 7. Generate capsules along principal axis
    float axisLength = maxProj - minProj;
    int numCapsules = std::min(maxCapsules, std::max(1, static_cast<int>(axisLength / (radius * 4.0f))));
    
    if (numCapsules == 1) {
        // Single capsule spanning entire length
        CollisionCapsule cap;
        cap.p0 = mean + principalDir * minProj;
        cap.p1 = mean + principalDir * maxProj;
        cap.radius = radius;
        capsules.push_back(cap);
    } else {
        // Multiple capsules with overlap
        float segmentLength = axisLength / (numCapsules - 0.5f); // Slight overlap
        
        for (int i = 0; i < numCapsules; ++i) {
            float t0 = minProj + i * segmentLength;
            float t1 = t0 + segmentLength;
            
            CollisionCapsule cap;
            cap.p0 = mean + principalDir * t0;
            cap.p1 = mean + principalDir * t1;
            cap.radius = radius;
            capsules.push_back(cap);
        }
    }
    
    return static_cast<int>(capsules.size());
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
    
    // Extract surface faces using Parallel Sort
    
    // Check if topology is dirty or cache is empty
    if (m_TopologyDirty || m_CachedSurfaceFaces.empty()) {
        m_CachedSurfaceFaces.clear();
        
        // 1. Generate all faces (4 per tetrahedron)
        // 2. Sort faces
        // 3. Count duplicates (count=1 -> surface)
        
        struct Face {
            int v0, v1, v2;
            int originalTetIndex; // Debug/tracking
            
            Face() : v0(0), v1(0), v2(0), originalTetIndex(-1) {}
            Face(int a, int b, int c, int tetIdx) : originalTetIndex(tetIdx) {
                // Sort to canonical form
                if (a > b) std::swap(a, b);
                if (b > c) std::swap(b, c);
                if (a > b) std::swap(a, b);
                v0 = a; v1 = b; v2 = c;
            }
            
            bool operator<(const Face& other) const {
                if (v0 != other.v0) return v0 < other.v0;
                if (v1 != other.v1) return v1 < other.v1;
                return v2 < other.v2;
            }
            
            bool operator==(const Face& other) const {
                return v0 == other.v0 && v1 == other.v1 && v2 == other.v2;
            }
        };
        
        // Generate all faces
        std::vector<Face> allFaces;
        allFaces.resize(m_TetrahedronCount * 4);
        
        // Parallel generation (using parallel for_each with index)
        // Using simple loop for generation as it's fast linear write
        for (int i = 0; i < m_TetrahedronCount; ++i) {
            int v0 = m_TetrahedronIndices[i * 4 + 0];
            int v1 = m_TetrahedronIndices[i * 4 + 1];
            int v2 = m_TetrahedronIndices[i * 4 + 2];
            int v3 = m_TetrahedronIndices[i * 4 + 3];
            
            allFaces[i * 4 + 0] = Face(v0, v1, v2, i);
            allFaces[i * 4 + 1] = Face(v0, v1, v3, i);
            allFaces[i * 4 + 2] = Face(v0, v2, v3, i);
            allFaces[i * 4 + 3] = Face(v1, v2, v3, i);
        }
        
        // Parallel Sort
        std::sort(std::execution::par, allFaces.begin(), allFaces.end());
        
        int numFaces = (int)allFaces.size();
        if (numFaces == 0) return 0.0f;
        
        // Scan for surface faces
        for (int i = 0; i < numFaces; ) {
            int j = i + 1;
            while (j < numFaces && allFaces[i] == allFaces[j]) {
                j++;
            }
            
            // Count is j - i
            if ((j - i) == 1) {
                // Surface face found
                const Face& f = allFaces[i];
                m_CachedSurfaceFaces.push_back({f.v0, f.v1, f.v2});
            }
            
            i = j;
        }
        
        m_TopologyDirty = false;
    }
    
    // Iterate cached faces and sum areas
    float totalArea = 0.0f;
    for (const auto& face : m_CachedSurfaceFaces) {
        const Vec3& p0 = currentPositions[face.v0];
        const Vec3& p1 = currentPositions[face.v1];
        const Vec3& p2 = currentPositions[face.v2];
        
        Vec3 edge1 = p1 - p0;
        Vec3 edge2 = p2 - p0;
        Vec3 cross = edge1.Cross(edge2);
        totalArea += 0.5f * cross.Length();
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

void PhysXSoftBody::SetSphereGenerationScale(const Vec3& scale) {
    m_SphereGenerationScale.x = std::max(0.1f, scale.x);
    m_SphereGenerationScale.y = std::max(0.1f, scale.y);
    m_SphereGenerationScale.z = std::max(0.1f, scale.z);
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
    
    glGenVertexArrays(1, &m_DebugSurfaceVAO);
    glGenBuffers(1, &m_DebugSurfaceVBO);
    
    glGenVertexArrays(1, &m_DebugSpheresVAO);
    glGenBuffers(1, &m_DebugSpheresVBO);
    
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
    if ((!m_DebugDrawHull && !m_DebugDrawSurface) || !shader) return;
    
    if (!m_DebugResourcesInitialized) {
        CreateDebugResources();
    }
    
    // --- Draw Hull ---
    if (m_DebugDrawHull) {
        // Update hull (expensive, so maybe only if dirty? For now every frame like before)
        // ideally we check if vertices changed.
        UpdateDebugBuffers();
        
        if (!m_DebugHullVertices.empty()) {
            Mat4 identity = Mat4::Identity();
            shader->SetMat4("model", identity.m);
            shader->SetVec3("color", 1.0f, 0.0f, 1.0f); // Magenta for hull
            
            glBindVertexArray(m_DebugVAO);
            glDrawArrays(GL_LINES, 0, (int)m_DebugHullVertices.size());
            glBindVertexArray(0);
        }
    }
    
    // --- Draw Surface ---
    if (m_DebugDrawSurface) {
        // Ensure topology is up to date
        CalculateSurfaceArea(); // This updates m_CachedSurfaceFaces if dirty
        
        if (!m_CachedSurfaceFaces.empty()) {
            std::vector<Vec3> lines;
            lines.reserve(m_CachedSurfaceFaces.size() * 6); // 3 lines per face, 2 verts per line
            
            std::vector<Vec3> positions(m_VertexCount);
            GetVertexPositions(positions.data());
            
            for (const auto& face : m_CachedSurfaceFaces) {
                Vec3 v0 = positions[face.v0];
                Vec3 v1 = positions[face.v1];
                Vec3 v2 = positions[face.v2];
                
                lines.push_back(v0); lines.push_back(v1);
                lines.push_back(v1); lines.push_back(v2);
                lines.push_back(v2); lines.push_back(v0);
            }
            
            // Upload
            glBindVertexArray(m_DebugSurfaceVAO);
            glBindBuffer(GL_ARRAY_BUFFER, m_DebugSurfaceVBO);
            glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Vec3), lines.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
            glEnableVertexAttribArray(0);
            
            // Draw
            Mat4 identity = Mat4::Identity();
            shader->SetMat4("model", identity.m);
            shader->SetVec3("color", 0.0f, 1.0f, 1.0f); // Cyan for surface
            
            glDrawArrays(GL_LINES, 0, (int)lines.size());
            glBindVertexArray(0);
        }
    }
    
    // --- Draw Cracks ---
    if (m_CrackVisualizationEnabled && m_CrackRenderer && m_PartialTearSystem) {
        const auto& cracks = m_PartialTearSystem->GetCracks();
        if (!cracks.empty()) {
            std::vector<Vec3> positions(m_VertexCount);
            GetVertexPositions(positions.data());
            
            Mat4 identity = Mat4::Identity();
            m_CrackRenderer->RenderCracks(
                cracks,
                positions.data(),
                m_TetrahedronIndices.data(),
                shader,
                identity,
                m_SimulationTime  // Pass time for animations
            );
        }
    }
    
    // --- Draw Collision Spheres/Capsules ---
    if (m_DebugDrawCollisionSpheres) {
        // Try capsules first (for elongated bodies)
        std::vector<CollisionCapsule> caps;
        int numCapsules = GetCollisionCapsules(caps, 16);
        
        if (numCapsules > 0) {
            // Visualize capsules
            std::vector<Vec3> lines;
            const int segments = 8;
            const float PI = 3.14159265f;
            
            for (const auto& cap : caps) {
                Vec3 axis = cap.p1 - cap.p0;
                float length = axis.Length();
                if (length < 1e-6f) continue;
                
                Vec3 dir = axis * (1.0f / length);
                
                // Find perpendicular vectors
                Vec3 perp1;
                if (std::abs(dir.x) < 0.9f) {
                    perp1 = Vec3(1, 0, 0).Cross(dir);
                } else {
                    perp1 = Vec3(0, 1, 0).Cross(dir);
                }
                perp1 = perp1 * (1.0f / perp1.Length());
                Vec3 perp2 = dir.Cross(perp1);
                
                // Draw cylinder rings
                for (int ring = 0; ring <= 2; ++ring) {
                    float t = ring * 0.5f;
                    Vec3 center = cap.p0 + axis * t;
                    
                    for (int j = 0; j < segments; ++j) {
                        float theta1 = (float)j / segments * 2.0f * PI;
                        float theta2 = (float)(j + 1) / segments * 2.0f * PI;
                        
                        Vec3 offset1 = (perp1 * std::cos(theta1) + perp2 * std::sin(theta1)) * cap.radius;
                        Vec3 offset2 = (perp1 * std::cos(theta2) + perp2 * std::sin(theta2)) * cap.radius;
                        
                        lines.push_back(center + offset1);
                        lines.push_back(center + offset2);
                    }
                }
                
                // Draw longitudinal lines
                for (int j = 0; j < 4; ++j) {
                    float theta = (float)j / 4.0f * 2.0f * PI;
                    Vec3 offset = (perp1 * std::cos(theta) + perp2 * std::sin(theta)) * cap.radius;
                    
                    lines.push_back(cap.p0 + offset);
                    lines.push_back(cap.p1 + offset);
                }
            }
            
            if (!lines.empty()) {
                glBindVertexArray(m_DebugSpheresVAO);
                glBindBuffer(GL_ARRAY_BUFFER, m_DebugSpheresVBO);
                glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Vec3), lines.data(), GL_DYNAMIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
                glEnableVertexAttribArray(0);
                
                Mat4 identity = Mat4::Identity();
                shader->SetMat4("model", identity.m);
                shader->SetVec3("color", 0.0f, 1.0f, 1.0f); // Cyan for capsules
                
                glDrawArrays(GL_LINES, 0, (int)lines.size());
                glBindVertexArray(0);
            }
        } else if (m_CollisionSpheresNeedUpdate || m_ClothCollisionEnabled) {
            // Fall back to spheres for non-elongated bodies
            std::vector<Vec3> centers;
            std::vector<float> radii;
            GetCollisionSpheres(centers, radii, m_MaxCollisionSpheres);
            
            std::vector<Vec3> lines;
            const int segments = 12;
            const float PI = 3.14159265f;
            
            for (size_t i = 0; i < centers.size(); ++i) {
                Vec3 c = centers[i];
                float r = radii[i];
                
                for (int j = 0; j < segments; ++j) {
                    float theta1 = (float)j / segments * 2.0f * PI;
                    float theta2 = (float)(j + 1) / segments * 2.0f * PI;
                    
                    float c1 = std::cos(theta1) * r;
                    float s1 = std::sin(theta1) * r;
                    float c2 = std::cos(theta2) * r;
                    float s2 = std::sin(theta2) * r;
                    
                    // XY
                    lines.push_back(c + Vec3(c1, s1, 0));
                    lines.push_back(c + Vec3(c2, s2, 0));
                    
                    // YZ
                    lines.push_back(c + Vec3(0, c1, s1));
                    lines.push_back(c + Vec3(0, c2, s2));
                    
                    // XZ
                    lines.push_back(c + Vec3(c1, 0, s1));
                    lines.push_back(c + Vec3(c2, 0, s2));
                }
            }
            
            if (!lines.empty()) {
                glBindVertexArray(m_DebugSpheresVAO);
                glBindBuffer(GL_ARRAY_BUFFER, m_DebugSpheresVBO);
                glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Vec3), lines.data(), GL_DYNAMIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
                glEnableVertexAttribArray(0);
                
                Mat4 identity = Mat4::Identity();
                shader->SetMat4("model", identity.m);
                shader->SetVec3("color", 1.0f, 1.0f, 0.0f); // Yellow for spheres
                
                glDrawArrays(GL_LINES, 0, (int)lines.size());
                glBindVertexArray(0);
            }
        }
    }
    
    // --- Draw Fiber Directions ---
    if (m_DebugDrawFibers && m_UseAnisotropicMaterial && m_AnisotropicMaterial) {
        if (!m_AnisotropicMaterial->IsInitialized()) {
            return;
        }
        
        std::vector<Vec3> lines;
        const float fiberLength = 0.2f; // Length of fiber direction lines
        
        // Get current vertex positions
        std::vector<Vec3> positions(m_VertexCount);
        GetVertexPositions(positions.data());
        
        // Draw fiber direction for each tetrahedron
        for (int tetIdx = 0; tetIdx < m_TetrahedronCount; ++tetIdx) {
            // Calculate tetrahedron center
            int v0 = m_TetrahedronIndices[tetIdx * 4 + 0];
            int v1 = m_TetrahedronIndices[tetIdx * 4 + 1];
            int v2 = m_TetrahedronIndices[tetIdx * 4 + 2];
            int v3 = m_TetrahedronIndices[tetIdx * 4 + 3];
            
            Vec3 center = (positions[v0] + positions[v1] + positions[v2] + positions[v3]) * 0.25f;
            
            // Get fiber data
            const FiberData& fiber = m_AnisotropicMaterial->GetFiberData(tetIdx);
            
            // Draw line in fiber direction
            Vec3 start = center - fiber.direction * (fiberLength * 0.5f);
            Vec3 end = center + fiber.direction * (fiberLength * 0.5f);
            
            lines.push_back(start);
            lines.push_back(end);
        }
        
        if (!lines.empty()) {
            glBindVertexArray(m_DebugSpheresVAO); // Reuse spheres VAO
            glBindBuffer(GL_ARRAY_BUFFER, m_DebugSpheresVBO);
            glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Vec3), lines.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
            glEnableVertexAttribArray(0);
            
            Mat4 identity = Mat4::Identity();
            shader->SetMat4("model", identity.m);
            shader->SetVec3("color", 1.0f, 0.0f, 1.0f); // Magenta for fibers
            
            glDrawArrays(GL_LINES, 0, (int)lines.size());
            glBindVertexArray(0);
        }
    }
}

// ============================================================================
// LOD System Methods
// ============================================================================

void PhysXSoftBody::SetLODConfig(const SoftBodyLODConfig& config) {
    if (!m_LODManager) {
        m_LODManager = std::make_unique<SoftBodyLODManager>();
    }
    
    m_LODManager->SetLODConfig(config);
    
    // Generate LOD meshes if we have tetrahedral data
    if (!m_TetrahedronIndices.empty() && !m_InitialPositions.empty()) {
        SoftBodyLODConfig& mutableConfig = const_cast<SoftBodyLODConfig&>(config);
        mutableConfig.GenerateLODMeshes(m_InitialPositions, m_TetrahedronIndices);
    }
    
    std::cout << "PhysXSoftBody: LOD configuration set with " << config.GetLODCount() << " levels" << std::endl;
}

const SoftBodyLODConfig* PhysXSoftBody::GetLODConfig() const {
    if (!m_LODManager) {
        return nullptr;
    }
    return &m_LODManager->GetLODConfig();
}

int PhysXSoftBody::GetCurrentLOD() const {
    if (!m_LODManager) {
        return 0;
    }
    return m_LODManager->GetCurrentLOD();
}

void PhysXSoftBody::ForceLOD(int lodIndex) {
    if (!m_LODManager) {
        return;
    }
    m_LODManager->ForceLOD(lodIndex);
}

// ============================================================================
// Serialization Methods
// ============================================================================

nlohmann::json PhysXSoftBody::Serialize() const {
    return SoftBodySerializer::SerializeToJson(this);
}

bool PhysXSoftBody::Deserialize(const nlohmann::json& json) {
    return SoftBodySerializer::DeserializeFromJson(json, this);
}

bool PhysXSoftBody::SaveToFile(const std::string& filename) const {
    return SoftBodySerializer::SaveToFile(this, filename);
}

bool PhysXSoftBody::LoadFromFile(const std::string& filename) {
    return SoftBodySerializer::LoadFromFile(filename, this);
}

// ============================================================================
// GPU Methods
// ============================================================================

void PhysXSoftBody::EnableDirectGpuApi(bool enable) {
    if (!m_Backend->IsGpuEnabled()) {
        if (enable) {
            std::cerr << "PhysXSoftBody: Cannot enable direct GPU API - GPU not available!" << std::endl;
        }
        m_UseDirectGpuApi = false;
        return;
    }
    
    m_UseDirectGpuApi = enable;
    
    if (enable) {
        std::cout << "PhysXSoftBody: Direct GPU API enabled for efficient data transfer" << std::endl;
    } else {
        std::cout << "PhysXSoftBody: Direct GPU API disabled, using standard CPU-GPU transfer" << std::endl;
    }
}

PhysXSoftBody::GpuMetrics PhysXSoftBody::GetGpuMetrics() const {
    GpuMetrics metrics;
    metrics.gpuSimulationTimeMs = m_GpuSimulationTimeMs;
    metrics.cpuGpuTransferTimeMs = m_CpuGpuTransferTimeMs;
    metrics.gpuMemoryUsageBytes = m_GpuMemoryUsage;
    metrics.usingDirectApi = m_UseDirectGpuApi;
    return metrics;
}

// Deformation Recording API

void PhysXSoftBody::StartRecording(float sampleRate, bool recordVelocities) {
    if (!m_Recorder) {
        return;
    }
    
    m_Recorder->StartRecording(m_VertexCount, recordVelocities);
    m_Recorder->SetSampleRate(sampleRate);
    m_RecordingStartTime = 0.0f;
}

void PhysXSoftBody::StopRecording() {
    if (m_Recorder) {
        m_Recorder->StopRecording();
    }
}

void PhysXSoftBody::PauseRecording() {
    if (m_Recorder) {
        m_Recorder->PauseRecording();
    }
}

void PhysXSoftBody::ResumeRecording() {
    if (m_Recorder) {
        m_Recorder->ResumeRecording();
    }
}

bool PhysXSoftBody::IsRecording() const {
    return m_Recorder && m_Recorder->IsRecording();
}

void PhysXSoftBody::StartPlayback() {
    if (!m_Recorder) {
        return;
    }
    
    m_Recorder->StartPlayback();
    m_IsPlayingBack = true;
}

void PhysXSoftBody::StopPlayback() {
    if (m_Recorder) {
        m_Recorder->StopPlayback();
    }
    m_IsPlayingBack = false;
}

void PhysXSoftBody::PausePlayback() {
    if (m_Recorder) {
        m_Recorder->PausePlayback();
    }
}

void PhysXSoftBody::SeekPlayback(float time) {
    if (m_Recorder) {
        m_Recorder->SeekPlayback(time);
    }
}

bool PhysXSoftBody::IsPlayingBack() const {
    return m_IsPlayingBack && m_Recorder && m_Recorder->IsPlayingBack();
}

float PhysXSoftBody::GetRecordingDuration() const {
    return m_Recorder ? m_Recorder->GetDuration() : 0.0f;
}

float PhysXSoftBody::GetPlaybackTime() const {
    return m_Recorder ? m_Recorder->GetPlaybackTime() : 0.0f;
}

void PhysXSoftBody::SetPlaybackSpeed(float speed) {
    if (m_Recorder) {
        m_Recorder->SetPlaybackSpeed(speed);
    }
}

void PhysXSoftBody::SetPlaybackLoopMode(LoopMode mode) {
    if (m_Recorder) {
        m_Recorder->SetLoopMode(mode);
    }
}

bool PhysXSoftBody::SaveRecording(const std::string& filename, bool binary) const {
    return m_Recorder ? m_Recorder->SaveToFile(filename, binary) : false;
}

bool PhysXSoftBody::LoadRecording(const std::string& filename) {
    return m_Recorder ? m_Recorder->LoadFromFile(filename) : false;
}

// Partial Tearing (Cracks) Implementation

void PhysXSoftBody::SetPartialTearingEnabled(bool enabled) {
    m_PartialTearingEnabled = enabled;
    std::cout << "Partial tearing " << (enabled ? "enabled" : "disabled") << std::endl;
}

bool PhysXSoftBody::IsPartialTearingEnabled() const {
    return m_PartialTearingEnabled;
}

void PhysXSoftBody::SetCrackThreshold(float threshold) {
    m_CrackThreshold = threshold;
}

float PhysXSoftBody::GetCrackThreshold() const {
    return m_CrackThreshold;
}

void PhysXSoftBody::SetCrackProgressionRate(float rate) {
    m_CrackProgressionRate = rate;
}

float PhysXSoftBody::GetCrackProgressionRate() const {
    return m_CrackProgressionRate;
}

const std::vector<PartialTearSystem::Crack>& PhysXSoftBody::GetCracks() const {
    static std::vector<PartialTearSystem::Crack> empty;
    if (m_PartialTearSystem) {
        return m_PartialTearSystem->GetCracks();
    }
    return empty;
}

int PhysXSoftBody::GetCrackCount() const {
    return m_PartialTearSystem ? m_PartialTearSystem->GetCrackCount() : 0;
}

void PhysXSoftBody::SetCrackHealingEnabled(bool enabled) {
    if (m_PartialTearSystem) {
        m_PartialTearSystem->SetHealingEnabled(enabled);
    }
}

void PhysXSoftBody::SetCrackHealingRate(float rate) {
    if (m_PartialTearSystem) {
        m_PartialTearSystem->SetHealingRate(rate);
    }
}

// Crack Visualization Implementation

void PhysXSoftBody::SetCrackVisualizationEnabled(bool enabled) {
    m_CrackVisualizationEnabled = enabled;
    std::cout << "Crack visualization " << (enabled ? "enabled" : "disabled") << std::endl;
}

bool PhysXSoftBody::IsCrackVisualizationEnabled() const {
    return m_CrackVisualizationEnabled;
}

void PhysXSoftBody::SetCrackRenderSettings(const CrackRenderer::RenderSettings& settings) {
    if (m_CrackRenderer) {
        m_CrackRenderer->SetRenderSettings(settings);
    }
}

#endif // USE_PHYSX
