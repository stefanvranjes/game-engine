#pragma once

#include "Particle.h"
#include "Texture.h"
#include "CollisionShape.h"
#include "ParticleTrail.h"
#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include <vector>
#include <memory>
#include <unordered_map>

enum class EmitterShape {
    Point,
    Sphere,
    Cone,
    Box
};

enum class PhysicsMode {
    Simple,
    Standard, // Collisions
    SPHFluid,
    Cloth,
    SoftBody
};

enum class BlendMode {
    Additive,
    Alpha
};

enum class TrailColorMode {
    ParticleColor,      // Use particle's current color
    FadeToTransparent,  // Fade from particle color to transparent
    GradientToEnd,      // Gradient from start to end color
    Custom              // Use custom trail color
};

struct GradientStop {
    float t; // 0.0 to 1.0
    Vec4 color;
};

// POD struct for GPU alignment (std430)
struct GPUParticle {
    Vec4 position;   // xyz, w=size
    Vec4 velocity;   // xyz, w=lifetime
    Vec4 color;      // rgba
    Vec4 properties; // x=age, y=lifeMax, z=active, w=density/pad
};

struct Spring {
    unsigned int p1;
    unsigned int p2;
    float restLength;
    float stiffness;
    float damping;
    unsigned int active; // padding/flag
    unsigned int padding2;
    unsigned int padding3;
};

struct EmitterLOD {
    float distance;            // Min distance for this level
    float emissionMultiplier;  // 0.0-1.0
    bool enableTurbulence;
    bool enableCollisions;
    bool enableTrails;
    bool enableLighting;
    bool enableShadows;
    
    EmitterLOD() 
        : distance(0.0f)
        , emissionMultiplier(1.0f)
        , enableTurbulence(true)
        , enableCollisions(true)
        , enableTrails(true)
        , enableLighting(true)
        , enableShadows(true)
    {}
};

class ParticleEmitter {
public:
    ParticleEmitter(const Vec3& position, int maxParticles = 1000);
    ~ParticleEmitter();

    void Update(float deltaTime, const Vec3& cameraPos = Vec3(0,0,0));
    
    // Pooling & Priority
    void SetMaxParticles(int count);
    int GetMaxParticles() const { return m_MaxParticles; }
    
    void SetPriority(int priority) { m_Priority = priority; }
    int GetPriority() const { return m_Priority; }
    
    // Kill oldest particles to free up space (for stealing/budgeting)
    // Returns number of particles actually killed
    int KillOldest(int count);
    
    const std::vector<Particle>& GetParticles() const { return m_Particles; }
    
    // Configuration
    void SetVelocityInheritance(float multiplier) { m_VelocityInheritance = multiplier; }
    float GetVelocityInheritance() const { return m_VelocityInheritance; }
    
    // Spawning
    void Burst(int count);

    // Getters/Setters for basic props...
    void SetPosition(const Vec3& pos) { m_Position = pos; }
    Vec3 GetPosition() const { return m_Position; }
    
    void SetSpawnRate(float rate) { m_SpawnRate = rate; }
    void SetParticleLifetime(float lifetime) { m_ParticleLifetime = lifetime; }
    void SetVelocityRange(const Vec3& min, const Vec3& max) { m_VelocityMin = min; m_VelocityMax = max; }
    void SetColorRange(const Vec4& start, const Vec4& end) { m_ColorStart = start; m_ColorEnd = end; }
    void SetSizeRange(float start, float end) { m_SizeStart = start; m_SizeEnd = end; }
    void SetGravity(const Vec3& gravity) { m_Gravity = gravity; }
    void SetEmitterShape(EmitterShape shape) { m_Shape = shape; }
    void SetConeAngle(float angle) { m_ConeAngle = angle; }
    void SetSphereRadius(float radius) { m_SphereRadius = radius; }
    void SetBoxSize(const Vec3& size) { m_BoxSize = size; }
    void SetBlendMode(BlendMode mode) { m_BlendMode = mode; }
    void SetTexture(std::shared_ptr<Texture> texture) { m_Texture = texture; }
    void SetParent(class GameObject* parent) { m_Parent = parent; }
    void SetAtlasSize(int rows, int cols) { m_AtlasRows = rows; m_AtlasCols = cols; }
    void SetAnimationSpeed(float speed) { m_AnimationSpeed = speed; }
    void SetLoopAnimation(bool loop) { m_LoopAnimation = loop; }
    
    BlendMode GetBlendMode() const { return m_BlendMode; }
    std::shared_ptr<Texture> GetTexture() const { return m_Texture; }
    GameObject* GetParent() const { return m_Parent; }
    int GetAtlasRows() const { return m_AtlasRows; }
    int GetAtlasCols() const { return m_AtlasCols; }
    float GetAnimationSpeed() const { return m_AnimationSpeed; }
    bool IsLoopAnimation() const { return m_LoopAnimation; }
    bool IsActive() const { return m_Active; }
    void SetActive(bool active) { m_Active = active; }
    
    // Collision configuration
    void AddCollisionShape(std::shared_ptr<CollisionShape> shape);
    void RemoveCollisionShape(std::shared_ptr<CollisionShape> shape);
    void ClearCollisionShapes();
    const std::vector<std::shared_ptr<CollisionShape>>& GetCollisionShapes() const { return m_CollisionShapes; }
    
    void SetEnableParticleCollisions(bool enable) { m_EnableParticleCollisions = enable; }
    bool GetEnableParticleCollisions() const { return m_EnableParticleCollisions; }
    
    void SetParticleCollisionRadius(float radius) { m_ParticleCollisionRadius = radius; }
    float GetParticleCollisionRadius() const { return m_ParticleCollisionRadius; }
    
    void SetParticleMass(float mass) { m_DefaultMass = mass; }
    void SetParticleRestitution(float restitution) { m_DefaultRestitution = restitution; }
    void SetParticleFriction(float friction) { m_DefaultFriction = friction; }
    
    // Advanced Physics (SPH)
    void SetPhysicsMode(PhysicsMode mode) { m_PhysicsMode = mode; }
    PhysicsMode GetPhysicsMode() const { return m_PhysicsMode; }
    
    void SetRestDensity(float rho) { m_RestDensity = rho; }
    void SetGasConstant(float k) { m_GasConstant = k; }
    void SetViscosity(float mu) { m_Viscosity = mu; }
    void SetSmoothingRadius(float h) { m_SmoothingRadius = h; }
    
    // Trail configuration
    void SetEnableTrails(bool enable) { m_EnableTrails = enable; }
    bool GetEnableTrails() const { return m_EnableTrails; }
    
    void SetTrailLength(int maxPoints) { m_TrailLength = maxPoints; }
    TrailColorMode GetTrailColorMode() const { return m_TrailColorMode; }
    
    void SetTrailColor(const Vec4& color) { m_TrailColor = color; }
    Vec4 GetTrailColor() const { return m_TrailColor; }
    
    // Trail turbulence
    void SetTrailTurbulence(float strength) { m_TrailTurbulence = strength; }
    float GetTrailTurbulence() const { return m_TrailTurbulence; }
    
    void SetTrailTurbulenceFrequency(float freq) { m_TrailTurbulenceFrequency = freq; }
    float GetTrailTurbulenceFrequency() const { return m_TrailTurbulenceFrequency; }
    
    void SetTrailTurbulenceSpeed(float speed) { m_TrailTurbulenceSpeed = speed; }
    float GetTrailTurbulenceSpeed() const { return m_TrailTurbulenceSpeed; }

    // LOD System
    void AddLOD(const EmitterLOD& lod) { m_LODs.push_back(lod); }
    void ClearLODs() { m_LODs.clear(); }
    const std::vector<EmitterLOD>& GetLODs() const { return m_LODs; }
    std::vector<EmitterLOD>& GetLODs() { return m_LODs; } // Mutable access for UI
    
    void SetUseLOD(bool use) { m_UseLOD = use; }
    bool GetUseLOD() const { return m_UseLOD; }
    int GetCurrentLODLevel() const { return m_CurrentLODLevel; }
    float GetCurrentDistance() const { return m_CurrentDistance; }

    
    // GPU Compute
    void SetUseGPUCompute(bool enable);
    bool GetUseGPUCompute() const { return m_UseGPUCompute; }
    bool IsGPUComputeAvailable() const;
    
    // Persistent GPU mode (keeps data on GPU, no CPU readback)
    void SetGPUPersistent(bool enable);
    bool GetGPUPersistent() const { return m_GPUPersistent; }
    
    // Get SSBO ID for rendering
    unsigned int GetParticleSSBO() const { return m_ParticleSSBO; }
    unsigned int GetSortSSBO() const { return m_SortSSBO; }
    unsigned int GetTrailSSBO() const { return m_TrailSSBO; }
    unsigned int GetAtomicCounterBuffer() const { return m_AtomicCounterBuffer; }
    unsigned int GetActiveParticleCount() const { return m_ActiveParticleCount; }
    unsigned int GetSortBufferSize() const { return m_SortBufferSize; }
    
    // Presets
    static std::shared_ptr<ParticleEmitter> CreateFire(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateSmoke(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateSparks(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateMagic(const Vec3& position);
    
    // Cloth/SoftBody Initialization
    void InitCloth(int width, int height, float spacing, float stiffness, float damping);
    void InitSoftBody(int width, int height, int depth, float spacing, float stiffness, float damping);

private:
    void SpawnParticle();
    Vec3 GetRandomVelocity();
    void HandleShapeCollisions(Particle& particle, float deltaTime);
    void HandleParticleCollisions(float deltaTime);
    void BuildSpatialGrid();
    void ResolveParticleCollision(Particle& p1, Particle& p2);
    float RandomFloat(float min, float max);
    
    std::vector<Particle> m_Particles;
    std::vector<GPUParticle> m_GPUParticles;
    int m_MaxParticles;
    
    // Emitter properties
    Vec3 m_Position;
    float m_SpawnRate;           // Particles per second
    float m_ParticleLifetime;
    Vec3 m_VelocityMin;
    Vec3 m_VelocityMax;
    Vec4 m_ColorStart;
    Vec4 m_ColorEnd;
    float m_SizeStart;
    float m_SizeEnd;
    Vec3 m_Gravity;
    EmitterShape m_Shape;
    float m_ConeAngle;
    float m_SphereRadius;
    Vec3 m_BoxSize;
    BlendMode m_BlendMode;
    std::shared_ptr<Texture> m_Texture;
    GameObject* m_Parent;
    bool m_Active;
    int m_Priority; // 0 = Low, 10 = High
    
    // Physics Enhancements
    float m_VelocityInheritance;
    Vec3 m_LastPosition;
    Vec3 m_CalculatedEmitterVelocity;
    
    // Visuals
    std::vector<GradientStop> m_ColorGradient;
    std::vector<float> m_SizeCurve; // Stores size values at t=index/size
    std::shared_ptr<ParticleEmitter> m_SubEmitterDeath;
    
    // Internal state
    float m_SpawnAccumulator;
    
    // Atlas animation
    int m_AtlasRows;
    int m_AtlasCols;
    float m_AnimationSpeed;
    bool m_LoopAnimation;
    
    // Collision system
    std::vector<std::shared_ptr<CollisionShape>> m_CollisionShapes;
    bool m_EnableParticleCollisions;
    float m_ParticleCollisionRadius;
    float m_DefaultMass;
    float m_DefaultRestitution;
    float m_DefaultFriction;
    
    // Spatial grid for particle-to-particle collisions
    struct SpatialGrid {
        std::unordered_map<int, std::vector<Particle*>> cells;
        float cellSize;
        
        int GetCellIndex(const Vec3& pos) const {
            int x = static_cast<int>(pos.x / cellSize);
            int y = static_cast<int>(pos.y / cellSize);
            int z = static_cast<int>(pos.z / cellSize);
            // Simple hash function
            return x * 73856093 ^ y * 19349663 ^ z * 83492791;
        }
    };
    SpatialGrid m_SpatialGrid;

    // GPU Collision System
    std::unique_ptr<class Shader> m_SpatialHashShader;
    std::unique_ptr<class Shader> m_BitonicSortShader;
    std::unique_ptr<class Shader> m_GridBuildShader;
    std::unique_ptr<class Shader> m_CollisionShader;

    unsigned int m_GridSSBO;
    unsigned int m_CollisionGridSize; // Total number of cells (e.g. 1M)
    float m_CollisionCellSize;        // Size of each cell
    
    // SPH Shaders
    std::unique_ptr<class Shader> m_SphDensityShader;
    std::unique_ptr<class Shader> m_SphForceShader;
    
    PhysicsMode m_PhysicsMode;
    float m_RestDensity;
    float m_GasConstant;
    float m_Viscosity;
    float m_SmoothingRadius;

    void InitGPUCollision();
    void DispatchGPUCollision(float deltaTime);
    
    unsigned int GetIndirectBuffer() const { return m_IndirectBuffer; }
    


    // Spring System (Cloth/SoftBody)
    std::unique_ptr<class Shader> m_SpringForceShader;
    std::unique_ptr<class Shader> m_ApplySpringForceShader;
    std::vector<Spring> m_Springs;
    unsigned int m_SpringSSBO;
    unsigned int m_ForceSSBO; // Atomic accumulation buffer
    
    // Indirect Draw
    unsigned int m_IndirectBuffer;
    std::unique_ptr<class Shader> m_IndirectArgsShader;
    
    // Trail settings
    bool m_EnableTrails;
    int m_TrailLength;
    float m_TrailLifetime;
    float m_TrailWidth;
    float m_TrailMinDistance;
    std::shared_ptr<Texture> m_TrailTexture;
    TrailColorMode m_TrailColorMode;
    Vec4 m_TrailColor;
    float m_TrailTurbulence;
    float m_TrailTurbulenceFrequency;
    float m_TrailTurbulenceSpeed;
    float m_Time; // For animated turbulence
    
    // LOD System
    std::vector<EmitterLOD> m_LODs;
    bool m_UseLOD;
    int m_CurrentLODLevel;
    float m_CurrentDistance;
    EmitterLOD m_ActiveLOD; // Currently active settings
    
    void UpdateLOD(float distance);

    // GPU Compute
    bool m_UseGPUCompute;
    bool m_GPUPersistent;
    unsigned int m_ParticleSSBO;
    unsigned int m_SortSSBO;
    unsigned int m_TrailSSBO;
    unsigned int m_AtomicCounterBuffer;
    unsigned int m_ActiveParticleCount;
    unsigned int m_SortBufferSize;
    std::unique_ptr<class Shader> m_ComputeShader;
    void InitGPUCompute();
    void ShutdownGPUCompute();
    void UpdateGPU(float deltaTime);
    void UpdateCPU(float deltaTime);
};
