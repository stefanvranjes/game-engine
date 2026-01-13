#pragma once

#include "Math/Vec3.h"
#include <memory>

/**
 * @brief Cloth descriptor for initialization
 */
struct ClothDesc {
    Vec3* particlePositions;    // Initial particle positions
    int particleCount;          // Number of particles
    int* triangleIndices;       // Triangle indices (3 per triangle)
    int triangleCount;          // Number of triangles
    float particleMass;         // Mass per particle
    Vec3 gravity;              // Gravity vector
    
    // Collision Settings
    bool enableSceneCollision;      // Collide with scene rigid bodies
    bool enableSelfCollision;       // Collide with self
    float selfCollisionDistance;    // Thickness for self collision
    float selfCollisionStiffness;   // Stiffness of self collision contacts
    
    // Two-Way Coupling
    bool enableTwoWayCoupling;      // Enable forces on rigid bodies
    float collisionMassScale;       // Scale mass for collision impulses (default 1.0)

    ClothDesc()
        : particlePositions(nullptr)
        , particleCount(0)
        , triangleIndices(nullptr)
        , triangleCount(0)
        , particleMass(1.0f)
        , gravity(0, -9.81f, 0)
        , enableSceneCollision(true)
        , enableSelfCollision(false)
        , selfCollisionDistance(0.01f)
        , selfCollisionStiffness(1.0f)
        , enableTwoWayCoupling(false)
        , collisionMassScale(1.0f)
    {}
};

/**
 * @brief Abstract interface for cloth simulation
 * 
 * Provides a common API for cloth physics across different backends.
 * PhysX 5.x provides GPU-accelerated cloth simulation.
 */
class IPhysicsCloth {
public:
    virtual ~IPhysicsCloth() = default;

    /**
     * @brief Initialize the cloth simulation
     * @param desc Cloth descriptor with mesh and physics parameters
     */
    virtual void Initialize(const ClothDesc& desc) = 0;

    /**
     * @brief Update cloth simulation
     * @param deltaTime Time step in seconds
     */
    virtual void Update(float deltaTime) = 0;

    /**
     * @brief Enable or disable cloth simulation
     * @param enabled If true, cloth will simulate
     */
    virtual void SetEnabled(bool enabled) = 0;

    /**
     * @brief Check if cloth simulation is enabled
     * @return True if enabled
     */
    virtual bool IsEnabled() const = 0;

    /**
     * @brief Get number of particles in cloth
     * @return Particle count
     */
    virtual int GetParticleCount() const = 0;

    /**
     * @brief Get current particle positions
     * @param positions Output array (must be pre-allocated with GetParticleCount() size)
     */
    virtual void GetParticlePositions(Vec3* positions) const = 0;

    /**
     * @brief Set particle positions (for initialization or reset)
     * @param positions Input array of positions
     */
    virtual void SetParticlePositions(const Vec3* positions) = 0;

    /**
     * @brief Get particle normals (for rendering)
     * @param normals Output array (must be pre-allocated)
     */
    virtual void GetParticleNormals(Vec3* normals) const = 0;

    /**
     * @brief Set stretch stiffness (resistance to stretching)
     * @param stiffness Stiffness value (0.0 = no resistance, 1.0 = maximum)
     */
    virtual void SetStretchStiffness(float stiffness) = 0;

    /**
     * @brief Set bend stiffness (resistance to bending)
     * @param stiffness Stiffness value (0.0 = no resistance, 1.0 = maximum)
     */
    virtual void SetBendStiffness(float stiffness) = 0;

    /**
     * @brief Set shear stiffness (resistance to shearing)
     * @param stiffness Stiffness value (0.0 = no resistance, 1.0 = maximum)
     */
    virtual void SetShearStiffness(float stiffness) = 0;

    /**
     * @brief Apply force to all particles
     * @param force Force vector in Newtons
     */
    virtual void AddForce(const Vec3& force) = 0;

    /**
     * @brief Set wind velocity affecting the cloth
     * @param velocity Wind velocity vector
     */
    virtual void SetWindVelocity(const Vec3& velocity) = 0;

    /**
     * @brief Set damping (energy loss)
     * @param damping Damping coefficient (0.0 = no damping, 1.0 = maximum)
     */
    virtual void SetDamping(float damping) = 0;

    /**
     * @brief Attach a particle to a rigid body
     * @param particleIndex Index of particle to attach
     * @param actor Backend-specific rigid body pointer
     * @param localPos Local position on the rigid body
     */
    virtual void AttachParticleToActor(int particleIndex, void* actor, const Vec3& localPos) = 0;

    /**
     * @brief Free a particle (remove attachment)
     * @param particleIndex Index of particle to free
     */
    virtual void FreeParticle(int particleIndex) = 0;

    /**
     * @brief Add collision sphere
     * @param center Sphere center in world space
     * @param radius Sphere radius
     */
    virtual void AddCollisionSphere(const Vec3& center, float radius) = 0;

    /**
     * @brief Add collision capsule
     * @param p0 Capsule endpoint 0
     * @param p1 Capsule endpoint 1
     * @param radius Capsule radius
     */
    virtual void AddCollisionCapsule(const Vec3& p0, const Vec3& p1, float radius) = 0;

    /**
     * @brief Enable or disable cloth tearing
     * @param tearable If true, cloth can tear when stretched too much
     */
    virtual void SetTearable(bool tearable) = 0;

    /**
     * @brief Set maximum stretch ratio before tearing
     * @param ratio Maximum stretch (1.0 = no stretch, 2.0 = 200% stretch)
     */
    virtual void SetMaxStretchRatio(float ratio) = 0;

    /**
     * @brief Manually tear cloth at specific particle
     * @param particleIndex Index of particle to tear
     * @return True if tear was successful
     */
    virtual bool TearAtParticle(int particleIndex) = 0;

    /**
     * @brief Tear cloth along a line
     * @param start Start position in world space
     * @param end End position in world space
     * @return Number of tears created
     */
    virtual int TearAlongLine(const Vec3& start, const Vec3& end) = 0;

    /**
     * @brief Get number of tears that have occurred
     * @return Tear count
     */
    virtual int GetTearCount() const = 0;

    /**
     * @brief Reset cloth to original state (remove all tears)
     */
    virtual void ResetTears() = 0;

    /**
     * @brief Get backend-specific cloth pointer
     * @return Opaque pointer to native cloth object
     */
    virtual void* GetNativeCloth() = 0;

    /**
     * @brief Enable/Disable collision with scene rigid bodies
     */
    virtual void SetSceneCollision(bool enabled) = 0;

    /**
     * @brief Enable/Disable self-collision
     */
    virtual void SetSelfCollision(bool enabled) = 0;

    /**
     * @brief Set self-collision distance (thickness)
     * @param distance Distance below which self-collision is detected
     */
    virtual void SetSelfCollisionDistance(float distance) = 0;

    /**
     * @brief Set self-collision stiffness
     * @param stiffness Stiffness of self-collision response (0.0 to 1.0)
     */
    virtual void SetSelfCollisionStiffness(float stiffness) = 0;

    /**
     * @brief Enable/Disable two-way coupling (forces on rigid bodies)
     */
    virtual void SetTwoWayCoupling(bool enabled) = 0;

    /**
     * @brief Set collision mass scale
     * @param scale Scale factor for collision impulses
     */
    virtual void SetCollisionMassScale(float scale) = 0;

    /**
     * @brief Enable/Disable two-sided rendering
     */
    virtual void SetTwoSidedRendering(bool enabled) = 0;

    /**
     * @brief Check if two-sided rendering is enabled
     */
    virtual bool GetTwoSidedRendering() const = 0;

    /**
     * @brief Enable/Disable subsurface scattering
     */
    virtual void SetEnableSubsurface(bool enabled) = 0;

    /**
     * @brief Check if subsurface scattering is enabled
     */
    virtual bool GetEnableSubsurface() const = 0;

    /**
     * @brief Set translucency
     */
    virtual void SetTranslucency(float translucency) = 0;

    /**
     * @brief Get translucency
     */
    virtual float GetTranslucency() const = 0;

    /**
     * @brief Set wrinkle scale
     */
    virtual void SetWrinkleScale(float scale) = 0;

    /**
     * @brief Get wrinkle scale
     */
    virtual float GetWrinkleScale() const = 0;

     /**
     * @brief Enable/Disable wrinkle detail
     */
    virtual void SetEnableWrinkleDetail(bool enabled) = 0;

    /**
     * @brief Check if wrinkle detail is enabled
     */
    virtual bool GetEnableWrinkleDetail() const = 0;

    /**
     * @brief Raycast against cloth triangles
     * @param from Start position
     * @param to End position
     * @param hit Hit result (output)
     * @return True if hit
     */
    virtual bool Raycast(const Vec3& from, const Vec3& to, struct PhysicsRaycastHit& hit) = 0;
};
