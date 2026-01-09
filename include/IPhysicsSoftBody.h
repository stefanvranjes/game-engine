#pragma once

#include "Math/Vec3.h"
#include &lt;memory&gt;
#include &lt;vector&gt;

class IPhysicsShape;
class IPhysicsRigidBody;

/**
 * @brief Soft body descriptor for initialization
 */
struct SoftBodyDesc {
    // Mesh data
    Vec3* vertexPositions;          // Surface mesh vertices
    int vertexCount;                // Number of surface vertices
    int* triangleIndices;           // Triangle indices (3 per triangle)
    int triangleCount;              // Number of triangles
    
    // Tetrahedral mesh (optional, will be auto-generated if null)
    Vec3* tetrahedronVertices;      // Tetrahedral mesh vertices
    int tetrahedronVertexCount;     // Number of tetrahedral vertices
    int* tetrahedronIndices;        // Tetrahedral indices (4 per tetrahedron)
    int tetrahedronCount;           // Number of tetrahedra
    
    // Physical properties
    float density;                  // Material density (kg/m³)
    float totalMass;                // Total mass (alternative to density)
    bool useDensity;                // If true, use density; otherwise use totalMass
    Vec3 gravity;                   // Gravity vector
    
    // Constraint stiffness (0.0 to 1.0)
    float volumeStiffness;          // Volume preservation (incompressibility)
    float shapeStiffness;           // Shape matching (elasticity)
    float deformationStiffness;     // Resistance to deformation
    
    // Deformation limits
    float maxStretch;               // Maximum stretch ratio (e.g., 1.5 = 150%)
    float maxCompress;              // Maximum compression ratio (e.g., 0.5 = 50%)
    
    // Damping
    float linearDamping;            // Linear velocity damping
    float angularDamping;           // Angular velocity damping
    
    // Collision settings
    bool enableSceneCollision;      // Collide with rigid bodies
    bool enableSelfCollision;       // Self-collision detection
    float collisionMargin;          // Collision detection margin
    
    // Simulation quality
    int solverIterations;           // Number of constraint solver iterations
    
    SoftBodyDesc()
        : vertexPositions(nullptr)
        , vertexCount(0)
        , triangleIndices(nullptr)
        , triangleCount(0)
        , tetrahedronVertices(nullptr)
        , tetrahedronVertexCount(0)
        , tetrahedronIndices(nullptr)
        , tetrahedronCount(0)
        , density(1000.0f)
        , totalMass(1.0f)
        , useDensity(false)
        , gravity(0, -9.81f, 0)
        , volumeStiffness(0.5f)
        , shapeStiffness(0.5f)
        , deformationStiffness(0.5f)
        , maxStretch(1.5f)
        , maxCompress(0.5f)
        , linearDamping(0.01f)
        , angularDamping(0.01f)
        , enableSceneCollision(true)
        , enableSelfCollision(false)
        , collisionMargin(0.01f)
        , solverIterations(4)
    {}
};

/**
 * @brief Abstract interface for soft body physics
 * 
 * Provides a common API for soft body simulation across different physics backends.
 * PhysX 5.x provides GPU-accelerated soft body simulation using Position-Based Dynamics.
 */
class IPhysicsSoftBody {
public:
    virtual ~IPhysicsSoftBody() = default;

    /**
     * @brief Initialize the soft body simulation
     * @param desc Soft body descriptor with mesh and physics parameters
     */
    virtual void Initialize(const SoftBodyDesc&amp; desc) = 0;

    /**
     * @brief Update soft body simulation
     * @param deltaTime Time step in seconds
     */
    virtual void Update(float deltaTime) = 0;

    /**
     * @brief Enable or disable soft body simulation
     * @param enabled If true, soft body will simulate
     */
    virtual void SetEnabled(bool enabled) = 0;

    /**
     * @brief Check if soft body simulation is enabled
     * @return True if enabled
     */
    virtual bool IsEnabled() const = 0;

    // ===== Vertex/Particle Access =====

    /**
     * @brief Get number of surface vertices
     * @return Vertex count
     */
    virtual int GetVertexCount() const = 0;

    /**
     * @brief Get current vertex positions
     * @param positions Output array (must be pre-allocated with GetVertexCount() size)
     */
    virtual void GetVertexPositions(Vec3* positions) const = 0;

    /**
     * @brief Set vertex positions (for initialization or reset)
     * @param positions Input array of positions
     */
    virtual void SetVertexPositions(const Vec3* positions) = 0;

    /**
     * @brief Get vertex normals (for rendering)
     * @param normals Output array (must be pre-allocated)
     */
    virtual void GetVertexNormals(Vec3* normals) const = 0;

    /**
     * @brief Get vertex velocities
     * @param velocities Output array (must be pre-allocated)
     */
    virtual void GetVertexVelocities(Vec3* velocities) const = 0;

    // ===== Material Properties =====

    /**
     * @brief Set volume stiffness (incompressibility)
     * @param stiffness Stiffness value (0.0 = compressible, 1.0 = incompressible)
     */
    virtual void SetVolumeStiffness(float stiffness) = 0;

    /**
     * @brief Get volume stiffness
     * @return Current volume stiffness
     */
    virtual float GetVolumeStiffness() const = 0;

    /**
     * @brief Set shape stiffness (elasticity)
     * @param stiffness Stiffness value (0.0 = no shape matching, 1.0 = rigid)
     */
    virtual void SetShapeStiffness(float stiffness) = 0;

    /**
     * @brief Get shape stiffness
     * @return Current shape stiffness
     */
    virtual float GetShapeStiffness() const = 0;

    /**
     * @brief Set deformation stiffness
     * @param stiffness Resistance to deformation (0.0 to 1.0)
     */
    virtual void SetDeformationStiffness(float stiffness) = 0;

    /**
     * @brief Get deformation stiffness
     * @return Current deformation stiffness
     */
    virtual float GetDeformationStiffness() const = 0;

    /**
     * @brief Set maximum stretch ratio
     * @param maxStretch Maximum stretch (1.0 = no stretch, 2.0 = 200% stretch)
     */
    virtual void SetMaxStretch(float maxStretch) = 0;

    /**
     * @brief Set maximum compression ratio
     * @param maxCompress Maximum compression (1.0 = no compression, 0.5 = 50% compression)
     */
    virtual void SetMaxCompress(float maxCompress) = 0;

    /**
     * @brief Set damping (energy loss)
     * @param linear Linear damping coefficient
     * @param angular Angular damping coefficient
     */
    virtual void SetDamping(float linear, float angular) = 0;

    // ===== Forces =====

    /**
     * @brief Apply force to all vertices
     * @param force Force vector in Newtons
     */
    virtual void AddForce(const Vec3&amp; force) = 0;

    /**
     * @brief Apply force to a specific vertex
     * @param vertexIndex Index of vertex
     * @param force Force vector in Newtons
     */
    virtual void AddForceAtVertex(int vertexIndex, const Vec3&amp; force) = 0;

    /**
     * @brief Apply impulse to all vertices
     * @param impulse Impulse vector in Newton-seconds
     */
    virtual void AddImpulse(const Vec3&amp; impulse) = 0;

    /**
     * @brief Apply impulse to a specific vertex
     * @param vertexIndex Index of vertex
     * @param impulse Impulse vector in Newton-seconds
     */
    virtual void AddImpulseAtVertex(int vertexIndex, const Vec3&amp; impulse) = 0;

    // ===== Attachments =====

    /**
     * @brief Attach a vertex to a rigid body
     * @param vertexIndex Index of vertex to attach
     * @param rigidBody Rigid body to attach to
     * @param localPos Local position on the rigid body
     */
    virtual void AttachVertexToRigidBody(int vertexIndex, IPhysicsRigidBody* rigidBody, const Vec3&amp; localPos) = 0;

    /**
     * @brief Detach a vertex (make it free)
     * @param vertexIndex Index of vertex to detach
     */
    virtual void DetachVertex(int vertexIndex) = 0;

    /**
     * @brief Fix a vertex in world space (infinite mass)
     * @param vertexIndex Index of vertex to fix
     * @param worldPos World position to fix at
     */
    virtual void FixVertex(int vertexIndex, const Vec3&amp; worldPos) = 0;

    /**
     * @brief Unfix a vertex (restore normal mass)
     * @param vertexIndex Index of vertex to unfix
     */
    virtual void UnfixVertex(int vertexIndex) = 0;

    // ===== Collision =====

    /**
     * @brief Enable/Disable collision with scene rigid bodies
     * @param enabled If true, collide with rigid bodies
     */
    virtual void SetSceneCollision(bool enabled) = 0;

    /**
     * @brief Enable/Disable self-collision
     * @param enabled If true, enable self-collision
     */
    virtual void SetSelfCollision(bool enabled) = 0;

    /**
     * @brief Set collision margin
     * @param margin Collision detection margin/thickness
     */
    virtual void SetCollisionMargin(float margin) = 0;

    /**
     * @brief Add collision sphere
     * @param center Sphere center in local space
     * @param radius Sphere radius
     */
    virtual void AddCollisionSphere(const Vec3&amp; center, float radius) = 0;

    /**
     * @brief Add collision capsule
     * @param p0 Capsule endpoint 0 in local space
     * @param p1 Capsule endpoint 1 in local space
     * @param radius Capsule radius
     */
    virtual void AddCollisionCapsule(const Vec3&amp; p0, const Vec3&amp; p1, float radius) = 0;

    // ===== Tearing/Fracture =====

    /**
     * @brief Enable or disable tearing
     * @param tearable If true, soft body can tear when stretched too much
     */
    virtual void SetTearable(bool tearable) = 0;

    /**
     * @brief Manually tear at a specific vertex
     * @param vertexIndex Index of vertex to tear
     * @return True if tear was successful
     */
    virtual bool TearAtVertex(int vertexIndex) = 0;

    /**
     * @brief Set tear threshold (stress ratio)
     * @param threshold Maximum stress before tearing (e.g., 2.0 = 200% stretch)
     */
    virtual void SetTearThreshold(float threshold) = 0;
    
    /**
     * @brief Get tear threshold
     */
    virtual float GetTearThreshold() const = 0;
    
    // ===== State Queries =====

    /**
     * @brief Get total mass of soft body
     * @return Total mass in kilograms
     */
    virtual float GetTotalMass() const = 0;

    /**
     * @brief Get current volume
     * @return Volume in cubic meters
     */
    virtual float GetVolume() const = 0;

    /**
     * @brief Get center of mass
     * @return Center of mass position
     */
    virtual Vec3 GetCenterOfMass() const = 0;

    /**
     * @brief Check if soft body is active (not sleeping)
     * @return True if active
     */
    virtual bool IsActive() const = 0;

    /**
     * @brief Set active state
     * @param active If true, wake up the soft body
     */
    virtual void SetActive(bool active) = 0;

    /**
     * @brief Get backend-specific soft body pointer
     * @return Opaque pointer to native soft body object
     */
    virtual void* GetNativeSoftBody() = 0;
};
