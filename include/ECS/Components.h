#pragma once

#include "Component.h"
#include "Math/Vec3.h"
#include "Math/Quat.h"
#include "Math/Mat4.h"

/**
 * @brief Transform component for entity positioning, rotation, and scaling.
 * 
 * Provides local transformation relative to parent entity (if any).
 * World transform is computed based on parent hierarchy.
 */
class TransformComponent : public ComponentBase<TransformComponent> {
public:
    TransformComponent()
        : m_Position(0.0f), m_Rotation(1.0f, 0.0f, 0.0f, 0.0f), m_Scale(1.0f) {}

    // Position
    void SetPosition(const Vec3& pos) { m_Position = pos; m_IsDirty = true; }
    Vec3 GetPosition() const { return m_Position; }
    void Translate(const Vec3& offset) { m_Position += offset; m_IsDirty = true; }

    // Rotation
    void SetRotation(const Quat& rot) { m_Rotation = rot; m_IsDirty = true; }
    Quat GetRotation() const { return m_Rotation; }

    // Scale
    void SetScale(const Vec3& scale) { m_Scale = scale; m_IsDirty = true; }
    Vec3 GetScale() const { return m_Scale; }

    // Matrix computation
    Mat4 GetLocalMatrix() const;
    Mat4 GetWorldMatrix() const { return m_WorldMatrix; }

    // Euler angles (convenient but slower than quaternions)
    void SetEulerAngles(const Vec3& angles);
    Vec3 GetEulerAngles() const;

    void SetWorldMatrix(const Mat4& world) { m_WorldMatrix = world; m_IsDirty = false; }
    void MarkDirty() { m_IsDirty = true; }
    bool IsDirty() const { return m_IsDirty; }

private:
    Vec3 m_Position;
    Quat m_Rotation;
    Vec3 m_Scale;
    mutable Mat4 m_WorldMatrix;
    mutable bool m_IsDirty = true;
};

/**
 * @brief Velocity component for physics-based movement.
 */
class VelocityComponent : public ComponentBase<VelocityComponent> {
public:
    VelocityComponent(const Vec3& velocity = Vec3(0.0f))
        : m_Velocity(velocity), m_AngularVelocity(0.0f) {}

    void SetVelocity(const Vec3& vel) { m_Velocity = vel; }
    Vec3 GetVelocity() const { return m_Velocity; }

    void SetAngularVelocity(const Vec3& angVel) { m_AngularVelocity = angVel; }
    Vec3 GetAngularVelocity() const { return m_AngularVelocity; }

    void ApplyForce(const Vec3& force) { m_Velocity += force; }

private:
    Vec3 m_Velocity;
    Vec3 m_AngularVelocity;
};

/**
 * @brief Basic sprite rendering component.
 */
class SpriteComponent : public ComponentBase<SpriteComponent> {
public:
    SpriteComponent() : m_TextureID(0), m_Color(1.0f), m_Opacity(1.0f) {}

    void SetTextureID(unsigned int texID) { m_TextureID = texID; }
    unsigned int GetTextureID() const { return m_TextureID; }

    void SetColor(const Vec3& color) { m_Color = color; }
    Vec3 GetColor() const { return m_Color; }

    void SetOpacity(float opacity) { m_Opacity = opacity; }
    float GetOpacity() const { return m_Opacity; }

private:
    unsigned int m_TextureID;
    Vec3 m_Color;
    float m_Opacity;
};

/**
 * @brief Mesh rendering component for 3D models.
 */
class MeshComponent : public ComponentBase<MeshComponent> {
public:
    MeshComponent() : m_MeshID(0), m_MaterialID(0) {}

    void SetMeshID(unsigned int meshID) { m_MeshID = meshID; }
    unsigned int GetMeshID() const { return m_MeshID; }

    void SetMaterialID(unsigned int matID) { m_MaterialID = matID; }
    unsigned int GetMaterialID() const { return m_MaterialID; }

    void SetCastShadow(bool cast) { m_CastShadow = cast; }
    bool GetCastShadow() const { return m_CastShadow; }

    void SetReceiveShadow(bool receive) { m_ReceiveShadow = receive; }
    bool GetReceiveShadow() const { return m_ReceiveShadow; }

private:
    unsigned int m_MeshID;
    unsigned int m_MaterialID;
    bool m_CastShadow = true;
    bool m_ReceiveShadow = true;
};

/**
 * @brief Rigidbody component for physics simulation.
 */
class RigidbodyComponent : public ComponentBase<RigidbodyComponent> {
public:
    enum class BodyType {
        Static,      // No physics, immovable
        Dynamic,     // Affected by physics
        Kinematic    // Animated, affects others but not affected
    };

    RigidbodyComponent(BodyType type = BodyType::Dynamic, float mass = 1.0f)
        : m_BodyType(type), m_Mass(mass), m_Drag(0.1f), m_AngularDrag(0.1f) {}

    void SetBodyType(BodyType type) { m_BodyType = type; }
    BodyType GetBodyType() const { return m_BodyType; }

    void SetMass(float mass) { m_Mass = std::max(0.001f, mass); }
    float GetMass() const { return m_Mass; }

    void SetDrag(float drag) { m_Drag = drag; }
    float GetDrag() const { return m_Drag; }

    void SetAngularDrag(float drag) { m_AngularDrag = drag; }
    float GetAngularDrag() const { return m_AngularDrag; }

    void SetUseGravity(bool use) { m_UseGravity = use; }
    bool GetUseGravity() const { return m_UseGravity; }

private:
    BodyType m_BodyType;
    float m_Mass;
    float m_Drag;
    float m_AngularDrag;
    bool m_UseGravity = true;
};

/**
 * @brief Collider component for collision detection.
 */
class ColliderComponent : public ComponentBase<ColliderComponent> {
public:
    enum class Shape {
        Box,
        Sphere,
        Capsule,
        Mesh
    };

    ColliderComponent(Shape shape = Shape::Box)
        : m_Shape(shape), m_IsTrigger(false) {}

    void SetShape(Shape shape) { m_Shape = shape; }
    Shape GetShape() const { return m_Shape; }

    void SetSize(const Vec3& size) { m_Size = size; }
    Vec3 GetSize() const { return m_Size; }

    void SetIsTrigger(bool isTrigger) { m_IsTrigger = isTrigger; }
    bool GetIsTrigger() const { return m_IsTrigger; }

private:
    Shape m_Shape;
    Vec3 m_Size = Vec3(1.0f);
    bool m_IsTrigger;
};

/**
 * @brief Light component for scene lighting.
 */
class LightComponent : public ComponentBase<LightComponent> {
public:
    enum class Type {
        Directional,
        Point,
        Spot
    };

    LightComponent(Type type = Type::Point)
        : m_Type(type), m_Color(1.0f), m_Intensity(1.0f),
          m_Range(10.0f), m_SpotAngle(45.0f) {}

    void SetType(Type type) { m_Type = type; }
    Type GetType() const { return m_Type; }

    void SetColor(const Vec3& color) { m_Color = color; }
    Vec3 GetColor() const { return m_Color; }

    void SetIntensity(float intensity) { m_Intensity = intensity; }
    float GetIntensity() const { return m_Intensity; }

    void SetRange(float range) { m_Range = range; }
    float GetRange() const { return m_Range; }

    void SetSpotAngle(float angle) { m_SpotAngle = angle; }
    float GetSpotAngle() const { return m_SpotAngle; }

    void SetCastShadow(bool cast) { m_CastShadow = cast; }
    bool GetCastShadow() const { return m_CastShadow; }

private:
    Type m_Type;
    Vec3 m_Color;
    float m_Intensity;
    float m_Range;
    float m_SpotAngle;
    bool m_CastShadow = true;
};

/**
 * @brief Tag component for marking entity types (e.g., "Player", "Enemy", "Collectible").
 * Use empty class for existential checks: HasComponent<TagComponent<Player>>()
 */
template<typename Tag>
class TagComponent : public ComponentBase<TagComponent<Tag>> {
    // Empty intentionally - just marks presence of a tag
};
