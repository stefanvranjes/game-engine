#pragma once

#include "Math/Vec3.h"
#include <memory>

// Base class for collision shapes
class CollisionShape {
public:
    virtual ~CollisionShape() = default;
    
    // Check if a point at given position with velocity collides with this shape
    // Returns true if collision occurred, updates position and velocity
    virtual bool CheckCollision(Vec3& position, Vec3& velocity, float radius, 
                               float restitution, float friction, float deltaTime) = 0;
};

// Infinite plane collision shape
class CollisionPlane : public CollisionShape {
public:
    CollisionPlane(const Vec3& normal, float distance);
    
    bool CheckCollision(Vec3& position, Vec3& velocity, float radius,
                       float restitution, float friction, float deltaTime) override;
    
    Vec3 GetNormal() const { return m_Normal; }
    float GetDistance() const { return m_Distance; }
    
private:
    Vec3 m_Normal;    // Plane normal (should be normalized)
    float m_Distance; // Distance from origin along normal
};

// Sphere collision shape
class CollisionSphere : public CollisionShape {
public:
    CollisionSphere(const Vec3& center, float radius);
    
    bool CheckCollision(Vec3& position, Vec3& velocity, float radius,
                       float restitution, float friction, float deltaTime) override;
    
    Vec3 GetCenter() const { return m_Center; }
    float GetRadius() const { return m_Radius; }
    void SetCenter(const Vec3& center) { m_Center = center; }
    
private:
    Vec3 m_Center;
    float m_Radius;
};

// Axis-aligned box collision shape
class CollisionBox : public CollisionShape {
public:
    CollisionBox(const Vec3& min, const Vec3& max);
    
    bool CheckCollision(Vec3& position, Vec3& velocity, float radius,
                       float restitution, float friction, float deltaTime) override;
    
    Vec3 GetMin() const { return m_Min; }
    Vec3 GetMax() const { return m_Max; }
    void SetBounds(const Vec3& min, const Vec3& max) { m_Min = min; m_Max = max; }
    
private:
    Vec3 m_Min;
    Vec3 m_Max;
    
    // Helper to find closest point on box to a given point
    Vec3 ClosestPoint(const Vec3& point) const;
};
