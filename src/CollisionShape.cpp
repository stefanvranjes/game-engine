#include "CollisionShape.h"
#include <cmath>
#include <algorithm>

// CollisionPlane implementation
CollisionPlane::CollisionPlane(const Vec3& normal, float distance)
    : m_Normal(normal.Normalized())
    , m_Distance(distance)
{
}

bool CollisionPlane::CheckCollision(Vec3& position, Vec3& velocity, float radius,
                                    float restitution, float friction, float deltaTime)
{
    // Calculate signed distance from particle to plane
    float distance = position.Dot(m_Normal) - m_Distance;
    
    // Check if particle is penetrating the plane
    if (distance < radius) {
        // Move particle out of plane
        position = position + m_Normal * (radius - distance);
        
        // Decompose velocity into normal and tangent components
        float velocityNormal = velocity.Dot(m_Normal);
        
        // Only apply collision if moving towards the plane
        if (velocityNormal < 0) {
            Vec3 normalVelocity = m_Normal * velocityNormal;
            Vec3 tangentVelocity = velocity - normalVelocity;
            
            // Apply restitution to normal component (bounce)
            normalVelocity = normalVelocity * -restitution;
            
            // Apply friction to tangent component
            tangentVelocity = tangentVelocity * (1.0f - friction);
            
            // Recombine velocity
            velocity = normalVelocity + tangentVelocity;
            
            return true;
        }
    }
    
    return false;
}

// CollisionSphere implementation
CollisionSphere::CollisionSphere(const Vec3& center, float radius)
    : m_Center(center)
    , m_Radius(radius)
{
}

bool CollisionSphere::CheckCollision(Vec3& position, Vec3& velocity, float radius,
                                     float restitution, float friction, float deltaTime)
{
    // Vector from sphere center to particle
    Vec3 toParticle = position - m_Center;
    float distance = toParticle.Length();
    float minDistance = m_Radius + radius;
    
    // Check if particle is penetrating the sphere
    if (distance < minDistance && distance > 0.001f) {
        // Collision normal points away from sphere center
        Vec3 normal = toParticle / distance;
        
        // Move particle out of sphere
        position = m_Center + normal * minDistance;
        
        // Decompose velocity into normal and tangent components
        float velocityNormal = velocity.Dot(normal);
        
        // Only apply collision if moving towards the sphere
        if (velocityNormal < 0) {
            Vec3 normalVelocity = normal * velocityNormal;
            Vec3 tangentVelocity = velocity - normalVelocity;
            
            // Apply restitution to normal component (bounce)
            normalVelocity = normalVelocity * -restitution;
            
            // Apply friction to tangent component
            tangentVelocity = tangentVelocity * (1.0f - friction);
            
            // Recombine velocity
            velocity = normalVelocity + tangentVelocity;
            
            return true;
        }
    }
    
    return false;
}

// CollisionBox implementation
CollisionBox::CollisionBox(const Vec3& min, const Vec3& max)
    : m_Min(min)
    , m_Max(max)
{
}

Vec3 CollisionBox::ClosestPoint(const Vec3& point) const
{
    return Vec3(
        std::max(m_Min.x, std::min(point.x, m_Max.x)),
        std::max(m_Min.y, std::min(point.y, m_Max.y)),
        std::max(m_Min.z, std::min(point.z, m_Max.z))
    );
}

bool CollisionBox::CheckCollision(Vec3& position, Vec3& velocity, float radius,
                                  float restitution, float friction, float deltaTime)
{
    // Find closest point on box to particle
    Vec3 closest = ClosestPoint(position);
    Vec3 toParticle = position - closest;
    float distance = toParticle.Length();
    
    // Check if particle is penetrating the box
    if (distance < radius && distance > 0.001f) {
        // Collision normal points away from box
        Vec3 normal = toParticle / distance;
        
        // Move particle out of box
        position = closest + normal * radius;
        
        // Decompose velocity into normal and tangent components
        float velocityNormal = velocity.Dot(normal);
        
        // Only apply collision if moving towards the box
        if (velocityNormal < 0) {
            Vec3 normalVelocity = normal * velocityNormal;
            Vec3 tangentVelocity = velocity - normalVelocity;
            
            // Apply restitution to normal component (bounce)
            normalVelocity = normalVelocity * -restitution;
            
            // Apply friction to tangent component
            tangentVelocity = tangentVelocity * (1.0f - friction);
            
            // Recombine velocity
            velocity = normalVelocity + tangentVelocity;
            
            return true;
        }
    }
    
    // Check if particle is inside the box
    if (position.x >= m_Min.x && position.x <= m_Max.x &&
        position.y >= m_Min.y && position.y <= m_Max.y &&
        position.z >= m_Min.z && position.z <= m_Max.z)
    {
        // Find the closest face
        float distToMinX = position.x - m_Min.x;
        float distToMaxX = m_Max.x - position.x;
        float distToMinY = position.y - m_Min.y;
        float distToMaxY = m_Max.y - position.y;
        float distToMinZ = position.z - m_Min.z;
        float distToMaxZ = m_Max.z - position.z;
        
        float minDist = std::min({distToMinX, distToMaxX, distToMinY, distToMaxY, distToMinZ, distToMaxZ});
        
        Vec3 normal;
        if (minDist == distToMinX) {
            normal = Vec3(-1, 0, 0);
            position.x = m_Min.x - radius;
        } else if (minDist == distToMaxX) {
            normal = Vec3(1, 0, 0);
            position.x = m_Max.x + radius;
        } else if (minDist == distToMinY) {
            normal = Vec3(0, -1, 0);
            position.y = m_Min.y - radius;
        } else if (minDist == distToMaxY) {
            normal = Vec3(0, 1, 0);
            position.y = m_Max.y + radius;
        } else if (minDist == distToMinZ) {
            normal = Vec3(0, 0, -1);
            position.z = m_Min.z - radius;
        } else {
            normal = Vec3(0, 0, 1);
            position.z = m_Max.z + radius;
        }
        
        // Reflect velocity
        float velocityNormal = velocity.Dot(normal);
        if (velocityNormal < 0) {
            Vec3 normalVelocity = normal * velocityNormal;
            Vec3 tangentVelocity = velocity - normalVelocity;
            
            normalVelocity = normalVelocity * -restitution;
            tangentVelocity = tangentVelocity * (1.0f - friction);
            
            velocity = normalVelocity + tangentVelocity;
        }
        
        return true;
    }
    
    return false;
}
