#pragma once

#include "Ray.h"
#include "Math/Vec3.h"
#include <vector>
#include <limits>

class PhysXSoftBody;

/**
 * @brief Picks vertices in 3D space using ray casting
 * 
 * Provides functionality to select vertices by clicking in the viewport.
 */
class VertexPicker {
public:
    struct PickResult {
        bool hit;
        int vertexIndex;
        float distance;
        Vec3 worldPosition;
        
        PickResult() 
            : hit(false)
            , vertexIndex(-1)
            , distance(std::numeric_limits<float>::max())
            , worldPosition(0, 0, 0)
        {}
    };
    
    VertexPicker();
    
    /**
     * @brief Pick closest vertex to ray
     * @param ray Ray from camera through mouse position
     * @param softBody Soft body to pick from
     * @param maxDistance Maximum distance from ray to vertex
     * @return Pick result with hit information
     */
    PickResult PickVertex(
        const Ray& ray,
        PhysXSoftBody* softBody,
        float maxDistance = 0.5f
    );
    
    /**
     * @brief Pick all vertices within radius of a point
     * @param center Center point
     * @param radius Selection radius
     * @param softBody Soft body to pick from
     * @return List of vertex indices
     */
    std::vector<int> PickVerticesInRadius(
        const Vec3& center,
        float radius,
        PhysXSoftBody* softBody
    );
    
    /**
     * @brief Set/get pick radius
     */
    void SetPickRadius(float radius) { m_PickRadius = radius; }
    float GetPickRadius() const { return m_PickRadius; }
    
private:
    float m_PickRadius;
    
    float CalculateDistanceToVertex(const Ray& ray, const Vec3& vertex);
};
