#pragma once

#include "IPhysicsCloth.h"
#include "Math/Vec3.h"
#include "Mesh.h"
#include <vector>

/**
 * @brief Utility class for generating cloth meshes
 * 
 * Provides factory methods for creating common cloth shapes
 * and converting existing meshes to cloth simulations.
 */
class ClothMeshGenerator {
public:
    /**
     * @brief Create a rectangular cloth mesh (flags, curtains, tablecloths)
     * @param width Width in meters
     * @param height Height in meters
     * @param segmentsX Number of segments along X axis
     * @param segmentsY Number of segments along Y axis
     * @param position World position of cloth center
     * @return Cloth descriptor ready for initialization
     */
    static ClothDesc CreateRectangularCloth(
        float width,
        float height,
        int segmentsX,
        int segmentsY,
        const Vec3& position = Vec3(0, 0, 0)
    );

    /**
     * @brief Create cloth from an existing mesh
     * @param mesh Source mesh
     * @return Cloth descriptor
     */
    static ClothDesc CreateFromMesh(const Mesh& mesh);

    /**
     * @brief Create a circular cloth (parachute, dome)
     * @param radius Radius in meters
     * @param segments Number of radial segments
     * @param position World position
     * @return Cloth descriptor
     */
    static ClothDesc CreateCircularCloth(
        float radius,
        int segments,
        const Vec3& position = Vec3(0, 0, 0)
    );

    /**
     * @brief Free cloth descriptor memory
     * @param desc Cloth descriptor to free
     */
    static void FreeClothDesc(ClothDesc& desc);

private:
    static void GenerateUVCoordinates(
        std::vector<Vec3>& positions,
        std::vector<Vec2>& uvs,
        float width,
        float height
    );
};
