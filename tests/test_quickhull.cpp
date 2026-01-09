#include <gtest/gtest.h>
#include "QuickHull.h"
#include "Math/Vec3.h"
#include <vector>
#include <cmath>

class QuickHullTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
    
    // Helper to create a cube
    std::vector<Vec3> CreateCube(float size) {
        std::vector<Vec3> points;
        float half = size * 0.5f;
        points.push_back(Vec3(-half, -half, -half));
        points.push_back(Vec3( half, -half, -half));
        points.push_back(Vec3( half,  half, -half));
        points.push_back(Vec3(-half,  half, -half));
        points.push_back(Vec3(-half, -half,  half));
        points.push_back(Vec3( half, -half,  half));
        points.push_back(Vec3( half,  half,  half));
        points.push_back(Vec3(-half,  half,  half));
        return points;
    }
};

TEST_F(QuickHullTest, Tetrahedron) {
    std::vector<Vec3> points;
    points.push_back(Vec3(0, 0, 0));
    points.push_back(Vec3(1, 0, 0));
    points.push_back(Vec3(0, 1, 0));
    points.push_back(Vec3(0, 0, 1));
    
    QuickHull qh;
    ConvexHull hull = qh.ComputeHull(points.data(), points.size());
    
    EXPECT_EQ(hull.faceCount, 4);
    // Surface area of unit tetrahedron at origin...
    // Base area (XY) = 0.5
    // Side area (XZ) = 0.5
    // Side area (YZ) = 0.5
    // Slanted face area = sqrt(3)/2 * side_len? No.
    // Slanted face vertices: (1,0,0), (0,1,0), (0,0,1). Side length sqrt(2).
    // Equilateral triangle with side sqrt(2). Area = sqrt(3)/4 * a^2 = sqrt(3)/4 * 2 = sqrt(3)/2 ≈ 0.866
    // Total area = 1.5 + 0.866 = 2.366
    
    float expectedArea = 0.5f + 0.5f + 0.5f + (std::sqrt(3.0f) / 2.0f);
    EXPECT_NEAR(hull.surfaceArea, expectedArea, 0.001f);
}

TEST_F(QuickHullTest, Cube) {
    std::vector<Vec3> points = CreateCube(2.0f);
    
    QuickHull qh;
    ConvexHull hull = qh.ComputeHull(points.data(), points.size());
    
    // A cube has 6 faces (quads), but QuickHull returns triangles.
    // Each quad is 2 triangles, so 12 triangles.
    EXPECT_EQ(hull.faceCount, 12);
    
    // Surface area = 6 * side^2 = 6 * 4 = 24
    EXPECT_NEAR(hull.surfaceArea, 24.0f, 0.001f);
}

TEST_F(QuickHullTest, CubeWithInternalPoints) {
    std::vector<Vec3> points = CreateCube(2.0f);
    // Add point inside
    points.push_back(Vec3(0, 0, 0));
    points.push_back(Vec3(0.5f, 0.5f, 0.5f));
    
    QuickHull qh;
    ConvexHull hull = qh.ComputeHull(points.data(), points.size());
    
    EXPECT_NEAR(hull.surfaceArea, 24.0f, 0.001f);
}

TEST_F(QuickHullTest, DegenerateCoplanar) {
    std::vector<Vec3> points;
    points.push_back(Vec3(0, 0, 0));
    points.push_back(Vec3(1, 0, 0));
    points.push_back(Vec3(0, 1, 0));
    points.push_back(Vec3(1, 1, 0)); // All z=0
    
    QuickHull qh;
    ConvexHull hull = qh.ComputeHull(points.data(), points.size());
    
    // Should handle gracefully (return empty hull or minimal valid hull depending on epsilon)
    // Our implementation returns empty hull for degenerate initial simplex
    EXPECT_EQ(hull.faceCount, 0);
    EXPECT_EQ(hull.surfaceArea, 0.0f);
}

TEST_F(QuickHullTest, DegenerateCollinear) {
    std::vector<Vec3> points;
    points.push_back(Vec3(0, 0, 0));
    points.push_back(Vec3(1, 0, 0));
    points.push_back(Vec3(2, 0, 0));
    points.push_back(Vec3(3, 0, 0));
    
    QuickHull qh;
    ConvexHull hull = qh.ComputeHull(points.data(), points.size());
    
    EXPECT_EQ(hull.faceCount, 0);
    EXPECT_EQ(hull.surfaceArea, 0.0f);
}
