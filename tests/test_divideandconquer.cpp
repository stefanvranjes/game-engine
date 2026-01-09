#include <gtest/gtest.h>
#include "DivideAndConquerHull.h"
#include "Math/Vec3.h"
#include <vector>
#include <cmath>

class DivideAndConquerTest : public ::testing::Test {
protected:
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

TEST_F(DivideAndConquerTest, Tetrahedron) {
    std::vector<Vec3> points;
    points.push_back(Vec3(0, 0, 0));
    points.push_back(Vec3(1, 0, 0));
    points.push_back(Vec3(0, 1, 0));
    points.push_back(Vec3(0, 0, 1));
    
    DivideAndConquerHull hull;
    ConvexHull result = hull.ComputeHull(points.data(), points.size());
    
    EXPECT_EQ(result.faceCount, 4);
    
    float expectedArea = 0.5f + 0.5f + 0.5f + (std::sqrt(3.0f) / 2.0f);
    EXPECT_NEAR(result.surfaceArea, expectedArea, 0.001f);
}

TEST_F(DivideAndConquerTest, Cube) {
    std::vector<Vec3> points = CreateCube(2.0f);
    
    DivideAndConquerHull hull;
    ConvexHull result = hull.ComputeHull(points.data(), points.size());
    
    EXPECT_EQ(result.faceCount, 12);
    EXPECT_NEAR(result.surfaceArea, 24.0f, 0.001f);
}

TEST_F(DivideAndConquerTest, RandomSphere) {
    int pointCount = 1000;
    std::vector<Vec3> points;
    for (int i = 0; i < pointCount; ++i) {
        float u = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float theta = 2.0f * 3.14159f * u;
        float phi = std::acos(2.0f * v - 1.0f);
        
        float x = std::sin(phi) * std::cos(theta);
        float y = std::sin(phi) * std::sin(theta);
        float z = std::cos(phi);
        
        points.push_back(Vec3(x, y, z) * 5.0f);
    }
    
    DivideAndConquerHull hull;
    ConvexHull result = hull.ComputeHull(points.data(), points.size());
    
    EXPECT_NEAR(result.surfaceArea, 314.159f, 5.0f);
}
