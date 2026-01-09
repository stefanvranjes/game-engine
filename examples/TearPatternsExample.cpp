// Example: Tear Patterns Demo
// Demonstrates straight, curved, and radial tear patterns

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"
#include "StraightTearPattern.h"
#include "CurvedTearPattern.h"
#include "RadialTearPattern.h"

void CreateTearPatternsDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Tear Patterns Demo ===" << std::endl;
    
    // Create cube mesh
    std::vector<Vec3> cubeVertices = {
        Vec3(-0.5f, -0.5f,  0.5f), Vec3( 0.5f, -0.5f,  0.5f),
        Vec3( 0.5f,  0.5f,  0.5f), Vec3(-0.5f,  0.5f,  0.5f),
        Vec3(-0.5f, -0.5f, -0.5f), Vec3( 0.5f, -0.5f, -0.5f),
        Vec3( 0.5f,  0.5f, -0.5f), Vec3(-0.5f,  0.5f, -0.5f)
    };
    
    std::vector<int> cubeIndices = {
        0, 1, 2, 0, 2, 3,  5, 4, 7, 5, 7, 6,
        4, 0, 3, 4, 3, 7,  1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,  4, 5, 1, 4, 1, 0
    };
    
    // ===== Example 1: Straight Line Tear =====
    
    auto tetMesh1 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(), cubeVertices.size(),
        cubeIndices.data(), cubeIndices.size() / 3,
        0.15f
    );
    
    SoftBodyDesc desc1;
    desc1.vertexPositions = cubeVertices.data();
    desc1.vertexCount = cubeVertices.size();
    desc1.tetrahedronVertices = tetMesh1.vertices.data();
    desc1.tetrahedronVertexCount = tetMesh1.vertices.size();
    desc1.tetrahedronIndices = tetMesh1.indices.data();
    desc1.tetrahedronCount = tetMesh1.tetrahedronCount;
    desc1.volumeStiffness = 0.5f;
    desc1.shapeStiffness = 0.4f;
    desc1.totalMass = 1.0f;
    desc1.useDensity = false;
    desc1.enableSceneCollision = true;
    desc1.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody1 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody1->Initialize(desc1);
    softBody1->SetTearable(true);
    
    auto obj1 = std::make_shared<GameObject>("StraightTear");
    obj1->SetSoftBody(softBody1);
    obj1->GetTransform().SetPosition(Vec3(-3, 5, 0));
    scene.AddGameObject(obj1);
    
    // Apply straight line tear
    softBody1->TearStraightLine(
        Vec3(-3.5f, 4.5f, 0),  // Start
        Vec3(-2.5f, 5.5f, 0),  // End
        0.15f                   // Width
    );
    
    std::cout << "Created straight line tear" << std::endl;
    
    
    // ===== Example 2: Curved Tear =====
    
    auto tetMesh2 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(), cubeVertices.size(),
        cubeIndices.data(), cubeIndices.size() / 3,
        0.15f
    );
    
    SoftBodyDesc desc2 = desc1;
    desc2.tetrahedronVertices = tetMesh2.vertices.data();
    desc2.tetrahedronVertexCount = tetMesh2.vertices.size();
    desc2.tetrahedronIndices = tetMesh2.indices.data();
    desc2.tetrahedronCount = tetMesh2.tetrahedronCount;
    
    auto softBody2 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody2->Initialize(desc2);
    softBody2->SetTearable(true);
    
    auto obj2 = std::make_shared<GameObject>("CurvedTear");
    obj2->SetSoftBody(softBody2);
    obj2->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(obj2);
    
    // Apply curved tear with high curvature
    softBody2->TearCurvedPath(
        Vec3(-0.5f, 4.5f, 0),  // Start
        Vec3(0.5f, 5.5f, 0),   // End
        0.8f                    // Curvature
    );
    
    std::cout << "Created curved tear" << std::endl;
    
    
    // ===== Example 3: Radial Burst =====
    
    auto tetMesh3 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(), cubeVertices.size(),
        cubeIndices.data(), cubeIndices.size() / 3,
        0.15f
    );
    
    SoftBodyDesc desc3 = desc1;
    desc3.tetrahedronVertices = tetMesh3.vertices.data();
    desc3.tetrahedronVertexCount = tetMesh3.vertices.size();
    desc3.tetrahedronIndices = tetMesh3.indices.data();
    desc3.tetrahedronCount = tetMesh3.tetrahedronCount;
    
    auto softBody3 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody3->Initialize(desc3);
    softBody3->SetTearable(true);
    
    auto obj3 = std::make_shared<GameObject>("RadialBurst");
    obj3->SetSoftBody(softBody3);
    obj3->GetTransform().SetPosition(Vec3(3, 5, 0));
    scene.AddGameObject(obj3);
    
    // Apply radial burst from center
    softBody3->TearRadialBurst(
        Vec3(3, 5, 0),  // Center
        6,              // Ray count
        0.4f            // Radius
    );
    
    std::cout << "Created radial burst tear" << std::endl;
    
    
    // ===== Example 4: Custom Pattern =====
    
    // Create custom curved pattern with explicit control point
    auto tetMesh4 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(), cubeVertices.size(),
        cubeIndices.data(), cubeIndices.size() / 3,
        0.15f
    );
    
    SoftBodyDesc desc4 = desc1;
    desc4.tetrahedronVertices = tetMesh4.vertices.data();
    desc4.tetrahedronVertexCount = tetMesh4.vertices.size();
    desc4.tetrahedronIndices = tetMesh4.indices.data();
    desc4.tetrahedronCount = tetMesh4.tetrahedronCount;
    
    auto softBody4 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody4->Initialize(desc4);
    softBody4->SetTearable(true);
    
    auto obj4 = std::make_shared<GameObject>("CustomCurve");
    obj4->SetSoftBody(softBody4);
    obj4->GetTransform().SetPosition(Vec3(6, 5, 0));
    scene.AddGameObject(obj4);
    
    // Create custom curved pattern
    CurvedTearPattern customPattern(0.12f, 0.5f);
    customPattern.SetControlPoint(Vec3(6, 6, 0));  // Explicit control
    
    softBody4->TearAlongPattern(
        customPattern,
        Vec3(5.5f, 4.5f, 0),
        Vec3(6.5f, 4.5f, 0)
    );
    
    std::cout << "Created custom curved tear" << std::endl;
    
    
    std::cout << "\n=== Pattern Demo Complete ===" << std::endl;
    std::cout << "4 soft bodies with different tear patterns:" << std::endl;
    std::cout << "1. Straight line tear" << std::endl;
    std::cout << "2. Curved tear (auto-generated control)" << std::endl;
    std::cout << "3. Radial burst (6 rays)" << std::endl;
    std::cout << "4. Custom curve (explicit control point)" << std::endl;
}
