// Example: Tear Healing Demo
// Demonstrates gradual tear healing over time

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"

void CreateTearHealingDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Tear Healing Demo ===" << std::endl;
    
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
    
    // ===== Example 1: Fast Healing =====
    
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
    softBody1->SetTearThreshold(1.3f);
    
    // Fast healing (1 second to full recovery)
    softBody1->SetHealingEnabled(true);
    softBody1->SetHealingRate(1.0f);   // 100% per second
    softBody1->SetHealingDelay(0.5f);  // 0.5 second delay
    
    auto obj1 = std::make_shared<GameObject>("FastHealing");
    obj1->SetSoftBody(softBody1);
    obj1->GetTransform().SetPosition(Vec3(-3, 5, 0));
    scene.AddGameObject(obj1);
    
    // Cause tear
    softBody1->TearStraightLine(Vec3(-3.5f, 4.5f, 0), Vec3(-2.5f, 5.5f, 0), 0.15f);
    
    std::cout << "Created fast healing soft body (1 sec recovery)" << std::endl;
    
    
    // ===== Example 2: Slow Healing =====
    
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
    softBody2->SetTearThreshold(1.3f);
    
    // Slow healing (20 seconds to full recovery)
    softBody2->SetHealingEnabled(true);
    softBody2->SetHealingRate(0.05f);  // 5% per second
    softBody2->SetHealingDelay(2.0f);  // 2 second delay
    
    auto obj2 = std::make_shared<GameObject>("SlowHealing");
    obj2->SetSoftBody(softBody2);
    obj2->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(obj2);
    
    // Cause tear
    softBody2->TearStraightLine(Vec3(-0.5f, 4.5f, 0), Vec3(0.5f, 5.5f, 0), 0.15f);
    
    std::cout << "Created slow healing soft body (20 sec recovery)" << std::endl;
    
    
    // ===== Example 3: No Healing (Control) =====
    
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
    softBody3->SetTearThreshold(1.3f);
    
    // No healing (permanent damage)
    softBody3->SetHealingEnabled(false);
    
    auto obj3 = std::make_shared<GameObject>("NoHealing");
    obj3->SetSoftBody(softBody3);
    obj3->GetTransform().SetPosition(Vec3(3, 5, 0));
    scene.AddGameObject(obj3);
    
    // Cause tear
    softBody3->TearStraightLine(Vec3(2.5f, 4.5f, 0), Vec3(3.5f, 5.5f, 0), 0.15f);
    
    std::cout << "Created non-healing soft body (permanent damage)" << std::endl;
    
    
    // ===== Example 4: Medium Healing with Monitoring =====
    
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
    softBody4->SetTearThreshold(1.3f);
    
    // Medium healing (5 seconds to full recovery)
    softBody4->SetHealingEnabled(true);
    softBody4->SetHealingRate(0.2f);   // 20% per second
    softBody4->SetHealingDelay(1.0f);  // 1 second delay
    
    auto obj4 = std::make_shared<GameObject>("MediumHealing");
    obj4->SetSoftBody(softBody4);
    obj4->GetTransform().SetPosition(Vec3(6, 5, 0));
    scene.AddGameObject(obj4);
    
    // Cause multiple tears
    softBody4->TearStraightLine(Vec3(5.5f, 4.5f, 0), Vec3(6.5f, 5.5f, 0), 0.1f);
    softBody4->TearCurvedPath(Vec3(5.5f, 5, 0), Vec3(6.5f, 5, 0), 0.5f);
    
    std::cout << "Created medium healing soft body with monitoring" << std::endl;
    
    
    std::cout << "\n=== Healing Demo Complete ===" << std::endl;
    std::cout << "4 soft bodies with different healing rates:" << std::endl;
    std::cout << "1. Fast healing (1 sec, 0.5s delay)" << std::endl;
    std::cout << "2. Slow healing (20 sec, 2s delay)" << std::endl;
    std::cout << "3. No healing (permanent)" << std::endl;
    std::cout << "4. Medium healing (5 sec, 1s delay)" << std::endl;
    std::cout << "\nWatch tears gradually heal over time!" << std::endl;
}

// Update loop example:
/*
void UpdateHealingDemo(float deltaTime) {
    // Healing is updated automatically in PhysXSoftBody::Update()
    // which is called by PhysXBackend::Update()
    
    // Optional: Monitor healing progress
    static float monitorTimer = 0.0f;
    monitorTimer += deltaTime;
    
    if (monitorTimer > 1.0f) {
        // Check healing status every second
        int healingCount = softBody->GetHealingTearCount();
        if (healingCount > 0) {
            std::cout << "Healing tears: " << healingCount << std::endl;
        }
        monitorTimer = 0.0f;
    }
}
*/
