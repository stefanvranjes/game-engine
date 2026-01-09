// Example: Variable Tear Resistance Demo
// Demonstrates region-based and gradient resistance

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"

void CreateVariableResistanceDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Variable Tear Resistance Demo ===" << std::endl;
    
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
    
    // ===== Example 1: Reinforced Center =====
    
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
    
    // Reinforce center (2x resistance)
    softBody1->SetRegionResistance(
        Vec3(-3, 5, 0),  // Center
        0.25f,           // Radius
        2.0f             // 2x resistance
    );
    
    auto obj1 = std::make_shared<GameObject>("ReinforcedCenter");
    obj1->SetSoftBody(softBody1);
    obj1->GetTransform().SetPosition(Vec3(-3, 5, 0));
    scene.AddGameObject(obj1);
    
    // Apply force to edges
    softBody1->AddForceAtVertex(0, Vec3(-200, 0, 0));
    softBody1->AddForceAtVertex(6, Vec3(200, 0, 0));
    
    std::cout << "Created soft body with reinforced center" << std::endl;
    
    
    // ===== Example 2: Weak Edges =====
    
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
    
    // Weaken edges (0.5x resistance)
    auto& resistanceMap2 = softBody2->GetResistanceMap();
    
    // Manually set weak edges
    for (int i = 0; i < tetMesh2.tetrahedronCount; ++i) {
        // Calculate tet center
        int v0 = tetMesh2.indices[i * 4 + 0];
        int v1 = tetMesh2.indices[i * 4 + 1];
        int v2 = tetMesh2.indices[i * 4 + 2];
        int v3 = tetMesh2.indices[i * 4 + 3];
        
        Vec3 center = (tetMesh2.vertices[v0] + tetMesh2.vertices[v1] +
                      tetMesh2.vertices[v2] + tetMesh2.vertices[v3]) * 0.25f;
        
        float distFromCenter = center.Length();
        
        // Weaken outer tetrahedra
        if (distFromCenter > 0.3f) {
            resistanceMap2.SetTetrahedronResistance(i, 0.5f);
        }
    }
    
    auto obj2 = std::make_shared<GameObject>("WeakEdges");
    obj2->SetSoftBody(softBody2);
    obj2->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(obj2);
    
    softBody2->AddForce(Vec3(0, -100, 0));
    
    std::cout << "Created soft body with weak edges" << std::endl;
    
    
    // ===== Example 3: Vertical Gradient =====
    
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
    
    // Gradient: strong at top, weak at bottom
    softBody3->SetResistanceGradient(
        Vec3(3, 5.5f, 0),  // Top
        Vec3(3, 4.5f, 0),  // Bottom
        2.0f,              // Strong at top
        0.5f               // Weak at bottom
    );
    
    auto obj3 = std::make_shared<GameObject>("VerticalGradient");
    obj3->SetSoftBody(softBody3);
    obj3->GetTransform().SetPosition(Vec3(3, 5, 0));
    scene.AddGameObject(obj3);
    
    // Fix top, apply force to bottom
    softBody3->FixVertex(3, Vec3(2.5f, 5.5f, 0.5f));
    softBody3->FixVertex(7, Vec3(2.5f, 5.5f, -0.5f));
    softBody3->AddForceAtVertex(0, Vec3(0, -300, 0));
    
    std::cout << "Created soft body with vertical gradient" << std::endl;
    
    
    // ===== Example 4: Multiple Regions =====
    
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
    
    // Multiple reinforced regions
    softBody4->SetRegionResistance(Vec3(5.5f, 5, 0), 0.15f, 2.5f);  // Right strong
    softBody4->SetRegionResistance(Vec3(6.5f, 5, 0), 0.15f, 2.5f);  // Left strong
    softBody4->SetRegionResistance(Vec3(6, 5.5f, 0), 0.15f, 0.5f);  // Top weak
    
    auto obj4 = std::make_shared<GameObject>("MultipleRegions");
    obj4->SetSoftBody(softBody4);
    obj4->GetTransform().SetPosition(Vec3(6, 5, 0));
    scene.AddGameObject(obj4);
    
    softBody4->AddForceAtVertex(3, Vec3(0, 200, 0));
    
    std::cout << "Created soft body with multiple regions" << std::endl;
    
    
    std::cout << "\n=== Resistance Demo Complete ===" << std::endl;
    std::cout << "4 soft bodies with variable resistance:" << std::endl;
    std::cout << "1. Reinforced center (2x)" << std::endl;
    std::cout << "2. Weak edges (0.5x)" << std::endl;
    std::cout << "3. Vertical gradient (2x to 0.5x)" << std::endl;
    std::cout << "4. Multiple regions" << std::endl;
}
