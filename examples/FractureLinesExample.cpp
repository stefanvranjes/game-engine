// Example: Fracture Lines Demo
// Demonstrates pre-scored lines for controlled tearing

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"
#include "FractureLine.h"

void CreateFractureLinesDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Fracture Lines Demo ===" << std::endl;
    
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
    
    // ===== Example 1: Single Fracture Line =====
    
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
    softBody1->SetTearThreshold(1.5f);
    
    // Create vertical fracture line
    FractureLine verticalLine(0.3f);  // 30% resistance
    verticalLine.SetWidth(0.15f);
    verticalLine.AddPoint(Vec3(-3, 4.5f, 0));
    verticalLine.AddPoint(Vec3(-3, 5.5f, 0));
    
    softBody1->AddFractureLine(verticalLine);
    
    auto obj1 = std::make_shared<GameObject>("SingleFracture");
    obj1->SetSoftBody(softBody1);
    obj1->GetTransform().SetPosition(Vec3(-3, 5, 0));
    scene.AddGameObject(obj1);
    
    // Apply force to tear along line
    softBody1->AddForceAtVertex(1, Vec3(300, 0, 0));
    softBody1->AddForceAtVertex(5, Vec3(-300, 0, 0));
    
    std::cout << "Created soft body with single vertical fracture line" << std::endl;
    
    
    // ===== Example 2: Cross Pattern =====
    
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
    softBody2->SetTearThreshold(1.5f);
    
    // Create cross pattern
    FractureLine horizontalLine(0.4f);
    horizontalLine.SetWidth(0.12f);
    horizontalLine.AddPoint(Vec3(-0.5f, 5, 0));
    horizontalLine.AddPoint(Vec3(0.5f, 5, 0));
    
    FractureLine verticalLine2(0.4f);
    verticalLine2.SetWidth(0.12f);
    verticalLine2.AddPoint(Vec3(0, 4.5f, 0));
    verticalLine2.AddPoint(Vec3(0, 5.5f, 0));
    
    softBody2->AddFractureLine(horizontalLine);
    softBody2->AddFractureLine(verticalLine2);
    
    auto obj2 = std::make_shared<GameObject>("CrossPattern");
    obj2->SetSoftBody(softBody2);
    obj2->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(obj2);
    
    // Apply radial force
    softBody2->AddForceAtVertex(0, Vec3(-200, -200, 0));
    softBody2->AddForceAtVertex(2, Vec3(200, 200, 0));
    softBody2->AddForceAtVertex(5, Vec3(200, -200, 0));
    softBody2->AddForceAtVertex(7, Vec3(-200, 200, 0));
    
    std::cout << "Created soft body with cross fracture pattern" << std::endl;
    
    
    // ===== Example 3: Curved Fracture Line =====
    
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
    softBody3->SetTearThreshold(1.5f);
    
    // Create curved fracture line
    FractureLine curvedLine(0.35f);
    curvedLine.SetWidth(0.14f);
    curvedLine.AddPoint(Vec3(2.5f, 4.5f, 0));
    curvedLine.AddPoint(Vec3(3, 5, 0));
    curvedLine.AddPoint(Vec3(3.5f, 5.5f, 0));
    
    softBody3->AddFractureLine(curvedLine);
    
    auto obj3 = std::make_shared<GameObject>("CurvedFracture");
    obj3->SetSoftBody(softBody3);
    obj3->GetTransform().SetPosition(Vec3(3, 5, 0));
    scene.AddGameObject(obj3);
    
    // Apply force along curve
    softBody3->AddForceAtVertex(0, Vec3(-250, -250, 0));
    softBody3->AddForceAtVertex(6, Vec3(250, 250, 0));
    
    std::cout << "Created soft body with curved fracture line" << std::endl;
    
    
    // ===== Example 4: Very Weak Perforation =====
    
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
    softBody4->SetTearThreshold(1.5f);
    
    // Create very weak perforation line
    FractureLine perforation(0.1f);  // Only 10% resistance!
    perforation.SetWidth(0.1f);
    perforation.AddPoint(Vec3(5.5f, 4.5f, 0));
    perforation.AddPoint(Vec3(6.5f, 5.5f, 0));
    
    softBody4->AddFractureLine(perforation);
    
    auto obj4 = std::make_shared<GameObject>("Perforation");
    obj4->SetSoftBody(softBody4);
    obj4->GetTransform().SetPosition(Vec3(6, 5, 0));
    scene.AddGameObject(obj4);
    
    // Light force should tear along perforation
    softBody4->AddForceAtVertex(0, Vec3(-150, 0, 0));
    softBody4->AddForceAtVertex(6, Vec3(150, 0, 0));
    
    std::cout << "Created soft body with perforation (very weak)" << std::endl;
    
    
    std::cout << "\n=== Fracture Lines Demo Complete ===" << std::endl;
    std::cout << "4 soft bodies with different fracture patterns:" << std::endl;
    std::cout << "1. Single vertical line (30% resistance)" << std::endl;
    std::cout << "2. Cross pattern (40% resistance)" << std::endl;
    std::cout << "3. Curved line (35% resistance)" << std::endl;
    std::cout << "4. Perforation (10% resistance)" << std::endl;
    std::cout << "\nTears will follow the pre-scored lines!" << std::endl;
}
