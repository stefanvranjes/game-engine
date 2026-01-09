// Example: Plasticity Demo
// Demonstrates permanent deformation without tearing

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"

void CreatePlasticityDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Plasticity Demo ===" << std::endl;
    
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
    
    // ===== Example 1: Clay-like Material =====
    
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
    desc1.volumeStiffness = 0.4f;
    desc1.shapeStiffness = 0.3f;
    desc1.totalMass = 1.0f;
    desc1.useDensity = false;
    desc1.enableSceneCollision = true;
    desc1.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody1 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody1->Initialize(desc1);
    
    // Clay: easy to deform permanently
    softBody1->SetPlasticityEnabled(true);
    softBody1->SetPlasticThreshold(1.2f);  // Yield at 120% stretch
    softBody1->SetPlasticityRate(0.1f);    // 10% deformation per frame
    softBody1->SetTearThreshold(2.5f);     // Hard to tear
    
    auto obj1 = std::make_shared<GameObject>("Clay");
    obj1->SetSoftBody(softBody1);
    obj1->GetTransform().SetPosition(Vec3(-3, 5, 0));
    scene.AddGameObject(obj1);
    
    // Apply deforming force
    softBody1->AddForceAtVertex(2, Vec3(0, 300, 0));
    softBody1->FixVertex(0, Vec3(-3.5f, 4.5f, 0.5f));
    
    std::cout << "Created clay-like material (easy plastic deformation)" << std::endl;
    
    
    // ===== Example 2: Metal-like Material =====
    
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
    desc2.volumeStiffness = 0.7f;
    desc2.shapeStiffness = 0.6f;
    
    auto softBody2 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody2->Initialize(desc2);
    
    // Metal: resists deformation but yields under high stress
    softBody2->SetPlasticityEnabled(true);
    softBody2->SetPlasticThreshold(1.5f);  // High yield point
    softBody2->SetPlasticityRate(0.03f);   // Slow deformation
    softBody2->SetTearThreshold(1.8f);     // Tears soon after yielding
    
    auto obj2 = std::make_shared<GameObject>("Metal");
    obj2->SetSoftBody(softBody2);
    obj2->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(obj2);
    
    // Apply strong force
    softBody2->AddForceAtVertex(2, Vec3(0, 500, 0));
    softBody2->FixVertex(0, Vec3(-0.5f, 4.5f, 0.5f));
    
    std::cout << \"Created metal-like material (high yield point)\" << std::endl;
    
    
    // ===== Example 3: Rubber-like Material (No Plasticity) =====
    
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
    desc3.volumeStiffness = 0.6f;
    desc3.shapeStiffness = 0.5f;
    
    auto softBody3 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody3->Initialize(desc3);
    
    // Rubber: very elastic, no plasticity
    softBody3->SetPlasticityEnabled(false);
    softBody3->SetTearThreshold(3.0f);  // Very stretchy before tearing
    
    auto obj3 = std::make_shared<GameObject>("Rubber");
    obj3->SetSoftBody(softBody3);
    obj3->GetTransform().SetPosition(Vec3(3, 5, 0));
    scene.AddGameObject(obj3);
    
    // Apply force
    softBody3->AddForceAtVertex(2, Vec3(0, 400, 0));
    softBody3->FixVertex(0, Vec3(2.5f, 4.5f, 0.5f));
    
    std::cout << \"Created rubber-like material (purely elastic)\" << std::endl;
    
    
    // ===== Example 4: Plastic with Reset =====
    
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
    
    // Plastic with ability to reset
    softBody4->SetPlasticityEnabled(true);
    softBody4->SetPlasticThreshold(1.3f);
    softBody4->SetPlasticityRate(0.08f);
    softBody4->SetTearThreshold(2.0f);
    
    auto obj4 = std::make_shared<GameObject>("PlasticWithReset");
    obj4->SetSoftBody(softBody4);
    obj4->GetTransform().SetPosition(Vec3(6, 5, 0));
    scene.AddGameObject(obj4);
    
    // Apply force
    softBody4->AddForceAtVertex(2, Vec3(0, 350, 0));
    softBody4->FixVertex(0, Vec3(5.5f, 4.5f, 0.5f));
    
    std::cout << \"Created plastic material with reset capability\" << std::endl;
    
    
    std::cout << \"\n=== Plasticity Demo Complete ===\" << std::endl;
    std::cout << \"4 soft bodies with different material behaviors:\" << std::endl;
    std::cout << \"1. Clay (low yield, high plasticity)\" << std::endl;
    std::cout << \"2. Metal (high yield, low plasticity)\" << std::endl;
    std::cout << \"3. Rubber (no plasticity, purely elastic)\" << std::endl;
    std::cout << \"4. Plastic with reset\" << std::endl;
    std::cout << \"\nWatch permanent deformation occur!\" << std::endl;
}

// Usage notes:
/*
// To reset a deformed soft body to original shape:
softBody->ResetRestShape();

// Plasticity parameters:
// - Threshold: Stress level at which plastic deformation begins
// - Rate: How quickly rest positions update (0.0 to 1.0)
// - Lower rate = slower, more gradual deformation
// - Higher rate = faster, more immediate deformation

// Material presets:
// Clay:    threshold=1.2, rate=0.1
// Metal:   threshold=1.5, rate=0.03
// Plastic: threshold=1.3, rate=0.08
*/
