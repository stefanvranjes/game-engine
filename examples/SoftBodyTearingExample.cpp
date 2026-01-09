// Example: Soft Body Tearing Demo
// Demonstrates stress detection and tear propagation

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "PhysXRigidBody.h"
#include "TetrahedralMeshGenerator.h"
#include "SoftBodyPieceManager.h"

void CreateTearingDemo(PhysXBackend* physxBackend, Scene& scene) {
    // ===== Example 1: Simple Tear Test =====
    
    std::cout << "=== Soft Body Tearing Demo ===" << std::endl;
    
    // Create cube mesh
    std::vector<Vec3> cubeVertices = {
        Vec3(-0.5f, -0.5f,  0.5f), Vec3( 0.5f, -0.5f,  0.5f),
        Vec3( 0.5f,  0.5f,  0.5f), Vec3(-0.5f,  0.5f,  0.5f),
        Vec3(-0.5f, -0.5f, -0.5f), Vec3( 0.5f, -0.5f, -0.5f),
        Vec3( 0.5f,  0.5f, -0.5f), Vec3(-0.5f,  0.5f, -0.5f)
    };
    
    std::vector<int> cubeIndices = {
        0, 1, 2, 0, 2, 3,  // Front
        5, 4, 7, 5, 7, 6,  // Back
        4, 0, 3, 4, 3, 7,  // Left
        1, 5, 6, 1, 6, 2,  // Right
        3, 2, 6, 3, 6, 7,  // Top
        4, 5, 1, 4, 1, 0   // Bottom
    };
    
    // Generate tetrahedral mesh
    auto tetMesh = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.2f  // Coarser mesh for easier tearing
    );
    
    // Create soft body with tearing enabled
    SoftBodyDesc desc;
    desc.vertexPositions = cubeVertices.data();
    desc.vertexCount = static_cast<int>(cubeVertices.size());
    desc.triangleIndices = cubeIndices.data();
    desc.triangleCount = static_cast<int>(cubeIndices.size() / 3);
    
    desc.tetrahedronVertices = tetMesh.vertices.data();
    desc.tetrahedronVertexCount = static_cast<int>(tetMesh.vertices.size());
    desc.tetrahedronIndices = tetMesh.indices.data();
    desc.tetrahedronCount = tetMesh.tetrahedronCount;
    
    // Material properties (softer for easier tearing)
    desc.volumeStiffness = 0.4f;
    desc.shapeStiffness = 0.3f;
    desc.deformationStiffness = 0.2f;
    
    desc.totalMass = 1.0f;
    desc.useDensity = false;
    desc.enableSceneCollision = true;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody->Initialize(desc);
    
    // Enable tearing with low threshold
    softBody->SetTearable(true);
    softBody->SetTearThreshold(1.3f);  // Tear at 130% stretch
    
    // Set tear callback
    softBody->SetTearCallback([](int tetIndex, float stress) {
        std::cout << "TEAR EVENT: Tetrahedron " << tetIndex 
                 << " tore with stress " << stress << std::endl;
        // Play sound effect, create particle effect, etc.
    });
    
    // Create piece manager
    auto pieceManager = std::make_shared<SoftBodyPieceManager>();
    
    // Set piece creation callback
    softBody->SetPieceCreatedCallback([&](auto newPiece) {
        std::cout << "New piece created!" << std::endl;
        
        auto newObj = std::make_shared<GameObject>("TornPiece");
        newObj->SetSoftBody(newPiece);
        
        pieceManager->AddPiece(newPiece, newObj);
        scene.AddGameObject(newObj);
    });
    
    // Create game object
    auto softBodyObj = std::make_shared<GameObject>("TearableCube");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 5, 0));
    
    scene.AddGameObject(softBodyObj);
    
    // Fix one corner to create tension
    softBody->FixVertex(0, Vec3(-0.5f, 5.5f, 0.5f));
    
    // Apply strong force to opposite corner to cause tearing
    softBody->AddForceAtVertex(6, Vec3(500, 0, 0));
    
    std::cout << "Created tearable soft body with threshold " 
              << softBody->GetTearThreshold() << std::endl;
    
    
    // ===== Example 2: Stress Visualization =====
    
    // Create another soft body for stress testing
    auto tetMesh2 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.15f
    );
    
    SoftBodyDesc desc2;
    desc2.vertexPositions = cubeVertices.data();
    desc2.vertexCount = static_cast<int>(cubeVertices.size());
    desc2.tetrahedronVertices = tetMesh2.vertices.data();
    desc2.tetrahedronVertexCount = static_cast<int>(tetMesh2.vertices.size());
    desc2.tetrahedronIndices = tetMesh2.indices.data();
    desc2.tetrahedronCount = tetMesh2.tetrahedronCount;
    
    desc2.volumeStiffness = 0.5f;
    desc2.shapeStiffness = 0.4f;
    desc2.deformationStiffness = 0.3f;
    desc2.totalMass = 0.8f;
    desc2.useDensity = false;
    desc2.enableSceneCollision = true;
    desc2.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody2 = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody2->Initialize(desc2);
    softBody2->SetTearable(true);
    softBody2->SetTearThreshold(1.5f);  // Higher threshold
    
    // Track tear count
    int tearCount = 0;
    softBody2->SetTearCallback([&tearCount](int tetIndex, float stress) {
        tearCount++;
        std::cout << "Tear #" << tearCount << " - Tet " << tetIndex 
                 << " (stress: " << stress << ")" << std::endl;
    });
    
    auto softBodyObj2 = std::make_shared<GameObject>("StressTest");
    softBodyObj2->SetSoftBody(softBody2);
    softBodyObj2->GetTransform().SetPosition(Vec3(3, 4, 0));
    
    scene.AddGameObject(softBodyObj2);
    
    // Apply continuous stretching force
    softBody2->FixVertex(0, Vec3(2.5f, 4.5f, 0.5f));
    softBody2->FixVertex(4, Vec3(2.5f, 4.5f, -0.5f));
    softBody2->AddForceAtVertex(2, Vec3(200, 100, 0));
    
    
    // ===== Example 3: Multiple Tears =====
    
    std::cout << "\n=== Setup Complete ===" << std::endl;
    std::cout << "Soft body 1: Low tear threshold (1.3x)" << std::endl;
    std::cout << "Soft body 2: Medium tear threshold (1.5x)" << std::endl;
    std::cout << "Watch for tear events in console!" << std::endl;
}

// Usage in update loop:
/*
void UpdateTearingDemo(float deltaTime, SoftBodyPieceManager* pieceManager) {
    // Update piece manager
    pieceManager->Update(deltaTime);
    
    // Cleanup small pieces (< 10% of original mass)
    static float cleanupTimer = 0.0f;
    cleanupTimer += deltaTime;
    
    if (cleanupTimer > 5.0f) {
        int removed = pieceManager->CleanupSmallPieces(0.1f);
        if (removed > 0) {
            std::cout << "Cleaned up " << removed << " small pieces" << std::endl;
        }
        cleanupTimer = 0.0f;
    }
}
*/
