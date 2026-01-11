// Example: Procedural Soft Body Tearing Demo
// Demonstrates automatic stress-based tearing with mesh splitting

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "PhysXRigidBody.h"
#include "TetrahedralMeshGenerator.h"
#include "SoftBodyPieceManager.h"
#include "ProceduralTearGenerator.h"
#include "TearPropagationSystem.h"
#include <iostream>

void CreateProceduralTearingDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Procedural Soft Body Tearing Demo ===" << std::endl;
    
    // Create cube mesh for soft body
    std::vector<Vec3> cubeVertices = {
        Vec3(-1.0f, -1.0f,  1.0f), Vec3( 1.0f, -1.0f,  1.0f),
        Vec3( 1.0f,  1.0f,  1.0f), Vec3(-1.0f,  1.0f,  1.0f),
        Vec3(-1.0f, -1.0f, -1.0f), Vec3( 1.0f, -1.0f, -1.0f),
        Vec3( 1.0f,  1.0f, -1.0f), Vec3(-1.0f,  1.0f, -1.0f)
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
        0.3f  // Medium resolution for good tearing
    );
    
    std::cout << "Generated tetrahedral mesh: " << tetMesh.vertices.size() 
             << " vertices, " << tetMesh.tetrahedronCount << " tetrahedra" << std::endl;
    
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
    
    // Material properties (moderately soft for tearing)
    desc.volumeStiffness = 0.5f;
    desc.shapeStiffness = 0.4f;
    desc.deformationStiffness = 0.3f;
    
    desc.totalMass = 2.0f;
    desc.useDensity = false;
    desc.enableSceneCollision = true;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody = std::make_shared<PhysXSoftBody>(physxBackend);
    if (!softBody->Initialize(desc)) {
        std::cerr << "Failed to initialize soft body" << std::endl;
        return;
    }
    
    // Enable procedural tearing
    softBody->SetTearable(true);
    softBody->SetTearThreshold(1.4f);  // Tear at 140% stretch
    
    std::cout << "Soft body created with tearing enabled (threshold: " 
             << softBody->GetTearThreshold() << ")" << std::endl;
    
    // Create piece manager for tracking torn pieces
    auto pieceManager = std::make_shared<SoftBodyPieceManager>();
    
    // Track tear events
    int tearCount = 0;
    softBody->SetTearCallback([&tearCount](int tetIndex, float stress) {
        tearCount++;
        std::cout << "TEAR #" << tearCount << ": Tetrahedron " << tetIndex 
                 << " tore with stress " << stress << std::endl;
    });
    
    // Set piece creation callback
    softBody->SetPieceCreatedCallback([&](auto newPiece) {
        std::cout << "New piece created from tear!" << std::endl;
        
        auto newObj = std::make_shared<GameObject>("TornPiece");
        newObj->SetSoftBody(newPiece);
        
        pieceManager->AddPiece(newPiece, newObj);
        scene.AddGameObject(newObj);
        
        std::cout << "Piece manager now tracking " << pieceManager->GetPieceCount() << " pieces" << std::endl;
    });
    
    // Create game object
    auto softBodyObj = std::make_shared<GameObject>("ProceduralTearCube");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 8, 0));
    
    scene.AddGameObject(softBodyObj);
    
    // ===== Setup 1: Fixed Corner with Pulling Force =====
    // Fix one corner
    softBody->FixVertex(0, Vec3(-1.0f, 8.0f, 1.0f));
    
    // Apply strong pulling force to opposite corner
    softBody->AddForceAtVertex(6, Vec3(800, 0, 0));
    
    std::cout << "\nSetup: Fixed corner with pulling force" << std::endl;
    std::cout << "Expected: Tear should propagate from stressed region" << std::endl;
    
    // ===== Create Ground Plane =====
    auto groundObj = std::make_shared<GameObject>("Ground");
    auto groundBody = std::make_shared<PhysXRigidBody>(physxBackend);
    groundBody->CreatePlane(Vec3(0, 1, 0), 0.0f);
    groundBody->SetStatic(true);
    groundObj->SetRigidBody(groundBody);
    scene.AddGameObject(groundObj);
    
    std::cout << "\n=== Demo Ready ===" << std::endl;
    std::cout << "Watch for automatic tearing as stress builds up!" << std::endl;
    std::cout << "Torn pieces will be tracked by the piece manager." << std::endl;
}

// Usage in update loop:
/*
void UpdateProceduralTearingDemo(float deltaTime, SoftBodyPieceManager* pieceManager) {
    // Update piece manager
    pieceManager->Update(deltaTime);
    
    // Cleanup very small pieces (< 3% of original mass)
    static float cleanupTimer = 0.0f;
    cleanupTimer += deltaTime;
    
    if (cleanupTimer > 3.0f) {
        int removed = pieceManager->CleanupSmallPieces(0.03f);
        if (removed > 0) {
            std::cout << "Cleaned up " << removed << " small pieces" << std::endl;
        }
        cleanupTimer = 0.0f;
    }
}
*/
