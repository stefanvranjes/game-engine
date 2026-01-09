// Example: Creating a soft body cube with PhysX
// This demonstrates how to use the soft body physics system

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "PhysXRigidBody.h"
#include "TetrahedralMeshGenerator.h"
#include "Mesh.h"

void CreateSoftBodyExample(PhysXBackend* physxBackend, Scene& scene) {
    // ===== Example 1: Soft Body Cube =====
    
    // Create cube mesh (surface mesh)
    std::vector<Vec3> cubeVertices = {
        // Front face
        Vec3(-0.5f, -0.5f,  0.5f), Vec3( 0.5f, -0.5f,  0.5f),
        Vec3( 0.5f,  0.5f,  0.5f), Vec3(-0.5f,  0.5f,  0.5f),
        // Back face
        Vec3(-0.5f, -0.5f, -0.5f), Vec3( 0.5f, -0.5f, -0.5f),
        Vec3( 0.5f,  0.5f, -0.5f), Vec3(-0.5f,  0.5f, -0.5f)
    };
    
    std::vector<int> cubeIndices = {
        // Front
        0, 1, 2, 0, 2, 3,
        // Back
        5, 4, 7, 5, 7, 6,
        // Left
        4, 0, 3, 4, 3, 7,
        // Right
        1, 5, 6, 1, 6, 2,
        // Top
        3, 2, 6, 3, 6, 7,
        // Bottom
        4, 5, 1, 4, 1, 0
    };
    
    // Generate tetrahedral mesh
    auto tetMesh = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.15f  // Voxel resolution
    );
    
    // Create soft body descriptor
    SoftBodyDesc desc;
    desc.vertexPositions = cubeVertices.data();
    desc.vertexCount = static_cast<int>(cubeVertices.size());
    desc.triangleIndices = cubeIndices.data();
    desc.triangleCount = static_cast<int>(cubeIndices.size() / 3);
    
    desc.tetrahedronVertices = tetMesh.vertices.data();
    desc.tetrahedronVertexCount = static_cast<int>(tetMesh.vertices.size());
    desc.tetrahedronIndices = tetMesh.indices.data();
    desc.tetrahedronCount = tetMesh.tetrahedronCount;
    
    // Material properties (rubber-like)
    desc.volumeStiffness = 0.7f;
    desc.shapeStiffness = 0.5f;
    desc.deformationStiffness = 0.4f;
    desc.maxStretch = 1.5f;
    desc.maxCompress = 0.5f;
    
    // Mass
    desc.useDensity = false;
    desc.totalMass = 1.0f;
    
    // Collision
    desc.enableSceneCollision = true;
    desc.enableSelfCollision = false;
    desc.collisionMargin = 0.01f;
    
    // Simulation quality
    desc.solverIterations = 4;
    
    // Gravity
    desc.gravity = Vec3(0, -9.81f, 0);
    
    // Create soft body
    auto softBody = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody->Initialize(desc);
    
    // Create game object
    auto softBodyObj = std::make_shared<GameObject>("SoftCube");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 5, 0));
    
    // Add to scene
    scene.AddGameObject(softBodyObj);
    
    std::cout << "Created soft body cube at position (0, 5, 0)" << std::endl;
    
    
    // ===== Example 2: Soft Body with Attachment =====
    
    // Create a rigid body (anchor point)
    auto rigidObj = std::make_shared<GameObject>("RigidAnchor");
    auto rigidBody = std::make_shared<PhysXRigidBody>(physxBackend);
    
    auto boxShape = std::make_shared<PhysicsCollisionShape>();
    boxShape->CreateBox(Vec3(0.2f, 0.2f, 0.2f));
    
    rigidBody->Initialize(BodyType::Static, 0.0f, boxShape);
    rigidObj->SetRigidBody(rigidBody);
    rigidObj->GetTransform().SetPosition(Vec3(0, 7, 0));
    
    scene.AddGameObject(rigidObj);
    
    // Attach top corner of soft body to rigid body
    softBody->AttachVertexToRigidBody(3, rigidBody.get(), Vec3(0, 0, 0));
    
    std::cout << "Attached soft body corner to rigid anchor" << std::endl;
    
    
    // ===== Example 3: Jelly-like Soft Body =====
    
    auto tetMesh2 = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.2f
    );
    
    SoftBodyDesc jellyDesc;
    jellyDesc.vertexPositions = cubeVertices.data();
    jellyDesc.vertexCount = static_cast<int>(cubeVertices.size());
    jellyDesc.triangleIndices = cubeIndices.data();
    jellyDesc.triangleCount = static_cast<int>(cubeIndices.size() / 3);
    jellyDesc.tetrahedronVertices = tetMesh2.vertices.data();
    jellyDesc.tetrahedronVertexCount = static_cast<int>(tetMesh2.vertices.size());
    jellyDesc.tetrahedronIndices = tetMesh2.indices.data();
    jellyDesc.tetrahedronCount = tetMesh2.tetrahedronCount;
    
    // Jelly material properties (very soft)
    jellyDesc.volumeStiffness = 0.3f;
    jellyDesc.shapeStiffness = 0.2f;
    jellyDesc.deformationStiffness = 0.1f;
    jellyDesc.linearDamping = 0.05f;
    jellyDesc.totalMass = 0.5f;
    jellyDesc.useDensity = false;
    jellyDesc.enableSceneCollision = true;
    jellyDesc.gravity = Vec3(0, -9.81f, 0);
    
    auto jellySoftBody = std::make_shared<PhysXSoftBody>(physxBackend);
    jellySoftBody->Initialize(jellyDesc);
    
    auto jellyObj = std::make_shared<GameObject>("JellyCube");
    jellyObj->SetSoftBody(jellySoftBody);
    jellyObj->GetTransform().SetPosition(Vec3(2, 3, 0));
    
    scene.AddGameObject(jellyObj);
    
    std::cout << "Created jelly-like soft body at position (2, 3, 0)" << std::endl;
    
    
    // ===== Example 4: Applying Forces =====
    
    // Apply a one-time impulse to the jelly
    jellySoftBody->AddImpulse(Vec3(5, 0, 0));
    
    // Apply continuous force to specific vertex
    // (This would typically be done in an update loop)
    // jellySoftBody->AddForceAtVertex(0, Vec3(0, 10, 0));
    
    std::cout << "Applied impulse to jelly soft body" << std::endl;
}

// Usage in main.cpp:
/*
int main() {
    Application app(1280, 720, "Soft Body Physics Demo");
    
    // Initialize PhysX backend
    PhysXBackend physxBackend;
    physxBackend.Initialize(Vec3(0, -9.81f, 0));
    
    // Create soft body examples
    CreateSoftBodyExample(&physxBackend, app.GetScene());
    
    // Main loop
    while (app.IsRunning()) {
        float deltaTime = 1.0f / 60.0f;
        
        // Update physics
        physxBackend.Update(deltaTime);
        
        // Update and render
        app.Update(deltaTime);
        app.Render();
    }
    
    return 0;
}
*/
