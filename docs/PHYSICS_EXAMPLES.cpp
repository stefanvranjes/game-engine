// Example: Using the Physics System
// This example shows how to create physics bodies and interact with them

#include "Application.h"
#include "GameObject.h"
#include "RigidBody.h"
#include "KinematicController.h"
#include "PhysicsCollisionShape.h"
#include "PhysicsSystem.h"
#include <iostream>

/**
 * Example 1: Creating a Simple Dynamic Box
 */
void Example_CreateDynamicBox(std::shared_ptr<Renderer> renderer) {
    // Create a game object
    auto box = std::make_shared<GameObject>("DynamicBox");
    box->GetTransform().SetPosition(Vec3(0, 5, 0));
    box->GetTransform().SetScale(Vec3(1, 1, 1));

    // Create a box collision shape (1x1x1 unit box)
    auto boxShape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));

    // Create a rigid body component
    auto rigidBody = std::make_shared<RigidBody>();
    rigidBody->Initialize(BodyType::Dynamic, 1.0f, boxShape);
    
    // Set physics properties
    rigidBody->SetFriction(0.5f);      // Medium friction
    rigidBody->SetRestitution(0.3f);   // Low bounce
    rigidBody->SetLinearDamping(0.01f);

    // Attach the rigid body to the game object
    box->SetRigidBody(rigidBody);

    // Add to scene
    renderer->AddGameObject(box);
    
    std::cout << "Created dynamic box at position (0, 5, 0)" << std::endl;
}

/**
 * Example 2: Creating a Static Floor
 */
void Example_CreateStaticFloor(std::shared_ptr<Renderer> renderer) {
    auto floor = std::make_shared<GameObject>("Floor");
    floor->GetTransform().SetPosition(Vec3(0, 0, 0));
    floor->GetTransform().SetScale(Vec3(20, 1, 20));

    // Large static box
    auto floorShape = PhysicsCollisionShape::CreateBox(Vec3(10, 0.5f, 10));

    auto floorBody = std::make_shared<RigidBody>();
    floorBody->Initialize(BodyType::Static, 0.0f, floorShape); // Static bodies have 0 mass
    floorBody->SetFriction(0.8f);

    floor->SetRigidBody(floorBody);
    renderer->AddGameObject(floor);
    
    std::cout << "Created static floor" << std::endl;
}

/**
 * Example 3: Creating a Player Character with Kinematic Controller
 */
void Example_CreatePlayerCharacter(std::shared_ptr<Renderer> renderer) {
    auto player = std::make_shared<GameObject>("Player");
    player->GetTransform().SetPosition(Vec3(0, 2, 0));

    // Capsule shape is ideal for humanoid characters
    auto capsuleShape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);

    auto controller = std::make_shared<KinematicController>();
    controller->Initialize(capsuleShape, 80.0f, 0.35f); // 80kg, 0.35m step height
    controller->SetMaxWalkSpeed(10.0f);  // 10 m/s max walk speed
    controller->SetFallSpeed(55.0f);     // Terminal velocity

    player->SetKinematicController(controller);
    renderer->AddGameObject(player);
    
    std::cout << "Created player character with kinematic controller" << std::endl;
}

/**
 * Example 4: Applying Forces and Impulses
 */
void Example_ApplyForcesAndImpulses(std::shared_ptr<RigidBody> body) {
    // Apply continuous force (like wind or engine thrust)
    body->ApplyForce(Vec3(100, 0, 0)); // 100N force in X direction

    // Apply instantaneous impulse (like explosion or jump)
    body->ApplyImpulse(Vec3(0, 500, 0)); // Upward impulse

    // Apply force at a specific point (creates torque)
    Vec3 contactPoint = Vec3(0, 1, 0);
    body->ApplyForceAtPoint(Vec3(50, 0, 0), contactPoint);

    // Set velocities directly (teleport)
    body->SetLinearVelocity(Vec3(5, 0, 0));
    body->SetAngularVelocity(Vec3(0, 3.14f, 0)); // Rotating

    std::cout << "Applied forces and impulses to body" << std::endl;
}

/**
 * Example 5: Controlling the Player Character
 */
void Example_UpdatePlayerInput(float deltaTime) {
    // This would be called in your input handling code
    static int moveX = 0, moveZ = 0;
    bool jumpPressed = false;

    // Get player from your scene (simplified)
    // auto player = renderer->GetGameObject("Player");
    // if (!player) return;

    // auto controller = player->GetKinematicController();
    // if (!controller) return;

    // Read input (pseudo-code, integrate with actual input system)
    // moveX = (GetKeyDown(KEY_D) ? 1 : 0) - (GetKeyDown(KEY_A) ? 1 : 0);
    // moveZ = (GetKeyDown(KEY_W) ? 1 : 0) - (GetKeyDown(KEY_S) ? 1 : 0);
    // jumpPressed = GetKeyDown(KEY_SPACE);

    // Calculate movement direction
    float moveSpeed = 5.0f; // m/s
    Vec3 moveDir = Vec3(moveX, 0, moveZ) * moveSpeed;

    // Update controller
    // controller->SetWalkDirection(moveDir);

    // Handle jumping (only if grounded)
    // if (jumpPressed && controller->IsGrounded()) {
    //     controller->Jump(Vec3(0, 10.0f, 0)); // Jump force
    // }

    std::cout << "Updated player input" << std::endl;
}

/**
 * Example 6: Creating Compound Shapes
 */
void Example_CreateCompoundShape(std::shared_ptr<Renderer> renderer) {
    // Create a composite object (like a T-shaped structure)
    auto compound = std::make_shared<GameObject>("CompoundObject");
    compound->GetTransform().SetPosition(Vec3(0, 3, 0));

    // Create compound shape
    auto compoundShape = PhysicsCollisionShape::CreateCompound();

    // Add a vertical box (stem of T)
    auto verticalBox = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 1.0f, 0.5f));
    compoundShape.AddChildShape(verticalBox, Vec3(0, 0.5f, 0));

    // Add a horizontal box (top of T)
    auto horizontalBox = PhysicsCollisionShape::CreateBox(Vec3(2.0f, 0.5f, 0.5f));
    compoundShape.AddChildShape(horizontalBox, Vec3(0, 2.0f, 0));

    // Create rigid body with compound shape
    auto compoundBody = std::make_shared<RigidBody>();
    compoundBody->Initialize(BodyType::Dynamic, 5.0f, compoundShape);

    compound->SetRigidBody(compoundBody);
    renderer->AddGameObject(compound);
    
    std::cout << "Created compound shape object" << std::endl;
}

/**
 * Example 7: Raycasting for Hit Detection
 */
void Example_RaycastDemo(Vec3 rayStart, Vec3 rayEnd) {
    RaycastHit hit;
    
    if (PhysicsSystem::Get().Raycast(rayStart, rayEnd, hit)) {
        std::cout << "Ray hit at: (" << hit.point.x << ", " 
                  << hit.point.y << ", " << hit.point.z << ")" << std::endl;
        std::cout << "Distance: " << hit.distance << "m" << std::endl;
        std::cout << "Surface normal: (" << hit.normal.x << ", " 
                  << hit.normal.y << ", " << hit.normal.z << ")" << std::endl;
        
        // If hit a rigid body, you could apply force
        if (hit.body) {
            std::cout << "Hit a rigid body!" << std::endl;
        }
    } else {
        std::cout << "No hit detected" << std::endl;
    }
}

/**
 * Example 8: Querying Physics System State
 */
void Example_QueryPhysicsState() {
    auto& physics = PhysicsSystem::Get();

    // Get number of bodies in simulation
    int numBodies = physics.GetNumRigidBodies();
    std::cout << "Total bodies in world: " << numBodies << std::endl;

    // Get current gravity
    Vec3 gravity = physics.GetGravity();
    std::cout << "Current gravity: (" << gravity.x << ", " 
              << gravity.y << ", " << gravity.z << ")" << std::endl;

    // Change gravity (for special effects)
    physics.SetGravity(Vec3(0, -20.0f, 0)); // Stronger gravity

    // Query all rigid bodies
    const auto& rigidBodies = physics.GetRigidBodies();
    for (auto body : rigidBodies) {
        std::cout << "Body mass: " << body->GetMass() << "kg" << std::endl;
    }
}

/**
 * Example 9: Kinematic Platform
 */
void Example_CreateMovingPlatform(std::shared_ptr<Renderer> renderer) {
    auto platform = std::make_shared<GameObject>("MovingPlatform");
    platform->GetTransform().SetPosition(Vec3(5, 2, 0));

    auto platformShape = PhysicsCollisionShape::CreateBox(Vec3(3, 0.5f, 1));

    auto platformBody = std::make_shared<RigidBody>();
    platformBody->Initialize(BodyType::Kinematic, 10.0f, platformShape);
    
    platform->SetRigidBody(platformBody);
    renderer->AddGameObject(platform);

    // In your update loop, move the platform
    // float t = glfwGetTime();
    // Vec3 newPos = Vec3(5 + sin(t) * 3, 2, 0);
    // platformBody->SyncTransformToPhysics(newPos, Quat());
}

/**
 * Example 10: Multiple Falling Objects
 */
void Example_CreateFallingObjects(std::shared_ptr<Renderer> renderer) {
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            auto box = std::make_shared<GameObject>("Box_" + std::to_string(i * 10 + j));
            float x = (i - 5) * 1.5f;
            float z = (j - 5) * 1.5f;
            float y = 10.0f + (i + j) * 0.5f;
            
            box->GetTransform().SetPosition(Vec3(x, y, z));
            
            auto shape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));
            auto body = std::make_shared<RigidBody>();
            body->Initialize(BodyType::Dynamic, 1.0f, shape);
            body->SetRestitution(0.5f);
            
            box->SetRigidBody(body);
            renderer->AddGameObject(box);
        }
    }
    
    std::cout << "Created 100 falling boxes" << std::endl;
}

/**
 * Main initialization function - Call this from your application startup
 */
void InitializePhysicsExamples(std::shared_ptr<Renderer> renderer) {
    std::cout << "\n=== Physics System Examples ===" << std::endl;

    // Create the basic scene
    Example_CreateStaticFloor(renderer);
    Example_CreateDynamicBox(renderer);
    Example_CreatePlayerCharacter(renderer);
    Example_CreateCompoundShape(renderer);
    Example_CreateMovingPlatform(renderer);

    // Uncomment to test with many objects
    // Example_CreateFallingObjects(renderer);

    std::cout << "Physics examples initialized!\n" << std::endl;
}

/**
 * Example usage in your game loop or input handler
 */
void UpdatePhysicsExamples(float deltaTime, std::shared_ptr<Renderer> renderer) {
    // Update player character (this would be integrated with your actual input)
    Example_UpdatePlayerInput(deltaTime);

    // Perform raycasts for hit detection
    // Example_RaycastDemo(rayStart, rayEnd);

    // Query system state occasionally
    static float queryTimer = 0.0f;
    queryTimer += deltaTime;
    if (queryTimer >= 1.0f) {
        // Example_QueryPhysicsState();
        queryTimer = 0.0f;
    }
}
