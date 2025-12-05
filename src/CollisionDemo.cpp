// Particle Collision System - Visual Demonstration
// This file demonstrates all collision features with visual examples

#include "Application.h"
#include "ParticleEmitter.h"
#include "CollisionShape.h"
#include <memory>

class CollisionDemo {
public:
    CollisionDemo(ParticleSystem* particleSystem) 
        : m_ParticleSystem(particleSystem)
        , m_CurrentDemo(0)
        , m_DemoTimer(0.0f)
    {}

    void Init() {
        // Start with demo 1
        LoadDemo1_GroundBounce();
    }

    void Update(float deltaTime) {
        m_DemoTimer += deltaTime;
        
        // Auto-switch demos every 10 seconds
        if (m_DemoTimer >= 10.0f) {
            NextDemo();
            m_DemoTimer = 0.0f;
        }
    }

    void NextDemo() {
        m_CurrentDemo = (m_CurrentDemo + 1) % 5;
        m_ParticleSystem->ClearEmitters();
        
        switch (m_CurrentDemo) {
            case 0: LoadDemo1_GroundBounce(); break;
            case 1: LoadDemo2_SphereObstacle(); break;
            case 2: LoadDemo3_BoxContainer(); break;
            case 3: LoadDemo4_ParticleCollisions(); break;
            case 4: LoadDemo5_ComplexScene(); break;
        }
    }

    std::string GetCurrentDemoName() const {
        switch (m_CurrentDemo) {
            case 0: return "Demo 1: Ground Bounce";
            case 1: return "Demo 2: Sphere Obstacle";
            case 2: return "Demo 3: Box Container";
            case 3: return "Demo 4: Particle-to-Particle";
            case 4: return "Demo 5: Complex Scene";
            default: return "Unknown";
        }
    }

private:
    // Demo 1: Basic ground bounce
    void LoadDemo1_GroundBounce() {
        auto sparks = ParticleEmitter::CreateSparks(Vec3(0, 8, 0));
        sparks->SetSpawnRate(100.0f);
        sparks->SetGravity(Vec3(0, -9.8f, 0));
        sparks->SetParticleRestitution(0.7f);  // Bouncy
        sparks->SetParticleFriction(0.2f);
        
        // Ground plane at y=0
        auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
        sparks->AddCollisionShape(ground);
        
        m_ParticleSystem->AddEmitter(sparks);
    }

    // Demo 2: Sphere obstacle
    void LoadDemo2_SphereObstacle() {
        auto fire = ParticleEmitter::CreateFire(Vec3(-5, 8, 0));
        fire->SetVelocityRange(Vec3(1, -2, -1), Vec3(3, 2, 1));
        fire->SetGravity(Vec3(0, -3.0f, 0));
        fire->SetParticleRestitution(0.5f);
        
        // Ground
        auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
        fire->AddCollisionShape(ground);
        
        // Sphere obstacle in the middle
        auto sphere = std::make_shared<CollisionSphere>(Vec3(0, 3, 0), 2.0f);
        fire->AddCollisionShape(sphere);
        
        m_ParticleSystem->AddEmitter(fire);
    }

    // Demo 3: Box container
    void LoadDemo3_BoxContainer() {
        auto magic = ParticleEmitter::CreateMagic(Vec3(0, 5, 0));
        magic->SetGravity(Vec3(0, -5.0f, 0));
        magic->SetParticleRestitution(0.8f);
        magic->SetParticleFriction(0.1f);
        
        // Create a box container
        auto box = std::make_shared<CollisionBox>(
            Vec3(-4, 0, -4),   // Min corner
            Vec3(4, 8, 4)      // Max corner
        );
        magic->AddCollisionShape(box);
        
        m_ParticleSystem->AddEmitter(magic);
    }

    // Demo 4: Particle-to-particle collisions
    void LoadDemo4_ParticleCollisions() {
        auto emitter = std::make_shared<ParticleEmitter>(Vec3(0, 10, 0), 250);
        emitter->SetSpawnRate(40.0f);
        emitter->SetParticleLifetime(8.0f);
        emitter->SetVelocityRange(Vec3(-2, -1, -2), Vec3(2, 1, 2));
        emitter->SetColorRange(
            Vec4(0.2f, 0.8f, 1.0f, 1.0f),  // Cyan
            Vec4(0.8f, 0.2f, 1.0f, 0.0f)   // Purple fade
        );
        emitter->SetSizeRange(0.3f, 0.6f);
        emitter->SetGravity(Vec3(0, -8.0f, 0));
        emitter->SetBlendMode(BlendMode::Additive);
        
        // Particle physics
        emitter->SetParticleMass(1.0f);
        emitter->SetParticleRestitution(0.7f);
        emitter->SetParticleFriction(0.15f);
        
        // Enable particle-to-particle collisions
        emitter->SetEnableParticleCollisions(true);
        emitter->SetParticleCollisionRadius(1.0f);
        
        // Ground to collect particles
        auto ground = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
        emitter->AddCollisionShape(ground);
        
        m_ParticleSystem->AddEmitter(emitter);
    }

    // Demo 5: Complex scene with multiple collision types
    void LoadDemo5_ComplexScene() {
        auto smoke = ParticleEmitter::CreateSmoke(Vec3(0, 1, 0));
        smoke->SetGravity(Vec3(0, -2.0f, 0));  // Downward gravity for demo
        smoke->SetParticleRestitution(0.4f);
        smoke->SetParticleFriction(0.3f);
        
        // Floor
        auto floor = std::make_shared<CollisionPlane>(Vec3(0, 1, 0), 0.0f);
        smoke->AddCollisionShape(floor);
        
        // Ceiling
        auto ceiling = std::make_shared<CollisionPlane>(Vec3(0, -1, 0), -10.0f);
        smoke->AddCollisionShape(ceiling);
        
        // Side walls
        auto wallLeft = std::make_shared<CollisionPlane>(Vec3(1, 0, 0), 6.0f);
        auto wallRight = std::make_shared<CollisionPlane>(Vec3(-1, 0, 0), 6.0f);
        smoke->AddCollisionShape(wallLeft);
        smoke->AddCollisionShape(wallRight);
        
        // Sphere obstacles
        auto sphere1 = std::make_shared<CollisionSphere>(Vec3(-3, 3, 0), 1.5f);
        auto sphere2 = std::make_shared<CollisionSphere>(Vec3(3, 5, 0), 1.5f);
        smoke->AddCollisionShape(sphere1);
        smoke->AddCollisionShape(sphere2);
        
        // Box obstacle
        auto box = std::make_shared<CollisionBox>(
            Vec3(-1, 6, -1),
            Vec3(1, 8, 1)
        );
        smoke->AddCollisionShape(box);
        
        m_ParticleSystem->AddEmitter(smoke);
    }

    ParticleSystem* m_ParticleSystem;
    int m_CurrentDemo;
    float m_DemoTimer;
};

// Integration example for Application.cpp:
/*
In Application.h, add:
    std::unique_ptr<CollisionDemo> m_CollisionDemo;

In Application::Init(), add:
    m_CollisionDemo = std::make_unique<CollisionDemo>(m_Renderer->GetParticleSystem());
    m_CollisionDemo->Init();

In Application::Update(), add:
    if (m_CollisionDemo) {
        m_CollisionDemo->Update(deltaTime);
    }

In Application::RenderEditorUI(), add demo info:
    ImGui::Begin("Collision Demo");
    ImGui::Text("Current Demo: %s", m_CollisionDemo->GetCurrentDemoName().c_str());
    if (ImGui::Button("Next Demo")) {
        m_CollisionDemo->NextDemo();
    }
    ImGui::End();
*/
