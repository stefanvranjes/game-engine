// Example: Partial Tearing (Cracks) Demo
// Demonstrates cracks that progressively damage without immediate separation

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "PhysXRigidBody.h"
#include "TetrahedralMeshGenerator.h"
#include <iostream>

void CreatePartialTearingDemo(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Partial Tearing (Cracks) Demo ===" << std::endl;
    
    // Create cube mesh
    std::vector<Vec3> cubeVertices = {
        Vec3(-1.0f, -1.0f,  1.0f), Vec3( 1.0f, -1.0f,  1.0f),
        Vec3( 1.0f,  1.0f,  1.0f), Vec3(-1.0f,  1.0f,  1.0f),
        Vec3(-1.0f, -1.0f, -1.0f), Vec3( 1.0f, -1.0f, -1.0f),
        Vec3( 1.0f,  1.0f, -1.0f), Vec3(-1.0f,  1.0f, -1.0f)
    };
    
    std::vector<int> cubeIndices = {
        0, 1, 2, 0, 2, 3,  5, 4, 7, 5, 7, 6,
        4, 0, 3, 4, 3, 7,  1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,  4, 5, 1, 4, 1, 0
    };
    
    // Generate tetrahedral mesh
    auto tetMesh = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.3f
    );
    
    // Create soft body
    SoftBodyDesc desc;
    desc.vertexPositions = cubeVertices.data();
    desc.vertexCount = static_cast<int>(cubeVertices.size());
    desc.triangleIndices = cubeIndices.data();
    desc.triangleCount = static_cast<int>(cubeIndices.size() / 3);
    desc.tetrahedronVertices = tetMesh.vertices.data();
    desc.tetrahedronVertexCount = static_cast<int>(tetMesh.vertices.size());
    desc.tetrahedronIndices = tetMesh.indices.data();
    desc.tetrahedronCount = tetMesh.tetrahedronCount;
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
    
    // ===== ENABLE PARTIAL TEARING =====
    softBody->SetPartialTearingEnabled(true);
    softBody->SetCrackThreshold(1.2f);      // Cracks form at 120% stretch
    softBody->SetTearThreshold(1.5f);       // Full tears at 150% stretch
    softBody->SetCrackProgressionRate(0.15f); // Damage increases 15% per second under stress
    
    // Optional: Enable crack healing
    softBody->SetCrackHealingEnabled(true);
    softBody->SetCrackHealingRate(0.05f);   // Heals 5% per second when not stressed
    
    // ===== ENABLE CRACK VISUALIZATION =====
    softBody->SetCrackVisualizationEnabled(true);
    
    // Configure crack appearance
    CrackRenderer::RenderSettings crackSettings;
    crackSettings.useDamageColor = true;
    crackSettings.lowDamageColor = Vec3(0.5f, 0.5f, 0.0f);  // Yellow at low damage
    crackSettings.highDamageColor = Vec3(1.0f, 0.0f, 0.0f); // Red at high damage
    crackSettings.baseWidth = 0.02f;
    crackSettings.maxWidth = 0.05f;
    crackSettings.useGlow = true;
    crackSettings.glowIntensity = 0.3f;
    crackSettings.minOpacity = 0.3f;
    crackSettings.maxOpacity = 1.0f;
    
    // Enable animations
    crackSettings.enablePulsing = true;
    crackSettings.pulseSpeed = 2.0f;        // 2 Hz - smooth breathing
    crackSettings.pulseAmplitude = 0.3f;    // 30% intensity variation
    
    crackSettings.enableFlickering = true;
    crackSettings.flickerSpeed = 10.0f;     // 10 Hz - rapid variations
    crackSettings.flickerAmplitude = 0.2f;  // 20% intensity variation
    
    crackSettings.enableGrowth = true;
    crackSettings.growthDuration = 0.5f;    // 0.5 second fade-in
    
    // Enable damage-based pulse speed
    crackSettings.damageAffectsSpeed = true;
    crackSettings.minPulseSpeed = 1.0f;     // Slow pulse at low damage
    crackSettings.maxPulseSpeed = 4.0f;     // Fast pulse at high damage
    
    // Enable color pulsing
    crackSettings.enableColorPulsing = true;
    crackSettings.pulseColorMin = Vec3(0.5f, 0.0f, 0.0f);   // Dark red at pulse min
    crackSettings.pulseColorMax = Vec3(1.0f, 0.4f, 0.0f);   // Bright orange at pulse max
    
    // Enable crack propagation animation
    crackSettings.enablePropagation = true;
    crackSettings.propagationDuration = 0.3f;  // 0.3 second growth
    crackSettings.propagationEasing = true;    // Smooth easing
    
    // Enable sound synchronization
    crackSettings.enableSoundSync = true;
    crackSettings.soundSyncThreshold = 0.9f;   // Trigger at 90% of pulse peak
    
    // Enable particle effects
    crackSettings.enableParticles = true;
    crackSettings.particleSpawnRate = 5.0f;    // 5 particles/second continuous
    crackSettings.particlesOnPulse = true;     // Burst on pulse peaks
    crackSettings.particleBurstCount = 3;      // 3 particles per burst
    
    softBody->SetCrackRenderSettings(crackSettings);
    
    // Set sound callback
    softBody->SetCrackSoundCallback([](int crackIndex, float damage, float intensity) {
        // This would trigger audio playback in a real implementation
        // Volume and pitch could be based on damage and intensity
        std::cout << "🔊 Sound: Crack " << crackIndex 
                  << " (damage: " << (damage * 100.0f) << "%, intensity: " << intensity << ")" 
                  << std::endl;
        
        // Example audio integration:
        // float volume = 0.3f + damage * 0.7f;  // 30% to 100% volume
        // float pitch = 0.8f + damage * 0.4f;   // Lower to higher pitch
        // AudioSystem::PlaySound("crack_pulse.wav", volume, pitch);
    });
    
    // Set particle callback
    softBody->SetCrackParticleCallback([](Vec3 position, Vec3 normal, Vec3 velocity, float damage, int count) {
        // This would spawn particles in a real implementation
        std::cout << "✨ Particles: " << count << " at (" 
                  << position.x << ", " << position.y << ", " << position.z << ") "
                  << "damage: " << (damage * 100.0f) << "%" << std::endl;
        
        // Example particle integration:
        // for (int i = 0; i < count; ++i) {
        //     Vec3 randomOffset = RandomInCone(normal, 30.0f);  // 30 degree cone
        //     Vec3 particleVel = velocity + randomOffset;
        //     Vec3 color = Vec3(1.0f, 0.5f * (1.0f - damage), 0.0f);  // Orange to red
        //     float lifetime = 0.5f + damage * 0.5f;  // 0.5-1.0 seconds
        //     ParticleSystem::Spawn(position, particleVel, color, lifetime);
        // }
    });
    
    std::cout << "Partial tearing enabled:" << std::endl;
    std::cout << "  Crack threshold: " << softBody->GetCrackThreshold() << std::endl;
    std::cout << "  Tear threshold: " << softBody->GetTearThreshold() << std::endl;
    std::cout << "  Progression rate: " << softBody->GetCrackProgressionRate() << "/s" << std::endl;
    
    // Track crack events
    softBody->SetTearCallback([](int tetIndex, float stress) {
        std::cout << "FULL TEAR: Tet " << tetIndex << " (stress: " << stress << ")" << std::endl;
    });
    
    // Create game object
    auto softBodyObj = std::make_shared<GameObject>("PartialTearCube");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 8, 0));
    scene.AddGameObject(softBodyObj);
    
    // Setup: Fixed corner with moderate pulling force
    softBody->FixVertex(0, Vec3(-1.0f, 8.0f, 1.0f));
    softBody->AddForceAtVertex(6, Vec3(400, 0, 0));  // Moderate force
    
    // Create ground
    auto groundObj = std::make_shared<GameObject>("Ground");
    auto groundBody = std::make_shared<PhysXRigidBody>(physxBackend);
    groundBody->CreatePlane(Vec3(0, 1, 0), 0.0f);
    groundBody->SetStatic(true);
    groundObj->SetRigidBody(groundBody);
    scene.AddGameObject(groundObj);
    
    std::cout << "\n=== Demo Ready ===" << std::endl;
    std::cout << "Expected behavior:" << std::endl;
    std::cout << "1. Cracks will form at stressed regions (120% stretch)" << std::endl;
    std::cout << "2. Cracks visually grow along edge over 0.3 seconds" << std::endl;
    std::cout << "3. Cracks progressively damage over time" << std::endl;
    std::cout << "4. Pulse speed increases with damage (1 Hz → 4 Hz)" << std::endl;
    std::cout << "5. Color pulses between dark red and bright orange" << std::endl;
    std::cout << "6. Sound triggers in sync with pulse peaks" << std::endl;
    std::cout << "7. Particles spawn continuously and burst on pulses" << std::endl;
    std::cout << "8. At 100% damage, cracks convert to full tears" << std::endl;
    std::cout << "9. Cracks may heal if stress is removed" << std::endl;
}

// Usage in update loop:
/*
void UpdatePartialTearingDemo(PhysXSoftBody* softBody) {
    // Get crack information for visualization
    const auto& cracks = softBody->GetCracks();
    int crackCount = softBody->GetCrackCount();
    
    if (crackCount > 0) {
        std::cout << "Active cracks: " << crackCount << std::endl;
        
        for (const auto& crack : cracks) {
            // Visualize crack (e.g., draw dark line at crack position)
            // Color intensity based on damage level
            float alpha = crack.damage;  // 0.0 = faint, 1.0 = fully visible
            
            // Render crack at crack.crackPosition with crack.crackNormal
            // Use crack.stiffnessMultiplier to show weakening
        }
    }
}
*/
