// Example: Deformation Recording and Playback
// This demonstrates how to record and play back soft body deformations

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "PhysXRigidBody.h"
#include "TetrahedralMeshGenerator.h"
#include "Mesh.h"
#include <iostream>

void DeformationRecordingExample(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Deformation Recording Example ===" << std::endl;
    
    // ===== Create Soft Body =====
    
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
    
    // Material properties (jelly-like for visible deformation)
    desc.volumeStiffness = 0.3f;
    desc.shapeStiffness = 0.2f;
    desc.deformationStiffness = 0.1f;
    desc.maxStretch = 2.0f;
    desc.maxCompress = 0.3f;
    desc.linearDamping = 0.05f;
    
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
    auto softBodyObj = std::make_shared<GameObject>("RecordableSoftBody");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 5, 0));
    
    // Add to scene
    scene.AddGameObject(softBodyObj);
    
    std::cout << "Created soft body at position (0, 5, 0)" << std::endl;
    
    
    // ===== Phase 1: Record Deformation =====
    
    std::cout << "\n--- Phase 1: Recording Deformation ---" << std::endl;
    
    // Start recording at 60 FPS, including velocities
    softBody->StartRecording(60.0f, true);
    std::cout << "Started recording at 60 FPS with velocities" << std::endl;
    
    // Apply impulse to cause deformation
    softBody->AddImpulse(Vec3(5, 0, 0));
    std::cout << "Applied impulse to soft body" << std::endl;
    
    // Simulate for 5 seconds (in actual application, this would be in the main loop)
    float recordingDuration = 5.0f;
    float deltaTime = 1.0f / 60.0f;
    int frameCount = static_cast<int>(recordingDuration / deltaTime);
    
    std::cout << "Simulating for " << recordingDuration << " seconds..." << std::endl;
    for (int i = 0; i < frameCount; ++i) {
        physxBackend->Update(deltaTime);
        softBody->Update(deltaTime);
        
        // Print progress every second
        if (i % 60 == 0) {
            std::cout << "  Recording frame " << i << "/" << frameCount 
                      << " (keyframes: " << softBody->GetRecorder()->GetKeyframeCount() << ")" << std::endl;
        }
    }
    
    // Stop recording
    softBody->StopRecording();
    std::cout << "Stopped recording" << std::endl;
    std::cout << "Total keyframes recorded: " << softBody->GetRecorder()->GetKeyframeCount() << std::endl;
    std::cout << "Recording duration: " << softBody->GetRecordingDuration() << " seconds" << std::endl;
    
    
    // ===== Phase 2: Save Recording =====
    
    std::cout << "\n--- Phase 2: Saving Recording ---" << std::endl;
    
    // Save in JSON format (human-readable)
    bool savedJson = softBody->SaveRecording("deformation_recording.json", false);
    std::cout << "Saved JSON recording: " << (savedJson ? "SUCCESS" : "FAILED") << std::endl;
    
    // Save in binary format (compact)
    bool savedBinary = softBody->SaveRecording("deformation_recording.bin", true);
    std::cout << "Saved binary recording: " << (savedBinary ? "SUCCESS" : "FAILED") << std::endl;
    
    
    // ===== Phase 3: Reset and Playback =====
    
    std::cout << "\n--- Phase 3: Playback ---" << std::endl;
    
    // Reset soft body to initial position
    softBodyObj->GetTransform().SetPosition(Vec3(0, 5, 0));
    std::cout << "Reset soft body to initial position" << std::endl;
    
    // Start playback
    softBody->StartPlayback();
    std::cout << "Started playback" << std::endl;
    
    // Playback at normal speed
    std::cout << "Playing back at 1x speed..." << std::endl;
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
        
        // Print progress every second
        static int lastSecond = -1;
        int currentSecond = static_cast<int>(softBody->GetPlaybackTime());
        if (currentSecond != lastSecond) {
            lastSecond = currentSecond;
            std::cout << "  Playback time: " << softBody->GetPlaybackTime() 
                      << "/" << softBody->GetRecordingDuration() << " seconds" << std::endl;
        }
    }
    std::cout << "Playback finished" << std::endl;
    
    
    // ===== Phase 4: Advanced Playback Features =====
    
    std::cout << "\n--- Phase 4: Advanced Playback Features ---" << std::endl;
    
    // Playback at 2x speed
    softBody->SetPlaybackSpeed(2.0f);
    softBody->StartPlayback();
    std::cout << "Playing back at 2x speed..." << std::endl;
    
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
    }
    std::cout << "Fast playback finished" << std::endl;
    
    // Playback with looping
    softBody->SetPlaybackLoopMode(LoopMode::Loop);
    softBody->SetPlaybackSpeed(1.0f);
    softBody->StartPlayback();
    std::cout << "Playing back with looping (will run for 3 loops)..." << std::endl;
    
    int loopCount = 0;
    float lastTime = 0.0f;
    while (loopCount < 3) {
        softBody->Update(deltaTime);
        
        // Detect loop restart
        float currentTime = softBody->GetPlaybackTime();
        if (currentTime < lastTime) {
            loopCount++;
            std::cout << "  Loop " << loopCount << " completed" << std::endl;
        }
        lastTime = currentTime;
    }
    
    softBody->StopPlayback();
    std::cout << "Looped playback finished" << std::endl;
    
    
    // ===== Phase 5: Seek and Pause =====
    
    std::cout << "\n--- Phase 5: Seek and Pause ---" << std::endl;
    
    // Seek to middle of recording
    softBody->StartPlayback();
    float midpoint = softBody->GetRecordingDuration() / 2.0f;
    softBody->SeekPlayback(midpoint);
    std::cout << "Seeked to midpoint: " << midpoint << " seconds" << std::endl;
    
    // Play for 1 second
    for (int i = 0; i < 60; ++i) {
        softBody->Update(deltaTime);
    }
    
    // Pause
    softBody->PausePlayback();
    std::cout << "Paused playback at: " << softBody->GetPlaybackTime() << " seconds" << std::endl;
    
    // Wait (simulate user interaction)
    std::cout << "Paused for 2 seconds..." << std::endl;
    
    // Resume
    softBody->ResumePlayback();
    std::cout << "Resumed playback" << std::endl;
    
    // Play to end
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
    }
    std::cout << "Playback finished" << std::endl;
    
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    std::cout << "Demonstrated features:" << std::endl;
    std::cout << "  - Recording deformation at 60 FPS" << std::endl;
    std::cout << "  - Saving to JSON and binary formats" << std::endl;
    std::cout << "  - Basic playback" << std::endl;
    std::cout << "  - Speed control (2x)" << std::endl;
    std::cout << "  - Looping playback" << std::endl;
    std::cout << "  - Seek and pause controls" << std::endl;
}

// Usage in main.cpp:
/*
int main() {
    Application app(1280, 720, "Deformation Recording Demo");
    
    // Initialize PhysX backend
    PhysXBackend physxBackend;
    physxBackend.Initialize(Vec3(0, -9.81f, 0));
    
    // Run deformation recording example
    DeformationRecordingExample(&physxBackend, app.GetScene());
    
    return 0;
}
*/
