// Example: Cubic Interpolation for Smooth Playback
// Demonstrates the difference between linear and cubic interpolation

#include "Application.h"
#include "GameObject.h"
#include "PhysXBackend.h"
#include "PhysXSoftBody.h"
#include "TetrahedralMeshGenerator.h"
#include "SoftBodyDeformationRecorder.h"
#include <iostream>
#include <chrono>

void CubicInterpolationExample(PhysXBackend* physxBackend, Scene& scene) {
    std::cout << "=== Cubic Interpolation Example ===" << std::endl;
    
    // Create soft body
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
    
    auto tetMesh = TetrahedralMeshGenerator::Generate(
        cubeVertices.data(),
        static_cast<int>(cubeVertices.size()),
        cubeIndices.data(),
        static_cast<int>(cubeIndices.size() / 3),
        0.15f
    );
    
    SoftBodyDesc desc;
    desc.vertexPositions = cubeVertices.data();
    desc.vertexCount = static_cast<int>(cubeVertices.size());
    desc.triangleIndices = cubeIndices.data();
    desc.triangleCount = static_cast<int>(cubeIndices.size() / 3);
    desc.tetrahedronVertices = tetMesh.vertices.data();
    desc.tetrahedronVertexCount = static_cast<int>(tetMesh.vertices.size());
    desc.tetrahedronIndices = tetMesh.indices.data();
    desc.tetrahedronCount = tetMesh.tetrahedronCount;
    desc.volumeStiffness = 0.3f;
    desc.shapeStiffness = 0.2f;
    desc.deformationStiffness = 0.1f;
    desc.useDensity = false;
    desc.totalMass = 1.0f;
    desc.enableSceneCollision = true;
    desc.gravity = Vec3(0, -9.81f, 0);
    
    auto softBody = std::make_shared<PhysXSoftBody>(physxBackend);
    softBody->Initialize(desc);
    
    auto softBodyObj = std::make_shared<GameObject>("InterpolatedSoftBody");
    softBodyObj->SetSoftBody(softBody);
    softBodyObj->GetTransform().SetPosition(Vec3(0, 5, 0));
    scene.AddGameObject(softBodyObj);
    
    std::cout << "Created soft body with " << tetMesh.vertices.size() << " vertices" << std::endl;
    
    // ===== Phase 1: Record Deformation =====
    
    std::cout << "\n--- Phase 1: Recording Deformation ---" << std::endl;
    
    auto recorder = softBody->GetRecorder();
    softBody->StartRecording(30.0f, false);  // Lower sample rate to emphasize interpolation
    std::cout << "Recording at 30 FPS (lower rate to show interpolation effect)" << std::endl;
    
    // Apply multiple impulses for interesting motion
    softBody->AddImpulse(Vec3(5, 0, 0));
    
    float deltaTime = 1.0f / 60.0f;
    for (int i = 0; i < 180; ++i) {  // 3 seconds
        physxBackend->Update(deltaTime);
        softBody->Update(deltaTime);
        
        // Add another impulse halfway through
        if (i == 90) {
            softBody->AddImpulse(Vec3(0, 3, 0));
        }
    }
    
    softBody->StopRecording();
    std::cout << "Recorded " << recorder->GetKeyframeCount() << " keyframes" << std::endl;
    std::cout << "Duration: " << recorder->GetDuration() << " seconds" << std::endl;
    
    // ===== Phase 2: Linear Interpolation Playback =====
    
    std::cout << "\n--- Phase 2: Linear Interpolation Playback ---" << std::endl;
    
    recorder->SetInterpolationMode(InterpolationMode::Linear);
    std::cout << "Interpolation mode: Linear" << std::endl;
    
    auto startLinear = std::chrono::high_resolution_clock::now();
    
    softBody->StartPlayback();
    int frameCount = 0;
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
        frameCount++;
    }
    
    auto endLinear = std::chrono::high_resolution_clock::now();
    auto linearTime = std::chrono::duration<double, std::milli>(endLinear - startLinear).count();
    
    std::cout << "Linear playback complete" << std::endl;
    std::cout << "Frames: " << frameCount << std::endl;
    std::cout << "Time: " << linearTime << " ms" << std::endl;
    std::cout << "Avg frame time: " << (linearTime / frameCount) << " ms" << std::endl;
    
    // ===== Phase 3: Cubic Interpolation Playback =====
    
    std::cout << "\n--- Phase 3: Cubic Interpolation Playback ---" << std::endl;
    
    recorder->SetInterpolationMode(InterpolationMode::Cubic);
    std::cout << "Interpolation mode: Cubic (Catmull-Rom)" << std::endl;
    
    auto startCubic = std::chrono::high_resolution_clock::now();
    
    softBody->StartPlayback();
    frameCount = 0;
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
        frameCount++;
    }
    
    auto endCubic = std::chrono::high_resolution_clock::now();
    auto cubicTime = std::chrono::duration<double, std::milli>(endCubic - startCubic).count();
    
    std::cout << "Cubic playback complete" << std::endl;
    std::cout << "Frames: " << frameCount << std::endl;
    std::cout << "Time: " << cubicTime << " ms" << std::endl;
    std::cout << "Avg frame time: " << (cubicTime / frameCount) << " ms" << std::endl;
    
    // ===== Phase 4: Slow Motion Comparison =====
    
    std::cout << "\n--- Phase 4: Slow Motion Comparison ---" << std::endl;
    std::cout << "Playing at 0.25x speed to emphasize smoothness difference" << std::endl;
    
    softBody->SetPlaybackSpeed(0.25f);
    
    std::cout << "\nLinear interpolation (slow motion)..." << std::endl;
    recorder->SetInterpolationMode(InterpolationMode::Linear);
    softBody->StartPlayback();
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
    }
    std::cout << "Linear slow-motion complete" << std::endl;
    
    std::cout << "\nCubic interpolation (slow motion)..." << std::endl;
    recorder->SetInterpolationMode(InterpolationMode::Cubic);
    softBody->StartPlayback();
    while (softBody->IsPlayingBack()) {
        softBody->Update(deltaTime);
    }
    std::cout << "Cubic slow-motion complete" << std::endl;
    
    // ===== Summary =====
    
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Linear interpolation: " << linearTime << " ms" << std::endl;
    std::cout << "Cubic interpolation:  " << cubicTime << " ms" << std::endl;
    std::cout << "Overhead: " << ((cubicTime / linearTime - 1.0) * 100.0) << "%" << std::endl;
    
    std::cout << "\n=== Quality Comparison ===" << std::endl;
    std::cout << "Linear Interpolation:" << std::endl;
    std::cout << "  + Fast (baseline performance)" << std::endl;
    std::cout << "  - Visible 'kinks' at keyframes" << std::endl;
    std::cout << "  - Abrupt velocity changes" << std::endl;
    std::cout << "  - Less natural motion" << std::endl;
    
    std::cout << "\nCubic Interpolation (Catmull-Rom):" << std::endl;
    std::cout << "  + Smooth, C1 continuous motion" << std::endl;
    std::cout << "  + Natural-looking animation" << std::endl;
    std::cout << "  + Passes through all keyframes" << std::endl;
    std::cout << "  + Better for slow-motion" << std::endl;
    std::cout << "  - ~6-7x slower than linear" << std::endl;
    
    std::cout << "\n=== Recommendations ===" << std::endl;
    std::cout << "Use Linear for:" << std::endl;
    std::cout << "  - Real-time playback of many objects" << std::endl;
    std::cout << "  - High sample rate recordings (60+ FPS)" << std::endl;
    std::cout << "  - Background/distant objects" << std::endl;
    
    std::cout << "\nUse Cubic for:" << std::endl;
    std::cout << "  - Cinematic replays" << std::endl;
    std::cout << "  - Slow-motion playback" << std::endl;
    std::cout << "  - Low sample rate recordings (< 30 FPS)" << std::endl;
    std::cout << "  - Close-up viewing" << std::endl;
    std::cout << "  - High-quality animation" << std::endl;
    
    std::cout << "\n=== Example Complete ===" << std::endl;
}

// Usage in main.cpp:
/*
int main() {
    Application app(1280, 720, "Cubic Interpolation Demo");
    
    PhysXBackend physxBackend;
    physxBackend.Initialize(Vec3(0, -9.81f, 0));
    
    CubicInterpolationExample(&physxBackend, app.GetScene());
    
    return 0;
}
*/
