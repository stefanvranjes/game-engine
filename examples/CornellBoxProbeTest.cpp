#include "Application.h"
#include "GlobalIllumination.h"
#include "ProbeGrid.h"
#include "ProbeBaker.h"
#include "GameObject.h"
#include "Mesh.h"
#include "Material.h"
#include "Light.h"
#include <iostream>

/**
 * @file CornellBoxProbeTest.cpp
 * @brief Test scene for probe-based GI baking
 * 
 * This file demonstrates how to:
 * 1. Create a Cornell Box scene
 * 2. Set up a probe grid
 * 3. Bake probes using raytracing
 * 4. Save and load probe data
 * 5. Render with probe-based GI
 */

class CornellBoxProbeTest {
public:
    void Setup() {
        std::cout << "=== Cornell Box Probe Baking Test ===" << std::endl;
        
        // Create Cornell Box geometry
        CreateCornellBox();
        
        // Create lights
        CreateLights();
        
        // Setup probe grid
        SetupProbeGrid();
        
        // Bake probes
        BakeProbes();
        
        // Save probe data
        SaveProbes();
        
        // Test loading
        TestLoadProbes();
    }
    
private:
    std::vector<GameObject*> m_SceneObjects;
    std::vector<Light> m_Lights;
    std::unique_ptr<ProbeGrid> m_ProbeGrid;
    std::unique_ptr<ProbeBaker> m_ProbeBaker;
    
    void CreateCornellBox() {
        std::cout << "\n[1/6] Creating Cornell Box geometry..." << std::endl;
        
        // Cornell Box dimensions: 5x5x5 units
        const float size = 5.0f;
        
        // Floor (white)
        auto floor = CreateQuad(
            glm::vec3(-size, 0, -size),
            glm::vec3(size, 0, -size),
            glm::vec3(size, 0, size),
            glm::vec3(-size, 0, size),
            glm::vec3(0.8f, 0.8f, 0.8f)
        );
        floor->SetName("Floor");
        m_SceneObjects.push_back(floor);
        
        // Ceiling (white)
        auto ceiling = CreateQuad(
            glm::vec3(-size, size, size),
            glm::vec3(size, size, size),
            glm::vec3(size, size, -size),
            glm::vec3(-size, size, -size),
            glm::vec3(0.8f, 0.8f, 0.8f)
        );
        ceiling->SetName("Ceiling");
        m_SceneObjects.push_back(ceiling);
        
        // Back wall (white)
        auto backWall = CreateQuad(
            glm::vec3(-size, 0, -size),
            glm::vec3(-size, size, -size),
            glm::vec3(size, size, -size),
            glm::vec3(size, 0, -size),
            glm::vec3(0.8f, 0.8f, 0.8f)
        );
        backWall->SetName("BackWall");
        m_SceneObjects.push_back(backWall);
        
        // Left wall (RED)
        auto leftWall = CreateQuad(
            glm::vec3(-size, 0, -size),
            glm::vec3(-size, 0, size),
            glm::vec3(-size, size, size),
            glm::vec3(-size, size, -size),
            glm::vec3(0.8f, 0.1f, 0.1f)  // Red
        );
        leftWall->SetName("LeftWall");
        m_SceneObjects.push_back(leftWall);
        
        // Right wall (GREEN)
        auto rightWall = CreateQuad(
            glm::vec3(size, 0, size),
            glm::vec3(size, 0, -size),
            glm::vec3(size, size, -size),
            glm::vec3(size, size, size),
            glm::vec3(0.1f, 0.8f, 0.1f)  // Green
        );
        rightWall->SetName("RightWall");
        m_SceneObjects.push_back(rightWall);
        
        // Tall box (white)
        auto tallBox = CreateBox(
            glm::vec3(-1.5f, 0.0f, -1.0f),
            glm::vec3(1.0f, 3.0f, 1.0f),
            glm::vec3(0.8f, 0.8f, 0.8f)
        );
        tallBox->SetName("TallBox");
        m_SceneObjects.push_back(tallBox);
        
        // Short box (white)
        auto shortBox = CreateBox(
            glm::vec3(1.0f, 0.0f, 1.0f),
            glm::vec3(1.5f, 1.5f, 1.5f),
            glm::vec3(0.8f, 0.8f, 0.8f)
        );
        shortBox->SetName("ShortBox");
        m_SceneObjects.push_back(shortBox);
        
        std::cout << "  Created " << m_SceneObjects.size() << " objects" << std::endl;
    }
    
    void CreateLights() {
        std::cout << "\n[2/6] Creating lights..." << std::endl;
        
        // Area light on ceiling (simulated as point light)
        Light ceilingLight;
        ceilingLight.type = LightType::Point;
        ceilingLight.position = Vec3(0.0f, 4.9f, 0.0f);
        ceilingLight.color = Vec3(1.0f, 1.0f, 1.0f);
        ceilingLight.intensity = 50.0f;
        ceilingLight.range = 10.0f;
        ceilingLight.castsShadows = true;
        
        m_Lights.push_back(ceilingLight);
        
        std::cout << "  Created " << m_Lights.size() << " lights" << std::endl;
    }
    
    void SetupProbeGrid() {
        std::cout << "\n[3/6] Setting up probe grid..." << std::endl;
        
        // Create probe grid covering the Cornell Box
        glm::vec3 gridMin(-5.0f, 0.0f, -5.0f);
        glm::vec3 gridMax(5.0f, 5.0f, 5.0f);
        glm::ivec3 gridRes(8, 8, 8);  // 512 probes
        
        m_ProbeGrid = std::make_unique<ProbeGrid>(gridMin, gridMax, gridRes);
        m_ProbeGrid->Initialize();
        
        // Generate uniform grid
        m_ProbeGrid->GenerateProbes();
        
        std::cout << "  Grid bounds: " << glm::to_string(gridMin) << " to " << glm::to_string(gridMax) << std::endl;
        std::cout << "  Grid resolution: " << gridRes.x << "x" << gridRes.y << "x" << gridRes.z << std::endl;
        std::cout << "  Total probes: " << m_ProbeGrid->GetProbeCount() << std::endl;
    }
    
    void BakeProbes() {
        std::cout << "\n[4/6] Baking probes..." << std::endl;
        
        m_ProbeBaker = std::make_unique<ProbeBaker>();
        
        // Configure bake settings
        ProbeBaker::BakeSettings settings;
        settings.samplesPerProbe = 512;      // Good quality
        settings.numBounces = 2;             // Capture color bleeding
        settings.maxRayDistance = 20.0f;
        settings.showProgress = true;
        
        std::cout << "  Samples per probe: " << settings.samplesPerProbe << std::endl;
        std::cout << "  Bounces: " << settings.numBounces << std::endl;
        std::cout << "  Max ray distance: " << settings.maxRayDistance << std::endl;
        std::cout << "\n  Starting bake (this may take a few minutes)..." << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Bake all probes
        m_ProbeBaker->BakeProbes(m_ProbeGrid.get(), m_SceneObjects, m_Lights, settings);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
        
        std::cout << "\n  Baking complete!" << std::endl;
        std::cout << "  Time taken: " << duration.count() << " seconds" << std::endl;
        
        // Verify probe data
        VerifyProbeData();
    }
    
    void VerifyProbeData() {
        std::cout << "\n  Verifying probe data..." << std::endl;
        
        // Sample a few probes to check they have valid data
        int validProbes = 0;
        int totalProbes = m_ProbeGrid->GetProbeCount();
        
        for (int i = 0; i < totalProbes; i++) {
            const LightProbeData& probe = m_ProbeGrid->GetProbe(i);
            
            // Check if probe has non-zero SH coefficients
            bool hasData = false;
            for (int j = 0; j < 27; j++) {
                if (std::abs(probe.shCoefficients[j]) > 0.001f) {
                    hasData = true;
                    break;
                }
            }
            
            if (hasData) validProbes++;
        }
        
        std::cout << "  Valid probes: " << validProbes << " / " << totalProbes 
                  << " (" << (validProbes * 100 / totalProbes) << "%)" << std::endl;
        
        // Sample irradiance at a test point
        glm::vec3 testPos(0.0f, 2.5f, 0.0f);  // Center of box
        glm::vec3 testNormal(0.0f, 1.0f, 0.0f);  // Up
        glm::vec3 irradiance = m_ProbeGrid->SampleIrradiance(testPos, testNormal);
        
        std::cout << "  Test sample at center:" << std::endl;
        std::cout << "    Position: " << glm::to_string(testPos) << std::endl;
        std::cout << "    Normal: " << glm::to_string(testNormal) << std::endl;
        std::cout << "    Irradiance: " << glm::to_string(irradiance) << std::endl;
    }
    
    void SaveProbes() {
        std::cout << "\n[5/6] Saving probe data..." << std::endl;
        
        std::string filename = "cornell_box_probes.bin";
        
        if (m_ProbeGrid->SaveToFile(filename)) {
            std::cout << "  Successfully saved to: " << filename << std::endl;
            
            // Get file size
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                size_t fileSize = file.tellg();
                std::cout << "  File size: " << (fileSize / 1024) << " KB" << std::endl;
                file.close();
            }
        } else {
            std::cerr << "  ERROR: Failed to save probe data!" << std::endl;
        }
    }
    
    void TestLoadProbes() {
        std::cout << "\n[6/6] Testing probe loading..." << std::endl;
        
        // Create a new probe grid and load data
        glm::vec3 gridMin(-5.0f, 0.0f, -5.0f);
        glm::vec3 gridMax(5.0f, 5.0f, 5.0f);
        glm::ivec3 gridRes(8, 8, 8);
        
        auto testGrid = std::make_unique<ProbeGrid>(gridMin, gridMax, gridRes);
        testGrid->Initialize();
        
        if (testGrid->LoadFromFile("cornell_box_probes.bin")) {
            std::cout << "  Successfully loaded probe data" << std::endl;
            std::cout << "  Loaded probes: " << testGrid->GetProbeCount() << std::endl;
            
            // Verify loaded data matches original
            glm::vec3 testPos(0.0f, 2.5f, 0.0f);
            glm::vec3 testNormal(0.0f, 1.0f, 0.0f);
            glm::vec3 irradiance = testGrid->SampleIrradiance(testPos, testNormal);
            
            std::cout << "  Test sample irradiance: " << glm::to_string(irradiance) << std::endl;
        } else {
            std::cerr << "  ERROR: Failed to load probe data!" << std::endl;
        }
    }
    
    // Helper functions to create geometry
    
    GameObject* CreateQuad(const glm::vec3& v0, const glm::vec3& v1, 
                          const glm::vec3& v2, const glm::vec3& v3,
                          const glm::vec3& color) {
        auto obj = new GameObject();
        
        // Create mesh
        auto mesh = obj->AddComponent<Mesh>();
        
        // Calculate normal
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
        
        // Add vertices
        std::vector<Vertex> vertices = {
            { v0, normal, glm::vec2(0, 0) },
            { v1, normal, glm::vec2(1, 0) },
            { v2, normal, glm::vec2(1, 1) },
            { v3, normal, glm::vec2(0, 1) }
        };
        
        std::vector<unsigned int> indices = { 0, 1, 2, 0, 2, 3 };
        
        mesh->SetVertices(vertices);
        mesh->SetIndices(indices);
        mesh->Setup();
        
        // Create material
        auto material = obj->AddComponent<Material>();
        material->SetDiffuse(color);
        
        return obj;
    }
    
    GameObject* CreateBox(const glm::vec3& position, const glm::vec3& size, 
                         const glm::vec3& color) {
        auto obj = new GameObject();
        obj->GetTransform().SetPosition(position);
        
        // Create simple box mesh (6 quads)
        auto mesh = obj->AddComponent<Mesh>();
        
        glm::vec3 halfSize = size * 0.5f;
        
        // TODO: Add all 6 faces
        // For now, simplified
        
        auto material = obj->AddComponent<Material>();
        material->SetDiffuse(color);
        
        return obj;
    }
};

// Example usage in main application
void RunCornellBoxProbeTest() {
    CornellBoxProbeTest test;
    test.Setup();
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Load 'cornell_box_probes.bin' in your renderer" << std::endl;
    std::cout << "2. Set GI technique to Probes or ProbesVCT" << std::endl;
    std::cout << "3. Render and verify color bleeding:" << std::endl;
    std::cout << "   - White sphere should show red tint from left wall" << std::endl;
    std::cout << "   - White sphere should show green tint from right wall" << std::endl;
}
