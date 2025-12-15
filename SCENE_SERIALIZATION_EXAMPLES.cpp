/**
 * @file SCENE_SERIALIZATION_EXAMPLES.cpp
 * @brief Example usage of scene serialization and prefab system
 * 
 * This file demonstrates common patterns for using the scene serialization
 * and prefab systems in the game engine.
 * 
 * These are example code snippets - not meant to be compiled directly.
 * Include these patterns in your actual game code.
 */

#include "SceneSerializer.h"
#include "Prefab.h"
#include "Renderer.h"
#include "GameObject.h"
#include <iostream>

// ============================================================================
// Example 1: Basic Scene Serialization
// ============================================================================

void Example_BasicSceneSave(Renderer* renderer) {
    std::cout << "\n=== Example 1: Save Scene ===" << std::endl;
    
    // Save scene as JSON (human-readable)
    renderer->SaveScene("scenes/level_01.scene.json");
    
    // Save scene as binary (compact)
    renderer->SaveScene("scenes/level_01.scene.bin", 
        SceneSerializer::SerializationFormat::BINARY);
    
    std::cout << "Scene saved successfully" << std::endl;
}

void Example_BasicSceneLoad(Renderer* renderer) {
    std::cout << "\n=== Example 2: Load Scene ===" << std::endl;
    
    // Load from JSON
    renderer->LoadScene("scenes/level_01.scene.json");
    
    // Or load from binary
    renderer->LoadScene("scenes/level_01.scene.bin");
    
    std::cout << "Scene loaded successfully" << std::endl;
}

// ============================================================================
// Example 3: Advanced Serialization Options
// ============================================================================

void Example_AdvancedSerialization(Renderer* renderer) {
    std::cout << "\n=== Example 3: Advanced Serialization ===" << std::endl;
    
    SceneSerializer serializer;
    
    // Create custom options
    SceneSerializer::SerializeOptions options;
    options.format = SceneSerializer::SerializationFormat::JSON;
    options.includeChildren = true;      // Include hierarchy
    options.includeLights = true;        // Include lighting setup
    options.includeAnimations = true;    // Include animators
    options.includePhysics = true;       // Include physics
    options.includeMaterials = true;     // Include materials
    options.prettyPrintJSON = true;      // Pretty-print JSON
    
    // Serialize with custom options
    auto root = renderer->GetRoot();
    auto& lights = renderer->GetLights();
    serializer.SerializeScene(root, "scenes/full_scene.json", lights, options);
    
    std::cout << "Scene serialized with custom options" << std::endl;
}

// ============================================================================
// Example 4: Individual GameObject Serialization
// ============================================================================

void Example_SerializeGameObject(std::shared_ptr<GameObject> myObject) {
    std::cout << "\n=== Example 4: Serialize Single GameObject ===" << std::endl;
    
    SceneSerializer serializer;
    
    // Serialize to JSON
    json objectJson = serializer.SerializeGameObjectToJson(myObject, true);
    std::cout << "Serialized to JSON: " << objectJson.dump(4) << std::endl;
    
    // Serialize to binary
    std::vector<uint8_t> binaryData = 
        serializer.SerializeGameObjectToBinary(myObject, true);
    std::cout << "Serialized to binary: " << binaryData.size() << " bytes" << std::endl;
    
    // Deserialize from JSON
    auto reconstructed = serializer.DeserializeGameObjectFromJson(objectJson);
    std::cout << "Reconstructed from JSON: " << reconstructed->GetName() << std::endl;
    
    // Deserialize from binary
    auto reconstructed2 = serializer.DeserializeGameObjectFromBinary(binaryData);
    std::cout << "Reconstructed from binary: " << reconstructed2->GetName() << std::endl;
}

// ============================================================================
// Example 5: Creating Prefabs
// ============================================================================

void Example_CreatePrefab(Renderer* renderer, std::shared_ptr<GameObject> playerObject) {
    std::cout << "\n=== Example 5: Create Prefab ===" << std::endl;
    
    // Simple prefab creation
    auto prefab = renderer->CreatePrefab("Player", playerObject);
    std::cout << "Created prefab: " << prefab->GetName() << std::endl;
    
    // Or with detailed metadata
    Prefab::Metadata metadata;
    metadata.name = "Player";
    metadata.version = "1.0.0";
    metadata.description = "Main player character with animations and physics";
    metadata.author = "Game Development Team";
    metadata.tags = {"character", "player", "animated", "physics"};
    
    auto detailedPrefab = renderer->GetPrefabManager()->CreatePrefab(
        playerObject, "Player", metadata);
    
    std::cout << "Created detailed prefab with metadata" << std::endl;
}

// ============================================================================
// Example 6: Saving and Loading Prefabs
// ============================================================================

void Example_SaveLoadPrefab(Renderer* renderer) {
    std::cout << "\n=== Example 6: Save and Load Prefab ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Get a prefab
    auto prefab = prefabManager->GetPrefab("Player");
    if (prefab) {
        // Save as JSON (for editing)
        prefabManager->SavePrefab(prefab, "Player", 
            SceneSerializer::SerializationFormat::JSON);
        std::cout << "Saved Player prefab as JSON" << std::endl;
        
        // Save as binary (for distribution)
        prefabManager->SavePrefab(prefab, "Player", 
            SceneSerializer::SerializationFormat::BINARY);
        std::cout << "Saved Player prefab as binary" << std::endl;
    }
    
    // Load prefab from disk
    auto loadedPrefab = prefabManager->LoadPrefab("assets/prefabs/Player.json");
    if (loadedPrefab) {
        prefabManager->RegisterPrefab(loadedPrefab, "Player");
        std::cout << "Loaded and registered Player prefab" << std::endl;
    }
    
    // Load all prefabs from directory
    int loadedCount = prefabManager->LoadAllPrefabs();
    std::cout << "Loaded " << loadedCount << " prefabs from directory" << std::endl;
}

// ============================================================================
// Example 7: Instantiating Prefabs
// ============================================================================

void Example_InstantiatePrefab(Renderer* renderer) {
    std::cout << "\n=== Example 7: Instantiate Prefab ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Get prefab
    auto enemyPrefab = prefabManager->GetPrefab("Goblin");
    if (!enemyPrefab) {
        std::cerr << "Prefab not found!" << std::endl;
        return;
    }
    
    // Basic instantiation
    auto enemy1 = enemyPrefab->Instantiate("Goblin_001");
    renderer->GetRoot()->AddChild(enemy1);
    std::cout << "Instantiated enemy: " << enemy1->GetName() << std::endl;
    
    // Instantiation with custom transform
    Vec3 position(10.0f, 0.0f, 5.0f);
    Vec3 rotation(0.0f, 45.0f, 0.0f);
    Vec3 scale(1.5f, 1.5f, 1.5f);
    
    auto enemy2 = enemyPrefab->InstantiateAt(position, rotation, scale, "Goblin_002");
    renderer->GetRoot()->AddChild(enemy2);
    std::cout << "Instantiated enemy with custom transform" << std::endl;
    
    // Using renderer convenience method
    auto enemy3 = renderer->InstantiatePrefab("Goblin", 
        Vec3(20.0f, 0.0f, 10.0f), "Goblin_003");
    std::cout << "Instantiated enemy via renderer" << std::endl;
}

// ============================================================================
// Example 8: Spawning Multiple Enemies from Prefab
// ============================================================================

void Example_SpawnWave(Renderer* renderer) {
    std::cout << "\n=== Example 8: Spawn Enemy Wave ===" << std::endl;
    
    auto enemyPrefab = renderer->GetPrefab("Goblin");
    if (!enemyPrefab) return;
    
    // Spawn wave of 10 enemies
    std::vector<std::shared_ptr<GameObject>> wave;
    for (int i = 0; i < 10; ++i) {
        Vec3 position(i * 5.0f, 0.0f, std::sin(i * 0.5f) * 3.0f);
        std::string name = "Goblin_" + std::to_string(i);
        
        auto enemy = enemyPrefab->InstantiateAt(position);
        enemy->GetTransform().position = position; // Override position
        
        // Randomize rotation
        enemy->GetTransform().rotation.y = (i % 4) * 90.0f;
        
        renderer->GetRoot()->AddChild(enemy);
        wave.push_back(enemy);
    }
    
    std::cout << "Spawned wave of " << wave.size() << " enemies" << std::endl;
}

// ============================================================================
// Example 9: Updating Prefabs
// ============================================================================

void Example_UpdatePrefab(Renderer* renderer, std::shared_ptr<GameObject> modifiedInstance) {
    std::cout << "\n=== Example 9: Update Prefab from Instance ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Get the prefab
    auto prefab = prefabManager->GetPrefab("Goblin");
    if (!prefab) return;
    
    // Instance has been modified in the editor or during gameplay
    // Update the prefab with new values
    prefab->UpdateFromInstance(modifiedInstance);
    std::cout << "Updated prefab from modified instance" << std::endl;
    
    // Save the updated prefab
    prefabManager->SavePrefab(prefab);
    std::cout << "Saved updated prefab" << std::endl;
    
    // Apply updated prefab to other instances
    // Note: In real code, you'd iterate through actual instances
    // for (auto& instance : allGoblinInstances) {
    //     prefab->ApplyToInstance(instance);
    // }
}

// ============================================================================
// Example 10: Searching Prefabs
// ============================================================================

void Example_SearchPrefabs(Renderer* renderer) {
    std::cout << "\n=== Example 10: Search Prefabs ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Search by name
    auto results = prefabManager->SearchByName("Enemy");
    std::cout << "Prefabs matching 'Enemy': " << results.size() << std::endl;
    for (const auto& name : results) {
        std::cout << "  - " << name << std::endl;
    }
    
    // Search by tag
    auto characters = prefabManager->SearchByTag("character");
    std::cout << "Prefabs tagged 'character': " << characters.size() << std::endl;
    for (const auto& name : characters) {
        std::cout << "  - " << name << std::endl;
    }
    
    // Get all prefabs
    auto allPrefabs = prefabManager->GetPrefabNames();
    std::cout << "Total prefabs loaded: " << allPrefabs.size() << std::endl;
}

// ============================================================================
// Example 11: Prefab Metadata Access
// ============================================================================

void Example_PrefabMetadata(Renderer* renderer) {
    std::cout << "\n=== Example 11: Access Prefab Metadata ===" << std::endl;
    
    auto prefab = renderer->GetPrefab("Player");
    if (!prefab) return;
    
    const auto& metadata = prefab->GetMetadata();
    
    std::cout << "Prefab Information:" << std::endl;
    std::cout << "  Name: " << metadata.name << std::endl;
    std::cout << "  Version: " << metadata.version << std::endl;
    std::cout << "  Author: " << metadata.author << std::endl;
    std::cout << "  Description: " << metadata.description << std::endl;
    std::cout << "  Created: " << metadata.created << std::endl;
    std::cout << "  Modified: " << metadata.modified << std::endl;
    std::cout << "  Tags: ";
    for (const auto& tag : metadata.tags) {
        std::cout << tag << ", ";
    }
    std::cout << std::endl;
}

// ============================================================================
// Example 12: Error Handling
// ============================================================================

void Example_ErrorHandling(Renderer* renderer) {
    std::cout << "\n=== Example 12: Error Handling ===" << std::endl;
    
    // Serialization error handling
    SceneSerializer serializer;
    auto root = renderer->GetRoot();
    auto& lights = renderer->GetLights();
    
    if (!serializer.SerializeScene(root, "invalid/path/scene.json", lights)) {
        std::cerr << "Serialization error: " << serializer.GetLastError() << std::endl;
    }
    
    // Prefab manager error handling
    auto prefabManager = renderer->GetPrefabManager();
    auto prefab = prefabManager->LoadPrefab("nonexistent.json");
    
    if (!prefab) {
        std::cerr << "Prefab load error: " << prefabManager->GetLastError() << std::endl;
    }
}

// ============================================================================
// Example 13: Batch Operations
// ============================================================================

void Example_BatchOperations(Renderer* renderer) {
    std::cout << "\n=== Example 13: Batch Operations ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Save all registered prefabs
    int savedCount = prefabManager->SaveAllPrefabs(
        SceneSerializer::SerializationFormat::JSON);
    std::cout << "Saved " << savedCount << " prefabs" << std::endl;
    
    // Load all from directory
    int loadedCount = prefabManager->LoadAllPrefabs();
    std::cout << "Loaded " << loadedCount << " prefabs" << std::endl;
    
    // Get count
    size_t count = prefabManager->GetPrefabCount();
    std::cout << "Total prefabs: " << count << std::endl;
}

// ============================================================================
// Example 14: Complex Scene with Multiple Prefabs
// ============================================================================

void Example_ComplexScene(Renderer* renderer) {
    std::cout << "\n=== Example 14: Build Complex Scene ===" << std::endl;
    
    auto prefabManager = renderer->GetPrefabManager();
    
    // Place player
    auto playerPrefab = prefabManager->GetPrefab("Player");
    if (playerPrefab) {
        auto player = playerPrefab->InstantiateAt(Vec3(0, 0, 0));
        renderer->GetRoot()->AddChild(player);
        std::cout << "Placed player" << std::endl;
    }
    
    // Place enemies
    auto enemyPrefab = prefabManager->GetPrefab("Goblin");
    if (enemyPrefab) {
        for (int i = 0; i < 5; ++i) {
            Vec3 pos(i * 10.0f, 0, 20.0f);
            auto enemy = enemyPrefab->InstantiateAt(pos);
            renderer->GetRoot()->AddChild(enemy);
        }
        std::cout << "Placed 5 enemies" << std::endl;
    }
    
    // Place environmental objects
    auto torchPrefab = prefabManager->GetPrefab("Torch");
    if (torchPrefab) {
        Vec3 positions[] = {
            Vec3(-10, 0, -10),
            Vec3(10, 0, -10),
            Vec3(-10, 0, 30),
            Vec3(10, 0, 30)
        };
        for (auto& pos : positions) {
            auto torch = torchPrefab->InstantiateAt(pos);
            renderer->GetRoot()->AddChild(torch);
        }
        std::cout << "Placed 4 torches" << std::endl;
    }
    
    // Save the composed scene
    renderer->SaveScene("scenes/level_01_populated.scene.json");
    std::cout << "Saved composed scene" << std::endl;
}

// ============================================================================
// Main: How to use these examples
// ============================================================================

/*
In your game code (e.g., in Application::Init() or RenderEditorUI()):

void Application::Initialize() {
    // ... other initialization ...
    
    // Load prefabs at startup
    Example_SaveLoadPrefab(m_Renderer.get());
    
    // Or create and save prefabs
    // Example_CreatePrefab(m_Renderer.get(), someGameObject);
    // Example_SaveLoadPrefab(m_Renderer.get());
}

void Application::Update(float deltaTime) {
    // ... game logic ...
    
    // Example: Spawn enemies when player enters trigger
    if (playerEnteredSpawnZone) {
        Example_SpawnWave(m_Renderer.get());
    }
}

void Application::RenderEditorUI() {
    // ... ImGui windows ...
    
    if (ImGui::Button("Save Scene")) {
        Example_BasicSceneSave(m_Renderer.get());
    }
    
    if (ImGui::Button("Load Scene")) {
        Example_BasicSceneLoad(m_Renderer.get());
    }
    
    if (ImGui::Button("Search Prefabs")) {
        Example_SearchPrefabs(m_Renderer.get());
    }
}
*/
