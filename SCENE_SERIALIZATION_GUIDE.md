# Scene Serialization & Prefab System

## Overview

The Game Engine now includes a comprehensive scene serialization and prefab system that allows you to:

1. **Save and load complete game scenes** with all GameObjects, transforms, materials, components, and lights
2. **Serialize in multiple formats**: Human-readable JSON or compact binary format
3. **Create reusable prefabs** from GameObjects for instantiation throughout your game
4. **Manage prefab libraries** with search, tagging, and versioning support
5. **Apply prefab updates** to all instances automatically

## Architecture

### Components

- **SceneSerializer** (`SceneSerializer.h/cpp`): Handles serialization/deserialization of scenes and individual GameObjects
- **Prefab** (`Prefab.h/cpp`): Represents a reusable blueprint created from a GameObject
- **PrefabManager** (`Prefab.h/cpp`): Manages prefab creation, storage, loading, and registration

### Data Flow

```
GameObject --> Serializer --> JSON/Binary File
                                     ↓
File --> Deserializer --> GameObject (reconstructed)

Prefab (template) --> Instantiate --> GameObject (independent copy)
```

## Scene Serialization

### Supported Data

Scenes can serialize:
- **Hierarchy**: Parent-child relationships and transforms
- **Transforms**: Position, rotation, scale
- **Materials**: Material properties and textures
- **Lights**: All light types with properties
- **Animations**: Animator and animation state
- **Physics**: RigidBody and KinematicController components
- **LOD System**: Level-of-detail configurations
- **Custom Properties**: UV offsets, scales, visibility, etc.

### Serialization Formats

#### JSON Format (Default)
Human-readable, suitable for editing and debugging:

```json
{
    "version": 1,
    "format": "json",
    "rootObject": {
        "name": "Root",
        "transform": {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        },
        "visible": true,
        "uvOffset": [0.0, 0.0],
        "uvScale": [1.0, 1.0],
        "children": [
            {
                "name": "Cube",
                "transform": {...},
                ...
            }
        ]
    },
    "lights": [
        {
            "type": 0,
            "position": [5.0, 5.0, 5.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": 1.0,
            ...
        }
    ]
}
```

#### Binary Format
Compact and efficient, optimized for production:

```
[MAGIC: 4 bytes] [VERSION: 4 bytes] [Object Data] [Light Count] [Lights...]
```

- Magic: 0x53434E45 ("SCNE")
- Version: For format compatibility
- Significantly smaller file size than JSON

### Usage Examples

#### Save Scene

```cpp
// Save as JSON (human-readable)
renderer->SaveScene("scenes/level_01.scene.json", 
    SceneSerializer::SerializationFormat::JSON);

// Save as binary (compact)
renderer->SaveScene("scenes/level_01.scene.bin", 
    SceneSerializer::SerializationFormat::BINARY);
```

#### Load Scene

```cpp
renderer->LoadScene("scenes/level_01.scene.json");
// Or load binary
renderer->LoadScene("scenes/level_01.scene.bin");
```

#### Advanced Serialization Options

```cpp
SceneSerializer serializer;

// Create custom options
SceneSerializer::SerializeOptions options;
options.format = SceneSerializer::SerializationFormat::JSON;
options.includeChildren = true;      // Serialize hierarchy
options.includeLights = true;        // Serialize lights
options.includeAnimations = true;    // Serialize animators
options.includePhysics = true;       // Serialize physics components
options.includeMaterials = true;     // Serialize materials
options.prettyPrintJSON = true;      // Pretty-print JSON output

// Serialize with options
serializer.SerializeScene(rootObject, "scenes/custom.scene.json", 
    lights, options);

// Deserialize
std::shared_ptr<GameObject> loadedRoot;
std::vector<Light> loadedLights;
serializer.DeserializeScene("scenes/custom.scene.json", 
    loadedRoot, loadedLights);
```

#### Serialize Individual GameObject

```cpp
SceneSerializer serializer;

// To JSON
json gameObjectJson = serializer.SerializeGameObjectToJson(myObject, 
    true); // true = include children

// To Binary
std::vector<uint8_t> binaryData = 
    serializer.SerializeGameObjectToBinary(myObject, true);

// Deserialize from JSON
auto reconstructed = serializer.DeserializeGameObjectFromJson(gameObjectJson);

// Deserialize from Binary
auto reconstructed = serializer.DeserializeGameObjectFromBinary(binaryData);
```

## Prefab System

### What is a Prefab?

A prefab is a saved, reusable template of a GameObject that can be instantiated multiple times. Changes to the prefab can optionally be applied to all instances.

### Prefab Features

- **Metadata**: Name, version, description, author, creation/modification dates, tags
- **Complete State**: Serializes entire GameObject with all components
- **Nested Prefabs**: Prefabs can contain child prefabs
- **Instantiation**: Create new independent copies from prefab
- **Updates**: Sync instances with updated prefab data
- **Searching**: Find prefabs by name or tags
- **Versioning**: Track prefab versions for asset management

### Creating Prefabs

#### From Renderer (Simple)

```cpp
// Create prefab from root scene object
auto prefab = renderer->CreatePrefab("MyCharacter");

// Or specify a different source object
auto prefab = renderer->CreatePrefab("MyCharacter", someGameObject);
```

#### Using PrefabManager (Advanced)

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Create with metadata
Prefab::Metadata metadata;
metadata.name = "Player";
metadata.description = "Main player character with animations";
metadata.author = "Game Team";
metadata.tags = {"character", "player", "animated"};

auto prefab = prefabManager->CreatePrefab(playerGameObject, "Player", metadata);

// Save to disk
prefabManager->SavePrefab(prefab, "Player", 
    SceneSerializer::SerializationFormat::JSON);
```

### Saving and Loading Prefabs

#### Save Prefab

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Save as JSON (human-readable, good for editing)
prefabManager->SavePrefab(prefab, "MyEnemy");

// Save as binary (compact, good for production)
prefabManager->SavePrefab(prefab, "MyEnemy",
    SceneSerializer::SerializationFormat::BINARY);
```

#### Load Prefab

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Load from file
auto loadedPrefab = prefabManager->LoadPrefab("assets/prefabs/MyEnemy.json");

// Register for later use
if (loadedPrefab) {
    prefabManager->RegisterPrefab(loadedPrefab, "MyEnemy");
}

// Load all prefabs from directory
int loadedCount = prefabManager->LoadAllPrefabs();
std::cout << "Loaded " << loadedCount << " prefabs" << std::endl;
```

### Instantiating Prefabs

#### Basic Instantiation

```cpp
// Get registered prefab
auto prefab = renderer->GetPrefab("MyEnemy");

// Instantiate at default location
auto enemy = prefab->Instantiate("Enemy_01");

// Add to scene
renderer->GetRoot()->AddChild(enemy);
```

#### Instantiation with Transform

```cpp
// Instantiate at specific position
Vec3 spawnPosition(10.0f, 0.0f, 5.0f);
Vec3 rotation(0.0f, 90.0f, 0.0f);  // Rotated 90 degrees on Y
Vec3 scale(1.5f, 1.5f, 1.5f);      // Scaled 1.5x

auto enemy = prefab->InstantiateAt(spawnPosition, rotation, scale, "Enemy_02");
renderer->GetRoot()->AddChild(enemy);
```

#### Using Renderer Helper

```cpp
// Renderer has convenient wrapper
auto enemy = renderer->InstantiatePrefab("MyEnemy", 
    Vec3(10.0f, 0.0f, 5.0f), "Enemy_Spawned");
```

### Updating Prefabs

#### Update Prefab from Instance

```cpp
// Modify an instance in the editor
auto instance = myEnemyInstance;
instance->GetTransform().scale = Vec3(2.0f, 2.0f, 2.0f);
// ... make other changes ...

// Update the prefab with new values
auto prefab = renderer->GetPrefab("MyEnemy");
prefab->UpdateFromInstance(instance);

// Save updated prefab
renderer->GetPrefabManager()->SavePrefab(prefab);
```

#### Apply Prefab to Instance

```cpp
// Create an instance
auto instance = prefab->Instantiate("MyEnemy_Instance");

// Later, sync instance with updated prefab
prefab->ApplyToInstance(instance, false); // false = don't preserve changes

// Or preserve local changes
prefab->ApplyToInstance(instance, true);  // true = keep local modifications
```

### Managing Prefab Libraries

#### Search Prefabs

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Search by name (substring match)
auto results = prefabManager->SearchByName("Enemy");
for (const auto& name : results) {
    std::cout << "Found prefab: " << name << std::endl;
}

// Search by tag
auto characterPrefabs = prefabManager->SearchByTag("character");
```

#### Batch Operations

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Save all registered prefabs
int savedCount = prefabManager->SaveAllPrefabs(
    SceneSerializer::SerializationFormat::JSON);

// Load all from directory
int loadedCount = prefabManager->LoadAllPrefabs();

// Get all prefab names
auto allPrefabs = prefabManager->GetPrefabNames();

// Clear all (unregister)
prefabManager->Clear();
```

#### Prefab Information

```cpp
auto prefabManager = renderer->GetPrefabManager();

// Get prefab
auto prefab = prefabManager->GetPrefab("MyEnemy");

// Access metadata
const auto& metadata = prefab->GetMetadata();
std::cout << "Name: " << metadata.name << std::endl;
std::cout << "Version: " << metadata.version << std::endl;
std::cout << "Author: " << metadata.author << std::endl;
std::cout << "Created: " << metadata.created << std::endl;
std::cout << "Modified: " << metadata.modified << std::endl;
std::cout << "Description: " << metadata.description << std::endl;

// Get count
size_t count = prefabManager->GetPrefabCount();

// Get error message
std::string error = prefabManager->GetLastError();
```

## Advanced Usage

### Multiple Instances from Single Prefab

```cpp
auto enemyPrefab = renderer->GetPrefab("Goblin");

// Spawn wave of enemies
std::vector<std::shared_ptr<GameObject>> enemies;
for (int i = 0; i < 10; ++i) {
    Vec3 position(i * 5.0f, 0.0f, 0.0f);
    auto enemy = enemyPrefab->InstantiateAt(position);
    enemies.push_back(enemy);
    renderer->GetRoot()->AddChild(enemy);
}
```

### Nested Prefabs

```cpp
// Create a complex object from simpler prefabs
auto carPrefab = renderer->GetPrefab("Car");
auto wheelPrefab = renderer->GetPrefab("Wheel");

auto carInstance = carPrefab->Instantiate("Car_01");

// Each wheel is already instantiated as part of the car
// But you can also modify them individually:
for (auto& wheel : carInstance->GetChildren()) {
    if (wheel->GetName().find("Wheel") != std::string::npos) {
        wheel->GetTransform().scale = Vec3(0.8f, 0.8f, 0.8f);
    }
}

renderer->GetRoot()->AddChild(carInstance);
```

### Prefab with LOD

```cpp
// Create prefab from object with LOD levels
auto detailedMesh = mesh;
auto simpleMesh = Mesh::CreateSimplified(detailedMesh);

auto gameObject = std::make_shared<GameObject>("Building");
gameObject->SetMesh(detailedMesh);

// Add LOD levels
gameObject->AddLOD(simpleMesh, 50.0f);  // Use simple mesh at 50+ units away

// Create prefab - LOD levels are automatically serialized
auto buildingPrefab = renderer->CreatePrefab("Building", gameObject);
```

### Error Handling

```cpp
auto serializer = std::make_unique<SceneSerializer>();

if (!serializer->SerializeScene(root, "scene.json", lights)) {
    std::cerr << "Error: " << serializer->GetLastError() << std::endl;
}

auto prefabManager = renderer->GetPrefabManager();
auto prefab = prefabManager->LoadPrefab("prefabs/NPC.json");
if (!prefab) {
    std::cerr << "Failed to load prefab: " 
              << prefabManager->GetLastError() << std::endl;
}
```

## File Structure

```
assets/
├── prefabs/
│   ├── Player.json
│   ├── Goblin.json
│   ├── Chest.json
│   └── Torch.json
└── scenes/
    ├── Level_01.scene.json
    ├── Level_02.scene.json
    └── Menu.scene.json
```

## Best Practices

1. **Save Scenes Regularly**: Use scene serialization for level saves and editor state
2. **Organize Prefabs**: Use tags and naming conventions for easy searching
3. **Version Control**: Keep prefab versions in metadata for tracking changes
4. **Backup Before Updates**: Save backups before applying prefab updates to instances
5. **Use JSON During Development**: Easy to inspect and debug
6. **Switch to Binary for Production**: Smaller file sizes and faster loading
7. **Lazy Load Prefabs**: Load only the prefabs you need for a level
8. **Prefab Dependencies**: Be aware of inter-prefab references when updating

## Performance Considerations

- **Binary Format**: ~60-80% smaller than JSON, faster deserialization
- **Selective Serialization**: Disable unused options (lights, animations) to reduce file size
- **Streaming**: Use for large scenes with asynchronous loading
- **Memory**: Prefab instances are fully independent copies (no shared state by default)

## Limitations and Future Enhancements

### Current Limitations
- Animator and RigidBody serialization are placeholder implementations
- Model mesh serialization references must be manually set
- Circular prefab references not prevented (manual validation required)

### Planned Features
- Complete animator state serialization
- Physics component full serialization with constraints
- Prefab inheritance hierarchy
- Automatic circular reference detection
- Delta compression for prefab updates
- Editor GUI for prefab creation/management
