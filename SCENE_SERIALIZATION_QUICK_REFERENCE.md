# Scene Serialization & Prefab System - Quick Reference

## Quick Start (TL;DR)

### Save and Load Scenes
```cpp
// Save as JSON
renderer->SaveScene("level.scene.json");

// Load from file
renderer->LoadScene("level.scene.json");
```

### Create and Use Prefabs
```cpp
// Create prefab from GameObject
auto prefab = renderer->CreatePrefab("Enemy", enemyObject);

// Save prefab
renderer->GetPrefabManager()->SavePrefab(prefab);

// Load prefab
auto loaded = renderer->GetPrefabManager()->LoadPrefab("assets/prefabs/Enemy.json");

// Instantiate
auto instance = prefab->InstantiateAt(Vec3(10, 0, 5));
renderer->GetRoot()->AddChild(instance);
```

---

## Key Classes

### SceneSerializer
Serializes/deserializes complete scenes and individual GameObjects.

**Key Methods:**
```cpp
bool SerializeScene(GameObject* root, const string& filename, 
                    const vector<Light>& lights, const SerializeOptions& opts);
bool DeserializeScene(const string& filename, GameObject*& outRoot, 
                      vector<Light>& outLights);
json SerializeGameObjectToJson(GameObject* obj, bool children);
GameObject* DeserializeGameObjectFromJson(const json& data);
```

**Serialization Formats:**
- `SerializationFormat::JSON` - Human-readable text
- `SerializationFormat::BINARY` - Compact binary (60-80% smaller)

**Options:**
```cpp
struct SerializeOptions {
    SerializationFormat format;
    bool includeChildren;
    bool includeLights;
    bool includeAnimations;
    bool includePhysics;
    bool includeMaterials;
    bool prettyPrintJSON;
};
```

### Prefab
A reusable blueprint that can be instantiated multiple times.

**Key Methods:**
```cpp
shared_ptr<GameObject> Instantiate(const string& name, bool applyTransform);
shared_ptr<GameObject> InstantiateAt(const Vec3& pos, const Vec3& rot, 
                                     const Vec3& scale, const string& name);
void UpdateFromInstance(shared_ptr<GameObject> source, bool updateMetadata);
void ApplyToInstance(shared_ptr<GameObject> instance, bool preserveChanges);
```

**Metadata:**
```cpp
struct Metadata {
    string name;
    string version;
    string description;
    string author;
    string created;        // ISO 8601 timestamp
    string modified;       // ISO 8601 timestamp
    vector<string> tags;   // For searching
};
```

### PrefabManager
Manages prefab creation, storage, searching, and loading.

**Key Methods:**
```cpp
shared_ptr<Prefab> CreatePrefab(GameObject* source, const string& name,
                                const Prefab::Metadata& metadata);
bool SavePrefab(shared_ptr<Prefab> prefab, const string& filename,
                SerializationFormat format);
shared_ptr<Prefab> LoadPrefab(const string& filename);
void RegisterPrefab(shared_ptr<Prefab> prefab, const string& name);
shared_ptr<Prefab> GetPrefab(const string& name);
vector<string> SearchByName(const string& nameFilter);
vector<string> SearchByTag(const string& tag);
int LoadAllPrefabs();
int SaveAllPrefabs(SerializationFormat format);
```

---

## Common Patterns

### Pattern 1: Save Current Scene
```cpp
renderer->SaveScene("scenes/level_01.scene.json");
```

### Pattern 2: Load Saved Scene
```cpp
renderer->LoadScene("scenes/level_01.scene.json");
```

### Pattern 3: Create Prefab from GameObject
```cpp
auto gameObj = std::make_shared<GameObject>("Enemy");
// ... configure gameObj ...

auto prefab = renderer->CreatePrefab("Goblin", gameObj);
renderer->GetPrefabManager()->SavePrefab(prefab);
```

### Pattern 4: Load All Prefabs at Startup
```cpp
auto prefabMgr = renderer->GetPrefabManager();
int count = prefabMgr->LoadAllPrefabs();
cout << "Loaded " << count << " prefabs" << endl;
```

### Pattern 5: Spawn from Prefab
```cpp
auto prefab = renderer->GetPrefab("Goblin");
auto enemy = prefab->InstantiateAt(Vec3(10, 0, 5), Vec3(0, 45, 0), Vec3(1, 1, 1));
renderer->GetRoot()->AddChild(enemy);
```

### Pattern 6: Spawn Multiple from Prefab
```cpp
auto prefab = renderer->GetPrefab("Goblin");
for (int i = 0; i < 10; ++i) {
    auto enemy = prefab->InstantiateAt(Vec3(i * 5, 0, 0));
    renderer->GetRoot()->AddChild(enemy);
}
```

### Pattern 7: Update Prefab from Modified Instance
```cpp
auto prefab = renderer->GetPrefab("Goblin");
// ... modify instance in editor ...
prefab->UpdateFromInstance(modifiedInstance);
renderer->GetPrefabManager()->SavePrefab(prefab);
```

### Pattern 8: Search for Prefabs
```cpp
auto prefabMgr = renderer->GetPrefabManager();
auto results = prefabMgr->SearchByName("Enemy");
auto characters = prefabMgr->SearchByTag("character");
```

---

## File Formats

### JSON Scene File
```json
{
    "version": 1,
    "format": "json",
    "rootObject": {
        "name": "Root",
        "transform": {
            "position": [0, 0, 0],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1]
        },
        "visible": true,
        "children": [...]
    },
    "lights": [...]
}
```

### Binary Scene File
- **Header**: Magic (0x53434E45) + Version + Root Object + Lights

### JSON Prefab File
```json
{
    "prefabMetadata": {
        "name": "Goblin",
        "version": "1.0",
        "description": "...",
        "author": "...",
        "created": "2024-01-01T...",
        "modified": "2024-01-02T...",
        "tags": ["enemy", "melee"]
    },
    "gameObject": { ... }
}
```

---

## Serialization Options

### Minimal Scene (Hierarchy Only)
```cpp
SceneSerializer::SerializeOptions opts;
opts.includeChildren = true;
opts.includeLights = false;
opts.includeAnimations = false;
opts.includePhysics = false;
opts.includeMaterials = false;
serializer.SerializeScene(root, "minimal.json", lights, opts);
```

### Complete Scene (Everything)
```cpp
SceneSerializer::SerializeOptions opts;
opts.format = SceneSerializer::SerializationFormat::JSON;
opts.includeChildren = true;
opts.includeLights = true;
opts.includeAnimations = true;
opts.includePhysics = true;
opts.includeMaterials = true;
opts.prettyPrintJSON = true;
serializer.SerializeScene(root, "complete.json", lights, opts);
```

### Binary for Production
```cpp
SceneSerializer::SerializeOptions opts;
opts.format = SceneSerializer::SerializationFormat::BINARY;
opts.prettyPrintJSON = false; // Ignored in binary mode
serializer.SerializeScene(root, "optimized.scene.bin", lights, opts);
```

---

## Error Handling

### Check Serialization Errors
```cpp
SceneSerializer serializer;
if (!serializer.SerializeScene(root, "scene.json", lights)) {
    cerr << "Error: " << serializer.GetLastError() << endl;
}
```

### Check Prefab Errors
```cpp
auto prefabMgr = renderer->GetPrefabManager();
auto prefab = prefabMgr->LoadPrefab("missing.json");
if (!prefab) {
    cerr << "Error: " << prefabMgr->GetLastError() << endl;
}
```

### Check Prefab Validity
```cpp
if (prefab && prefab->IsValid()) {
    // Safe to use
} else {
    cout << "Prefab is invalid" << endl;
}
```

---

## Directory Structure

```
assets/
├── prefabs/              # Prefab library
│   ├── Player.json
│   ├── Goblin.json
│   ├── Skeleton.json
│   ├── Chest.json
│   ├── Torch.json
│   └── Door.json
└── scenes/               # Scene files
    ├── Level_01.scene.json
    ├── Level_02.scene.json
    ├── Menu.scene.json
    └── Level_01.scene.bin
```

---

## Performance Tips

| Aspect | Recommendation |
|--------|---|
| **Format** | Use JSON during development, binary for production |
| **File Size** | Binary is 60-80% smaller than JSON |
| **Load Speed** | Binary deserializes faster |
| **Editability** | JSON is human-readable and editable |
| **Lazy Loading** | Load only needed prefabs per level |
| **Streaming** | Use async loading for large scenes |
| **Memory** | Prefab instances are independent copies |

---

## API Reference Quick Lookup

### Renderer Methods
```cpp
void SaveScene(const string& filename, SerializationFormat format);
void LoadScene(const string& filename);
shared_ptr<Prefab> CreatePrefab(const string& name, GameObject* source);
shared_ptr<Prefab> GetPrefab(const string& name);
shared_ptr<GameObject> InstantiatePrefab(const string& name, const Vec3& pos, 
                                         const string& instanceName);
PrefabManager* GetPrefabManager();
```

### SceneSerializer Methods
```cpp
bool SerializeScene(GameObject* root, const string& filename, 
                    const vector<Light>& lights, const SerializeOptions& opts);
bool DeserializeScene(const string& filename, GameObject*& outRoot, 
                      vector<Light>& outLights);
json SerializeGameObjectToJson(GameObject* obj, bool children);
vector<uint8_t> SerializeGameObjectToBinary(GameObject* obj, bool children);
GameObject* DeserializeGameObjectFromJson(const json& data);
GameObject* DeserializeGameObjectFromBinary(const vector<uint8_t>& data);
const string& GetLastError() const;
```

### Prefab Methods
```cpp
shared_ptr<GameObject> Instantiate(const string& name, bool applyTransform);
shared_ptr<GameObject> InstantiateAt(const Vec3& pos, const Vec3& rot, 
                                     const Vec3& scale, const string& name);
void UpdateFromInstance(shared_ptr<GameObject> source, bool updateMetadata);
void ApplyToInstance(shared_ptr<GameObject> instance, bool preserveChanges);
const Metadata& GetMetadata() const;
const string& GetName() const;
const json& GetSerializedData() const;
bool IsValid() const;
```

### PrefabManager Methods
```cpp
shared_ptr<Prefab> CreatePrefab(GameObject* source, const string& name, 
                                const Metadata& metadata);
bool SavePrefab(shared_ptr<Prefab> prefab, const string& filename, 
                SerializationFormat format);
shared_ptr<Prefab> LoadPrefab(const string& filename);
void RegisterPrefab(shared_ptr<Prefab> prefab, const string& name);
bool UnregisterPrefab(const string& name);
shared_ptr<Prefab> GetPrefab(const string& name);
bool HasPrefab(const string& name) const;
vector<string> GetPrefabNames() const;
vector<string> SearchByName(const string& filter) const;
vector<string> SearchByTag(const string& tag) const;
int SaveAllPrefabs(SerializationFormat format);
int LoadAllPrefabs();
void Clear();
size_t GetPrefabCount() const;
const string& GetLastError() const;
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Prefab not found** | Ensure it's loaded with `LoadAllPrefabs()` or explicitly registered |
| **Large JSON files** | Use binary format for production |
| **Can't edit prefab** | JSON format is editable, just modify and reload |
| **Instances not syncing** | Use `ApplyToInstance()` to sync from updated prefab |
| **Missing components** | Ensure all components are serializable in SceneSerializer |
| **File not found** | Check path and ensure directory exists |
| **Circular references** | Manually prevent by careful prefab design |

---

## Related Documentation

- [Full Scene Serialization Guide](SCENE_SERIALIZATION_GUIDE.md)
- [Example Code](SCENE_SERIALIZATION_EXAMPLES.cpp)
- GameObject Documentation
- Renderer Documentation
- Material System Documentation
