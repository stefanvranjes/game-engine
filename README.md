# Game Engine

A modern C++20 OpenGL 3.3+ game engine with deferred rendering, advanced visual effects, skeletal animation, physics simulation, and optional multiplayer support.

## Features

### Rendering Pipeline
- **Deferred Rendering** with G-Buffer (position, normal, albedo+spec, emissive)
- **PBR (Physically-Based Rendering)** with metallic/roughness workflows
- **Skeletal Animation** with matrix palette skinning (up to 100 bones)
- **Occlusion Culling** via hardware queries with adaptive refresh rates
- **Cascaded Shadow Mapping** for directional lights with visualizations
- **Cubemap Shadows** for point lights
- **Multiple Light Types**: Directional, point, and spot lights (up to 32 active)

### Advanced Post-Processing
- **SSAO** (Screen Space Ambient Occlusion)
- **SSR** (Screen Space Reflections)
- **TAA** (Temporal Anti-Aliasing) with frame history blending
- **Volumetric Fog** with light scattering
- **Bloom** with extraction and blending
- **Shader Hot-Reload** for rapid iteration

### Game Systems
- **Scene Graph** with transform hierarchy and parent-child relationships
- **LOD System** for performance optimization (Level of Detail)
- **Impostor Rendering** for distant object optimization
- **Material System** with per-object customization and UV atlasing
- **Billboard Rendering** with sprite atlas animation
- **Light Probes** for indirect lighting baking
- **Reflection Probes** for environment capture

### Animation & Effects
- **Skeletal Animation** with state transitions and blending
- **Animation Events** for frame-triggered callbacks (e.g., footstep SFX)
- **Sprite Atlas Animation** with frame sequences and looping
- **Particle System** with dual CPU/GPU compute paths
- **Particle Physics**: gravity, wind, attractors, collisions via spatial grid
- **Cloth Simulation** support
- **Trail Rendering** with history buffers
- **Global Particle Budget** (default 100k) with per-emitter limits

### Audio System
- **3D Spatial Audio** via Miniaudio
- **Per-GameObject Audio Sources**
- **Audio Listener** with camera-based positioning

### Multiplayer Layer
- **Server/Client Architecture** using ENet (UDP-based networking)
- **Message Serialization** with binary protocol support
- **Peer Management** with automatic connection state tracking
- **Thread-Safe Message Queue** and callback handlers
- **Keep-Alive Mechanisms** and automatic reconnection

### Physics Engine
- **Rigid Body Dynamics** with constraint solving
- **Collision Detection** (box, sphere, capsule, mesh colliders)
- **Joint System** (revolute, prismatic, fixed)
- **Raycasting** and shape casting
- **Spatial Partitioning** for broad-phase optimization

### Tools & Editor
- **ImGui Integration** for in-engine editor and debugging
- **Material Editor** with real-time updates
- **Animation State Machine** visualization and editing
- **Performance Profiler** with GPU/CPU metrics
- **Debug Rendering** for collision shapes and skeletons

## Requirements

- **C++20** compiler (MSVC, Clang)
- **CMake 3.15+**
- **Windows** (primary platform)
- **OpenGL 3.3+** capable GPU
- **Visual Studio 2019+** (for MSVC builds)

## Dependencies

All dependencies are automatically fetched via CMake FetchContent:

| Library | Version | Purpose |
|---------|---------|---------|
| GLFW | 3.3.8 | Windowing and input |
| nlohmann/json | 3.11.2 | JSON serialization |
| tinygltf | 2.8.13 | glTF model loading |
| Miniaudio | 0.11.21 | Audio playback and 3D sound |
| ENet | 7.19 | UDP-based networking |
| ImGui | Latest | Editor UI and debugging |
| GoogleTest | Latest | Unit testing (optional) |

## Getting Started

### Build Instructions

#### Using Batch Scripts (Windows)
```batch
# Build in Debug mode
build.bat

# Run the engine
run_gameengine.bat

# Build with custom options
build_with_options.bat
```

#### Using CMake Directly
```bash
# Debug build
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Debug

# Release build
cmake --build build --config Release

# Run the executable
./build/Debug/GameEngine.exe
```

#### Using PowerShell
```powershell
# Build script available
.\build.ps1
```

### Project Structure

```
game-engine/
├── include/              # Header files for engine components
├── src/                  # Implementation files
├── shaders/              # GLSL shader programs
├── assets/               # Game assets (models, textures, audio)
├── docs/                 # Documentation and guides
├── game-engine-multiplayer/  # Networking components
├── tests/                # Unit tests
└── CMakeLists.txt        # Build configuration
```

## Quick Start Example

### Basic Application Setup

```cpp
#include "Application.h"
#include "GameObject.h"
#include "Camera.h"

int main() {
    // Initialize the application
    Application app(1280, 720, "My Game");
    
    // Create a game object
    auto gameObj = std::make_shared<GameObject>("Player");
    gameObj->GetTransform().SetPosition({0, 0, 5});
    
    // Apply a material
    gameObj->SetMaterial(app.GetRenderer().GetMaterialLibrary().GetMaterial("default"));
    
    // Add to scene
    app.GetScene().AddGameObject(gameObj);
    
    // Main loop
    while (app.IsRunning()) {
        app.Update(1.0f / 60.0f);  // 60 FPS timestep
        app.Render();
    }
    
    return 0;
}
```

### Networking Example

```cpp
#include "game-engine-multiplayer/include/NetworkManager.hpp"
#include "game-engine-multiplayer/include/Message.hpp"

// As a client
NetworkManager netMgr(NetworkManager::Mode::Client, 12345);
netMgr.setMessageHandler([](const Message& msg, uint32_t peer_id) {
    std::cout << "Received: " << msg.getContent() << std::endl;
});

// Start networking in thread
std::thread netThread([&netMgr]() { netMgr.initialize(); });
netThread.detach();

// In game loop
netMgr.update();

// Send state update
nlohmann::json state = {{"pos", {x, y, z}}, {"rotation", rot}};
Message msg(MessageType::State, state.dump());
netMgr.sendMessage(msg);
```

### Particle System Example

```cpp
auto emitter = std::make_shared<ParticleEmitter>(
    ParticleEmitterConfig{
        .emission_rate = 100,
        .lifetime = 2.0f,
        .initial_velocity = glm::vec3(0, 5, 0),
        .max_particles = 1000,
        .use_gpu_compute = true
    }
);

auto particleSystem = gameObj->AddComponent<ParticleSystem>();
particleSystem->AddEmitter(emitter);
```

## Architecture

### Rendering Pipeline Flow

1. **Occlusion Pass**: Issue GPU queries on visible objects
2. **Geometry Pass**: Render to G-Buffer with skinned meshes
3. **Lighting Pass**: Compute PBR with shadow maps and probes
4. **Post-Processing**: SSAO → SSR → TAA → Bloom → Volumetric Fog
5. **Transparent**: Forward render with blend modes (particles, text last)

### Networking Architecture

The multiplayer layer provides a clean abstraction over ENet:

- **Server**: Accepts connections, manages peers, broadcasts state
- **Client**: Connects to server, sends/receives messages
- **Message Protocol**: Binary format with type tagging and length prefixes
- **Thread Model**: Networking runs on dedicated thread, game loop safe

### Physics Engine

- Constraint-based solver for rigid bodies and joints
- Spatial hashing for efficient broad-phase collision detection
- Multiple collider shapes with convex mesh support
- Sleep states for performance optimization

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed setup instructions
- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - System architecture overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/API_OVERVIEW.md](docs/API_OVERVIEW.md) - API reference
- [docs/PHYSICS_INTEGRATION_GUIDE.md](docs/PHYSICS_INTEGRATION_GUIDE.md) - Physics system guide
- [docs/PROFILING_README.md](docs/PROFILING_README.md) - Profiling and optimization
- [game-engine-multiplayer/README.md](game-engine-multiplayer/README.md) - Networking guide

## Performance Considerations

### Optimization Techniques

- **LOD System**: Automatically reduces geometry complexity for distant objects
- **Occlusion Culling**: Skips rendering of occluded objects
- **Impostor Rendering**: Uses simplified impostor textures for very distant objects
- **GPU Particles**: Offload particle simulation to GPU compute shaders
- **Spatial Partitioning**: Efficient broad-phase collision detection
- **Cascaded Shadow Maps**: Better shadow quality at reduced cost
- **Adaptive Refresh Rates**: Occlusion queries refresh less frequently on stable frames

### Profiling

Built-in GPU profiler tracks:
- Frame timing and CPU/GPU frame times
- Render pass durations
- Memory usage
- Draw call counts
- Shader compilation times

## Building & Running Tests

```bash
# Build with tests enabled
cmake -B build -DBUILD_TESTS=ON
cmake --build build --config Debug

# Run tests
ctest --build-config Debug
```

## Common Issues

### Shader Compilation Errors
- Ensure shaders are in the `shaders/` directory
- Check shader syntax with your graphics debugger (RenderDoc recommended)
- Enable hot-reload via ImGui for rapid iteration

### Physics Instability
- Adjust solver iterations and substeps in [PhysicsEngine.h](include/PhysicsEngine.h)
- Monitor constraint violations in profiler
- Ensure proper mass ratios for rigid bodies

### Networking Timeouts
- Check firewall settings for UDP port
- Monitor peer activity with `Peer::GetLastActivityTime()`
- Verify message deserialization with logging

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for our code of conduct and contribution process.

## License

This project is provided as-is. See individual components for specific licensing information.

## Acknowledgments

- Built with modern C++20 features and best practices
- OpenGL community for graphics API insights
- ENet for reliable UDP networking
- Miniaudio for spatial audio support
- ImGui for editor integration
- All open-source dependencies and their maintainers

## Getting Help

- Check the [docs/](docs/) folder for detailed guides
- Review [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for system design
- See [examples/](game-engine-multiplayer/examples/) for code samples
- Open an issue for bugs or feature requests
