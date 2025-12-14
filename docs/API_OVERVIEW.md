# Game Engine API Documentation

Welcome to the Game Engine API documentation. This reference guide covers all public APIs and systems.

## Quick Navigation

### Core Systems
- **[Renderer](namespaceGameEngine.html#renderer-system)** - Deferred rendering pipeline with PBR materials
- **[GameObject](classGameEngine_1_1GameObject.html)** - Scene graph and entity component system
- **[Animator](classGameEngine_1_1Animator.html)** - Skeletal animation with state machines
- **[ParticleSystem](classGameEngine_1_1ParticleSystem.html)** - Dual CPU/GPU particle simulation
- **[AudioSystem](classGameEngine_1_1AudioSystem.html)** - 3D spatial audio via Miniaudio

### Graphics & Effects
- **[PostProcessing](classGameEngine_1_1PostProcessing.html)** - SSAO, SSR, TAA, Bloom, Volumetric Fog
- **[Light](classGameEngine_1_1Light.html)** - Directional, point, and spot lights
- **[Material](classGameEngine_1_1MaterialLibrary.html)** - PBR material system
- **[Camera](classGameEngine_1_1Camera.html)** - View and projection management

### Physics & Collision
- **[CollisionShape](classGameEngine_1_1CollisionShape.html)** - Physics collision volumes
- **[RigidBody](classGameEngine_1_1RigidBody.html)** - Dynamic and kinematic bodies
- **[Physics](namespaceGameEngine.html#physics)** - Physics engine core

### Multiplayer & Networking
- **[NetworkManager](classGameEngine_1_1NetworkManager.html)** - Client/Server networking
- **[Message](classGameEngine_1_1Message.html)** - Network message protocol
- **[Peer](classGameEngine_1_1Peer.html)** - Connection management

### Utilities
- **[Transform](classGameEngine_1_1Transform.html)** - Position, rotation, scale hierarchy
- **[Frustum](classGameEngine_1_1Frustum.html)** - Camera frustum culling
- **[Profiler](classGameEngine_1_1Profiler.html)** - Performance profiling and telemetry

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Application                             │
│                   (Main Game Loop)                           │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      ┌─────▼──────┐  ┌────▼────┐  ┌─────▼──────┐
      │  Renderer  │  │ Animator │  │   Audio    │
      │ (Deferred) │  │ (Skeletal)  │  System   │
      └─────┬──────┘  └────┬────┘  └─────┬──────┘
            │              │             │
      ┌─────┴──────────────┴─────────────┴─────┐
      │        Scene Graph (GameObjects)       │
      │    - Transform Hierarchy               │
      │    - Component System                  │
      │    - Culling & LOD                     │
      └──────────────────────────────────────┬─┘
            │
      ┌─────▼─────────────────────────────────┐
      │    Rendering Pipeline                 │
      │  ┌─────────────────────────────────┐  │
      │  │ 1. Occlusion Culling (GPU)      │  │
      │  │ 2. Geometry Pass (G-Buffer)     │  │
      │  │ 3. Lighting Pass (PBR)          │  │
      │  │ 4. Post-Processing Effects      │  │
      │  │ 5. Transparent Objects Forward  │  │
      │  │ 6. UI Overlay (ImGui)           │  │
      │  └─────────────────────────────────┘  │
      └──────────────────────────────────────┘
```

## Module Organization

### Rendering Module
Handles all graphics rendering, effects, and screen output.

**Key Classes:**
- `Renderer` - Main rendering pipeline coordinator
- `PostProcessing` - Post-process effect chains
- `GBuffer` - Deferred rendering target
- `Light` - Light source definition
- `Material` - Surface properties and shaders
- `Shader` - GLSL shader program management

**Key Files:**
- [include/Renderer.h](../../include/Renderer.h)
- [src/Renderer.cpp](../../src/Renderer.cpp)
- [include/PostProcessing.h](../../include/PostProcessing.h)

### Scene Module
Scene graph and entity management.

**Key Classes:**
- `GameObject` - Scene entities with components
- `Transform` - Position, rotation, scale hierarchy
- `Component` - Base for all game components
- `Scene` - Collection of GameObjects

**Key Files:**
- [include/GameObject.h](../../include/GameObject.h)
- [include/Transform.h](../../include/Transform.h)

### Animation Module
Skeletal and sprite animation systems.

**Key Classes:**
- `Animator` - Animation state machine
- `Animation` - Frame-based animation clips
- `AnimationStateMachine` - State transitions
- `Bone` - Skeletal hierarchy
- `BlendTree` - Advanced animation blending

**Key Files:**
- [include/Animator.h](../../include/Animator.h)
- [include/Animation.h](../../include/Animation.h)
- [include/BlendTree.h](../../include/BlendTree.h)

### Particle Module
CPU and GPU particle simulation.

**Key Classes:**
- `ParticleSystem` - Main particle coordinator
- `ParticleEmitter` - Particle spawning and properties
- `ParticleData` - Per-particle attributes
- `ParticlePhysics` - Physics simulation (gravity, wind, collisions)

**Key Files:**
- [include/ParticleSystem.h](../../include/ParticleSystem.h)
- [include/ParticleEmitter.h](../../include/ParticleEmitter.h)

### Physics Module
Collision detection and rigid body dynamics.

**Key Classes:**
- `RigidBody` - Dynamic and kinematic objects
- `CollisionShape` - Collision primitives (box, sphere, capsule, mesh)
- `Physics` - Physics engine coordinator
- `PhysicsDebugDraw` - Visualization

**Key Files:**
- [include/RigidBody.h](../../include/RigidBody.h)
- [include/CollisionShape.h](../../include/CollisionShape.h)

### Audio Module
3D spatial audio system.

**Key Classes:**
- `AudioSystem` - Audio engine coordinator
- `AudioSource` - Sound playback on GameObjects
- `AudioListener` - Listener position and orientation
- `AudioClip` - Audio asset wrapper

**Key Files:**
- [include/AudioSystem.h](../../include/AudioSystem.h)
- [include/AudioSource.h](../../include/AudioSource.h)

### Networking Module
Client/Server multiplayer support.

**Key Classes:**
- `NetworkManager` - Connection management
- `Message` - Message serialization protocol
- `Peer` - Remote peer representation
- `Server` - Server endpoint
- `Client` - Client endpoint

**Key Files:**
- [game-engine-multiplayer/include/NetworkManager.hpp](../../game-engine-multiplayer/include/NetworkManager.hpp)
- [game-engine-multiplayer/include/Message.hpp](../../game-engine-multiplayer/include/Message.hpp)

## Common Tasks

### Creating a GameObject

```cpp
#include "Application.h"

// In your game code
GameObject* player = scene->CreateGameObject("Player");
player->GetTransform().SetPosition(glm::vec3(0, 1, 0));
player->AddComponent<AnimatorComponent>();
player->AddComponent<RigidBodyComponent>();
```

### Setting Up Rendering

```cpp
#include "Renderer.h"

Renderer renderer(window_width, window_height);
renderer.SetClearColor(glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
renderer.EnableShadows(true);
renderer.EnableSSAO(true);
renderer.EnableSSR(true);
```

### Playing Animation

```cpp
#include "Animator.h"

auto animator = gameObject->GetComponent<Animator>();
if (animator) {
    animator->SetState("Walk");
    animator->SetBlendValue("Speed", 1.0f);
}
```

### Creating Particles

```cpp
#include "ParticleSystem.h"

auto emitter = particleSystem->CreateEmitter();
emitter->SetEmissionRate(100);
emitter->SetLifetime(2.0f);
emitter->SetVelocity(glm::vec3(0, 2, 0));
emitter->Emit(50);
```

### Network Communication

```cpp
#include "NetworkManager.hpp"

NetworkManager netMgr(NetworkManager::Mode::Client, 12345);
netMgr.setMessageHandler([](const Message& msg, uint32_t peer_id) {
    auto data = json::parse(msg.getContent());
    // Handle message
});

Message msg(MessageType::State, "{\"health\": 100}");
netMgr.sendMessage(msg);
```

## Performance Tips

1. **Use LOD** - Set up Level of Detail to reduce geometry for distant objects
2. **Enable Occlusion Culling** - Automatically skips rendering of occluded objects
3. **Batch Particles** - Combine emitters for better GPU performance
4. **Use Impostor Rendering** - For distant complex objects
5. **Profile First** - Use Profiler class to identify bottlenecks
6. **GPU Compute** - Enable for particle systems with 10k+ particles

## Debugging

Enable debug modes:

```cpp
renderer.SetShowCascades(true);      // Visualize shadow cascades
renderer.SetShowOcclusion(true);     // Visualize occlusion culling
renderer.SetWireframeMode(true);     // Wireframe rendering
physics.SetDebugDraw(true);          // Physics collision shapes
```

## API Stability

- **Stable APIs** - Core rendering, scene management, animation
- **In Progress** - Networking optimizations, physics improvements
- **Experimental** - Advanced post-processing effects, machine learning integration

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the engine.

## License

Refer to the project LICENSE file for terms and conditions.

---

**Last Updated:** 2025  
**API Version:** 1.0  
**Engine Version:** C++20, OpenGL 3.3+
