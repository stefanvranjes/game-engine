# Game Engine Copilot Instructions

## Architecture Overview

This is a C++20 OpenGL 3.3+ game engine with deferred rendering, advanced visual effects, and optional multiplayer support.

### Core Systems

**Rendering Pipeline** ([Renderer.h](include/Renderer.h), [Renderer.cpp](src/Renderer.cpp)):
- Deferred rendering with G-Buffer (position, normal, albedo+spec, emissive)
- PBR materials with skeletal animation support
- Occlusion culling via hardware queries with adaptive refresh rates
- Multiple shadow systems: cascaded shadowmaps (directional), cubemap shadows (point lights), spot shadows

**Advanced Post-Processing** ([PostProcessing.h](include/PostProcessing.h)):
- SSAO (Screen Space Ambient Occlusion)
- SSR (Screen Space Reflections)
- TAA (Temporal Anti-Aliasing) with frame history blending
- Volumetric Fog with light scattering
- Bloom extraction and blending

**GameObject System** ([GameObject.h](include/GameObject.h)):
- Scene graph with transform hierarchy and parent-child relationships
- Material-per-object with UV atlasing support (offset/scale for sprite sheets)
- LOD (Level of Detail) system: objects switch meshes/models based on camera distance
- Impostor rendering: generates simplified 3D impostor textures for distant objects

**Animation System** ([Animator.h](include/Animator.h), [Animation.h](include/Animation.h)):
- Skeletal animation with matrix palette skinning (up to 100 bones)
- Animation blending/state transitions
- Animation events: trigger callbacks at specific frames (e.g., footstep SFX)
- Sprite atlas animation with frame sequences and looping

**Particle System** ([ParticleSystem.h](include/ParticleSystem.h), [ParticleEmitter.h](include/ParticleEmitter.h)):
- Dual CPU/GPU compute paths with bitonic sorting for depth-correct rendering
- Physics: gravity, wind, attractors, collisions via spatial grid
- Cloth simulation support
- Trail rendering with history buffers
- Global particle budget (default 100k) with per-emitter limits

**Lighting & Probes** ([Light.h](include/Light.h), [LightProbe.h](include/LightProbe.h)):
- Directional, point, and spot lights (max 32 active)
- Light probes for indirect lighting baking
- Reflection probes for environment capture

**Audio System** ([AudioSystem.h](include/AudioSystem.h), [AudioListener.h](include/AudioListener.h)):
- 3D spatial audio via Miniaudio
- Per-GameObject AudioSource components
- Audio listener updates from camera position

**Multiplayer Layer** ([game-engine-multiplayer/](game-engine-multiplayer/)):
- Server/Client architecture using ENet (UDP-based networking)
- Message serialization framework with binary protocol support
- Peer management with automatic connection state tracking
- Thread-safe message queue and callback handlers
- Built-in ping/pong and keep-alive mechanisms

## Build System & Workflow

**CMake** (C++20, MSVC/Clang compatible):
```
build.bat          # Incremental Debug build to build/Debug/GameEngine.exe
run_gameengine.bat # Executes build/Debug/GameEngine.exe
cmake --build build --config Release  # Release build
```

**Dependencies** (fetched via FetchContent in [CMakeLists.txt](CMakeLists.txt)):
- GLFW 3.3.8 (windowing)
- nlohmann/json 3.11.2 (JSON serialization for complex network payloads)
- tinygltf 2.8.13 (glTF model loading)
- Miniaudio 0.11.21 (audio)
- ENet 7.19 (UDP-based multiplayer networking with congestion control)
- ImGui (editor UI, bundled sources in src/imgui/)
- GoogleTest (optional, for unit tests)

## Networking Integration

The multiplayer layer is now integrated directly into the main GameEngine executable. All networking components are compiled with the engine and accessible from the main app loop.

### NetworkManager ([NetworkManager.hpp](game-engine-multiplayer/include/NetworkManager.hpp))
- **Initialization**: `NetworkManager(Mode::Server/Client, port, max_peers)`
- **Threading**: `initialize()` blocks until `shutdown()` called—run in dedicated thread
- **Message Handling**: `setMessageHandler()` registers callback for received messages
- **Updates**: Call `update()` regularly from game loop to process network events

**Usage Pattern**:
```cpp
// In Application or game component
NetworkManager netMgr(NetworkManager::Mode::Client, 12345);
netMgr.setMessageHandler([this](const Message& msg, uint32_t peer_id) {
    // Handle incoming message from server/peer
    std::cout << "Received: " << msg.getContent() << std::endl;
});
// Start in thread
std::thread netThread([&netMgr]() { netMgr.initialize(); });
netThread.detach();

// In game loop (Application::Update):
netMgr.update(); // Poll events
if (playerMoved) {
    auto state = nlohmann::json({{"pos", {x,y,z}}, {"rot", rotation}});
    Message msg(MessageType::State, state.dump());
    netMgr.sendMessage(msg);
}
```

### Message Protocol ([Message.hpp](game-engine-multiplayer/include/Message.hpp))
- **Format**: `[1 byte type][4 bytes length (uint32 LE)][variable payload]`
- **Types**: Chat, Join, Leave, Ping, State (extensible enum)
- **Serialization**: `Message::serialize()` → `std::vector<uint8_t>`, `deserialize()` ← binary data
- **Content**: String-based payload (use `nlohmann::json` for complex data)

### Server ([Server.hpp](game-engine-multiplayer/include/Server.hpp))
- Accepts connections on specified port, maintains peer list
- `broadcastMessage()` sends to all connected clients
- `Peer` objects track individual client state (socket, last activity, etc.)
- Runs asynchronously—connection/disconnection handled automatically
- Max 32 peers by default (configurable)

### Client ([Client.hpp](game-engine-multiplayer/include/Client.hpp))
- `connectToServer(address, port)` initiates connection
- `sendMessage()` queues outgoing messages
- `receiveMessage()` blocks until message available (use in network thread)
- Automatically handles reconnection attempts via ENet

### Peer Management ([Peer.hpp](game-engine-multiplayer/include/Peer.hpp))
- Represents connected client or server endpoint
- Tracks: socket state, last message time, peer ID, disconnection reason
- `IsConnected()` for connectivity checks
- `GetLastActivityTime()` for timeout detection

## Key Patterns & Conventions

**Smart Pointer Ownership**:
- `std::unique_ptr` for exclusive ownership (Renderer systems, shaders)
- `std::shared_ptr` for shared GameObjects/assets (textures, models, animations)
- Use `enable_shared_from_this` on GameObject to get safe shared_ptr in callbacks

**Shader Hot-Reload** ([Renderer::UpdateShaders()](src/Renderer.cpp#L1089)):
- Call during editor UI when shaders change (ImGui button trigger)
- Recompiles all shader programs from disk

**Deferred Rendering Flow**:
1. **Occlusion Pass**: Issue GPU queries on visible objects, read last frame's results
2. **Geometry Pass**: Render to G-Buffer with skinned meshes (bone matrices in uniform array)
3. **Lighting Pass**: Compute PBR with shadow maps and probes
4. **Post-Processing**: Apply SSAO, SSR, TAA, bloom, volumetric fog in sequence
5. **Transparent**: Forward render with blend modes (particles, text, billboards last)

**Material System**:
- Materials stored in [MaterialLibrary](include/MaterialLibrary.h)
- Set via `GameObject::SetMaterial()` or mesh-embedded defaults
- PBR parameters: albedo, metallic, roughness, normal map

**Billboard Rendering**:
- Detected via LODLevel flag: `isBillboard = true`
- Uses `billboard.vert/frag` shaders
- Rotates quad toward camera in world space
- Supports sprite atlas animation with rows/cols

**Particle GPU Compute**:
- Check `ParticleEmitter::IsGPUComputeAvailable()` before enabling
- Compute shaders in [shaders/](shaders/) with `particle_*` prefix
- Requires texture bindings for position, velocity, collision data

## Critical File References

| Component | Header | Implementation |
|-----------|--------|-----------------|
| Main App Loop | [Application.h](include/Application.h) | [Application.cpp](src/Application.cpp) |
| Rendering | [Renderer.h](include/Renderer.h) | [Renderer.cpp](src/Renderer.cpp) |
| Scene Objects | [GameObject.h](include/GameObject.h) | [GameObject.cpp](src/GameObject.cpp) |
| Skinning | [Animator.h](include/Animator.h) | [Animator.cpp](src/Animator.cpp) |
| Particles | [ParticleSystem.h](include/ParticleSystem.h) | [ParticleSystem.cpp](src/ParticleSystem.cpp) |
| Audio | [AudioSystem.h](include/AudioSystem.h) | [AudioSystem.cpp](src/AudioSystem.cpp) |
| Networking | [NetworkManager.hpp](game-engine-multiplayer/include/NetworkManager.hpp) | [NetworkManager.cpp](game-engine-multiplayer/src/NetworkManager.cpp) |
| Network Messages | [Message.hpp](game-engine-multiplayer/include/Message.hpp) | [Message.cpp](game-engine-multiplayer/src/Message.cpp) |
| Server | [Server.hpp](game-engine-multiplayer/include/Server.hpp) | [Server.cpp](game-engine-multiplayer/src/Server.cpp) |
| Client | [Client.hpp](game-engine-multiplayer/include/Client.hpp) | [Client.cpp](game-engine-multiplayer/src/Client.cpp) |

## Common Tasks

**Implement Networked Game State**:
1. Define message structure (e.g., player position, animation state, action)
2. Serialize via `nlohmann::json`: `json payload = {{"pos", {x,y,z}}, {"anim", anim_id}}`
3. Send as `Message(MessageType::State, payload.dump())`
4. Deserialize in handler: `auto data = json::parse(msg.getContent())`
5. Apply state to GameObjects: `obj->GetTransform().SetPosition(data["pos"])`
6. For frequent updates (100+ Hz), consider binary encoding over JSON

**Add Server-Side Validation**:
- Implement cheat detection in `Server::broadcastMessage()` before relay
- Validate client position changes against movement speed limits
- Store authoritative GameObject state on server, clients request actions
- Example: Client sends "attack" command → server validates target distance → broadcasts result

**Debug Network Issues**:
- Enable ENet verbose logging by setting `enet_verbose_mode = 1`
- Check `Peer::GetLastActivityTime()` to detect timeout/lag spikes
- Log all `Message::serialize()/deserialize()` calls to verify protocol compliance
- Use Wireshark to inspect UDP packets on port (default 12345)

**Add a Post-Processing Effect**:
1. Create shader pair in [shaders/](shaders/)
2. Add class deriving from effect pattern (see [SSAO.h](include/SSAO.h), [SSR.h](include/SSR.h))
3. Register in Renderer and expose via getter/setter
4. Integrate in render pipeline (Renderer::Render() around line 1235)

**Extend Animation System**:
- Add blend parameters to [BlendTree.h](include/BlendTree.h)
- Update [Animator::Update()](src/Animator.cpp) for state machine logic
- Trigger events via `Animation::TriggerEvent()` callback mechanism

**Optimize Particle Performance**:
- Enable GPU compute in emitter setup
- Check global budget with `ParticleSystem::GetTotalActiveParticles()`
- Use spatial grid for collisions (enabled by default)
- Consider CPU vs GPU tradeoff: GPU better for 10k+ particles

**Debug Rendering Issues**:
- Toggle cascaded shadow maps with `SetShowCascades(true)` to visualize
- Check G-Buffer contents in shader (albedo, normal, depth layers)
- Enable wireframe: set cull face mode or use debug geometry pass
- Verify bone matrices uploaded: `max 100` in shaders, check skeletal animation flag in geometry pass

## Shader Conventions

- **Vertex inputs** (layout 0-4): position, texCoord, normal, boneIDs, boneWeights
- **Instance data** (layout 5): instanced model matrices for batched rendering
- **Uniforms**: `u_Model`, `u_MVP`, `u_View`, `u_Projection`, `u_ViewPos` (standard names)
- **G-Buffer layout**: location 0-3 for position, normal, albedo+spec, emissive
- **Include guards**: All headers use `#pragma once`
