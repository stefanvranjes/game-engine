# PhysX Integration Guide

## Overview

This game engine now supports **two physics backends**:
1. **Bullet3D** (default) - Open-source, CPU-based physics
2. **PhysX 5.x** (NVIDIA) - Industry-standard with GPU acceleration support

Both backends share a common abstraction layer, allowing you to switch between them at compile time.

## Quick Start

### Building with Bullet3D (Default)

```bash
cd c:\Users\Stefan\Documents\GitHub\game-engine
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Building with PhysX

```bash
cd c:\Users\Stefan\Documents\GitHub\game-engine
mkdir build-physx && cd build-physx
cmake .. -DPHYSICS_BACKEND=PHYSX
cmake --build . --config Release
```

> **Note**: First-time PhysX build will download ~500MB from GitHub and may take 10-15 minutes.

## Backend Comparison

| Feature | Bullet3D | PhysX 5.x |
|---------|----------|-----------|
| **Performance (CPU)** | Good | Excellent |
| **GPU Acceleration** | ❌ | ✅ (NVIDIA GPUs) |
| **Memory Usage** | Lower | Higher |
| **Advanced Features** | Basic | Cloth, Particles, Destruction |
| **Visual Debugger** | Limited | PhysX Visual Debugger (PVD) |
| **License** | zlib | BSD-3-Clause |
| **Build Time** | Fast (~2 min) | Slow (~10 min first time) |

## PhysX Visual Debugger (PVD)

PhysX includes a powerful visual debugger for real-time physics inspection.

### Setup PVD

1. Download PhysX Visual Debugger from [NVIDIA Developer](https://developer.nvidia.com/physx-visual-debugger)
2. Run PVD application
3. Start your game engine (it will auto-connect to `localhost:5425`)
4. View real-time physics simulation, collision shapes, and performance metrics

### PVD Features

- Real-time visualization of all physics objects
- Collision shape debugging
- Contact point visualization
- Performance profiling
- Memory usage tracking

## GPU Acceleration (PhysX Only)

PhysX can offload physics simulation to NVIDIA GPUs via CUDA.

### Requirements

- NVIDIA GPU with CUDA support (GTX 600 series or newer)
- CUDA Toolkit installed

### Enabling GPU Acceleration

Currently, GPU acceleration is disabled by default. To enable:

1. Edit `CMakeLists.txt`:
   ```cmake
   set(PX_GENERATE_GPU_PROJECTS ON CACHE BOOL "" FORCE)
   ```

2. Rebuild:
   ```bash
   cmake .. -DPHYSICS_BACKEND=PHYSX
   cmake --build . --config Release
   ```

## API Usage

The abstraction layer provides identical APIs for both backends:

### Creating Rigid Bodies

```cpp
#include "PhysicsSystem.h"
#include "RigidBody.h"
#include "PhysicsCollisionShape.h"

// Create physics system (automatically selects backend)
PhysicsSystem::Get().Initialize({0, -9.81f, 0});

// Create a dynamic box
auto box = std::make_shared<GameObject>("Box");
auto shape = PhysicsCollisionShape::CreateBox(Vec3(0.5f, 0.5f, 0.5f));
auto body = std::make_shared<RigidBody>();
body->Initialize(BodyType::Dynamic, 1.0f, shape);
box->SetRigidBody(body);
```

### Character Controller

```cpp
#include "KinematicController.h"

auto player = std::make_shared<GameObject>("Player");
auto shape = PhysicsCollisionShape::CreateCapsule(0.3f, 1.8f);
auto controller = std::make_shared<KinematicController>();
controller->Initialize(shape, 80.0f, 0.35f);
player->SetKinematicController(controller);

// In update loop
controller->SetWalkDirection({inputX, 0, inputZ} * 5.0f);
if (jumpPressed && controller->IsGrounded()) {
    controller->Jump({0, 10.0f, 0});
}
```

## Performance Tuning

### Bullet3D Optimization

- Use static bodies for fixed geometry
- Enable sleeping for inactive objects
- Reduce collision margin for better accuracy

### PhysX Optimization

- Enable GPU acceleration for large scenes (1000+ bodies)
- Use scene queries for raycasting instead of iterating objects
- Tune `PxSceneDesc` parameters for your use case

## Troubleshooting

### PhysX Build Fails

**Problem**: CMake fails to configure PhysX

**Solution**:
1. Ensure you have Visual Studio 2019 or newer
2. Check internet connection (PhysX downloads from GitHub)
3. Try cleaning build directory: `rm -rf build-physx`

### PVD Not Connecting

**Problem**: PhysX Visual Debugger shows "No connection"

**Solution**:
1. Ensure PVD is running before starting the game
2. Check firewall settings (allow port 5425)
3. Verify PVD is listening on `127.0.0.1:5425`

### Performance Issues with PhysX

**Problem**: PhysX runs slower than Bullet3D

**Solution**:
1. Enable GPU acceleration (if you have NVIDIA GPU)
2. Reduce number of substeps in `Update(deltaTime, substeps)`
3. Use simpler collision shapes (sphere > capsule > box > mesh)

## Migration Guide

### From Bullet3D to PhysX

No code changes required! Just rebuild with `-DPHYSICS_BACKEND=PHYSX`.

### From PhysX to Bullet3D

No code changes required! Just rebuild with `-DPHYSICS_BACKEND=BULLET` or omit the flag.

## Advanced Features

### PhysX-Specific Features

For advanced PhysX features not in the abstraction layer:

```cpp
#ifdef USE_PHYSX
#include "PhysXBackend.h"

auto* backend = static_cast<PhysXBackend*>(PhysicsSystem::Get().GetBackend());
physx::PxScene* scene = backend->GetScene();
// Use PhysX-specific APIs
#endif
```

### Bullet-Specific Features

```cpp
#ifdef USE_BULLET
#include "BulletBackend.h"

auto* backend = static_cast<BulletBackend*>(PhysicsSystem::Get().GetBackend());
btDynamicsWorld* world = backend->GetDynamicsWorld();
// Use Bullet-specific APIs
#endif
```

## Support

- **Bullet3D**: [GitHub Issues](https://github.com/bulletphysics/bullet3/issues)
- **PhysX**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/gameworks/physx/24)
- **Engine Issues**: Create issue in your repository

## Next Steps

1. Try both backends and compare performance
2. Experiment with PhysX Visual Debugger
3. Test GPU acceleration (if available)
4. Profile physics performance in your scenes
5. Explore advanced features (constraints, vehicles, soft bodies)
