# NVRHI Integration - Developer Guide

## 🎯 TL;DR

Phase 1 of NVRHI integration is **COMPLETE**. The graphics abstraction layer is ready for use. Start with **[NVRHI_QUICK_START.md](NVRHI_QUICK_START.md)** for a 5-minute overview.

## 📚 Documentation Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [NVRHI_QUICK_START.md](NVRHI_QUICK_START.md) | Quick overview & examples | 5 min ⭐ |
| [NVRHI_INTEGRATION_GUIDE.md](NVRHI_INTEGRATION_GUIDE.md) | Comprehensive guide | 20 min |
| [NVRHI_PHASE2_PLAN.md](NVRHI_PHASE2_PLAN.md) | Next steps with examples | 15 min |
| [NVRHI_IMPLEMENTATION_CHECKLIST.md](NVRHI_IMPLEMENTATION_CHECKLIST.md) | Progress tracking | 10 min |
| [NVRHI_PHASE1_SUMMARY.md](NVRHI_PHASE1_SUMMARY.md) | What was delivered | 10 min |

## 🚀 Get Started in 3 Steps

### Step 1: Build
```bash
cmake -B build -DGRAPHICS_BACKEND=D3D12
cmake --build build --config Debug
```

### Step 2: Read Quick Start
Open [NVRHI_QUICK_START.md](NVRHI_QUICK_START.md) and review examples

### Step 3: Choose Your Path

**I want to use NVRHI in my code now:**
→ Use helpers in `include/Graphics/GraphicsCompat.h`

**I want to understand the architecture:**
→ Read [NVRHI_INTEGRATION_GUIDE.md](NVRHI_INTEGRATION_GUIDE.md)

**I want to start Phase 2:**
→ Follow [NVRHI_PHASE2_PLAN.md](NVRHI_PHASE2_PLAN.md)

## 📂 What's in the Code

### Header Files (include/Graphics/)
```
GraphicsCommon.h     ← Start here for types/enums
GraphicsDevice.h     ← Main device interface
GraphicsCompat.h     ← OpenGL→NVRHI helpers (easiest migration path!)
NVRHIBackend.h       ← Full NVRHI implementation
```

### Implementation
```
src/NVRHIBackend.cpp ← All the NVRHI wrapper code (2100 lines)
```

## 💡 Minimal Example

```cpp
#include "Graphics/GraphicsInit.h"

// Initialize
auto device = Graphics::InitializeGraphics(
    Graphics::GraphicsBackend::D3D12,
    1920, 1080,
    windowHandle
);

// Create texture
auto texture = device->CreateTexture({
    .width = 512,
    .height = 512,
    .format = Graphics::TextureFormat::R8G8B8A8_UNORM
});

// Create and submit commands
auto cmdList = device->CreateCommandList();
cmdList->Begin();
cmdList->SetViewport({0, 0, 1920, 1080});
cmdList->Draw(3);
cmdList->End();
cmdList->Submit();

// Cleanup
Graphics::ShutdownGraphics();
```

## 🔨 Build Configuration

### Supported Backends
```bash
-DGRAPHICS_BACKEND=D3D12    # Windows, best performance (default)
-DGRAPHICS_BACKEND=D3D11    # Windows legacy support
-DGRAPHICS_BACKEND=VULKAN   # Cross-platform
```

### Verify Build
```bash
cmake -B build -DGRAPHICS_BACKEND=D3D12
cmake --build build --config Debug
# Should complete without errors
```

## 📊 Code Organization

```
include/Graphics/
├── GraphicsCommon.h       Types, enums, descriptors
├── GraphicsResource.h     Abstract resource interfaces
├── GraphicsDevice.h       Device interface
├── NVRHIBackend.h         NVRHI implementations
├── GraphicsInit.h         Easy initialization
└── GraphicsCompat.h       Migration helpers

src/
└── NVRHIBackend.cpp       Full NVRHI implementation

Documentation/
├── NVRHI_DOCUMENTATION_INDEX.md     📍 Navigation hub
├── NVRHI_QUICK_START.md             ⭐ Start here
├── NVRHI_INTEGRATION_GUIDE.md       📖 Comprehensive
├── NVRHI_PHASE2_PLAN.md             🗺️ Next steps
├── NVRHI_IMPLEMENTATION_CHECKLIST.md ✅ Progress
├── NVRHI_PHASE1_SUMMARY.md          📋 Status
└── NVRHI_EXECUTIVE_SUMMARY.md       👔 High level
```

## 🎓 Learning Path

### Beginner (New to NVRHI)
1. Read [NVRHI_QUICK_START.md](NVRHI_QUICK_START.md) (5 min)
2. Review code example above (2 min)
3. Try the minimal example (10 min)

### Intermediate (Integrating into codebase)
1. Review [NVRHI_INTEGRATION_GUIDE.md](NVRHI_INTEGRATION_GUIDE.md) (20 min)
2. Check `include/Graphics/GraphicsCompat.h` for helpers (10 min)
3. Start with TextureManager migration (see Phase 2 plan)

### Advanced (Full integration)
1. Review full [NVRHI_IMPLEMENTATION_CHECKLIST.md](NVRHI_IMPLEMENTATION_CHECKLIST.md) (15 min)
2. Follow [NVRHI_PHASE2_PLAN.md](NVRHI_PHASE2_PLAN.md) step by step (varies)
3. Profile with GPU debuggers (NSight, RenderDoc)

## ⚡ Quick API Reference

### Device Operations
```cpp
auto device = Graphics::GetDevice();

// Create resources
auto texture = device->CreateTexture(textureDesc);
auto buffer = device->CreateBuffer(bufferDesc);
auto shader = device->CreateShader(shaderDesc);

// Get info
uint32_t width = device->GetScreenWidth();
uint64_t memUsed = device->GetGPUMemoryUsed();

// Monitor
float temp = device->GetGPUTemperature();
```

### Command Recording
```cpp
auto cmdList = device->CreateCommandList();

cmdList->Begin();

cmdList->SetViewport({x, y, w, h});
cmdList->SetRenderTargets({colorTargets}, depthTarget);
cmdList->ClearRenderTarget(target, {1, 1, 1, 1});

cmdList->SetPipeline(pipeline);
cmdList->SetConstantBuffer(0, buffer);
cmdList->SetTexture(0, texture);

cmdList->Draw(vertexCount);

cmdList->End();
cmdList->Submit();
```

### Migration Helpers
```cpp
#include "Graphics/GraphicsCompat.h"

// Instead of learning NVRHI details:
auto vb = Graphics::GLToNVRHIAdapter::CreateVertexBuffer(size, data);
auto cb = Graphics::GLToNVRHIAdapter::CreateConstantBuffer(256, nullptr);
auto texture = Graphics::GLToNVRHIAdapter::CreateTexture2D(w, h, fmt, data);

// Or use macros:
auto cb = CREATE_CONSTANT_BUFFER(256, nullptr);
auto size = GET_DEVICE_SCREEN_SIZE(w, h);
```

## 🔍 Debugging

### Enable Graphics Debug
```cpp
auto device = Graphics::InitializeGraphics(backend, width, height, handle);
device->SetDebugName(texture, "MyTexture");
device->SetDebugName(buffer, "MyBuffer");
```

### Profile with GPU Debuggers
```bash
# Windows: NVIDIA NSight
nsight-systems --trace gpu application.exe

# Or use RenderDoc (works with all backends)
renderdoc application.exe
```

### Check GPU Memory
```cpp
auto device = Graphics::GetDevice();
std::cout << "GPU Memory Used: " << device->GetGPUMemoryUsed() << " bytes" << std::endl;
std::cout << "GPU Memory Available: " << device->GetGPUMemoryAvailable() << " bytes" << std::endl;
```

## ❓ Common Questions

**Q: Do I have to use NVRHI for everything?**
A: No! You can migrate incrementally. Use OpenGL and NVRHI together.

**Q: Which backend is fastest?**
A: D3D12 on Windows, Vulkan on Linux. Both are fast.

**Q: Can I switch backends at runtime?**
A: Not yet, but it's planned for Phase 6.

**Q: Is performance impacted by the abstraction?**
A: No, it's negligible. NVRHI is optimized by NVIDIA.

**Q: What about ray tracing?**
A: Foundation is ready for OptiX integration in future phases.

## 🚦 Current Status

### Phase 1: Foundation ✅
- ✅ Graphics abstraction layer
- ✅ NVRHI backend
- ✅ CMake integration
- ✅ Documentation

### Phase 2: Renderer Integration 📋
- 📋 Ready to start
- See [NVRHI_PHASE2_PLAN.md](NVRHI_PHASE2_PLAN.md)

### Phases 3-7: Full Migration 🗺️
- All systems will eventually use NVRHI
- Gradual integration planned

## 📞 Need Help?

1. **Quick Question?** → Check [NVRHI_QUICK_START.md](NVRHI_QUICK_START.md)
2. **Integration Help?** → See [NVRHI_PHASE2_PLAN.md](NVRHI_PHASE2_PLAN.md)
3. **API Details?** → Read header file documentation in `include/Graphics/`
4. **Architecture?** → Read [NVRHI_INTEGRATION_GUIDE.md](NVRHI_INTEGRATION_GUIDE.md)
5. **Official NVRHI?** → https://github.com/NVIDIA-Omniverse/nvrhi

## 📋 Checklist for Your First Integration

- [ ] Read NVRHI_QUICK_START.md
- [ ] Build project with NVRHI (`cmake -B build && cmake --build build`)
- [ ] Review GraphicsCommon.h for available types
- [ ] Review GraphicsCompat.h for migration helpers
- [ ] Try the minimal example above
- [ ] Read NVRHI_PHASE2_PLAN.md for your next step
- [ ] Update relevant code with Graphics API calls

## 🎯 Next Steps

1. **This meeting**: Read NVRHI_QUICK_START.md
2. **This week**: Review documentation, understand architecture
3. **Next week**: Start Phase 2 (Renderer integration)
4. **Following weeks**: Port systems one by one

## 📚 Reference

- **Headers**: `include/Graphics/*.h` (with inline documentation)
- **Implementation**: `src/NVRHIBackend.cpp`
- **Quick Start**: [NVRHI_QUICK_START.md](NVRHI_QUICK_START.md)
- **Full Guide**: [NVRHI_INTEGRATION_GUIDE.md](NVRHI_INTEGRATION_GUIDE.md)
- **Navigation**: [NVRHI_DOCUMENTATION_INDEX.md](NVRHI_DOCUMENTATION_INDEX.md)

---

## Summary

✅ **Phase 1 is complete and production-ready**
- Clean graphics abstraction layer
- Full NVRHI implementation
- Comprehensive documentation
- Ready for Phase 2

🚀 **You're ready to start using NVRHI!**
- Build with NVRHI support
- Use migration helpers for easy integration
- Follow Phase 2 plan for renderer integration
- Migrate systems one at a time

📖 **Start with NVRHI_QUICK_START.md and go from there!**

Good luck! 🎮
