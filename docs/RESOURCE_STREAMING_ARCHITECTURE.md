# Resource Streaming & Virtual Filesystem - Architecture Diagram

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                                 │
│  (Game Logic, Entity System, Rendering Pipeline, Audio System)             │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                    ┌──────▼──────────┐
                    │ ResourceManager │
                    │  (Centralized)  │
                    └──────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
   │Texture    │    │ Audio       │   │Model        │
   │Manager    │    │ Manager     │   │Manager      │
   └────┬─────┘    └──────┬──────┘   └──────┬──────┘
        │                 │                  │
        └─────────────────┼──────────────────┘
                          │
                ┌─────────▼────────────┐
                │ Resource Streaming   │
                │ Manager              │
                │                      │
                │ - Priority Queue     │
                │ - Worker Threads     │
                │ - Memory Management  │
                │ - LRU Cache          │
                └─────────┬────────────┘
                          │
                ┌─────────▼────────────┐
                │ Virtual File System  │
                │                      │
                │ - Mount Points       │
                │ - Path Resolution    │
                │ - Async I/O          │
                └─────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────────┐
        │                 │                 │              │
   ┌────▼────┐    ┌──────▼──────┐   ┌─────▼──────┐   ┌───▼────┐
   │Physical │    │Asset         │   │Memory      │   │Custom  │
   │File     │    │Package       │   │File System │   │Provider│
   │Provider │    │Provider      │   │Provider    │   │        │
   └────┬────┘    └──────┬──────┘   └─────┬──────┘   └───┬────┘
        │                │                │              │
   ┌────▼────┐    ┌──────▼──────┐   ┌─────▼──────┐
   │./assets │    │game.pak     │   │Temp Buffers│
   │./shaders│    │models.pak   │   │            │
   │./sounds │    │textures.pak │   │            │
   └─────────┘    └─────────────┘   └────────────┘
```

## Resource Loading Pipeline

```
┌─────────────────────┐
│ RequestLoad()       │
│ (High Priority)     │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│ Priority Queue           │
│ ┌──────────────────────┐ │
│ │ Critical    (0)      │ │
│ │ High        (1)  ◄───┼─── New Request Inserted
│ │ Normal      (2)      │ │
│ │ Low         (3)      │ │
│ │ Deferred    (4)      │ │
│ └──────────────────────┘ │
└──────────┬───────────────┘
           │
           ▼ (Worker Threads Pop)
┌──────────────────────────┐
│ Worker Thread Pool       │
│ ┌─ Thread 0 ─────────┐   │
│ │ VFS I/O, Parse     │   │
│ ├─ Thread 1 ─────────┤   │
│ │ VFS I/O, Parse     │   │
│ ├─ Thread 2 ─────────┤   │
│ │ VFS I/O, Parse     │   │
│ └─ Thread 3 ─────────┘   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ OnLoadComplete()         │
│ (Parse & GPU Upload)     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Resource Cache           │
│ ┌──────────────────────┐ │
│ │ Path → Resource Ptr  │ │
│ │ Track Memory Usage   │ │
│ │ Record Access Time   │ │
│ └──────────────────────┘ │
└──────────┬───────────────┘
           │
           ▼ (When Budget Exceeded)
┌──────────────────────────┐
│ LRU Eviction             │
│ Sort by Access Time      │
│ Unload Oldest Unused     │
│ Repeat Until Under Budget│
└──────────────────────────┘
```

## Virtual File System Architecture

```
Application Request:
"vfs.ReadFile("/assets/models/player.gltf")"
           │
           ▼
┌─────────────────────────────────────────┐
│ VirtualFileSystem::FindProvider()       │
│ - Check mount point prefixes            │
│ - Find longest match: "/assets"         │
│ - Return PhysicalFileSystemProvider     │
└────────────┬────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│ Provider::ReadFile()                     │
│ "/assets/models/player.gltf"             │
│         ↓                                │
│ Remove mount point: "/models/player.gltf"│
│         ↓                                │
│ Map to physical: "./assets/models/player"│
│         ↓                                │
│ Open and read file                       │
│         ↓                                │
│ Return data buffer                       │
└──────────────────────────────────────────┘

Mount Point Priority (Higher = Checked First):
┌────────────────────────────┐
│ Mount Order:               │
│ 1. /pak0 (Highest Pr.)    │
│ 2. /assets                │
│ 3. /cache                 │
│ 4. /temp (Lowest Pr.)     │
└────────────────────────────┘
```

## Memory Management Strategy

```
┌──────────────────────────────────────────┐
│ Set Memory Budget: 512 MB                 │
└──────────────────────────────────────────┘
           │
           ▼ Each Frame
┌──────────────────────────────────────────┐
│ Update() - Main Thread                   │
│ ├─ Check current usage                   │
│ ├─ If usage > budget:                    │
│ │  └─ Call ManageMemory()                │
│ └─ Process callbacks & stats             │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ LRU Eviction Algorithm                   │
│                                          │
│ 1. Collect all loaded resources          │
│ 2. Sort by last access time              │
│ 3. Unload oldest until budget OK         │
│                                          │
│ Time: |------|------|------|------>  NOW│
│ Load: |    R1    |  R2  |   R3    |     │
│ LRU:  | Oldest   |      |  Newest |     │
│ Evict:|----------| (Remove first)        │
└──────────────────────────────────────────┘
```

## Asset Package Format

```
Binary Layout (APKG v1):

┌─────────────────────────────────────────┐
│ HEADER (16 bytes)                       │
├─────────────────────────────────────────┤
│ Magic:    "APKG" (0x504B4741)  [4 bytes]│
│ Version:  1                     [4 bytes]│
│ AssetCnt: N                     [4 bytes]│
│ Reserved: 0                     [4 bytes]│
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ DIRECTORY (Variable)                    │
├─────────────────────────────────────────┤
│ For each asset:                         │
│  PathLen        [4 bytes]               │
│  Path           [PathLen bytes]         │
│  DataOffset     [8 bytes]               │
│  CompressedSz   [8 bytes]               │
│  UncompressedSz [8 bytes]               │
│  Compression    [4 bytes] (0=None,1=LZ4)
│  CRC32          [4 bytes]               │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ DATA (Variable)                         │
├─────────────────────────────────────────┤
│ [Compressed/Uncompressed Asset Data]    │
│ [Offsets referenced from DIRECTORY]     │
└─────────────────────────────────────────┘

Example:
┌─────────────────────────────┐
│ Header: 16 bytes            │
│ Dir Entry 1: 98 bytes       │
│ Dir Entry 2: 110 bytes      │
│ Dir Entry 3: 85 bytes       │
│ Offset to Data: 309 bytes   │
├─────────────────────────────┤
│ Asset1: 2,048 KB @ 309      │
│ Asset2: 4,096 KB @ 2,097,433
│ Asset3: 1,024 KB @ 6,194,185
└─────────────────────────────┘
```

## Memory Budget Management

```
Time-based Memory Profile:

Memory
  │     ┌─────── Budget Limit (512 MB)
  │     │
  │   ┌─┘     ┌──────
  │   │       │
  ├──┼───────┼────── Peak Usage
  │  │  LRU  │
  │  │Eviction
  │  │   ↓   │
  │  │   ├───┤  ┌─ Second Eviction
  │  │   │   │  │
  │  │   │   ├──┤
  │  │   │   │
  ├──┴───┴───┴──────── Optimal (80% Budget)
  │
  └──────────────────────► Time (Frames)

Actions:
• < 80% Budget: Load freely
• 80-100% Budget: Monitor, gentle preload
• > 100% Budget: Trigger LRU eviction immediately
```

## Threading Model

```
┌─────────────────────────────────────────┐
│ Main Thread                             │
├─────────────────────────────────────────┤
│ ├─ Game Logic                           │
│ ├─ Render Frame                         │
│ ├─ ResourceStreamingMgr->Update()       │
│ │  ├─ Check worker completion           │
│ │  ├─ Process callbacks                 │
│ │  └─ Manage memory (LRU eviction)      │
│ └─ Next Frame...                        │
└─────────────────────────────────────────┘
           ▲
           │ Queue Resource Requests
           │ (Thread-safe)
           │
┌──────────▼──────────────────────────────┐
│ Worker Thread Pool (4 threads)          │
├─────────────────────────────────────────┤
│ ┌─ Thread 0 ───┐ ┌─ Thread 1 ───┐      │
│ │ Pop from Q    │ │ Pop from Q    │      │
│ │ VFS I/O       │ │ VFS I/O       │      │
│ │ Resource Parse│ │ Resource Parse│      │
│ │ Call Callback │ │ Call Callback │      │
│ └───────────────┘ └───────────────┘      │
│                                         │
│ ┌─ Thread 2 ───┐ ┌─ Thread 3 ───┐      │
│ │ Pop from Q    │ │ Pop from Q    │      │
│ │ VFS I/O       │ │ VFS I/O       │      │
│ │ Resource Parse│ │ Resource Parse│      │
│ │ Call Callback │ │ Call Callback │      │
│ └───────────────┘ └───────────────┘      │
│                                         │
│ Shared Resources (Thread-Safe):         │
│ - std::priority_queue<Request>          │
│ - std::map<path, Resource>              │
│ - Mutex protection on writes            │
└─────────────────────────────────────────┘
```

## Integration Flow

```
┌──────────────────────────────────┐
│ Application::Init()              │
│                                  │
│ 1. Create VirtualFileSystem      │
│ 2. Mount /assets directory       │
│ 3. Create ResourceStreamingMgr   │
│ 4. Initialize worker threads     │
│ 5. Set memory budget             │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Application::LoadScene()         │
│                                  │
│ 1. Preload critical assets       │
│ 2. Create resource instances     │
│ 3. Request load with callbacks   │
│ 4. Return immediately (async)    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Application::Update(deltaTime)   │
│ (Each Frame)                     │
│                                  │
│ 1. ResourceStreamingMgr->Update()│
│ 2. Process completion callbacks  │
│ 3. Handle LRU evictions          │
│ 4. Collect statistics            │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Application::Shutdown()          │
│                                  │
│ 1. ResourceStreamingMgr->UnloadAll()
│ 2. Wait for worker threads       │
│ 3. Cleanup VirtualFileSystem     │
└──────────────────────────────────┘
```

## Performance Characteristics

```
Load Time Distribution:

VFS Overhead:          < 1%
├─ Mount resolution
├─ Path normalization
└─ Callback dispatch

I/O Time:             60-80%
├─ Physical read
├─ Memory copy
└─ Package extraction

Resource Processing: 10-30%
├─ Data parsing
├─ GPU resource creation
└─ Validation

Total Load Latency: 1-100ms (depends on size)
```

## Data Flow for Asset Load

```
User Code:
    CreateResource("player.gltf")
         ▼
Worker Thread:
    ReadFile("/assets/models/player.gltf")
         ▼
    VFS::ReadFile()
         ▼
    PhysicalFileSystemProvider::ReadFile()
         ▼
    std::ifstream reads ./assets/models/player.gltf
         ▼
    Returns std::vector<uint8_t> data
         ▼
Main Thread:
    Resource::OnLoadComplete(data)
         ▼
    Parse glTF binary
    Create meshes
    Upload to GPU
         ▼
    m_State = Loaded
         ▼
User Callback:
    OnLoadComplete([this](bool success) { ... })
         ▼
Game Uses Resource:
    auto model = resource->GetModel()
```

## See Also

- [RESOURCE_STREAMING_GUIDE.md](docs/RESOURCE_STREAMING_GUIDE.md)
- [RESOURCE_STREAMING_QUICK_REFERENCE.md](docs/RESOURCE_STREAMING_QUICK_REFERENCE.md)
- [RESOURCE_STREAMING_DELIVERY_SUMMARY.md](RESOURCE_STREAMING_DELIVERY_SUMMARY.md)
- [RESOURCE_STREAMING_INTEGRATION_CHECKLIST.md](RESOURCE_STREAMING_INTEGRATION_CHECKLIST.md)
