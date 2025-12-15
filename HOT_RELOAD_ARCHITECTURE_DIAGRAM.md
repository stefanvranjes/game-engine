# Hot-Reload System - Visual Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   GAME ENGINE EDITOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Application (Main Loop)                 │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │   Update()                                     │ │  │
│  │  │   - m_HotReloadManager->Update()              │ │  │
│  │  │   - Process hot-reload events                 │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                      ▲                               │  │
│  │                      │ calls each frame              │  │
│  │                      │                               │  │
│  └──────────────────────┼───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │         AssetHotReloadManager                        │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  - Initialize(renderer, textureManager)      │   │  │
│  │  │  - SetEnabled(bool)                          │   │  │
│  │  │  - WatchShaderDirectory(path)                │   │  │
│  │  │  - WatchTextureDirectory(path)               │   │  │
│  │  │  - Update()                                  │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │                      │                               │  │
│  │        ┌─────────────┼─────────────┐               │  │
│  │        │             │             │               │  │
│  │        ▼             ▼             ▼               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ │  │
│  │  │FileWatcher  │ │TextureManager│ │Renderer      │ │  │
│  │  │             │ │              │ │              │ │  │
│  │  │• Monitor    │ │• Hot-reload  │ │• UpdateShader│ │  │
│  │  │  files      │ │  textures    │ │  s()         │ │  │
│  │  │• Detect     │ │• Callback    │ │• Shader      │ │  │
│  │  │  changes    │ │  system      │ │  compilation │ │  │
│  │  └─────────────┘ └─────────────┘ └──────────────┘ │  │
│  │        │               │               │           │  │
│  │        │               │               │           │  │
│  │        └───────────────┼───────────────┘           │  │
│  │                        │                           │  │
│  │                        ▼                           │  │
│  │        Detected Changes → Callbacks Invoked        │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ImGui Editor UI - Asset Hot-Reload Panel          │  │
│  │  ├─ Enable/Disable Toggle                          │  │
│  │  ├─ Status: ACTIVE / INACTIVE                      │  │
│  │  ├─ Watched Files: 23                              │  │
│  │  ├─ Reloads: 47                                    │  │
│  │  ├─ Watching:                                      │  │
│  │  │  ├─ shaders/                                    │  │
│  │  │  └─ assets/                                     │  │
│  │  └─ Reset Counter [Button]                         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ▲                                   ▲
         │                                   │
         │ File changes detected         File polling
         │                               (100ms)
         │                                   │
         └──────────────┬────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│             File System (Editor Environment)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  shaders/                                                   │
│  ├─ lighting.frag     ◄── Edit & Save                      │
│  ├─ pbr.frag          ◄── Edit & Save                      │
│  └─ post_process.frag ◄── Edit & Save                      │
│                                                             │
│  assets/                                                    │
│  ├─ textures/                                              │
│  │  ├─ material.png   ◄── Edit in Photoshop               │
│  │  └─ normal.png     ◄── Edit in GIMP                    │
│  └─ models/                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
File Modification
       │
       ▼
FileWatcher::Update()
  ├─ Check file timestamps
  ├─ Compare with last known time
  └─ If changed → invoke callback
       │
       ▼
AssetHotReloadManager callback
  ├─ If shader → OnShaderChanged()
  │   └─ Call Renderer::UpdateShaders()
  │       └─ Each shader calls CheckForUpdates()
  │           └─ Recompile if newer version exists
  │
  └─ If texture → OnTextureChanged()
      └─ Call TextureManager::ReloadTexture()
          └─ Unload old texture
          └─ Queue for async reload
              └─ UploadToGPU() next Update()
                  └─ Material updates automatically
```

## Component Interactions

```
┌──────────────────────┐
│  Application::Init   │
└──────────┬───────────┘
           │
           ├─► Create AssetHotReloadManager
           │
           ├─► Initialize with Renderer + TextureManager
           │
           ├─► SetEnabled(true)
           │
           ├─► WatchShaderDirectory("shaders/")
           │
           └─► WatchTextureDirectory("assets/")

┌──────────────────────┐
│  Application::Run    │
│   (Main Game Loop)   │
└──────────┬───────────┘
           │
           ├─► While(!shouldClose) {
           │     ├─ Update()
           │     │  └─ m_HotReloadManager->Update()
           │     │     ├─ FileWatcher checks files
           │     │     ├─ Invokes callbacks if changed
           │     │     └─ TextureManager processes uploads
           │     │
           │     ├─ Render()
           │     │  └─ Displays updated content
           │     │
           │     └─ RenderEditorUI()
           │        └─ "Asset Hot-Reload" ImGui panel
           │   }

```

## Thread Model

```
┌─────────────────────────────────────────────┐
│         Main Thread (Game Loop)             │
├─────────────────────────────────────────────┤
│                                             │
│  Application::Update() [Each Frame]         │
│  ├─ m_HotReloadManager->Update()            │
│  │  ├─ FileWatcher::Update()                │
│  │  │  └─ Check file timestamps (100ms)    │
│  │  ├─ Invoke callbacks                     │
│  │  └─ TextureManager::Update()             │
│  │     └─ Upload textures to GPU            │
│  ├─ Renderer::Update()                      │
│  └─ Render()                                │
│                                             │
└─────────────────────────────────────────────┘
         │                           │
         │ (Thread-safe via mutex)   │
         │                           │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
    FileWatcher                TextureManager
    (Poll files)               (Async load)
```

## Class Relationships

```
┌──────────────────────────────────────────────────────────┐
│                   Application                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │ - m_HotReloadManager: AssetHotReloadManager        │  │
│  │ - m_Renderer: Renderer                             │  │
│  │ - m_TextureManager: TextureManager                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────────────┘
               │
               │ owns
               ▼
┌──────────────────────────────────────────────────────────┐
│              AssetHotReloadManager                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │ - m_FileWatcher: FileWatcher                       │  │
│  │ - m_Renderer: Renderer*                            │  │
│  │ - m_TextureManager: TextureManager*                │  │
│  │ - m_Enabled: bool                                  │  │
│  │ - m_ReloadCount: uint32_t                          │  │
│  │ - m_OnAssetReloaded: Callback                      │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────────────┘
               │
               │ owns               references
               ▼                         ▼
        ┌────────────────┐      ┌──────────────────┐
        │  FileWatcher   │      │  Renderer        │
        ├────────────────┤      ├──────────────────┤
        │ - m_Watches    │      │ - UpdateShaders()│
        │ - m_WatchesMtx │      └──────────────────┘
        │ - Update()     │
        │ - Callbacks    │
        └────────────────┘      ┌──────────────────┐
                                │ TextureManager   │
                                ├──────────────────┤
                                │ - SetHotReload   │
                                │ - WatchDirectory │
                                │ - ReloadTexture()│
                                └──────────────────┘
```

## State Diagram

```
              Initialize
                 │
                 ▼
        ┌────────────────┐
        │  [Disabled]    │◄──────────────────┐
        └────────┬───────┘                   │
                 │                      SetEnabled(false)
            SetEnabled(true)
                 │
                 ▼
        ┌────────────────┐
        │   [Enabled]    │
        │                │
        │ • Polling      │
        │ • Callbacks    │
        │ • Reloads      │
        └────────────────┘
              │
              │ File change detected
              ▼
        ┌────────────────┐
        │  [Processing]  │
        │                │
        │ • Reload shader│
        │ • Reload tex   │
        │ • Invoke CB    │
        └────┬───────────┘
             │
             ▼
        ┌────────────────┐
        │   [Complete]   │
        │                │
        │ • Update stats │
        │ • Resume       │
        └────────┬───────┘
                 │
                 └─────► [Enabled]
```

## File Type Support

```
Shaders (Watched)                Textures (Watched)
─────────────────                ──────────────────
.glsl ──┐                        .png  ──┐
.vert   ├─► FileWatcher          .jpg    ├─► FileWatcher
.frag   │   (Extension Filter)   .jpeg   │   (Extension Filter)
.geom   │                        .bmp    │
.comp   │                        .tga    │
.tese   │                        .hdr    │
.tesc ──┘                        .exr ──┘

                    │
                    ▼
            Callback Invoked
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    OnShaderChanged      OnTextureChanged
         │                     │
         ▼                     ▼
    Renderer::            TextureManager::
    UpdateShaders()       ReloadTexture()
```

## Performance Model

```
Per-Frame Cost Analysis:

FileWatcher::Update()
├─ Lock mutex: <1μs
├─ Check timestamps: ~100-500μs (100ms interval)
└─ Invoke callbacks: ~10-100μs

TextureManager::Update()
├─ Process pending uploads: Variable (depends on texture size)
└─ Memory management: <1ms

Shader Compilation (when needed)
└─ 5-50ms (depending on complexity)

Total Overhead (when idle):
├─ Polling: ~0.1-2ms per 100ms interval
└─ Memory: <1MB

With Active Reloading:
├─ Shader compilation: 5-50ms one-time
├─ Texture upload: 1-20ms per texture
└─ No FPS impact when idle
```

## Data Structures

```
FileWatcher Internal State:
┌──────────────────────────────┐
│ m_Watches (Handle → Watch)   │
├──────────────────────────────┤
│ 1: path="shaders/pbr.frag"   │
│    lastModified=1702656000   │
│    callback=OnShaderChanged  │
│                              │
│ 2: path="assets/textures/"   │
│    extensionFilter=".png"    │
│    lastModified=1702656100   │
│    callback=OnTextureChanged │
│                              │
│ ... (more watches)           │
└──────────────────────────────┘

AssetHotReloadManager State:
┌──────────────────────────────┐
│ m_Enabled: bool              │
│ m_ReloadCount: uint32_t      │
│ m_Renderer: Renderer*        │
│ m_TextureManager: TexMgr*    │
│ m_FileWatcher: FileWatcher   │
│ m_OnAssetReloaded: Callback  │
└──────────────────────────────┘
```

---

This visual architecture shows how the hot-reload system integrates seamlessly into the game engine's main loop and editor environment.
