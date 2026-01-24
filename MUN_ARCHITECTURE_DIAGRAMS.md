# Mun System Architecture & Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Game Application                         │
│  (Application.cpp, main game loop)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ Init()
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              MunScriptSystem::GetInstance()                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Initialization                                           │   │
│  │ - Check Mun compiler available                          │   │
│  │ - Get compiler version                                  │   │
│  │ - Set default compilation options                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Script Management                                        │   │
│  │ - LoadScript() / CompileScript()                         │   │
│  │ - Unload / Reload                                        │   │
│  │ - Function caching                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ File Watching                                            │   │
│  │ - WatchScriptFile()                                      │   │
│  │ - WatchScriptDirectory()                                 │   │
│  │ - CheckForChanges() (100ms poll)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Hot-Reload Pipeline                                      │   │
│  │ - Detect file modification                              │   │
│  │ - CompileMunSource()                                     │   │
│  │ - UnloadLibrary()                                        │   │
│  │ - LoadCompiledLibrary()                                  │   │
│  │ - Trigger callbacks                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Statistics & Debugging                                   │   │
│  │ - Compilation stats                                      │   │
│  │ - Error tracking                                         │   │
│  │ - Memory profiling                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Compilation Pipeline

```
                    ┌─────────────────────┐
                    │ .mun Source File    │
                    │ (e.g., gameplay.mun)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ File Changed?       │
                    │ (100ms Poll Check)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────┐
                    │ CompileMunSource()      │
                    │ - Invoke: mun build ... │
                    │ - Options: optimize,    │
                    │   verbose, metadata     │
                    └──────────┬──────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
        ┌───────▼──────────┐      ┌──────────▼─────────┐
        │ Compilation OK?  │      │ Compilation Failed │
        └───────┬──────────┘      └────────────────────┘
                │                         │
                │ YES                     │ NO
                │                    ┌────▼────────────┐
                │                    │ SetError()      │
                │                    │ stats.failed++  │
                │                    │ Report Error    │
                │                    └─────────────────┘
                │
    ┌───────────▼──────────────┐
    │ Generate Output:          │
    │ mun-target/script.dll     │
    │ mun-target/script.dylib   │
    │ mun-target/script.so      │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │ UnloadLibrary()           │
    │ UNLOAD_LIBRARY(old_hdl)   │
    │ Release old binary        │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │ LoadCompiledLibrary()     │
    │ LOAD_LIBRARY(new_path)    │
    │ Get library handle        │
    │ Cache function pointers   │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │ Trigger Callbacks         │
    │ OnScriptReloaded()        │
    │ Update game systems       │
    └───────────┬──────────────┘
                │
    ┌───────────▼──────────────┐
    │ Update Statistics         │
    │ stats.successful++        │
    │ stats.totalReloads++      │
    │ stats.lastCompileTime    │
    └───────────┬──────────────┘
                │
                ▼
        Ready for next frame!
```

## Hot-Reload Timeline

```
TIME  ACTION                              STATUS
────────────────────────────────────────────────────────────
 0ms  ▶ Player edits gameplay.mun        EDITING
      
 1ms  Script in memory buffer

250ms ▶ User presses SAVE                FILE SAVED
      File written to disk

250ms  FileWatcher monitoring             WAITING
       (next check in ~100ms)

350ms  ▶ MunScriptSystem::Update()        DETECTING
       CheckForChanges() runs
       File mtime changed!

351ms  ▶ RecompileAndReload() starts      COMPILING
       Invokes: mun build gameplay.mun
       --output-dir mun-target
       --release

450ms  ▶ mun compiler running             COMPILING
       (typical: 200-300ms for incremental)

550ms  ▶ Compiler finished                COMPILED
       mun-target/gameplay.dll created

551ms  ▶ UnloadLibrary() called           UNLOADING
       FreeLibrary(old_handle)
       Old .dll released

552ms  ▶ LoadCompiledLibrary() called     LOADING
       LoadLibraryA(new_path)
       New .dll loaded into memory

553ms  ▶ OnScriptReloaded() callback      CALLBACK
       Game systems notified
       Can reset state, re-init

554ms  ▶ Hot-reload complete!            READY
       Game uses updated code
       Zero frame rate impact

─────────────────────────────────────────────────────────────
      Total: 304ms from save to active
      Frame Impact: 0ms (no blocking)
```

## File Watching Mechanism

```
┌──────────────────────────────────────────────────┐
│       Watched Files Vector                       │
│  ┌────────────────────────────────────────────┐  │
│  │ scripts/gameplay.mun                       │  │
│  │ scripts/ai.mun                             │  │
│  │ scripts/physics.mun                        │  │
│  └────────────────────────────────────────────┘  │
└────────────────┬─────────────────────────────────┘
                 │
              100ms Poll
                 │
    ┌────────────▼────────────┐
    │ For Each Watched File:  │
    │                         │
    │ ┌─────────────────────┐ │
    │ │ Get Current mtime   │ │
    │ └─────────────────────┘ │
    │         │               │
    │    ┌────▼─────┐         │
    │    │ Compare   │         │
    │    │ to last   │         │
    │    │ mtime     │         │
    │    └────┬──────┘         │
    │         │                │
    │    ┌────┴─────┐          │
    │    │ Changed?  │          │
    │    └────┬──────┘          │
    │         │                │
    │    YES  │  NO            │
    │    ┌────▼───┐  ┌─────┐  │
    │    │Trigger │  │Skip │  │
    │    │Reload  │  └─────┘  │
    │    └────────┘            │
    └────────────────────────┘
             │
      ┌──────▼──────────┐
      │ Schedule Next   │
      │ Check (100ms)   │
      └─────────────────┘
```

## Memory Layout - Loaded Scripts

```
┌─────────────────────────────────────────────────────┐
│           m_loadedScripts Map                       │
│  ┌────────────────────────────────────────────────┐ │
│  │ "gameplay" → LoadedScript                      │ │
│  │   ├─ sourceFile: "scripts/gameplay.mun"        │ │
│  │   ├─ compiledLib: "mun-target/gameplay.dll"    │ │
│  │   ├─ libHandle: 0x7fff1234 (void*)             │ │
│  │   ├─ lastModified: 2024-01-24 15:30:45         │ │
│  │   └─ needsReload: false                         │ │
│  ├────────────────────────────────────────────────┤ │
│  │ "ai" → LoadedScript                            │ │
│  │   ├─ sourceFile: "scripts/ai.mun"              │ │
│  │   ├─ compiledLib: "mun-target/ai.dll"          │ │
│  │   ├─ libHandle: 0x7fff5678 (void*)             │ │
│  │   ├─ lastModified: 2024-01-24 15:25:10         │ │
│  │   └─ needsReload: false                         │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
            ▲                        ▲
            │                        │
      ┌─────┴──────┐         ┌──────┴─────┐
      │ .dll/.so/  │         │ Function   │
      │ .dylib in  │         │ Pointers   │
      │   memory   │         │ Cache      │
      └────────────┘         └────────────┘
```

## Integration Flow in Application

```
┌────────────────────────────────┐
│  Application::Init()           │
└────────────┬───────────────────┘
             │
    ┌────────▼──────────────┐
    │ Create MunScriptSystem │
    │ std::make_unique<>    │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ mun.Init()            │
    │ - Check compiler      │
    │ - Get version         │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ mun.LoadScript()      │
    │ - Compile .mun        │
    │ - Load .dll           │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────┐
    │ mun.WatchScriptDir()  │
    │ - Start file watching │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────────────────┐
    │ Application::Update(deltaTime)     │
    └────────┬───────────────────────────┘
             │
    ┌────────▼──────────────────┐
    │ mun.Update(deltaTime)     │
    │ - CheckForChanges()       │
    │ - Trigger recompile if    │
    │   file changed            │
    └────────┬───────────────────┘
             │ (repeat every frame)
             │
    ┌────────▼──────────────────┐
    │ Application::Shutdown()    │
    └────────┬───────────────────┘
             │
    ┌────────▼──────────────────┐
    │ mun.Shutdown()            │
    │ - Unload all scripts      │
    │ - Cleanup resources       │
    └───────────────────────────┘
```

## Compilation Options Configuration

```
┌────────────────────────────────────────────────┐
│    CompilationOptions struct                   │
│  ┌──────────────────────────────────────────┐  │
│  │ bool optimize = true/false               │  │
│  │   └─ Controls: Release vs Debug mode     │  │
│  │      Debug:   200-500ms per compile      │  │
│  │      Release: 1-3s per compile           │  │
│  │      Performance: Release is 10-50% faster│ │
│  ├──────────────────────────────────────────┤  │
│  │ string targetDir = "mun-target"          │  │
│  │   └─ Output directory for .dll/.so       │  │
│  │      Auto-created if doesn't exist       │  │
│  ├──────────────────────────────────────────┤  │
│  │ bool verbose = true/false                │  │
│  │   └─ Show compiler output in console     │  │
│  │      Useful for debugging                │  │
│  ├──────────────────────────────────────────┤  │
│  │ bool emitMetadata = true                 │  │
│  │   └─ Generate type metadata              │  │
│  │      For runtime reflection (future)     │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  Configuration:                                │
│  MunScriptSystem::CompilationOptions opts;    │
│  opts.optimize = true;                        │
│  opts.targetDir = "output/";                  │
│  mun.SetCompilationOptions(opts);             │
└────────────────────────────────────────────────┘
```

## Error Handling & Recovery

```
                ┌─────────────────────┐
                │ LoadScript()        │
                │ Compilation         │
                └──────────┬──────────┘
                           │
                ┌──────────▼─────────────┐
                │ Successful?            │
                └──────────┬──────────┬──┘
                           │          │
                        YES│          │NO
                           │          │
        ┌──────────────────┘          └──────────────┐
        │                                            │
        ▼                                            ▼
    ┌─────────────┐                          ┌────────────────┐
    │ Return true │                          │ SetError()     │
    │ Continue    │                          │ Store message  │
    └─────────────┘                          └────────┬───────┘
                                                       │
                                            ┌──────────▼──────┐
                                            │ HasErrors() = true
                                            │ GetLastError()  │
                                            │ returns message │
                                            └─────────────────┘
                                                       │
                                            ┌──────────▼──────┐
                                            │ Game can check: │
                                            │ if (mun.HasErr)│
                                            │ { fallback }    │
                                            └─────────────────┘
```

## Performance Profile Example

```
One Gameplay Script Lifecycle
─────────────────────────────────────────────────────

Script: gameplay.mun (200 lines)
Format: Mun source → compiled binary

Timeline:
┌─────────────────────────────────────────────────┐
│ FIRST LOAD                                      │
│                                                 │
│ Compilation:    450ms    ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░ │
│ Library Load:    10ms    ▓░░░░░░░░░░░░░░░░░░░░ │
│ Function Cache:   5ms    ░░░░░░░░░░░░░░░░░░░░ │
│ ─────────────────────────────────────────────  │
│ Total:          465ms                           │
├─────────────────────────────────────────────────┤
│ INCREMENTAL RELOAD (after edit)                │
│                                                 │
│ Compilation:    250ms    ▓▓▓▓▓░░░░░░░░░░░░░░░ │
│ Library Load:    10ms    ▓░░░░░░░░░░░░░░░░░░░ │
│ Function Cache:   5ms    ░░░░░░░░░░░░░░░░░░░░ │
│ ─────────────────────────────────────────────  │
│ Total:          265ms                           │
├─────────────────────────────────────────────────┤
│ RELEASE BUILD (with optimizations)             │
│                                                 │
│ Compilation:   1500ms    ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ │
│ Library Load:    10ms    ▓░░░░░░░░░░░░░░░░░░░ │
│ Function Cache:   5ms    ░░░░░░░░░░░░░░░░░░░░ │
│ ─────────────────────────────────────────────  │
│ Total:         1515ms                           │
├─────────────────────────────────────────────────┤
│ RUNTIME EXECUTION (per function call)          │
│                                                 │
│ Function Call:   <1us   ░░░░░░░░░░░░░░░░░░░░░ │
│ GC Pause:         0us   (None - ownership)    │
│ Memory Overhead:  ~5MB  (Loaded library)      │
│                                                 │
│ ✓ Native code performance                     │
│ ✓ No garbage collection                        │
│ ✓ Predictable frame time                       │
└─────────────────────────────────────────────────┘
```

## Platform Abstraction

```
┌────────────────────────────────────────────┐
│    MunScriptSystem (Platform Abstraction)  │
└──────────────────┬───────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
   ┌──▼──┐    ┌────▼────┐  ┌───▼───┐
   │     │    │         │  │       │
┌──▼──────┐ ┌───▼────────┐ ┌──▼──────┐
│ Windows │ │   macOS    │ │ Linux  │
│         │ │            │ │        │
│ LOAD:   │ │ LOAD:      │ │ LOAD:  │
│ LoadLib │ │ dlopen()   │ │dlopen()│
│         │ │            │ │        │
│ GET:    │ │ GET:       │ │ GET:   │
│GetProc  │ │ dlsym()    │ │ dlsym()│
│         │ │            │ │        │
│ FREE:   │ │ FREE:      │ │ FREE:  │
│FreeLib  │ │ dlclose()  │ │dlclose│
│         │ │            │ │        │
│ .dll    │ │ .dylib     │ │ .so    │
└─────────┘ └────────────┘ └────────┘
```

## Statistics Collection

```
┌──────────────────────────────────────────┐
│    CompilationStats                      │
│                                          │
│  totalCompiles: 15                       │
│  ├─ Count of all compilation attempts   │
│  │                                       │
│  successfulCompiles: 14                  │
│  ├─ Count of successful builds          │
│  │                                       │
│  failedCompiles: 1                       │
│  ├─ Count of failed attempts             │
│  │                                       │
│  totalReloads: 10                        │
│  ├─ Count of successful hot-reloads     │
│  │                                       │
│  totalCompileTime: 3.245 seconds         │
│  ├─ Sum of all compilation times        │
│  │  (Initial + all reloads)             │
│  │                                       │
│  lastCompileTime: 0.215 seconds          │
│  ├─ Duration of most recent compile     │
│  │  Useful for performance monitoring   │
│  │                                       │
│  ResetStats()                            │
│  ├─ Clear all counters                   │
│  │  Useful between development sessions │
│                                          │
└──────────────────────────────────────────┘
```

---

This architecture enables:
- ✅ Fast iteration (250-500ms per edit)
- ✅ No engine restart needed
- ✅ Type-safe compiled code
- ✅ Zero runtime overhead
- ✅ Cross-platform support
- ✅ Robust error handling
