# Go Language Support & Concurrent Systems

## Overview

The Game Engine now includes **native Go language support** with full access to Go's powerful concurrency primitives. This enables developers to write high-performance, concurrent game logic using goroutines, channels, and other Go features.

## Why Go for Game Development?

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **Goroutines** | Lightweight concurrency (100k+ in a single program) |
| **Channels** | Type-safe message passing between goroutines |
| **Native Parallelism** | Automatic multi-core utilization via runtime scheduler |
| **Simple Syntax** | Clean, readable code with fast compilation |
| **Memory Efficiency** | Garbage-collected with predictable performance |
| **Compiled Performance** | Near-C++ speeds with easier development |
| **Cross-Platform** | Single codebase runs on Windows, Linux, macOS |

### Ideal Use Cases

1. **NPC AI Systems** - Multiple independent behavior trees running concurrently
2. **Network Synchronization** - Concurrent player state updates and replication
3. **Physics Processing** - Parallel physics simulations per actor/region
4. **Asset Loading** - Asynchronous streaming and texture loading
5. **Data Analytics** - Real-time telemetry and game metrics
6. **Server Architecture** - Multiplayer server backends

## Integration Architecture

```
Game Engine (C++20)
    ↓
ScriptLanguageRegistry
    ↓
GoScriptSystem (C++ wrapper)
    ↓ [cgo FFI]
    ↓
Go Runtime Library
    ├─ Goroutines (M:N threading)
    ├─ Channels (typed message passing)
    ├─ Game Engine Bindings
    └─ Concurrent Game Systems
```

## Getting Started

### 1. Creating a Go Script

Create a `.go` file in your scripts directory:

```go
// scripts/npc_behavior.go
package main

import "fmt"

// ExportedFunc - uppercase means it's callable from C++
func UpdateNPCBehavior(npcID int32, deltaTime float32) {
    fmt.Printf("NPC %d updating (dt: %f)\n", npcID, deltaTime)
}

func ConcurrentUpdate() {
    // Create goroutines for parallel NPC updates
    done := make(chan bool)
    
    for i := 0; i < 10; i++ {
        go func(id int) {
            // Concurrent work here
            fmt.Printf("NPC %d processing...\n", id)
            done <- true
        }(i)
    }
    
    // Wait for all to complete
    for i := 0; i < 10; i++ {
        <-done
    }
}
```

### 2. Compile to Shared Library

```bash
# Windows
go build -o scripts/npc_behavior.dll -buildmode=c-shared scripts/npc_behavior.go

# Linux
go build -o scripts/npc_behavior.so -buildmode=c-shared scripts/npc_behavior.go

# macOS
go build -o scripts/npc_behavior.dylib -buildmode=c-shared scripts/npc_behavior.go
```

### 3. Load and Use in C++

```cpp
auto& registry = ScriptLanguageRegistry::GetInstance();

// Load Go script (auto-detected by .go extension)
registry.ExecuteScript("scripts/npc_behavior.go");

// Call Go function synchronously
auto system = dynamic_cast<GoScriptSystem*>(registry.GetScriptSystem(ScriptLanguage::Go));
system->CallFunction("UpdateNPCBehavior", {npcID, deltaTime});

// Or spawn concurrent goroutine
int gid = system->StartGoroutine("ConcurrentUpdate");
system->WaitGoroutine(gid); // Wait for completion
```

## Core Concepts

### Goroutines - Lightweight Concurrency

Goroutines are Go's lightweight threads, managed by the Go runtime scheduler. They're far cheaper than OS threads:

```go
package main

func ConcurrentPhysicsUpdate(actorCount int32) {
    // Spawn a goroutine for each actor group
    for groupID := 0; groupID < 4; groupID++ {
        go updateActorGroup(groupID, actorCount/4)
    }
}

func updateActorGroup(groupID int, count int32) {
    // Each group updates its actors in parallel
    for i := int32(0); i < count; i++ {
        updateActor(i)
    }
}
```

**From C++:**
```cpp
GoScriptSystem* goSystem = dynamic_cast<GoScriptSystem*>(
    registry.GetScriptSystem(ScriptLanguage::Go)
);

// Start concurrent physics update
int goroutineID = goSystem->StartGoroutine("ConcurrentPhysicsUpdate", nullptr);

// Don't wait - continue game loop
// Physics updates happen in background
```

### Channels - Message Passing

Channels enable safe communication between goroutines without shared memory:

```go
package main

func PlayerMovementController() {
    // Create channels for communication
    commands := make(chan PlayerCommand, 100) // Buffered channel
    results := make(chan Vector3)
    
    // Spawn worker goroutines
    for i := 0; i < 4; i++ {
        go moveProcessor(i, commands, results)
    }
    
    // Send commands
    commands <- PlayerCommand{action: "move_forward", value: 5.0}
    
    // Receive results
    newPos := <-results
}

func moveProcessor(id int, commands chan PlayerCommand, results chan Vector3) {
    for cmd := range commands {
        // Process movement command
        newPos := processMovement(cmd)
        results <- newPos
    }
}
```

**From C++:**
```cpp
// Create channel for player commands
goSystem->CreateChannel("player_commands");

// Send movement data via channel
std::string cmdJson = R"({"action": "move", "x": 10.0, "y": 0.0})";
goSystem->SendToChannel("player_commands", cmdJson);

// Later, receive processed result
std::string result = goSystem->ReceiveFromChannel("player_result", 100); // 100ms timeout
```

### WaitGroups - Synchronization

WaitGroups synchronize multiple goroutines:

```go
package main

import "sync"

func ParallelAssetLoading(assetCount int32) {
    var wg sync.WaitGroup
    
    // Add tasks to wait group
    wg.Add(int(assetCount))
    
    // Launch loading goroutines
    for i := int32(0); i < assetCount; i++ {
        go func(id int32) {
            defer wg.Done() // Signal completion
            LoadAsset(id)
        }(i)
    }
    
    // Wait for all to complete
    wg.Wait()
}

func LoadAsset(id int32) {
    // Load asset concurrently
    // ...
}
```

**From C++:**
```cpp
// Create wait group
goSystem->CreateWaitGroup("asset_load", 10); // 10 assets

// Call Go function that spawns goroutines
goSystem->StartGoroutine("ParallelAssetLoading", nullptr);

// Wait for all assets to load (with timeout)
int result = goSystem->WaitGoroutine(goroutineID, 5000); // 5 second timeout
```

## Game Systems Examples

### 1. Concurrent NPC AI System

```go
package main

import "fmt"

// NPCBehaviorTree - one running per NPC
func NPCBehaviorTree(npcID int32) {
    stateChannel := make(chan string, 10)
    
    // Behavior goroutines
    go patrol(npcID, stateChannel)
    go detectPlayer(npcID, stateChannel)
    go updateAnimation(npcID, stateChannel)
    
    // State machine
    for state := range stateChannel {
        fmt.Printf("NPC %d state: %s\n", npcID, state)
    }
}

func patrol(npcID int32, states chan string) {
    for {
        // Move along patrol path
        states <- "patrolling"
    }
}

func detectPlayer(npcID int32, states chan string) {
    // Periodically check for player
    states <- "attacking"
}

func updateAnimation(npcID int32, states chan string) {
    // Update animation state
}
```

**Usage:**
```cpp
// Start NPC behavior trees concurrently
std::vector<int> npcGoroutines;
for (int npcID = 0; npcID < 50; npcID++) {
    int gid = goSystem->StartGoroutine("NPCBehaviorTree", (void*)(intptr_t)npcID);
    npcGoroutines.push_back(gid);
}

// Game loop
while (running) {
    // Update C++ systems...
    
    // Go goroutines update in parallel
    registry.Update(deltaTime);
}
```

### 2. Network Replication System

```go
package main

import "encoding/json"

type PlayerState struct {
    Position [3]float32 `json:"position"`
    Rotation [4]float32 `json:"rotation"`
    Animation string     `json:"animation"`
}

func ReplicatePlayerState(playerID int32) {
    serverChannel := make(chan PlayerState)
    clientChannel := make(chan PlayerState)
    
    // Separate goroutines for send/receive
    go sendPlayerState(playerID, serverChannel)
    go receivePlayerState(playerID, clientChannel)
    
    // Sync with other systems
}

func sendPlayerState(playerID int32, ch chan PlayerState) {
    for state := range ch {
        data, _ := json.Marshal(state)
        // Send to network manager
    }
}

func receivePlayerState(playerID int32, ch chan PlayerState) {
    // Listen for network updates
    // Parse and send to channel
}
```

### 3. Physics Processing

```go
package main

func ParallelPhysicsUpdate(deltaTime float32) {
    physicsChannel := make(chan PhysicsActor, 1000)
    resultChannel := make(chan PhysicsResult)
    
    // Spawn worker goroutines (one per CPU core)
    numWorkers := runtime.NumCPU()
    for i := 0; i < numWorkers; i++ {
        go physicsWorker(physicsChannel, resultChannel)
    }
}

func physicsWorker(in chan PhysicsActor, out chan PhysicsResult) {
    for actor := range in {
        // Update actor physics
        result := UpdatePhysics(actor)
        out <- result
    }
}
```

## API Reference

### GoScriptSystem Class

#### Lifecycle
```cpp
void Init();                           // Initialize Go runtime
void Shutdown();                       // Cleanup
void Update(float deltaTime);          // Call every frame
```

#### Script Execution
```cpp
bool RunScript(const std::string& filepath);        // Load .go file
bool ExecuteString(const std::string& source);      // Execute Go code
```

#### Goroutine Management
```cpp
int StartGoroutine(const std::string& functionName, void* userData = nullptr);
int WaitGoroutine(int goroutineId);
int KillGoroutine(int goroutineId);
const Goroutine* GetGoroutineStatus(int goroutineId) const;
size_t GetActiveGoroutineCount() const;
```

#### Channel Operations
```cpp
int CreateChannel(const std::string& channelName);
int SendToChannel(const std::string& channelName, const std::string& jsonData);
std::string ReceiveFromChannel(const std::string& channelName, int timeoutMs = 0);
int CloseChannel(const std::string& channelName);
```

#### Function Calling
```cpp
std::any CallFunction(const std::string& functionName, 
                     const std::vector<std::any>& args);
```

#### Diagnostics
```cpp
uint64_t GetMemoryUsage() const;
double GetLastExecutionTime() const;
bool HasErrors() const;
std::string GetLastError() const;
```

### GoRuntime Helper Class

For lower-level access to Go runtime:

```cpp
class GoRuntime {
    bool Initialize();
    void Shutdown();
    
    int SpawnGoroutine(const std::string& functionName);
    int CreateChannel(const std::string& channelName, size_t bufferSize);
    int CreateWaitGroup(const std::string& groupName, int initialCount);
    
    std::string CallGoFunction(const std::string& functionName, 
                              const std::string& parameters);
};
```

## Best Practices

### 1. Goroutine Naming Convention

```go
// Prefix with system name for clarity
func NPC_BehaviorUpdate(npcID int32) { }
func Physics_ActorUpdate(actorID int32) { }
func Network_PlayerSync(playerID int32) { }
func Audio_SoundProcessor() { }
```

### 2. Error Handling in Goroutines

```go
func SafeGoroutine(name string, fn func() error) {
    defer func() {
        if r := recover(); r != nil {
            fmt.Printf("Goroutine %s panicked: %v\n", name, r)
        }
    }()
    
    if err := fn(); err != nil {
        fmt.Printf("Goroutine %s error: %v\n", name, err)
    }
}
```

### 3. Timeout Handling

```go
func ConcurrentOperationWithTimeout(timeoutMs int64) {
    done := make(chan bool)
    timer := time.NewTimer(time.Duration(timeoutMs) * time.Millisecond)
    
    go func() {
        // Do work
        done <- true
    }()
    
    select {
    case <-done:
        fmt.Println("Completed")
    case <-timer.C:
        fmt.Println("Timeout!")
    }
}
```

### 4. Resource Cleanup

```go
func GoroutineWithCleanup(resourceID int32) {
    defer func() {
        // Always cleanup resources
        FreeResource(resourceID)
    }()
    
    // Use resource
    UseResource(resourceID)
}
```

### 5. Channel Patterns

```go
// Worker pool pattern
func WorkerPool(jobCount int) {
    jobs := make(chan Job, jobCount)
    results := make(chan Result, jobCount)
    
    // Launch workers
    for i := 0; i < 4; i++ {
        go worker(jobs, results)
    }
    
    // Distribute jobs
    for job := range jobs {
        results <- processJob(job)
    }
}

// Fan-out/Fan-in pattern
func FanOutFanIn(input chan Data) chan Result {
    workers := 4
    results := make(chan Result, workers)
    
    // Fan-out
    for i := 0; i < workers; i++ {
        go func(id int) {
            for data := range input {
                results <- process(data)
            }
        }(i)
    }
    
    return results
}
```

## Performance Considerations

### Memory Usage

- **Per Goroutine**: ~2 KB (vs ~1-8 MB for OS thread)
- **Per Channel**: ~48 bytes base + element size
- **Total Overhead**: Typically <50 MB for 1000+ concurrent operations

### Scheduling

```
Go Runtime
├─ M - OS Threads (usually # CPU cores)
├─ P - Processors (max GOMAXPROCS)
└─ G - Goroutines (1000s+)
```

The Go scheduler is work-stealing and handles goroutine preemption automatically.

### Optimization Tips

1. **Use buffered channels** when throughput matters
2. **Limit goroutine count** for NPC behaviors (use worker pools)
3. **Profile with pprof** for hot spots
4. **Close channels when done** to signal workers
5. **Avoid goroutine leaks** with timeouts/cancellation

## Troubleshooting

### Issue: Go runtime crashes on shutdown

**Solution**: Ensure all goroutines complete before shutdown
```cpp
goSystem->Update(0.0f); // Process completion events
std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

### Issue: Channel deadlock

**Solution**: Always have a receiver or use timeout
```go
select {
case data := <-channel:
    // Process
case <-time.After(1 * time.Second):
    // Timeout
}
```

### Issue: High memory usage

**Solution**: Limit goroutine count and clean up
```cpp
if (goSystem->GetActiveGoroutineCount() > 1000) {
    // Wait for some to complete
    goSystem->Update(0.01f);
}
```

## Integration with Engine Systems

### With Audio System

```go
func PlaySoundConcurrent(soundID int32, worldPos [3]float32) {
    go func() {
        // Load and play sound in background
        AudioEngine.Play(soundID, worldPos)
    }()
}
```

### With Rendering System

```go
func ConcurrentImpostorGeneration(modelID int32) {
    results := make(chan ImpostorTexture, 1)
    
    go func() {
        impostor := RenderEngine.GenerateImpostor(modelID)
        results <- impostor
    }()
    
    // Continue rendering while impostor generates
}
```

### With Physics System

```go
func ParallelRaycasts(rayCount int32) {
    resultChannel := make(chan RaycastResult, rayCount)
    
    for i := int32(0); i < rayCount; i++ {
        go func(id int32) {
            result := Physics.Raycast(rays[id])
            resultChannel <- result
        }(i)
    }
}
```

## File Extensions and Auto-Detection

- `.go` - Go source files (auto-detected)
- `.dll`, `.so`, `.dylib` - Compiled Go libraries

The engine automatically detects Go files by extension and routes to GoScriptSystem.

## Compilation and Distribution

### Single-File Distribution

Compile Go scripts to native binaries for distribution:

```bash
# Compile to standalone executable
go build -o game_npc_ai scripts/npc_ai.go

# Or shared library (recommended)
go build -o libgame_npc_ai.so -buildmode=c-shared scripts/npc_ai.go
```

### Linking Against Game Engine Libraries

Use cgo to link against engine libraries:

```go
package main

//#cgo LDFLAGS: -L. -lengine_core
//#include "engine_core.h"
import "C"

func UpdateGameState() {
    C.EngineUpdateFrame()
}
```

## Comparison with Other Languages

| Feature | Go | Lua | Python | Rust |
|---------|----|----|--------|------|
| Native Concurrency | ✅ Goroutines | ❌ Coroutines | ❌ GIL | ✅ Native |
| Learning Curve | Easy | Easy | Easy | Hard |
| Performance | Fast | Medium | Slow | Very Fast |
| Hot-Reload | ❌ Recompile | ✅ | ✅ | ❌ Recompile |
| Memory | Efficient | Very Low | Large | Variable |
| Ideal for | Concurrent systems | Game logic | AI/Data | Performance-critical |

## Further Resources

- [Go Documentation](https://golang.org/doc/)
- [Concurrency Patterns](https://go.dev/blog/pipelines)
- [Effective Go](https://golang.org/doc/effective_go)

## Summary

Go language support brings powerful native concurrency to the engine, enabling:

✅ Concurrent NPC AI with 50+ independent behavior trees  
✅ Parallel physics processing across multiple cores  
✅ Non-blocking network replication  
✅ Background asset loading and streaming  
✅ Scalable server backends for multiplayer  

Use Go for systems that benefit from lightweight concurrency and clean syntax!
