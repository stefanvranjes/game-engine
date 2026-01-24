# Go Language Integration - Quick Start

## 5-Minute Setup

### 1. Install Go

**Windows**:
```bash
# Download from https://golang.org/dl/
# Or use Chocolatey:
choco install golang
```

**Linux**:
```bash
sudo apt-get install golang-go
```

**macOS**:
```bash
brew install go
```

Verify:
```bash
go version  # Should be 1.21+
```

### 2. Build Go Scripts

**Windows**:
```cmd
build_go_scripts.bat
```

**Linux/macOS**:
```bash
chmod +x build_go_scripts.sh
./build_go_scripts.sh
```

This creates `.dll` (Windows), `.so` (Linux), or `.dylib` (macOS) files in `scripts/build/`.

### 3. Load in Engine

```cpp
#include "ScriptLanguageRegistry.h"
#include "GoScriptSystem.h"

// Initialize
auto& registry = ScriptLanguageRegistry::GetInstance();
registry.Init();

// Load Go script (auto-detected by .go extension)
registry.ExecuteScript("scripts/example_go_systems.go");

// Get Go system
auto goSystem = dynamic_cast<GoScriptSystem*>(
    registry.GetScriptSystem(ScriptLanguage::Go)
);

// Spawn concurrent NPC behavior
int gid = goSystem->StartGoroutine("ExampleNPCBehavior", (void*)1);

// Update each frame
while (running) {
    registry.Update(deltaTime);
}
```

## Example Use Cases

### Concurrent NPC AI

```cpp
// Spawn 50 NPCs running AI concurrently
std::vector<int> npcGoroutines;
for (int i = 0; i < 50; i++) {
    int gid = goSystem->StartGoroutine("ExampleNPCBehavior", (void*)(intptr_t)i);
    npcGoroutines.push_back(gid);
}

// Each NPC runs independently with goroutines for:
// - Patrol behavior
// - Player detection  
// - Animation state
// - Pathfinding
```

### Parallel Physics

```cpp
// Process 1000 physics actors in parallel
std::vector<std::any> args = {deltaTime, 1000};
goSystem->CallFunction("ExampleParallelPhysicsUpdate", args);

// Go spawns worker goroutines automatically,
// utilizing all CPU cores
```

### Network Replication

```cpp
// Start network sync for a player
int gid = goSystem->StartGoroutine("ExampleNetworkReplication", (void*)playerId);

// Player state is now synchronized in background
// while game loop continues
```

### Asset Loading

```cpp
// Load 100 assets concurrently
std::vector<std::any> args = {100};
goSystem->CallFunction("ExampleAssetLoading", args);

// Assets load in background via worker goroutines
```

## File Structure

```
game-engine/
├── include/
│   ├── GoScriptSystem.h          ← Main C++ interface
│   └── GoRuntime.h               ← Lower-level Go runtime
├── src/
│   ├── GoScriptSystem.cpp        ← Implementation
│   └── GoRuntime.cpp             ← Implementation
├── scripts/
│   ├── go.mod                    ← Go module definition
│   ├── example_go_systems.go     ← Example implementations
│   └── build/
│       ├── example_go_systems.dll ← Compiled output
│       └── ...
├── GO_LANGUAGE_GUIDE.md          ← Detailed documentation
└── GO_BUILD_GUIDE.md             ← Build instructions
```

## Key Concepts

### Goroutines

Lightweight threads managed by Go runtime:

```go
func NPC_BehaviorUpdate() {
    go patrolLogic()      // Runs concurrently
    go detectPlayer()     // Runs concurrently  
    go animationUpdate()  // Runs concurrently
}
```

100,000+ goroutines can run on a single machine!

### Channels

Type-safe message passing between goroutines:

```go
stateChannel := make(chan string, 10)

go func() {
    stateChannel <- "attacking"
}()

state := <-stateChannel  // Receive: "attacking"
```

### WaitGroups

Synchronize multiple goroutines:

```go
var wg sync.WaitGroup
wg.Add(4)

for i := 0; i < 4; i++ {
    go func() {
        defer wg.Done()
        doWork()
    }()
}

wg.Wait()  // Block until all done
```

## Performance Characteristics

| Aspect | Value |
|--------|-------|
| Goroutine Memory | ~2 KB |
| Channel Memory | ~48 bytes + element size |
| Goroutine Startup | <1 microsecond |
| Context Switch | <100 nanoseconds |
| Max Active Goroutines | 1M+ |

Go outperforms OS threads by orders of magnitude!

## Common Patterns

### Pattern 1: Concurrent Workers

```go
func WorkerPool(jobs chan Job) {
    for job := range jobs {
        processJob(job)
    }
}

// In main
jobs := make(chan Job, 100)
for i := 0; i < 4; i++ {
    go WorkerPool(jobs)
}
```

### Pattern 2: Fan-Out/Fan-In

```go
// Fan-out: distribute work
for i := 0; i < 4; i++ {
    go worker(i, input, output)
}

// Fan-in: collect results
for i := 0; i < expected; i++ {
    result := <-output
}
```

### Pattern 3: Timeout Handling

```go
select {
case result := <-ch:
    // Got result
case <-time.After(1 * time.Second):
    // Timeout
}
```

## Debugging

### Check Running Goroutines

```cpp
size_t count = goSystem->GetActiveGoroutineCount();
std::cout << "Active goroutines: " << count << std::endl;
```

### Check for Errors

```cpp
if (goSystem->HasErrors()) {
    std::cout << "Error: " << goSystem->GetLastError() << std::endl;
}
```

### Monitor Memory

```cpp
uint64_t memUsage = goSystem->GetMemoryUsage();
std::cout << "Go memory: " << memUsage / 1024 / 1024 << " MB" << std::endl;
```

## Next Steps

1. **Read [GO_LANGUAGE_GUIDE.md](GO_LANGUAGE_GUIDE.md)** for detailed examples
2. **Read [GO_BUILD_GUIDE.md](GO_BUILD_GUIDE.md)** for build configuration
3. **Study [scripts/example_go_systems.go](scripts/example_go_systems.go)** for patterns
4. **Modify scripts** for your game logic
5. **Profile** with Go's pprof tool

## Troubleshooting

### Go not found

```bash
# Verify Go is installed
go version

# Check PATH includes Go bin
echo $PATH | grep -i go
```

### Build fails with "gcc not found"

Install a C compiler:
- **Windows**: MinGW or MSVC
- **Linux**: `sudo apt install build-essential`
- **macOS**: `xcode-select --install`

### Goroutines not starting

```cpp
// Always call this after starting goroutines
registry.Update(0.001f);

// Give Go scheduler time to start
std::this_thread::sleep_for(std::chrono::milliseconds(10));
```

### Deadlock on channel

```go
// Always handle receive with timeout
select {
case data := <-ch:
    // Use data
case <-time.After(timeout):
    // Handle timeout
}
```

## Integration with Other Systems

### With Physics

```cpp
// Parallel rigid body updates
goSystem->StartGoroutine("PhysicsParallelUpdate");
```

### With Audio

```cpp
// Play sounds concurrently
goSystem->CallFunction("PlaySoundAsync", {soundID, position});
```

### With Rendering

```cpp
// Generate impostors in background
goSystem->StartGoroutine("GenerateImpostors");
```

### With Networking

```cpp
// Replicate player state concurrently
goSystem->StartGoroutine("NetworkSync");
```

## Performance Tips

1. **Use buffered channels** for high throughput
2. **Limit goroutines** - use worker pools
3. **Profile regularly** with pprof
4. **Monitor memory** - goroutines are cheap but not free
5. **Close channels** when done to signal workers

## Comparison: Go vs Other Languages

### Go vs Lua

| Feature | Go | Lua |
|---------|----|----|
| Concurrency | Native (goroutines) | Coroutines (manual) |
| Performance | Fast (compiled) | Slower (interpreted) |
| Syntax | Clean, modern | Simple, lightweight |
| Ideal for | Concurrent systems | General game logic |

### Go vs Python

| Feature | Go | Python |
|---------|----|----|
| Concurrency | True parallelism | GIL (single-threaded) |
| Performance | Fast | Slow |
| Syntax | Clean | Flexible |
| Ideal for | Concurrent systems | AI/Data science |

### Go vs Rust

| Feature | Go | Rust |
|---------|----|----|
| Concurrency | Easy (goroutines) | Complex (async/await) |
| Performance | Very fast | Fastest |
| Syntax | Simple | Steep learning curve |
| Ideal for | Game logic | Performance-critical |

## Resources

- [Official Go Docs](https://golang.org/doc/)
- [Go Concurrency Patterns](https://go.dev/blog/pipelines)
- [Effective Go](https://golang.org/doc/effective_go)
- [Go by Example](https://gobyexample.com/)

## Summary

Go brings powerful native concurrency to the engine:

✅ 50+ concurrent NPCs with independent AI  
✅ Parallel physics processing across all cores  
✅ Non-blocking asset loading  
✅ Scalable server backends  
✅ Clean, readable code  

Start building concurrent game systems today!
