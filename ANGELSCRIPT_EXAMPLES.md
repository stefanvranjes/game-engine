# AngelScript Implementation Examples

## Example 1: Basic Game Loop Integration

### Application.h

```cpp
#include "AngelScriptSystem.h"
#include "ScriptLanguageRegistry.h"

class Application {
private:
    AngelScriptSystem* m_AngelScript = nullptr;
    float m_GameTime = 0.0f;
    bool m_Running = true;

public:
    void Init() {
        std::cout << "Initializing application..." << std::endl;
        
        // Initialize script registry
        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Init();
        
        // Get AngelScript system
        m_AngelScript = dynamic_cast<AngelScriptSystem*>(
            registry.GetScriptSystem(ScriptLanguage::AngelScript)
        );

        if (!m_AngelScript) {
            std::cerr << "Failed to get AngelScript system!" << std::endl;
            return;
        }

        // Set up error handling
        m_AngelScript->SetErrorHandler([this](const std::string& error) {
            std::cerr << "[AngelScript Error] " << error << std::endl;
        });

        // Set up print output
        m_AngelScript->SetPrintHandler([](const std::string& msg) {
            std::cout << "[Game Script] " << msg << std::endl;
        });

        // Load initialization script
        if (!m_AngelScript->RunScript("scripts/game_init.as")) {
            std::cerr << "Failed to load game_init.as" << std::endl;
            return;
        }

        // Call initialization function
        std::vector<std::any> args;
        m_AngelScript->CallFunction("OnGameStart", args);

        std::cout << "Application initialized successfully!" << std::endl;
    }

    void Update(float deltaTime) {
        m_GameTime += deltaTime;

        if (!m_AngelScript) return;

        // Call update function with delta time
        std::vector<std::any> args = {deltaTime, m_GameTime};
        m_AngelScript->CallFunction("OnUpdate", args);
    }

    void Render() {
        if (!m_AngelScript) return;

        std::vector<std::any> args;
        m_AngelScript->CallFunction("OnRender", args);
    }

    void Shutdown() {
        if (!m_AngelScript) return;

        std::vector<std::any> args;
        m_AngelScript->CallFunction("OnGameEnd", args);

        auto& registry = ScriptLanguageRegistry::GetInstance();
        registry.Shutdown();

        std::cout << "Application shutdown complete." << std::endl;
    }

    bool IsRunning() const { return m_Running; }
};
```

### scripts/game_init.as

```angelscript
// Global game state
class GameState {
    float timeElapsed = 0.0f;
    uint32 frameCount = 0;
    int score = 0;
    bool isPaused = false;
    string gameStatus = "Running";
}

// Global instance
GameState@ g_GameState = @GameState();

// Called when game starts
void OnGameStart() {
    print("========================================");
    print("    GAME STARTED");
    print("========================================");
    print("Welcome to AngelScript Game!");
    print("");
    
    g_GameState.timeElapsed = 0.0f;
    g_GameState.frameCount = 0;
    g_GameState.score = 0;
    g_GameState.isPaused = false;
    g_GameState.gameStatus = "Running";
}

// Called every frame
void OnUpdate(float dt, float totalTime) {
    if (g_GameState.isPaused) return;

    g_GameState.timeElapsed = totalTime;
    g_GameState.frameCount++;

    // Update game logic (called every 60 frames = ~1 second at 60 FPS)
    if (g_GameState.frameCount % 60 == 0) {
        int seconds = int(totalTime);
        if (seconds % 10 == 0) {
            print("Time: " + seconds + "s, Score: " + g_GameState.score);
        }
    }
}

// Called when rendering
void OnRender() {
    // Rendering logic
}

// Called when game ends
void OnGameEnd() {
    print("");
    print("========================================");
    print("    GAME ENDED");
    print("========================================");
    print("Final Score: " + g_GameState.score);
    print("Game Time: " + int(g_GameState.timeElapsed) + " seconds");
    print("Total Frames: " + g_GameState.frameCount);
    print("");
}

// Accessors
int GetScore() {
    return g_GameState.score;
}

void AddScore(int points) {
    g_GameState.score += points;
    print("Score increased by " + points + "! Total: " + g_GameState.score);
}

void PauseGame() {
    g_GameState.isPaused = true;
    g_GameState.gameStatus = "Paused";
    print("Game paused!");
}

void ResumeGame() {
    g_GameState.isPaused = false;
    g_GameState.gameStatus = "Running";
    print("Game resumed!");
}
```

---

## Example 2: Scripted Game Objects

### scripts/player.as

```angelscript
// Player class with full game logic
class Player {
    string name = "";
    float health = 100.0f;
    float maxHealth = 100.0f;
    float speed = 5.0f;
    float attackPower = 10.0f;
    Vec3 position = Vec3(0, 0, 0);
    Vec3 velocity = Vec3(0, 0, 0);
    bool isAlive = true;
    
    // Constructor
    Player(string playerName, float initialSpeed) {
        name = playerName;
        speed = initialSpeed;
        health = maxHealth;
        print("Player '" + name + "' created with speed " + speed);
    }
    
    // Update player state
    void Update(float dt) {
        if (!isAlive) return;
        
        // Update position based on velocity
        position = position + velocity * dt;
        
        // Damping velocity
        velocity = velocity * 0.95f;
    }
    
    // Move in direction
    void Move(Vec3 direction, float dt) {
        velocity = direction * speed;
    }
    
    // Take damage
    void TakeDamage(float damage) {
        health -= damage;
        print(name + " took " + damage + " damage. Health: " + health);
        
        if (health <= 0) {
            health = 0;
            Die();
        }
    }
    
    // Heal player
    void Heal(float amount) {
        float healed = amount;
        if (health + healed > maxHealth) {
            healed = maxHealth - health;
        }
        health += healed;
        print(name + " healed for " + healed + " HP. Health: " + health);
    }
    
    // Die
    void Die() {
        isAlive = false;
        print(name + " has been defeated!");
    }
    
    // Attack enemy
    void Attack(Enemy@ target) {
        if (!isAlive || target is null) return;
        
        float damage = attackPower;
        print(name + " attacks " + target.GetName() + " for " + damage + " damage!");
        target.TakeDamage(damage);
    }
    
    // Getters
    string GetName() { return name; }
    float GetHealth() { return health; }
    float GetMaxHealth() { return maxHealth; }
    bool IsAlive() { return isAlive; }
    Vec3 GetPosition() { return position; }
    float GetSpeed() { return speed; }
    
    // Setters
    void SetHealth(float h) { health = h; if (health > maxHealth) health = maxHealth; }
    void SetSpeed(float s) { speed = s; }
}

// Create global player instance
Player@ g_Player = null;

void InitializePlayer() {
    g_Player = Player("Hero", 5.0f);
    g_Player.SetHealth(100.0f);
}

void UpdatePlayer(float dt) {
    if (g_Player !is null) {
        g_Player.Update(dt);
    }
}

Player@ GetPlayer() {
    return g_Player;
}
```

### scripts/enemies.as

```angelscript
// Enemy class
class Enemy {
    string name = "";
    float health = 50.0f;
    float maxHealth = 50.0f;
    float speed = 3.0f;
    float attackPower = 5.0f;
    Vec3 position = Vec3(10, 0, 0);
    bool isAlive = true;
    uint32 updateCounter = 0;
    
    Enemy(string enemyName, float spd) {
        name = enemyName;
        speed = spd;
        health = maxHealth;
    }
    
    void Update(float dt) {
        if (!isAlive) return;
        
        updateCounter++;
        
        // Simple AI: move towards origin
        if (position.x > 0.1f) {
            position.x -= speed * dt;
        }
    }
    
    void TakeDamage(float damage) {
        health -= damage;
        print(name + " took " + damage + " damage. Health: " + health);
        
        if (health <= 0) {
            health = 0;
            Die();
        }
    }
    
    void Die() {
        isAlive = false;
        print(name + " has been defeated!");
    }
    
    string GetName() { return name; }
    float GetHealth() { return health; }
    bool IsAlive() { return isAlive; }
    Vec3 GetPosition() { return position; }
}

// Enemy array
array<Enemy@> g_Enemies;

void SpawnEnemy(string name, float speed) {
    Enemy@ enemy = Enemy(name, speed);
    g_Enemies.insertLast(enemy);
    print("Enemy spawned: " + name);
}

void UpdateEnemies(float dt) {
    for (uint i = 0; i < g_Enemies.length(); i++) {
        if (g_Enemies[i] !is null && g_Enemies[i].IsAlive()) {
            g_Enemies[i].Update(dt);
        }
    }
}

uint GetEnemyCount() {
    return g_Enemies.length();
}

uint GetAliveEnemyCount() {
    uint count = 0;
    for (uint i = 0; i < g_Enemies.length(); i++) {
        if (g_Enemies[i] !is null && g_Enemies[i].IsAlive()) {
            count++;
        }
    }
    return count;
}
```

---

## Example 3: Event System

### scripts/events.as

```angelscript
// Event callback types
typedef void EventCallback();
typedef void EventCallbackWithInt(int value);
typedef void EventCallbackWithString(string message);

// Event system
class EventSystem {
    // Global listeners
    array<EventCallback@> onPlayerDied;
    array<EventCallback@> onGamePaused;
    array<EventCallback@> onGameResumed;
    
    // Listeners with parameters
    array<EventCallbackWithInt@> onScoreChanged;
    array<EventCallbackWithString@> onMessageReceived;
    
    // Subscribe to events
    void SubscribePlayerDied(EventCallback@ callback) {
        if (callback !is null) {
            onPlayerDied.insertLast(callback);
        }
    }
    
    void SubscribeGamePaused(EventCallback@ callback) {
        if (callback !is null) {
            onGamePaused.insertLast(callback);
        }
    }
    
    void SubscribeScoreChanged(EventCallbackWithInt@ callback) {
        if (callback !is null) {
            onScoreChanged.insertLast(callback);
        }
    }
    
    // Emit events
    void EmitPlayerDied() {
        print("Event: Player Died");
        for (uint i = 0; i < onPlayerDied.length(); i++) {
            onPlayerDied[i]();
        }
    }
    
    void EmitGamePaused() {
        print("Event: Game Paused");
        for (uint i = 0; i < onGamePaused.length(); i++) {
            onGamePaused[i]();
        }
    }
    
    void EmitScoreChanged(int newScore) {
        print("Event: Score Changed to " + newScore);
        for (uint i = 0; i < onScoreChanged.length(); i++) {
            onScoreChanged[i](newScore);
        }
    }
}

// Global event system
EventSystem@ g_EventSystem = @EventSystem();

// Event handlers
void OnPlayerDiedHandler() {
    print("  -> Showing game over screen");
}

void OnGamePausedHandler() {
    print("  -> Opening pause menu");
    print("  -> Pausing background music");
}

void OnScoreChangedHandler(int score) {
    print("  -> Updating UI: Score = " + score);
}

void InitializeEventSystem() {
    g_EventSystem.SubscribePlayerDied(@OnPlayerDiedHandler);
    g_EventSystem.SubscribeGamePaused(@OnGamePausedHandler);
    g_EventSystem.SubscribeScoreChanged(@OnScoreChangedHandler);
    
    print("Event system initialized");
}

// Trigger events from game code
void TriggerPlayerDied() {
    g_EventSystem.EmitPlayerDied();
}

void TriggerGamePaused() {
    g_EventSystem.EmitGamePaused();
}

void TriggerScoreChanged(int newScore) {
    g_EventSystem.EmitScoreChanged(newScore);
}
```

---

## Example 4: State Machine Pattern

### scripts/state_machine.as

```angelscript
// Player states
enum PlayerState {
    Idle,
    Moving,
    Attacking,
    Defending,
    Dead
}

// State machine for player
class PlayerStateMachine {
    PlayerState currentState = Idle;
    float stateTimer = 0.0f;
    
    PlayerStateMachine() {
        print("PlayerStateMachine created");
    }
    
    void Update(float dt) {
        stateTimer += dt;
        
        // State-specific logic
        switch (currentState) {
            case Idle:
                UpdateIdleState(dt);
                break;
            case Moving:
                UpdateMovingState(dt);
                break;
            case Attacking:
                UpdateAttackingState(dt);
                break;
            case Defending:
                UpdateDefendingState(dt);
                break;
            case Dead:
                UpdateDeadState(dt);
                break;
        }
    }
    
    void ChangeState(PlayerState newState) {
        if (currentState == newState) return;
        
        OnStateExit(currentState);
        currentState = newState;
        stateTimer = 0.0f;
        OnStateEnter(currentState);
    }
    
    void OnStateEnter(PlayerState state) {
        print("Entering state: " + GetStateName(state));
        
        switch (state) {
            case Idle:
                print("  -> Player is idle");
                break;
            case Moving:
                print("  -> Player is moving");
                break;
            case Attacking:
                print("  -> Player is attacking");
                break;
            case Defending:
                print("  -> Player is defending");
                break;
            case Dead:
                print("  -> Player is dead");
                break;
        }
    }
    
    void OnStateExit(PlayerState state) {
        print("Exiting state: " + GetStateName(state));
    }
    
    void UpdateIdleState(float dt) {
        // In idle state
    }
    
    void UpdateMovingState(float dt) {
        // In moving state - switch to idle after some time
        if (stateTimer > 2.0f) {
            ChangeState(Idle);
        }
    }
    
    void UpdateAttackingState(float dt) {
        // In attacking state
        if (stateTimer > 0.5f) {
            ChangeState(Idle);
        }
    }
    
    void UpdateDefendingState(float dt) {
        // In defending state
    }
    
    void UpdateDeadState(float dt) {
        // Dead state - no transitions
    }
    
    string GetStateName(PlayerState state) {
        if (state == Idle) return "Idle";
        else if (state == Moving) return "Moving";
        else if (state == Attacking) return "Attacking";
        else if (state == Defending) return "Defending";
        else if (state == Dead) return "Dead";
        else return "Unknown";
    }
    
    PlayerState GetCurrentState() { return currentState; }
    string GetCurrentStateName() { return GetStateName(currentState); }
}

// Global state machine
PlayerStateMachine@ g_PlayerStateMachine = @PlayerStateMachine();

void UpdatePlayerStateMachine(float dt) {
    g_PlayerStateMachine.Update(dt);
}

void PlayerStartMoving() {
    g_PlayerStateMachine.ChangeState(Moving);
}

void PlayerStartAttacking() {
    g_PlayerStateMachine.ChangeState(Attacking);
}

void PlayerStartDefending() {
    g_PlayerStateMachine.ChangeState(Defending);
}

void PlayerDie() {
    g_PlayerStateMachine.ChangeState(Dead);
}
```

---

## Example 5: Using the ScriptLanguageRegistry

### C++ Code

```cpp
#include "ScriptLanguageRegistry.h"

void DemoMultiLanguageScripting() {
    auto& registry = ScriptLanguageRegistry::GetInstance();
    registry.Init();

    // Execute with different languages
    registry.ExecuteScript("scripts/config.lua", ScriptLanguage::Lua);
    registry.ExecuteScript("scripts/game_logic.as", ScriptLanguage::AngelScript);
    registry.ExecuteScript("scripts/ui.wren", ScriptLanguage::Wren);

    // Get specific system
    auto* angelScript = registry.GetScriptSystem(ScriptLanguage::AngelScript);
    if (angelScript) {
        angelScript->RunScript("scripts/game_init.as");
    }

    registry.Shutdown();
}
```

---

## Example 6: Error Handling and Debugging

### C++ Code

```cpp
#include "AngelScriptSystem.h"
#include <iostream>

void DemoErrorHandling() {
    auto& angel = AngelScriptSystem::GetInstance();
    angel.Init();

    // Set up error handler
    angel.SetErrorHandler([](const std::string& error) {
        std::cerr << "ERROR: " << error << std::endl;
    });

    // Set up print handler
    angel.SetPrintHandler([](const std::string& msg) {
        std::cout << "[Script Output] " << msg << std::endl;
    });

    // Enable debugging
    angel.SetDebugEnabled(true);

    // Try to load a script
    if (!angel.RunScript("scripts/problematic.as")) {
        std::cerr << "Failed to load script!" << std::endl;
        if (angel.HasErrors()) {
            std::cerr << "Last error: " << angel.GetLastError() << std::endl;
        }
    }

    // Try to call a function
    if (angel.HasFunction("MyFunction")) {
        std::vector<std::any> args = {42, "test"};
        angel.CallFunction("MyFunction", args);
        
        if (angel.HasErrors()) {
            std::cerr << "Function error: " << angel.GetLastError() << std::endl;
        }
    } else {
        std::cout << "Function 'MyFunction' not found" << std::endl;
    }

    angel.Shutdown();
}
```

---

## Example 7: Performance Profiling

### C++ Code

```cpp
#include "AngelScriptSystem.h"
#include <chrono>
#include <iostream>

void DemoPerformanceProfiling() {
    auto& angel = AngelScriptSystem::GetInstance();
    angel.Init();

    // Enable optimization
    angel.SetOptimizationEnabled(true);

    // Load script
    angel.RunScript("scripts/heavy_computation.as");

    // Profile multiple calls
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::any> args = {i, 0.016f};
        angel.CallFunction("HeavyComputation", args);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(end - start).count();
        
        std::cout << "Call " << i << ": " << (elapsed * 1000.0) << " ms" << std::endl;
    }

    // Get statistics
    auto stats = angel.GetCompileStats();
    std::cout << "\nCompilation Statistics:" << std::endl;
    std::cout << "  Total functions: " << stats.totalFunctions << std::endl;
    std::cout << "  Total classes: " << stats.totalClasses << std::endl;
    std::cout << "  Total modules: " << stats.totalModules << std::endl;
    std::cout << "  Memory used: " << stats.memoryUsed << " bytes" << std::endl;

    std::cout << "Last execution time: " << angel.GetLastExecutionTime() * 1000.0 << " ms" << std::endl;

    // Force garbage collection
    angel.ForceGarbageCollection();

    angel.Shutdown();
}
```

---

**For more examples and detailed usage, see [ANGELSCRIPT_INTEGRATION_GUIDE.md](ANGELSCRIPT_INTEGRATION_GUIDE.md)**
