#pragma once

#include "IScriptSystem.h"
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <any>
#include <stdint.h>

// Forward declarations for Go runtime
extern "C" {
    typedef void* GoScriptHandle;
    typedef void (*GoCallback)(const char* eventName, void* userData);
    
    // Go runtime initialization
    GoScriptHandle GoScriptInit();
    void GoScriptShutdown(GoScriptHandle handle);
    
    // Script execution
    int GoScriptLoadFile(GoScriptHandle handle, const char* filepath);
    int GoScriptExecuteString(GoScriptHandle handle, const char* source);
    
    // Goroutine/concurrent execution
    int GoScriptStartGoroutine(GoScriptHandle handle, const char* functionName, void* userData);
    int GoScriptWaitGoroutine(GoScriptHandle handle, int goroutineId);
    int GoScriptKillGoroutine(GoScriptHandle handle, int goroutineId);
    
    // Channel operations
    int GoScriptChannelSend(GoScriptHandle handle, const char* channelName, const char* jsonData);
    const char* GoScriptChannelReceive(GoScriptHandle handle, const char* channelName, int timeoutMs);
    
    // Function calling
    const char* GoScriptCallFunction(GoScriptHandle handle, const char* functionName, const char* jsonArgs);
    
    // Memory and diagnostics
    uint64_t GoScriptGetMemoryUsage(GoScriptHandle handle);
    double GoScriptGetLastExecutionTime(GoScriptHandle handle);
    const char* GoScriptGetLastError(GoScriptHandle handle);
}

/**
 * @class GoScriptSystem
 * @brief Go language scripting system with native concurrency support
 * 
 * Provides Go/Go language scripting with native support for:
 * - Goroutines (lightweight concurrency primitives)
 * - Channels (typed message passing)
 * - WaitGroups (synchronization)
 * - Concurrent game systems (AI, physics, animation)
 * 
 * Go is ideal for:
 * - Concurrent NPC behavior trees
 * - Parallel physics updates
 * - Network synchronization
 * - Async resource loading
 * - Distributed game logic across multiple cores
 * 
 * Usage Example:
 * ```cpp
 * auto goSystem = std::make_unique<GoScriptSystem>();
 * goSystem->Init();
 * 
 * // Load Go script with concurrent logic
 * goSystem->RunScript("scripts/npc_behavior.go");
 * 
 * // Start concurrent goroutine for NPC AI
 * goSystem->StartGoroutine("NPC_Update", npcData);
 * 
 * // Send data to goroutine via channel
 * goSystem->SendToChannel("npc_commands", commandJson);
 * 
 * // Update each frame
 * goSystem->Update(deltaTime);
 * ```
 */
class GoScriptSystem : public IScriptSystem {
public:
    /**
     * Represents a running goroutine
     */
    struct Goroutine {
        int id;
        std::string functionName;
        bool isRunning;
        uint64_t startTime;
        std::string lastError;
    };

    /**
     * Represents a typed channel for inter-goroutine communication
     */
    struct Channel {
        std::string name;
        std::queue<std::string> messages;
        std::mutex mutex;
        bool isOpen;
    };

    GoScriptSystem();
    virtual ~GoScriptSystem();

    // Lifecycle
    void Init() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Script execution
    bool RunScript(const std::string& filepath) override;
    bool ExecuteString(const std::string& source) override;

    // Language metadata
    ScriptLanguage GetLanguage() const override;
    ScriptExecutionMode GetExecutionMode() const override;
    std::string GetLanguageName() const override { return "Go"; }
    std::string GetFileExtension() const override { return ".go"; }

    // Type registration and interop
    void RegisterTypes() override;
    bool HasType(const std::string& typeName) const override;
    std::any CallFunction(const std::string& functionName, 
                          const std::vector<std::any>& args) override;

    // Go-specific: Goroutine management
    /**
     * Start a concurrent goroutine (Go's lightweight thread)
     * @param functionName Name of Go function to run concurrently
     * @param userData Optional user data to pass to goroutine
     * @return Goroutine ID (negative on error)
     */
    int StartGoroutine(const std::string& functionName, void* userData = nullptr);

    /**
     * Wait for a goroutine to complete
     * @param goroutineId ID returned from StartGoroutine
     * @return 0 on success, negative on error
     */
    int WaitGoroutine(int goroutineId);

    /**
     * Cancel a running goroutine
     * @param goroutineId ID of goroutine to cancel
     * @return 0 on success, negative on error
     */
    int KillGoroutine(int goroutineId);

    /**
     * Get status of a goroutine
     * @param goroutineId ID to check
     * @return Goroutine struct if found, nullptr otherwise
     */
    const Goroutine* GetGoroutineStatus(int goroutineId) const;

    /**
     * Get count of active goroutines
     */
    size_t GetActiveGoroutineCount() const;

    // Go-specific: Channel operations
    /**
     * Create a new typed channel for inter-goroutine communication
     * @param channelName Name of channel
     * @return 0 on success, negative on error
     */
    int CreateChannel(const std::string& channelName);

    /**
     * Send data to a channel (non-blocking)
     * @param channelName Name of target channel
     * @param jsonData JSON-serialized data to send
     * @return 0 on success, negative on error
     */
    int SendToChannel(const std::string& channelName, const std::string& jsonData);

    /**
     * Receive data from a channel (blocking with timeout)
     * @param channelName Name of source channel
     * @param timeoutMs Maximum wait time in milliseconds (0 = non-blocking)
     * @return JSON string of received data, or empty string on timeout/error
     */
    std::string ReceiveFromChannel(const std::string& channelName, int timeoutMs = 0);

    /**
     * Close a channel (prevents further sends)
     * @param channelName Name of channel to close
     * @return 0 on success, negative on error
     */
    int CloseChannel(const std::string& channelName);

    // Hot-reload support
    bool SupportsHotReload() const override { return false; } // Go recompilation required
    void ReloadScript(const std::string& filepath) override;

    // Performance profiling
    uint64_t GetMemoryUsage() const override;
    double GetLastExecutionTime() const override { return lastExecutionTime; }

    // Error handling
    bool HasErrors() const override;
    std::string GetLastError() const override;

private:
    GoScriptHandle goHandle;
    std::map<int, Goroutine> activeGoroutines;
    std::map<std::string, std::unique_ptr<Channel>> channels;
    mutable std::mutex goroutineMutex;
    mutable std::mutex channelMutex;
    
    double lastExecutionTime;
    int nextGoroutineId;
    std::string lastError;
    bool isInitialized;

    // Helper functions
    std::string SerializeArgs(const std::vector<std::any>& args);
    std::any DeserializeResult(const std::string& jsonResult);
    void UpdateGoroutineStatus();
};
