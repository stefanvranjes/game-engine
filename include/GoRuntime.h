#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

/**
 * @class GoRuntime
 * @brief Manages Go runtime lifecycle and concurrent task execution
 * 
 * Provides facilities for:
 * - Loading and compiling Go programs
 * - Managing goroutine execution
 * - Channel-based communication
 * - WaitGroup synchronization
 * - Concurrent task scheduling
 * 
 * This class serves as the bridge between C++ game engine and Go concurrent systems.
 */
class GoRuntime {
public:
    /**
     * Task representation for concurrent execution
     */
    struct Task {
        std::string id;
        std::string functionName;
        std::string parameters;
        std::function<void()> callback;
        bool isRunning;
        bool isCanceled;
    };

    /**
     * WaitGroup for synchronizing goroutines
     */
    struct WaitGroup {
        std::string name;
        int count;
        std::mutex mutex;
        std::condition_variable cv;
        bool isDone;
    };

    GoRuntime();
    ~GoRuntime();

    /**
     * Initialize Go runtime (call once at engine startup)
     * @return true if initialization successful
     */
    bool Initialize();

    /**
     * Shutdown Go runtime (call at engine shutdown)
     */
    void Shutdown();

    /**
     * Check if runtime is initialized
     */
    bool IsInitialized() const { return initialized; }

    /**
     * Load a Go program from file
     * @param filepath Path to .go file or compiled binary
     * @return true if loaded successfully
     */
    bool LoadProgram(const std::string& filepath);

    /**
     * Unload a loaded Go program
     * @param programName Name of program to unload
     */
    void UnloadProgram(const std::string& programName);

    // Goroutine management
    /**
     * Spawn a new goroutine to execute a Go function
     * @param functionName Name of Go function to execute (e.g., "MyPackage.MyFunc")
     * @param parameters JSON-formatted parameters for the function
     * @return Goroutine ID (negative on error)
     */
    int SpawnGoroutine(const std::string& functionName, const std::string& parameters = "");

    /**
     * Wait for goroutine to complete
     * @param goroutineId ID from SpawnGoroutine
     * @param timeoutMs Maximum time to wait (-1 = indefinite)
     * @return 0 if completed, -1 if timeout or error
     */
    int WaitForGoroutine(int goroutineId, int timeoutMs = -1);

    /**
     * Cancel/kill a running goroutine
     * @param goroutineId ID from SpawnGoroutine
     * @return 0 on success
     */
    int CancelGoroutine(int goroutineId);

    /**
     * Get status of a goroutine
     * @param goroutineId ID to query
     * @return true if still running
     */
    bool IsGoroutineRunning(int goroutineId) const;

    /**
     * Get count of active goroutines
     */
    size_t GetActiveGoroutineCount() const;

    // Channel operations
    /**
     * Create a typed channel for inter-goroutine communication
     * @param channelName Unique channel identifier
     * @param bufferSize 0 for unbuffered, >0 for buffered channel
     * @return 0 on success
     */
    int CreateChannel(const std::string& channelName, size_t bufferSize = 0);

    /**
     * Send data to a channel
     * @param channelName Target channel
     * @param data JSON-encoded data to send
     * @return 0 on success, -1 on error
     */
    int SendToChannel(const std::string& channelName, const std::string& data);

    /**
     * Receive data from a channel
     * @param channelName Source channel
     * @param timeoutMs Maximum wait (-1 = indefinite)
     * @return Data received, or empty string on timeout
     */
    std::string ReceiveFromChannel(const std::string& channelName, int timeoutMs = -1);

    /**
     * Close a channel (prevents further sends)
     * @param channelName Channel to close
     * @return 0 on success
     */
    int CloseChannel(const std::string& channelName);

    /**
     * Check if channel has pending data
     * @param channelName Channel to check
     * @return true if data available
     */
    bool HasChannelData(const std::string& channelName) const;

    // WaitGroup synchronization
    /**
     * Create a new WaitGroup for synchronizing multiple goroutines
     * @param groupName Unique group identifier
     * @param initialCount Initial counter value
     * @return 0 on success
     */
    int CreateWaitGroup(const std::string& groupName, int initialCount);

    /**
     * Increment WaitGroup counter
     * @param groupName WaitGroup to modify
     * @param delta Amount to add
     * @return 0 on success
     */
    int WaitGroupAdd(const std::string& groupName, int delta);

    /**
     * Decrement WaitGroup counter
     * @param groupName WaitGroup to modify
     * @return 0 on success
     */
    int WaitGroupDone(const std::string& groupName);

    /**
     * Wait for WaitGroup counter to reach zero
     * @param groupName WaitGroup to wait on
     * @param timeoutMs Maximum wait (-1 = indefinite)
     * @return 0 if counter reached zero, -1 on timeout
     */
    int WaitGroupWait(const std::string& groupName, int timeoutMs = -1);

    /**
     * Destroy a WaitGroup
     * @param groupName Group to destroy
     */
    void DestroyWaitGroup(const std::string& groupName);

    // Concurrent task scheduling
    /**
     * Schedule a task for concurrent execution
     * @param functionName Go function to execute
     * @param callback Optional C++ callback when task completes
     * @return Task ID
     */
    std::string ScheduleTask(const std::string& functionName, 
                            std::function<void()> callback = nullptr);

    /**
     * Cancel a scheduled task
     * @param taskId ID from ScheduleTask
     */
    void CancelTask(const std::string& taskId);

    /**
     * Get status of a task
     * @param taskId ID to query
     * @return true if task is still running
     */
    bool IsTaskRunning(const std::string& taskId) const;

    // Utility
    /**
     * Call a Go function synchronously from C++ (blocking)
     * @param functionName Go function to call
     * @param parameters JSON parameters
     * @return Result as JSON string
     */
    std::string CallGoFunction(const std::string& functionName, const std::string& parameters = "");

    /**
     * Get memory usage of Go runtime
     */
    uint64_t GetMemoryUsage() const;

    /**
     * Get last error message
     */
    std::string GetLastError() const { return lastError; }

    /**
     * Frame update (process completed goroutines and execute callbacks)
     */
    void Update();

private:
    bool initialized;
    std::map<std::string, int> loadedPrograms;
    std::map<int, Task> runningGoroutines;
    std::map<std::string, std::queue<std::string>> channels;
    std::map<std::string, std::unique_ptr<WaitGroup>> waitGroups;
    std::map<std::string, Task> scheduledTasks;
    
    mutable std::mutex goroutineMutex;
    mutable std::mutex channelMutex;
    mutable std::mutex taskMutex;
    
    int nextGoroutineId;
    mutable std::string lastError;

    // Helper functions
    void ProcessCompletedGoroutines();
    void ExecuteTaskCallbacks();
};
