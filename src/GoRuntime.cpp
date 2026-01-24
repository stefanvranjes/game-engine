#include "GoRuntime.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>

GoRuntime::GoRuntime()
    : initialized(false),
      nextGoroutineId(1) {
}

GoRuntime::~GoRuntime() {
    if (initialized) {
        Shutdown();
    }
}

bool GoRuntime::Initialize() {
    if (initialized) {
        lastError = "Go runtime already initialized";
        return true;
    }

    try {
        // Initialize Go runtime
        // This would normally call into Go via cgo
        initialized = true;
        std::cout << "GoRuntime initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        lastError = std::string("Failed to initialize Go runtime: ") + e.what();
        std::cerr << "GoRuntime: " << lastError << std::endl;
        return false;
    }
}

void GoRuntime::Shutdown() {
    if (!initialized) return;

    try {
        // Cancel all running goroutines
        {
            std::lock_guard<std::mutex> lock(goroutineMutex);
            for (auto& [id, task] : runningGoroutines) {
                if (task.isRunning) {
                    task.isCanceled = true;
                }
            }
            runningGoroutines.clear();
        }

        // Close all channels
        {
            std::lock_guard<std::mutex> lock(channelMutex);
            channels.clear();
        }

        // Destroy all wait groups
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            waitGroups.clear();
            loadedPrograms.clear();
        }

        initialized = false;
        std::cout << "GoRuntime shutdown complete" << std::endl;
    } catch (const std::exception& e) {
        lastError = std::string("GoRuntime shutdown exception: ") + e.what();
        std::cerr << "GoRuntime: " << lastError << std::endl;
    }
}

bool GoRuntime::LoadProgram(const std::string& filepath) {
    if (!initialized) {
        lastError = "Go runtime not initialized";
        return false;
    }

    try {
        // Extract program name from filepath
        size_t lastSlash = filepath.find_last_of("/\\");
        std::string programName = lastSlash != std::string::npos ? 
                                  filepath.substr(lastSlash + 1) : filepath;

        // In a real implementation, this would:
        // 1. Load the compiled Go binary or shared library
        // 2. Initialize the Go package
        // 3. Register exported functions
        
        std::lock_guard<std::mutex> lock(taskMutex);
        loadedPrograms[programName] = 1; // Dummy value

        std::cout << "GoRuntime: Loaded program " << programName << std::endl;
        return true;
    } catch (const std::exception& e) {
        lastError = std::string("Failed to load program: ") + e.what();
        std::cerr << "GoRuntime: " << lastError << std::endl;
        return false;
    }
}

void GoRuntime::UnloadProgram(const std::string& programName) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        loadedPrograms.erase(programName);
        std::cout << "GoRuntime: Unloaded program " << programName << std::endl;
    } catch (const std::exception& e) {
        lastError = std::string("Failed to unload program: ") + e.what();
        std::cerr << "GoRuntime: " << lastError << std::endl;
    }
}

int GoRuntime::SpawnGoroutine(const std::string& functionName, const std::string& parameters) {
    if (!initialized) {
        lastError = "Go runtime not initialized";
        return -1;
    }

    try {
        int goroutineId = nextGoroutineId++;
        
        Task task;
        task.id = std::to_string(goroutineId);
        task.functionName = functionName;
        task.parameters = parameters;
        task.isRunning = true;
        task.isCanceled = false;

        {
            std::lock_guard<std::mutex> lock(goroutineMutex);
            runningGoroutines[goroutineId] = task;
        }

        std::cout << "GoRuntime: Spawned goroutine " << goroutineId 
                  << " for " << functionName << std::endl;
        return goroutineId;
    } catch (const std::exception& e) {
        lastError = std::string("SpawnGoroutine exception: ") + e.what();
        std::cerr << "GoRuntime: " << lastError << std::endl;
        return -1;
    }
}

int GoRuntime::WaitForGoroutine(int goroutineId, int timeoutMs) {
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            {
                std::lock_guard<std::mutex> lock(goroutineMutex);
                auto it = runningGoroutines.find(goroutineId);
                if (it == runningGoroutines.end() || !it->second.isRunning) {
                    return 0; // Goroutine completed
                }
            }

            if (timeoutMs >= 0) {
                auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeoutMs) {
                    return -1; // Timeout
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception& e) {
        lastError = std::string("WaitForGoroutine exception: ") + e.what();
        return -1;
    }
}

int GoRuntime::CancelGoroutine(int goroutineId) {
    try {
        std::lock_guard<std::mutex> lock(goroutineMutex);
        auto it = runningGoroutines.find(goroutineId);
        if (it != runningGoroutines.end()) {
            it->second.isCanceled = true;
            it->second.isRunning = false;
            std::cout << "GoRuntime: Canceled goroutine " << goroutineId << std::endl;
            return 0;
        }
        return -1; // Goroutine not found
    } catch (const std::exception& e) {
        lastError = std::string("CancelGoroutine exception: ") + e.what();
        return -1;
    }
}

bool GoRuntime::IsGoroutineRunning(int goroutineId) const {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    auto it = runningGoroutines.find(goroutineId);
    return it != runningGoroutines.end() && it->second.isRunning;
}

size_t GoRuntime::GetActiveGoroutineCount() const {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    size_t count = 0;
    for (const auto& [id, task] : runningGoroutines) {
        if (task.isRunning) {
            count++;
        }
    }
    return count;
}

int GoRuntime::CreateChannel(const std::string& channelName, size_t bufferSize) {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        if (channels.find(channelName) != channels.end()) {
            lastError = "Channel already exists: " + channelName;
            return -1;
        }

        channels[channelName] = std::queue<std::string>();
        std::cout << "GoRuntime: Created channel " << channelName 
                  << " with buffer size " << bufferSize << std::endl;
        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("CreateChannel exception: ") + e.what();
        return -1;
    }
}

int GoRuntime::SendToChannel(const std::string& channelName, const std::string& data) {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        auto it = channels.find(channelName);
        if (it == channels.end()) {
            lastError = "Channel not found: " + channelName;
            return -1;
        }

        it->second.push(data);
        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("SendToChannel exception: ") + e.what();
        return -1;
    }
}

std::string GoRuntime::ReceiveFromChannel(const std::string& channelName, int timeoutMs) {
    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        while (true) {
            {
                std::lock_guard<std::mutex> lock(channelMutex);
                auto it = channels.find(channelName);
                if (it != channels.end() && !it->second.empty()) {
                    std::string data = it->second.front();
                    it->second.pop();
                    return data;
                }
            }

            if (timeoutMs >= 0) {
                auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeoutMs) {
                    return ""; // Timeout
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception& e) {
        lastError = std::string("ReceiveFromChannel exception: ") + e.what();
        return "";
    }
}

int GoRuntime::CloseChannel(const std::string& channelName) {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        if (channels.erase(channelName) > 0) {
            std::cout << "GoRuntime: Closed channel " << channelName << std::endl;
            return 0;
        }
        return -1; // Channel not found
    } catch (const std::exception& e) {
        lastError = std::string("CloseChannel exception: ") + e.what();
        return -1;
    }
}

bool GoRuntime::HasChannelData(const std::string& channelName) const {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        auto it = channels.find(channelName);
        return it != channels.end() && !it->second.empty();
    } catch (const std::exception& e) {
        return false;
    }
}

int GoRuntime::CreateWaitGroup(const std::string& groupName, int initialCount) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        if (waitGroups.find(groupName) != waitGroups.end()) {
            lastError = "WaitGroup already exists: " + groupName;
            return -1;
        }

        auto wg = std::make_unique<WaitGroup>();
        wg->name = groupName;
        wg->count = initialCount;
        wg->isDone = (initialCount == 0);
        waitGroups[groupName] = std::move(wg);

        std::cout << "GoRuntime: Created WaitGroup " << groupName 
                  << " with count " << initialCount << std::endl;
        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("CreateWaitGroup exception: ") + e.what();
        return -1;
    }
}

int GoRuntime::WaitGroupAdd(const std::string& groupName, int delta) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        auto it = waitGroups.find(groupName);
        if (it == waitGroups.end()) {
            lastError = "WaitGroup not found: " + groupName;
            return -1;
        }

        it->second->count += delta;
        if (it->second->count < 0) {
            lastError = "WaitGroup count went negative";
            return -1;
        }

        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("WaitGroupAdd exception: ") + e.what();
        return -1;
    }
}

int GoRuntime::WaitGroupDone(const std::string& groupName) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        auto it = waitGroups.find(groupName);
        if (it == waitGroups.end()) {
            lastError = "WaitGroup not found: " + groupName;
            return -1;
        }

        it->second->count--;
        if (it->second->count == 0) {
            it->second->isDone = true;
            it->second->cv.notify_all();
        }

        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("WaitGroupDone exception: ") + e.what();
        return -1;
    }
}

int GoRuntime::WaitGroupWait(const std::string& groupName, int timeoutMs) {
    try {
        std::unique_lock<std::mutex> lock(taskMutex);
        auto it = waitGroups.find(groupName);
        if (it == waitGroups.end()) {
            lastError = "WaitGroup not found: " + groupName;
            return -1;
        }

        auto& wg = it->second;
        if (wg->isDone) {
            return 0;
        }

        if (timeoutMs < 0) {
            wg->cv.wait(lock, [&wg]() { return wg->isDone; });
            return 0;
        } else {
            bool result = wg->cv.wait_for(lock, 
                std::chrono::milliseconds(timeoutMs),
                [&wg]() { return wg->isDone; });
            return result ? 0 : -1;
        }
    } catch (const std::exception& e) {
        lastError = std::string("WaitGroupWait exception: ") + e.what();
        return -1;
    }
}

void GoRuntime::DestroyWaitGroup(const std::string& groupName) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        waitGroups.erase(groupName);
        std::cout << "GoRuntime: Destroyed WaitGroup " << groupName << std::endl;
    } catch (const std::exception& e) {
        lastError = std::string("DestroyWaitGroup exception: ") + e.what();
    }
}

std::string GoRuntime::ScheduleTask(const std::string& functionName,
                                     std::function<void()> callback) {
    try {
        std::string taskId = "task_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        
        Task task;
        task.id = taskId;
        task.functionName = functionName;
        task.callback = callback;
        task.isRunning = true;
        task.isCanceled = false;

        {
            std::lock_guard<std::mutex> lock(taskMutex);
            scheduledTasks[taskId] = task;
        }

        std::cout << "GoRuntime: Scheduled task " << taskId 
                  << " for " << functionName << std::endl;
        return taskId;
    } catch (const std::exception& e) {
        lastError = std::string("ScheduleTask exception: ") + e.what();
        return "";
    }
}

void GoRuntime::CancelTask(const std::string& taskId) {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        auto it = scheduledTasks.find(taskId);
        if (it != scheduledTasks.end()) {
            it->second.isCanceled = true;
            it->second.isRunning = false;
        }
    } catch (const std::exception& e) {
        lastError = std::string("CancelTask exception: ") + e.what();
    }
}

bool GoRuntime::IsTaskRunning(const std::string& taskId) const {
    try {
        std::lock_guard<std::mutex> lock(taskMutex);
        auto it = scheduledTasks.find(taskId);
        return it != scheduledTasks.end() && it->second.isRunning;
    } catch (const std::exception& e) {
        return false;
    }
}

std::string GoRuntime::CallGoFunction(const std::string& functionName, const std::string& parameters) {
    if (!initialized) {
        lastError = "Go runtime not initialized";
        return "";
    }

    try {
        // This would call directly into Go and wait for result
        // Implementation depends on cgo bindings
        return "";
    } catch (const std::exception& e) {
        lastError = std::string("CallGoFunction exception: ") + e.what();
        return "";
    }
}

uint64_t GoRuntime::GetMemoryUsage() const {
    try {
        std::lock_guard<std::mutex> lock(goroutineMutex);
        return (uint64_t)runningGoroutines.size() * 1024 * 1024; // Rough estimate
    } catch (const std::exception& e) {
        return 0;
    }
}

void GoRuntime::Update() {
    ProcessCompletedGoroutines();
    ExecuteTaskCallbacks();
}

void GoRuntime::ProcessCompletedGoroutines() {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    auto it = runningGoroutines.begin();
    while (it != runningGoroutines.end()) {
        if (!it->second.isRunning) {
            it = runningGoroutines.erase(it);
        } else {
            ++it;
        }
    }
}

void GoRuntime::ExecuteTaskCallbacks() {
    std::lock_guard<std::mutex> lock(taskMutex);
    auto it = scheduledTasks.begin();
    while (it != scheduledTasks.end()) {
        if (!it->second.isRunning && it->second.callback) {
            try {
                it->second.callback();
            } catch (const std::exception& e) {
                lastError = std::string("Task callback exception: ") + e.what();
            }
            it = scheduledTasks.erase(it);
        } else {
            ++it;
        }
    }
}
