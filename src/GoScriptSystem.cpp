#include "GoScriptSystem.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <fstream>

// Go bridge fallback stubs
extern "C" {
    void* GoScriptInit() { return nullptr; }
    void GoScriptShutdown(void* handle) { }
    int GoScriptLoadFile(void* handle, const char* filepath) { return -1; }
    int GoScriptExecuteString(void* handle, const char* source) { return -1; }
    int GoScriptStartGoroutine(void* handle, const char* functionName, void* userData) { return -1; }
    int GoScriptWaitGoroutine(void* handle, int goroutineId) { return -1; }
    int GoScriptKillGoroutine(void* handle, int goroutineId) { return -1; }
    int GoScriptChannelSend(void* handle, const char* channelName, const char* jsonData) { return -1; }
    const char* GoScriptChannelReceive(void* handle, const char* channelName, int timeoutMs) { return nullptr; }
    const char* GoScriptCallFunction(void* handle, const char* functionName, const char* jsonArgs) { return nullptr; }
    uint64_t GoScriptGetMemoryUsage(void* handle) { return 0; }
    double GoScriptGetLastExecutionTime(void* handle) { return 0.0; }
    const char* GoScriptGetLastError(void* handle) { return nullptr; }
}

GoScriptSystem::GoScriptSystem()
    : goHandle(nullptr),
      lastExecutionTime(0.0),
      nextGoroutineId(1),
      isInitialized(false) {
}

GoScriptSystem::~GoScriptSystem() {
    if (isInitialized) {
        Shutdown();
    }
}

void GoScriptSystem::Init() {
    if (isInitialized) return;

    try {
        goHandle = GoScriptInit();
        if (!goHandle) {
            lastError = "Failed to initialize Go runtime";
            std::cerr << "GoScriptSystem: " << lastError << std::endl;
            return;
        }

        RegisterTypes();
        isInitialized = true;
        std::cout << "GoScriptSystem initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        lastError = std::string("Init exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
    }
}

void GoScriptSystem::Shutdown() {
    if (!isInitialized) return;

    try {
        // Kill all active goroutines
        {
            std::lock_guard<std::mutex> lock(goroutineMutex);
            for (auto& [id, goroutine] : activeGoroutines) {
                if (goroutine.isRunning) {
                    GoScriptKillGoroutine(goHandle, id);
                }
            }
            activeGoroutines.clear();
        }

        // Close all channels
        {
            std::lock_guard<std::mutex> lock(channelMutex);
            for (auto& [name, channel] : channels) {
                GoScriptChannelReceive(goHandle, name.c_str(), 0);
            }
            channels.clear();
        }

        if (goHandle) {
            GoScriptShutdown(goHandle);
            goHandle = nullptr;
        }

        isInitialized = false;
        std::cout << "GoScriptSystem shutdown complete" << std::endl;
    } catch (const std::exception& e) {
        lastError = std::string("Shutdown exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
    }
}

void GoScriptSystem::Update(float deltaTime) {
    if (!isInitialized || !goHandle) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        UpdateGoroutineStatus();
    } catch (const std::exception& e) {
        lastError = std::string("Update exception: ") + e.what();
        std::cerr << "GoScriptSystem::Update: " << lastError << std::endl;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    lastExecutionTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

bool GoScriptSystem::RunScript(const std::string& filepath) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return false;
    }

    try {
        int result = GoScriptLoadFile(goHandle, filepath.c_str());
        if (result != 0) {
            lastError = std::string("Failed to load Go script: ") + filepath;
            std::cerr << "GoScriptSystem: " << lastError << std::endl;
            return false;
        }

        std::cout << "GoScriptSystem: Loaded script " << filepath << std::endl;
        return true;
    } catch (const std::exception& e) {
        lastError = std::string("RunScript exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
        return false;
    }
}

bool GoScriptSystem::ExecuteString(const std::string& source) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return false;
    }

    try {
        int result = GoScriptExecuteString(goHandle, source.c_str());
        if (result != 0) {
            lastError = "Failed to execute Go source code";
            std::cerr << "GoScriptSystem: " << lastError << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        lastError = std::string("ExecuteString exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
        return false;
    }
}

ScriptLanguage GoScriptSystem::GetLanguage() const {
    return ScriptLanguage::Go;
}

ScriptExecutionMode GoScriptSystem::GetExecutionMode() const {
    return ScriptExecutionMode::NativeCompiled;
}

void GoScriptSystem::RegisterTypes() {
    // Register standard Go game engine types
    // These would be exposed from the Go binding layer
    std::cout << "GoScriptSystem: Registered Go language types" << std::endl;
}

bool GoScriptSystem::HasType(const std::string& typeName) const {
    // In a real implementation, this would check registered Go types
    return !typeName.empty();
}

std::any GoScriptSystem::CallFunction(const std::string& functionName, 
                                       const std::vector<std::any>& args) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return std::any();
    }

    try {
        std::string jsonArgs = SerializeArgs(args);
        const char* result = GoScriptCallFunction(goHandle, functionName.c_str(), jsonArgs.c_str());
        
        if (!result) {
            lastError = std::string("CallFunction failed for: ") + functionName;
            return std::any();
        }

        return DeserializeResult(result);
    } catch (const std::exception& e) {
        lastError = std::string("CallFunction exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
        return std::any();
    }
}

int GoScriptSystem::StartGoroutine(const std::string& functionName, void* userData) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return -1;
    }

    try {
        int goId = GoScriptStartGoroutine(goHandle, functionName.c_str(), userData);
        if (goId < 0) {
            lastError = std::string("Failed to start goroutine: ") + functionName;
            return -1;
        }

        std::lock_guard<std::mutex> lock(goroutineMutex);
        Goroutine g;
        g.id = goId;
        g.functionName = functionName;
        g.isRunning = true;
        g.startTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        activeGoroutines[goId] = g;

        std::cout << "GoScriptSystem: Started goroutine " << goId << " (" << functionName << ")" << std::endl;
        return goId;
    } catch (const std::exception& e) {
        lastError = std::string("StartGoroutine exception: ") + e.what();
        std::cerr << "GoScriptSystem: " << lastError << std::endl;
        return -1;
    }
}

int GoScriptSystem::WaitGoroutine(int goroutineId) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return -1;
    }

    try {
        int result = GoScriptWaitGoroutine(goHandle, goroutineId);
        if (result == 0) {
            std::lock_guard<std::mutex> lock(goroutineMutex);
            auto it = activeGoroutines.find(goroutineId);
            if (it != activeGoroutines.end()) {
                it->second.isRunning = false;
            }
        }
        return result;
    } catch (const std::exception& e) {
        lastError = std::string("WaitGoroutine exception: ") + e.what();
        return -1;
    }
}

int GoScriptSystem::KillGoroutine(int goroutineId) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return -1;
    }

    try {
        int result = GoScriptKillGoroutine(goHandle, goroutineId);
        if (result == 0) {
            std::lock_guard<std::mutex> lock(goroutineMutex);
            activeGoroutines.erase(goroutineId);
        }
        return result;
    } catch (const std::exception& e) {
        lastError = std::string("KillGoroutine exception: ") + e.what();
        return -1;
    }
}

const GoScriptSystem::Goroutine* GoScriptSystem::GetGoroutineStatus(int goroutineId) const {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    auto it = activeGoroutines.find(goroutineId);
    if (it != activeGoroutines.end()) {
        return &it->second;
    }
    return nullptr;
}

size_t GoScriptSystem::GetActiveGoroutineCount() const {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    size_t count = 0;
    for (const auto& [id, goroutine] : activeGoroutines) {
        if (goroutine.isRunning) {
            count++;
        }
    }
    return count;
}

int GoScriptSystem::CreateChannel(const std::string& channelName) {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        if (channels.find(channelName) != channels.end()) {
            return -1; // Channel already exists
        }

        auto channel = std::make_unique<Channel>();
        channel->name = channelName;
        channel->isOpen = true;
        channels[channelName] = std::move(channel);

        std::cout << "GoScriptSystem: Created channel " << channelName << std::endl;
        return 0;
    } catch (const std::exception& e) {
        lastError = std::string("CreateChannel exception: ") + e.what();
        return -1;
    }
}

int GoScriptSystem::SendToChannel(const std::string& channelName, const std::string& jsonData) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return -1;
    }

    try {
        int result = GoScriptChannelSend(goHandle, channelName.c_str(), jsonData.c_str());
        return result;
    } catch (const std::exception& e) {
        lastError = std::string("SendToChannel exception: ") + e.what();
        return -1;
    }
}

std::string GoScriptSystem::ReceiveFromChannel(const std::string& channelName, int timeoutMs) {
    if (!isInitialized || !goHandle) {
        lastError = "Go runtime not initialized";
        return "";
    }

    try {
        const char* data = GoScriptChannelReceive(goHandle, channelName.c_str(), timeoutMs);
        if (!data) {
            return "";
        }
        return std::string(data);
    } catch (const std::exception& e) {
        lastError = std::string("ReceiveFromChannel exception: ") + e.what();
        return "";
    }
}

int GoScriptSystem::CloseChannel(const std::string& channelName) {
    try {
        std::lock_guard<std::mutex> lock(channelMutex);
        auto it = channels.find(channelName);
        if (it != channels.end()) {
            it->second->isOpen = false;
            channels.erase(it);
            return 0;
        }
        return -1; // Channel not found
    } catch (const std::exception& e) {
        lastError = std::string("CloseChannel exception: ") + e.what();
        return -1;
    }
}

void GoScriptSystem::ReloadScript(const std::string& filepath) {
    // Go requires recompilation, so we can't hot-reload
    lastError = "Go scripts require recompilation for updates";
    std::cout << "GoScriptSystem: " << lastError << std::endl;
}

uint64_t GoScriptSystem::GetMemoryUsage() const {
    if (!isInitialized || !goHandle) {
        return 0;
    }
    return GoScriptGetMemoryUsage(goHandle);
}

bool GoScriptSystem::HasErrors() const {
    if (!isInitialized || !goHandle) {
        return !lastError.empty();
    }
    const char* err = GoScriptGetLastError(goHandle);
    return err != nullptr && std::strlen(err) > 0;
}

std::string GoScriptSystem::GetLastError() const {
    if (!lastError.empty()) {
        return lastError;
    }
    if (!isInitialized || !goHandle) {
        return "Go runtime not initialized";
    }
    const char* err = GoScriptGetLastError(goHandle);
    return err ? std::string(err) : "";
}

std::string GoScriptSystem::SerializeArgs(const std::vector<std::any>& args) {
    // Simple JSON serialization for arguments
    // In production, use nlohmann/json
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) oss << ",";
        
        // Handle basic types
        if (args[i].type() == typeid(int)) {
            oss << std::any_cast<int>(args[i]);
        } else if (args[i].type() == typeid(float)) {
            oss << std::any_cast<float>(args[i]);
        } else if (args[i].type() == typeid(double)) {
            oss << std::any_cast<double>(args[i]);
        } else if (args[i].type() == typeid(std::string)) {
            oss << "\"" << std::any_cast<std::string>(args[i]) << "\"";
        } else if (args[i].type() == typeid(bool)) {
            oss << (std::any_cast<bool>(args[i]) ? "true" : "false");
        } else {
            oss << "null";
        }
    }
    oss << "]";
    return oss.str();
}

std::any GoScriptSystem::DeserializeResult(const std::string& jsonResult) {
    // Simple result deserialization
    // In production, use nlohmann/json for full parsing
    if (jsonResult == "null" || jsonResult.empty()) {
        return std::any();
    }
    return std::any(jsonResult);
}

void GoScriptSystem::UpdateGoroutineStatus() {
    std::lock_guard<std::mutex> lock(goroutineMutex);
    
    // Remove completed goroutines
    auto it = activeGoroutines.begin();
    while (it != activeGoroutines.end()) {
        if (!it->second.isRunning) {
            it = activeGoroutines.erase(it);
        } else {
            ++it;
        }
    }
}
