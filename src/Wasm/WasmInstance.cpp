#include "Wasm/WasmInstance.h"
#include "Wasm/WasmModule.h"
#include <algorithm>
#include <chrono>
#include <cstring>

// wasm3 includes
#include <wasm3.h>
#include <m3_env.h>

WasmInstance::WasmInstance(std::shared_ptr<WasmModule> module)
    : m_Module(module) {
    
    if (!module || !module->m_Module) {
        return;
    }

    try {
        // Create execution environment for this instance
        // This would involve setting up a new runtime or reusing one
        // The exact approach depends on wasm3 API usage

        // Allocate default memory
        m_MemorySize = 256 * 1024;  // 256 KB default
        m_Memory = new uint8_t[m_MemorySize];
        if (m_Memory) {
            std::memset(m_Memory, 0, m_MemorySize);
        }
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Failed to create WasmInstance: ") + e.what();
    }
}

WasmInstance::~WasmInstance() {
    if (m_Memory) {
        delete[] m_Memory;
        m_Memory = nullptr;
    }
    m_MemorySize = 0;
}

WasmValue WasmInstance::Call(const std::string& functionName) {
    std::vector<WasmValue> args;
    return Call(functionName, args);
}

WasmValue WasmInstance::Call(const std::string& functionName, const std::vector<WasmValue>& args) {
    if (!m_Execution) {
        m_LastError = "WasmInstance not properly initialized";
        return WasmValue::I32(0);
    }

    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Look up function in module
        M3Function* func = nullptr;
        // This would require wasm3 API to find function by name
        
        if (!func) {
            m_LastError = std::string("Function not found: ") + functionName;
            return WasmValue::I32(0);
        }

        // Call function with arguments
        // wasm3 call mechanism would be used here
        M3Result result = m3Err_none;  // Placeholder
        
        if (result != m3Err_none) {
            m_LastError = std::string("Function call failed: ") + result;
            return WasmValue::I32(0);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        m_LastCallDuration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        // Update profiling
        if (m_ProfilingEnabled) {
            auto& data = m_ProfilingMap[functionName];
            data.functionName = functionName;
            data.callCount++;
            data.totalTime += m_LastCallDuration;
            data.averageTime = data.totalTime / data.callCount;
            if (m_LastCallDuration < data.minTime || data.minTime == 0) {
                data.minTime = m_LastCallDuration;
            }
            if (m_LastCallDuration > data.maxTime) {
                data.maxTime = m_LastCallDuration;
            }
        }

        m_LastError.clear();
        return WasmValue::I32(0);  // Placeholder return
    }
    catch (const std::exception& e) {
        m_LastError = std::string("Exception during function call: ") + e.what();
        return WasmValue::I32(0);
    }
}

std::vector<uint8_t> WasmInstance::ReadMemory(uint32_t offset, uint32_t size) const {
    std::vector<uint8_t> result;
    
    if (!m_Memory || offset + size > m_MemorySize) {
        return result;  // Return empty on bounds error
    }

    result.assign(m_Memory + offset, m_Memory + offset + size);
    return result;
}

bool WasmInstance::WriteMemory(uint32_t offset, const std::vector<uint8_t>& data) {
    if (!m_Memory || offset + data.size() > m_MemorySize) {
        m_LastError = "Memory write out of bounds";
        return false;
    }

    std::memcpy(m_Memory + offset, data.data(), data.size());
    return true;
}

std::string WasmInstance::ReadString(uint32_t offset) const {
    if (!m_Memory || offset >= m_MemorySize) {
        return "";
    }

    std::string result;
    for (size_t i = offset; i < m_MemorySize && m_Memory[i] != '\0'; ++i) {
        result += static_cast<char>(m_Memory[i]);
    }
    return result;
}

bool WasmInstance::WriteString(uint32_t offset, const std::string& str) {
    if (!m_Memory || offset + str.size() + 1 > m_MemorySize) {
        m_LastError = "String write out of bounds";
        return false;
    }

    std::memcpy(m_Memory + offset, str.c_str(), str.size() + 1);
    return true;
}

uint32_t WasmInstance::Malloc(uint32_t size) {
    // This requires either a simple bump allocator or calling the WASM malloc function
    // For now, return 0 to indicate failure
    m_LastError = "Malloc not implemented";
    return 0;
}

void WasmInstance::Free(uint32_t offset) {
    // Would interface with WASM free function or allocator
}

WasmValue WasmInstance::GetGlobal(const std::string& name) const {
    m_LastError = "GetGlobal not implemented";
    return WasmValue::I32(0);
}

bool WasmInstance::SetGlobal(const std::string& name, const WasmValue& value) {
    m_LastError = "SetGlobal not implemented";
    return false;
}

WasmInstance::InstanceStats WasmInstance::GetStats() const {
    InstanceStats stats;
    stats.totalMemory = m_MemorySize;
    stats.usedMemory = 0;  // Would need to track allocations
    stats.callDepth = 0;
    stats.totalCalls = 0;
    stats.lastCallTime = 0;
    return stats;
}

void WasmInstance::Reset() {
    if (m_Memory) {
        std::memset(m_Memory, 0, m_MemorySize);
    }
    m_LastError.clear();
    m_LastCallDuration = 0.0;
    m_ProfilingMap.clear();
}

bool WasmInstance::RegisterHostCallback(const std::string& name, HostCallback callback) {
    if (!callback) {
        return false;
    }
    m_HostCallbacks[name] = callback;
    return true;
}

std::vector<WasmInstance::ProfilingData> WasmInstance::GetProfilingData() const {
    std::vector<ProfilingData> data;
    for (const auto& pair : m_ProfilingMap) {
        data.push_back(pair.second);
    }
    return data;
}

void WasmInstance::PackSingleArgument(std::vector<WasmValue>& args, int32_t val) {
    args.push_back(WasmValue::I32(val));
}

void WasmInstance::PackSingleArgument(std::vector<WasmValue>& args, int64_t val) {
    args.push_back(WasmValue::I64(val));
}

void WasmInstance::PackSingleArgument(std::vector<WasmValue>& args, float val) {
    args.push_back(WasmValue::F32(val));
}

void WasmInstance::PackSingleArgument(std::vector<WasmValue>& args, double val) {
    args.push_back(WasmValue::F64(val));
}

