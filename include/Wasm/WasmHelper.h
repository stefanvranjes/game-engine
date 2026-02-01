#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <vector>

class WasmInstance;
class WasmModule;

/**
 * @namespace WasmHelper
 * Helper utilities for WASM integration and module development
 */
namespace WasmHelper {
    // Internal helpers to avoid dereferencing WasmInstance in the header
    std::vector<uint8_t> InternalReadMemory(std::shared_ptr<::WasmInstance> instance, uint32_t offset, uint32_t size);
    bool InternalWriteMemory(std::shared_ptr<::WasmInstance> instance, uint32_t offset, const std::vector<uint8_t>& data);
    uint32_t InternalMalloc(std::shared_ptr<::WasmInstance> instance, uint32_t size);

    /**
     * Convert WASM memory offset and type to typed access
     */
    template <typename T>
    T ReadValue(std::shared_ptr<::WasmInstance> instance, uint32_t offset) {
        auto data = InternalReadMemory(instance, offset, sizeof(T));
        if (data.size() < sizeof(T)) {
            return T();
        }
        return *reinterpret_cast<const T*>(data.data());
    }

    template <typename T>
    bool WriteValue(std::shared_ptr<::WasmInstance> instance, uint32_t offset, const T& value) {
        auto ptr = reinterpret_cast<const uint8_t*>(&value);
        std::vector<uint8_t> data(ptr, ptr + sizeof(T));
        return InternalWriteMemory(instance, offset, data);
    }

    /**
     * Convert between WASM and GLM types
     */
    uint32_t AllocateVec3(std::shared_ptr<::WasmInstance> instance, float x, float y, float z);
    uint32_t AllocateVec4(std::shared_ptr<::WasmInstance> instance, float x, float y, float z, float w);

    /**
     * Helper for module debugging
     */
    void PrintWasmMemory(std::shared_ptr<::WasmInstance> instance, 
                         uint32_t offset, uint32_t size);

    void PrintWasmModule(std::shared_ptr<::WasmModule> module);

    /**
     * Validate module compatibility with engine
     */
    bool ValidateModuleInterface(std::shared_ptr<::WasmModule> module);
}

