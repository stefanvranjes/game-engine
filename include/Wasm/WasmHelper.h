#pragma once

#include <string>
#include <cstdint>

/**
 * @namespace WasmHelper
 * Helper utilities for WASM integration and module development
 */
namespace WasmHelper {
    /**
     * Convert WASM memory offset and type to typed access
     */
    template <typename T>
    T ReadValue(std::shared_ptr<class WasmInstance> instance, uint32_t offset) {
        auto data = instance->ReadMemory(offset, sizeof(T));
        if (data.size() < sizeof(T)) {
            return T();
        }
        return *reinterpret_cast<const T*>(data.data());
    }

    template <typename T>
    bool WriteValue(std::shared_ptr<class WasmInstance> instance, uint32_t offset, const T& value) {
        auto ptr = reinterpret_cast<const uint8_t*>(&value);
        std::vector<uint8_t> data(ptr, ptr + sizeof(T));
        return instance->WriteMemory(offset, data);
    }

    /**
     * Convert between WASM and GLM types
     */
    inline uint32_t AllocateVec3(std::shared_ptr<class WasmInstance> instance, 
                                 float x, float y, float z) {
        // Would allocate and initialize a vec3 in WASM memory
        return instance->Malloc(sizeof(float) * 3);
    }

    inline uint32_t AllocateVec4(std::shared_ptr<class WasmInstance> instance,
                                 float x, float y, float z, float w) {
        return instance->Malloc(sizeof(float) * 4);
    }

    /**
     * Helper for module debugging
     */
    void PrintWasmMemory(std::shared_ptr<class WasmInstance> instance, 
                         uint32_t offset, uint32_t size);

    void PrintWasmModule(std::shared_ptr<class WasmModule> module);

    /**
     * Validate module compatibility with engine
     */
    bool ValidateModuleInterface(std::shared_ptr<class WasmModule> module);
}

