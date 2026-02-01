#include "Wasm/WasmHelper.h"
#include "Wasm/WasmInstance.h"
#include "Wasm/WasmModule.h"
#include <iostream>
#include <iomanip>

namespace WasmHelper {

void PrintWasmMemory(std::shared_ptr<::WasmInstance> instance, uint32_t offset, uint32_t size) {
    if (!instance) {
        return;
    }

    auto data = instance->ReadMemory(offset, size);
    
    std::cout << "WASM Memory [" << offset << " - " << (offset + size - 1) << "]:" << std::endl;
    std::cout << std::hex;
    
    for (uint32_t i = 0; i < data.size(); ++i) {
        if (i % 16 == 0) {
            std::cout << "\n0x" << std::setfill('0') << std::setw(4) << (offset + i) << ": ";
        }
        std::cout << std::setfill('0') << std::setw(2) << (int)data[i] << " ";
    }
    
    std::cout << std::dec << std::endl;
}

void PrintWasmModule(std::shared_ptr<::WasmModule> module) {
    if (!module) {
        return;
    }

    auto info = module->GetInfo();
    
    std::cout << "=== WASM Module Info ===" << std::endl;
    std::cout << "Name: " << info.name << std::endl;
    std::cout << "File: " << info.filepath << std::endl;
    std::cout << "Version: " << info.version << std::endl;
    std::cout << "Size: " << info.sizeBytes << " bytes" << std::endl;
    std::cout << "Functions: " << info.numFunctions << std::endl;
    std::cout << "Exports: " << info.numExports << std::endl;
    std::cout << "Validated: " << (info.validated ? "Yes" : "No") << std::endl;
    
    if (!info.exportedFunctions.empty()) {
        std::cout << "Exported Functions:" << std::endl;
        for (const auto& func : info.exportedFunctions) {
            std::cout << "  - " << func << std::endl;
        }
    }
}

std::vector<uint8_t> InternalReadMemory(std::shared_ptr<::WasmInstance> instance, uint32_t offset, uint32_t size) {
    if (!instance) return {};
    return instance->ReadMemory(offset, size);
}

bool InternalWriteMemory(std::shared_ptr<::WasmInstance> instance, uint32_t offset, const std::vector<uint8_t>& data) {
    if (!instance) return false;
    return instance->WriteMemory(offset, data);
}

uint32_t InternalMalloc(std::shared_ptr<::WasmInstance> instance, uint32_t size) {
    if (!instance) return 0;
    return instance->Malloc(size);
}

uint32_t AllocateVec3(std::shared_ptr<::WasmInstance> instance, float x, float y, float z) {
    if (!instance) return 0;
    return instance->Malloc(sizeof(float) * 3);
}

uint32_t AllocateVec4(std::shared_ptr<::WasmInstance> instance, float x, float y, float z, float w) {
    if (!instance) return 0;
    return instance->Malloc(sizeof(float) * 4);
}

bool ValidateModuleInterface(std::shared_ptr<::WasmModule> module) {
    if (!module) {
        return false;
    }

    // Check for required exports
    bool hasInit = module->HasExportedFunction("init");
    bool hasUpdate = module->HasExportedFunction("update");
    bool hasShutdown = module->HasExportedFunction("shutdown");

    // At least one lifecycle function should exist
    if (!hasInit && !hasUpdate && !hasShutdown) {
        std::cout << "Warning: Module doesn't export lifecycle functions (init/update/shutdown)" << std::endl;
    }

    return module->Validate();
}

}

