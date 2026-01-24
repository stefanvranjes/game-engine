#include "Wasm/WasmModule.h"
#include "Wasm/WasmInstance.h"
#include <algorithm>

// wasm3 includes
#include <m3.h>
#include <m3_env.h>

WasmValue WasmValue::I32(int32_t v) {
    WasmValue val;
    val.type = WasmValueType::I32;
    val.value = v;
    return val;
}

WasmValue WasmValue::I64(int64_t v) {
    WasmValue val;
    val.type = WasmValueType::I64;
    val.value = v;
    return val;
}

WasmValue WasmValue::F32(float v) {
    WasmValue val;
    val.type = WasmValueType::F32;
    val.value = v;
    return val;
}

WasmValue WasmValue::F64(double v) {
    WasmValue val;
    val.type = WasmValueType::F64;
    val.value = v;
    return val;
}

WasmValue WasmValue::Ptr(uintptr_t ptr) {
    WasmValue val;
    val.type = WasmValueType::I32;
    val.value = static_cast<int32_t>(ptr);
    return val;
}

// WasmModule Implementation

WasmModule::WasmModule(const std::string& name, const std::string& filepath)
    : m_Name(name), m_FilePath(filepath) {
}

std::vector<std::string> WasmModule::GetExportedFunctions() const {
    CacheExports();
    return m_ExportedFunctions;
}

std::vector<std::string> WasmModule::GetExportedMemories() const {
    std::vector<std::string> memories;
    // wasm3 doesn't directly expose memory exports in the same way
    // Memory is typically exported as "memory"
    if (m_Module) {
        memories.push_back("memory");
    }
    return memories;
}

std::vector<std::string> WasmModule::GetExportedGlobals() const {
    std::vector<std::string> globals;
    // Extract globals from module
    // This requires deeper wasm3 API usage
    return globals;
}

std::vector<std::string> WasmModule::GetExportedTables() const {
    std::vector<std::string> tables;
    // Extract tables from module
    return tables;
}

bool WasmModule::HasExportedFunction(const std::string& name) const {
    CacheExports();
    return std::find(m_ExportedFunctions.begin(), m_ExportedFunctions.end(), name) 
           != m_ExportedFunctions.end();
}

bool WasmModule::GetFunctionSignature(const std::string& name, FunctionSignature& outSig) const {
    CacheExports();
    auto it = m_FunctionSignatures.find(name);
    if (it != m_FunctionSignatures.end()) {
        outSig = it->second;
        return true;
    }
    return false;
}

std::shared_ptr<WasmInstance> WasmModule::CreateInstance() {
    auto instance = std::make_shared<WasmInstance>(std::make_shared<WasmModule>(*this));
    if (instance && instance->GetMemory()) {
        return instance;
    }
    return nullptr;
}

bool WasmModule::Validate() const {
    if (!m_Module || m_ModuleData.empty()) {
        return false;
    }

    // Check magic number
    if (m_ModuleData.size() < 4 ||
        m_ModuleData[0] != 0x00 || m_ModuleData[1] != 0x61 ||
        m_ModuleData[2] != 0x73 || m_ModuleData[3] != 0x6d) {
        return false;
    }

    // Check version
    if (m_ModuleData.size() < 8) {
        return false;
    }
    uint32_t version = m_ModuleData[4] | (m_ModuleData[5] << 8) | 
                       (m_ModuleData[6] << 16) | (m_ModuleData[7] << 24);
    if (version != 1) {
        return false;  // Only version 1 is supported
    }

    return true;
}

WasmModule::ModuleInfo WasmModule::GetInfo() const {
    ModuleInfo info;
    info.name = m_Name;
    info.filepath = m_FilePath;
    info.version = m_Version;
    info.sizeBytes = m_ModuleData.size();
    info.validated = Validate();
    info.exportedFunctions = GetExportedFunctions();
    info.numExports = info.exportedFunctions.size();
    info.numFunctions = 0;  // Would need to parse to get accurate count
    return info;
}

void WasmModule::CacheExports() const {
    if (m_ExportsCached) {
        return;
    }

    m_ExportedFunctions.clear();
    m_FunctionSignatures.clear();

    if (!m_Module) {
        m_ExportsCached = true;
        return;
    }

    // wasm3 provides M3Module which has export information
    // This would require deeper wasm3 API exploration to implement fully
    // For now, we'll populate this when instances call exported functions

    m_ExportsCached = true;
}

