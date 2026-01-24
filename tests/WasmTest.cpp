#pragma once

#include "Wasm/WasmScriptSystem.h"
#include "Wasm/WasmEngineBindings.h"
#include <gtest/gtest.h>
#include <memory>

/**
 * @class WasmTest
 * Unit tests for WASM support system
 */
class WasmTest : public ::testing::Test {
protected:
    void SetUp() override {
        WasmScriptSystem::GetInstance().Init();
    }

    void TearDown() override {
        WasmScriptSystem::GetInstance().Shutdown();
    }
};

// Test WASM runtime initialization
TEST_F(WasmTest, RuntimeInitialization) {
    auto& runtime = WasmRuntime::GetInstance();
    EXPECT_TRUE(runtime.IsInitialized());
}

// Test module loading (requires test.wasm)
TEST_F(WasmTest, ModuleLoading) {
    auto& wasmSys = WasmScriptSystem::GetInstance();
    
    // This would require a test WASM module
    // For now, test the system structure
    EXPECT_TRUE(wasmSys.GetLoadedModules().empty());
}

// Test memory access
TEST_F(WasmTest, MemoryAccess) {
    // Would require a loaded instance
    // Testing memory bounds checking, reads/writes
}

// Test function calls
TEST_F(WasmTest, FunctionCalls) {
    // Would require a loaded instance
    // Testing function invocation and return values
}

// Test lifecycle hooks
TEST_F(WasmTest, LifecycleHooks) {
    // Test init/update/shutdown callbacks
}

// Test profiling
TEST_F(WasmTest, Profiling) {
    // Test performance data collection
}

// Test error handling
TEST_F(WasmTest, ErrorHandling) {
    auto& runtime = WasmRuntime::GetInstance();
    runtime.ClearLastError();
    EXPECT_TRUE(runtime.GetLastError().empty());
}

// Test bindings registration
TEST_F(WasmTest, EngineBindings) {
    WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();
    // Test binding registration
}

