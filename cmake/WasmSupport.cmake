# CMake Configuration Snippet for WASM Support
# Add this to your CMakeLists.txt

# WebAssembly (WASM) Support
option(ENABLE_WASM_SUPPORT "Enable WebAssembly support via wasm3" ON)

if(ENABLE_WASM_SUPPORT)
    # Fetch wasm3 - lightweight WASM interpreter
    FetchContent_Declare(
        wasm3
        GIT_REPOSITORY https://github.com/wasm3/wasm3.git
        GIT_TAG v0.5.0
    )
    
    # Configure wasm3
    set(WASM3_FEATURES_SIMD ON CACHE BOOL "Enable SIMD support" FORCE)
    set(WASM3_FEATURES_BULK_MEMORY ON CACHE BOOL "Enable bulk memory operations" FORCE)
    set(WASM3_FEATURES_MEMORY64 OFF CACHE BOOL "Disable 64-bit memory (not widely supported)" FORCE)
    
    FetchContent_MakeAvailable(wasm3)
    
    # Add WASM source files
    set(WASM_SOURCES
        src/Wasm/WasmRuntime.cpp
        src/Wasm/WasmModule.cpp
        src/Wasm/WasmInstance.cpp
        src/Wasm/WasmScriptSystem.cpp
        src/Wasm/WasmEngineBindings.cpp
        src/Wasm/WasmHelper.cpp
    )
    
    # Add WASM include directories
    include_directories(
        ${wasm3_SOURCE_DIR}/source
    )
    
    # Add compile definitions
    add_compile_definitions(
        ENABLE_WASM_SUPPORT
        WASM3_FEATURE_SIMD=1
    )
    
    # Link wasm3
    target_link_libraries(GameEngine PRIVATE m3)
    
    # Platform-specific WASM support
    if(MSVC)
        # MSVC flags for WASM
        target_compile_options(GameEngine PRIVATE /fp:precise)
    else()
        # GCC/Clang flags
        target_compile_options(GameEngine PRIVATE -frounding-math)
    endif()
    
    message(STATUS "WASM Support: ENABLED (wasm3)")
else()
    message(STATUS "WASM Support: DISABLED")
endif()

