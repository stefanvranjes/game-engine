# Script Debugger CMake Integration Guide

## Automatic Inclusion

The Script Debugger is automatically compiled and linked when building the game engine. No additional configuration is required.

### Files Automatically Included

```cmake
# In CMakeLists.txt, the following are automatically added:

set(ENGINE_SOURCES
    ${ENGINE_SOURCES}
    src/ScriptDebugger.cpp
    src/ScriptDebuggerUI.cpp
)
```

## Manual Integration (if needed)

If you're integrating into a separate project:

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(YourProject)

# Enable C++17 for std::any support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add engine includes
target_include_directories(YourProject PRIVATE ${ENGINE_INCLUDE_DIR})

# Add debugger sources
target_sources(YourProject PRIVATE
    ${ENGINE_SOURCE_DIR}/ScriptDebugger.cpp
    ${ENGINE_SOURCE_DIR}/ScriptDebuggerUI.cpp
)

# Link ImGui (required)
target_link_libraries(YourProject PRIVATE imgui)
```

## Conditional Compilation

### Debug Only Build

```cmake
# Optionally only include in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_sources(YourProject PRIVATE
        src/ScriptDebugger.cpp
        src/ScriptDebuggerUI.cpp
    )
    target_compile_definitions(YourProject PRIVATE ENABLE_SCRIPT_DEBUGGER)
endif()
```

### Feature Flag

```cmake
# Allow disabling via option
option(ENABLE_SCRIPT_DEBUGGER "Enable Script Debugger UI" ON)

if(ENABLE_SCRIPT_DEBUGGER)
    target_sources(YourProject PRIVATE
        src/ScriptDebugger.cpp
        src/ScriptDebuggerUI.cpp
    )
    target_compile_definitions(YourProject PRIVATE ENABLE_SCRIPT_DEBUGGER)
endif()
```

## With Custom Build Settings

```cmake
# For custom optimization levels
if(MSVC)
    target_compile_options(ScriptDebugger PRIVATE /W4 /WX)
else()
    target_compile_options(ScriptDebugger PRIVATE -Wall -Wextra -Werror)
endif()

# For LTO (Link Time Optimization)
if(ENABLE_LTO)
    set_property(TARGET ScriptDebugger PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
```

## Compiler Requirements

```cmake
# Enforce minimum C++ standard
target_compile_features(GameEngine PRIVATE cxx_std_17)

# Check compiler compatibility
if(MSVC AND MSVC_VERSION LESS 1920)
    message(FATAL_ERROR "MSVC 2019 (v142) or newer required for C++17 support")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message(FATAL_ERROR "GCC 9 or newer required for std::any support")
endif()
```

## Runtime Configuration

The debugger can be configured at runtime:

```cpp
// In Application.cpp
m_ScriptDebuggerUI = std::make_unique<ScriptDebuggerUI>();
m_ScriptDebuggerUI->Init();

#ifdef DEBUG
// Auto-show debugger in debug builds (optional)
m_ScriptDebuggerUI->Show();
#endif

// Custom theme for dark mode
m_ScriptDebuggerUI->SetThemeDarkMode(true);
```

## Dependencies

### Required
- C++17 or later
- ImGui 1.8 or later
- Standard Library (no external deps)

### Optional Integration
- AngelScript (for AngelScript debugging)
- Lua/LuaJIT (for Lua debugging)
- Python (for Python debugging)
- Other script systems as needed

## Performance Considerations

### Optimization Flags

For release builds with debugger:

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Enable optimizations
    if(MSVC)
        target_compile_options(GameEngine PRIVATE /O2 /Ob2)
    else()
        target_compile_options(GameEngine PRIVATE -O3 -march=native)
    endif()
endif()
```

### Linker Optimization

```cmake
# Strip unused symbols (Linux/macOS)
if(UNIX AND NOT APPLE)
    target_link_options(GameEngine PRIVATE -Wl,--gc-sections)
endif()
```

## Installation

If packaging the engine:

```cmake
# Install headers
install(FILES 
    include/ScriptDebugger.h
    include/ScriptDebuggerUI.h
    DESTINATION include)

# Install libraries (if built separately)
install(TARGETS ScriptDebugger
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
```

## Testing Integration

```cmake
# Unit tests for debugger
if(ENABLE_TESTING)
    add_executable(ScriptDebuggerTests
        tests/ScriptDebuggerTest.cpp
        src/ScriptDebugger.cpp
        src/ScriptDebuggerUI.cpp
    )
    
    target_link_libraries(ScriptDebuggerTests PRIVATE gtest gtest_main imgui)
    add_test(NAME ScriptDebuggerTests COMMAND ScriptDebuggerTests)
endif()
```

## Cross-Platform Support

### Windows
```cmake
if(WIN32)
    target_compile_definitions(GameEngine PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
```

### Linux
```cmake
if(UNIX AND NOT APPLE)
    # No special configuration needed
endif()
```

### macOS
```cmake
if(APPLE)
    target_compile_options(GameEngine PRIVATE -fno-strict-aliasing)
endif()
```

## Troubleshooting CMake

### Include Path Issues
```cmake
# Explicitly add include paths if build fails
target_include_directories(GameEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
```

### Missing ImGui
```cmake
# Find ImGui
find_package(imgui REQUIRED)
target_link_libraries(GameEngine PRIVATE imgui::imgui)
```

### C++ Standard Issues
```cmake
# Verify C++17 is enabled
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(CMAKE_CXX_STANDARD LESS 17)
    message(FATAL_ERROR "C++17 or later required")
endif()
```

## Example Full CMakeLists Configuration

```cmake
cmake_minimum_required(VERSION 3.15)
project(GameEngine)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Script Debugger sources
set(DEBUGGER_SOURCES
    src/ScriptDebugger.cpp
    src/ScriptDebuggerUI.cpp
)

# Main executable
add_executable(GameEngine 
    src/main.cpp
    src/Application.cpp
    ${DEBUGGER_SOURCES}
    # ... other sources
)

# Include directories
target_include_directories(GameEngine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui
)

# Link libraries
find_package(imgui REQUIRED)
target_link_libraries(GameEngine PRIVATE 
    imgui::imgui
    # ... other libraries
)

# Optional: Debug-only features
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(GameEngine PRIVATE DEBUG_ENABLED)
endif()

# Installation
install(TARGETS GameEngine DESTINATION bin)
```

## Version History

### v1.0
- Initial release with core debugging features
- ImGui-based UI
- Breakpoint management
- Call stack inspection
- Variable watching
- Console output

## See Also

- [SCRIPT_DEBUGGER_GUIDE.md](SCRIPT_DEBUGGER_GUIDE.md) - User guide
- [SCRIPT_DEBUGGER_IMPLEMENTATION.md](SCRIPT_DEBUGGER_IMPLEMENTATION.md) - Technical details
- [CMakeLists.txt](CMakeLists.txt) - Main build file
