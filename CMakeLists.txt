cmake_minimum_required(VERSION 3.10)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(GameEngine VERSION 0.1.0 LANGUAGES CXX C ASM_MASM)

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW) 
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(NOMINMAX)
add_compile_definitions(WIN32_LEAN_AND_MEAN)
add_compile_definitions(TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION)
add_compile_definitions(TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION)

# Go language support
option(ENABLE_GO_SUPPORT "Enable Go language scripting support" ON)
if(ENABLE_GO_SUPPORT)
    # Check for Go installation
    find_program(GO_EXECUTABLE go)
    if(GO_EXECUTABLE)
        message(STATUS "Go found: ${GO_EXECUTABLE}")
        add_compile_definitions(GO_LANGUAGE_SUPPORT)
    else()
        message(WARNING "Go not found - Go scripting support disabled")
        set(ENABLE_GO_SUPPORT OFF)
    endif()
endif()

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/FS /W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Sanitizers configuration
option(USE_ASAN "Enable Address Sanitizer" OFF)
option(USE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(USE_MSAN "Enable Memory Sanitizer" OFF)
option(USE_TSAN "Enable Thread Sanitizer" OFF)

if(USE_ASAN AND NOT MSVC)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(USE_UBSAN AND NOT MSVC)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

if(USE_MSAN AND NOT MSVC)
    add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
    add_link_options(-fsanitize=memory)
endif()

if(USE_TSAN AND NOT MSVC)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# Clang-Tidy configuration
option(ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)

if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_COMMAND clang-tidy)
    if(CLANG_TIDY_COMMAND)
        set(CMAKE_CXX_CLANG_TIDY
            "${CLANG_TIDY_COMMAND}"
            "-checks=readability-*,performance-*,modernize-*,bugprone-*,portability-*"
            "-header-filter=.*"
            "-warnings-as-errors=*"
        )
    endif()
endif()

# FetchContent
include(FetchContent)

# Fetch GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch GLFW

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw)

# Fetch nlohmann_json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Fetch tinygltf
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.8.13
)
# Prevent tinygltf from defining STB_IMAGE_IMPLEMENTATION (we define it in Texture.cpp)
set(TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION ON CACHE BOOL "" FORCE)
set(TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# Fetch miniaudio
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.21
)
FetchContent_MakeAvailable(miniaudio)

# Fetch ENet
FetchContent_Declare(
    enet
    GIT_REPOSITORY https://github.com/lsalzman/enet.git
    GIT_TAG v1.3.17
)
FetchContent_MakeAvailable(enet)

# Fetch spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)


# Fetch ImGui (Docking Branch)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG docking
)
FetchContent_GetProperties(imgui)
if(NOT imgui_populated)
    FetchContent_Populate(imgui)
    
    # Define imgui target
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    
    target_include_directories(imgui PUBLIC 
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    
    target_link_libraries(imgui PRIVATE glfw)

    
    # We want #include "imgui/imgui.h" to work, so we add a symlink or an include alias
    # Simpler: just add imgui_SOURCE_DIR as a directory named 'imgui' in the include search
    # But since we can't easily rename FetchContent dirs, we add the parent of the fetched dir
    # and hope the dir is called 'imgui-src' or we just fix the project-wide includes.
    # Actually, a better way is to provide a local include/imgui that just forwards if needed,
    # or just add the source dir directly.
endif()


# Graphics Backend Selection (NVRHI)
option(GRAPHICS_BACKEND "Graphics backend for NVRHI: D3D12, D3D11, VULKAN" "D3D12")
message(STATUS "Graphics Backend: ${GRAPHICS_BACKEND}")

# Fetch NVRHI (NVIDIA Rendering Hardware Interface)
# https://github.com/NVIDIA-RTX/NVRHI.git
FetchContent_Declare(
    nvrhi
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NVRHI.git
    GIT_TAG main
)

# Build NVRHI backends based on platform
if(WIN32)
    set(NVRHI_WITH_DX12 ON CACHE BOOL "" FORCE)
    set(NVRHI_WITH_DX11 ON CACHE BOOL "" FORCE)
endif()
set(NVRHI_WITH_VULKAN ON CACHE BOOL "" FORCE)
set(NVRHI_WITH_OPENGL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nvrhi)
set(NVRHI_SOURCE_DIR ${nvrhi_SOURCE_DIR})

# Fetch Assimp (Open Asset Import Library)
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG v5.4.3
)
FetchContent_GetProperties(assimp)
if(NOT assimp_populated)
    FetchContent_Populate(assimp)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
    set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
    # Only build necessary importers to speed up compilation
    set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_FBX_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_COLLADA_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_BLEND_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_IQM_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_MD5_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${assimp_SOURCE_DIR} assimp_build EXCLUDE_FROM_ALL)
endif()

# Physics Backend Selection
option(PHYSICS_BACKEND "Physics backend to use: BULLET or PHYSX" "PHYSX")

if(PHYSICS_BACKEND STREQUAL "PHYSX")
    message(STATUS "Using PhysX physics backend")
    add_compile_definitions(USE_PHYSX)
    
    # Try to enable CUDA language for GPU queries (optional)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit found - GPU memory queries will use actual CUDA API")
            add_compile_definitions(HAS_CUDA_TOOLKIT)
            
            # Enable NVTX profiling markers (header-only, included with CUDA Toolkit)
            add_compile_definitions(HAS_NVTX)
            message(STATUS "NVTX profiling markers enabled")
        else()
            message(WARNING "CUDA Toolkit not found - GPU memory queries will use estimates")
        endif()
    else()
        message(WARNING "CUDA compiler not found - GPU memory queries will use estimates")
    endif()
    
    # Fetch PhysX SDK
    FetchContent_Declare(
        physx
        GIT_REPOSITORY https://github.com/NVIDIA-Omniverse/PhysX.git
        GIT_TAG 106.0-physx-5.4.1
    )
    FetchContent_GetProperties(physx)
    if(NOT physx_populated)
        FetchContent_Populate(physx)
        
        # PhysX build configuration
        set(PHYSX_ROOT_DIR ${physx_SOURCE_DIR}/physx)
        set(PXSHARED_ROOT_DIR ${physx_SOURCE_DIR}/pxshared)
        set(TARGET_BUILD_PLATFORM "windows")
        set(PX_BUILDSNIPPETS OFF CACHE BOOL "" FORCE)
        set(PX_BUILDSAMPLES OFF CACHE BOOL "" FORCE)
        set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "" FORCE)
        set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "" FORCE)

        set(PX_OUTPUT_LIB_DIR ${CMAKE_BINARY_DIR}/lib CACHE PATH "" FORCE)
        set(PX_OUTPUT_BIN_DIR ${CMAKE_BINARY_DIR}/bin CACHE PATH "" FORCE)
        set(PX_OUTPUT_DLL_DIR ${CMAKE_BINARY_DIR}/bin CACHE PATH "" FORCE)
        set(PX_COPY_EXTERNAL_DLL OFF CACHE BOOL "" FORCE)
        set(PUBLIC_RELEASE OFF CACHE BOOL "" FORCE)
        message(STATUS "DEBUG: PX_COPY_EXTERNAL_DLL=${PX_COPY_EXTERNAL_DLL}")
        message(STATUS "DEBUG: PUBLIC_RELEASE=${PUBLIC_RELEASE}")
        
        # Add PhysX subdirectory
        add_subdirectory(${PHYSX_ROOT_DIR}/compiler/public ${CMAKE_BINARY_DIR}/physx_build)
    endif()
else()
    message(STATUS "Using Bullet3D physics backend (default)")
    add_compile_definitions(USE_BULLET)
    
    # Fetch Bullet3D Physics Engine
    FetchContent_Declare(
        bullet3
        GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
        GIT_TAG master
    )
    FetchContent_GetProperties(bullet3)
    if(NOT bullet3_populated)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
        set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
        set(BUILD_CPU_DEMOS OFF CACHE BOOL "" FORCE)
        set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)
        set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
        set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(bullet3)
    endif()
endif()

# Fetch Box2D
FetchContent_Declare(
    box2d
    GIT_REPOSITORY https://github.com/erincatto/box2d.git
    GIT_TAG v2.4.1
)
set(BOX2D_BUILD_TESTBED OFF CACHE BOOL "" FORCE)
set(BOX2D_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(box2d)

if(PHYSICS_BACKEND STREQUAL "BOX2D")
    message(STATUS "Using Box2D physics backend")
    add_compile_definitions(USE_BOX2D)
endif()

# LuaJIT Support - 10x+ performance improvement over standard Lua
set(ENABLE_LUAJIT OFF CACHE BOOL "Enable LuaJIT (JIT-compiled Lua) for 10x+ performance" FORCE)

if(ENABLE_LUAJIT)
    # Fetch LuaJIT - Just-In-Time compiled Lua for significantly better performance
    FetchContent_Declare(
        luajit
        GIT_REPOSITORY https://github.com/LuaJIT/LuaJIT.git
        GIT_TAG v2.1
    )
    FetchContent_GetProperties(luajit)
    if(NOT luajit_populated)
        FetchContent_Populate(luajit)
        
        # Build LuaJIT as a static library
        # LuaJIT uses a custom build system, so we'll use its CMakeLists or define targets
        file(GLOB LUAJIT_CORE_SOURCES
            "${luajit_SOURCE_DIR}/src/lj_*.c"
            "${luajit_SOURCE_DIR}/src/lib*.c"
        )
        
        # Exclude VM and other specific sources that need special handling
        list(FILTER LUAJIT_CORE_SOURCES EXCLUDE REGEX "lj_trace\\.c")
        
        # Build the main LuaJIT library
        add_library(luajit STATIC ${LUAJIT_CORE_SOURCES})
        target_include_directories(luajit PUBLIC "${luajit_SOURCE_DIR}/src")
        
        # Platform-specific definitions for LuaJIT
        target_compile_definitions(luajit PRIVATE
            LUAJIT_ENABLE_LUA52COMPAT
            LUAJIT_ENABLE_GC64
        )
        
        # Windows specific settings
        if(WIN32)
            target_compile_definitions(luajit PRIVATE
                _CRT_SECURE_NO_WARNINGS
                _WINSOCK_DEPRECATED_NO_WARNINGS
            )
        endif()
        
        # Set LuaJIT as preferred Lua implementation
        set(LUA_INCLUDE_DIR "${luajit_SOURCE_DIR}/src")
        set(LUA_LIBRARY luajit)
        message(STATUS "LuaJIT 2.1 enabled - 10x+ performance vs standard Lua")
    endif()
else()
    # Fallback to standard Lua if LuaJIT disabled
    message(STATUS "LuaJIT disabled, using standard Lua 5.4")
    
    FetchContent_Declare(
        lua
        GIT_REPOSITORY https://github.com/lua/lua.git
        GIT_TAG v5.4.6
    )
    FetchContent_GetProperties(lua)
    if(NOT lua_populated)
        FetchContent_Populate(lua)
        
        file(GLOB LUA_SOURCES 
            "${lua_SOURCE_DIR}/*.c"
        )
        # Exclude lua.c and luac.c for the library build
        list(FILTER LUA_SOURCES EXCLUDE REGEX "lua\\.c")
        list(FILTER LUA_SOURCES EXCLUDE REGEX "luac\\.c")

        add_library(lua STATIC ${LUA_SOURCES})
        target_include_directories(lua PUBLIC ${lua_SOURCE_DIR})
        
        # Lua needs to be compiled as C++ if being linked into C++ project with C++ exceptions potential
        target_compile_definitions(lua PUBLIC LUA_USE_WINDOWS)
        
        set(LUA_INCLUDE_DIR ${lua_SOURCE_DIR})
        set(LUA_LIBRARY lua)
    endif()
endif()

# Fetch Wren - lightweight scripting language for gameplay logic
FetchContent_Declare(
    wren
    GIT_REPOSITORY https://github.com/wren-lang/wren.git
    GIT_TAG 0.4.0
)
FetchContent_GetProperties(wren)
if(NOT wren_populated)
    FetchContent_Populate(wren)
    
    # Build Wren as a static library
    file(GLOB WREN_SOURCES
        "${wren_SOURCE_DIR}/src/vm/*.c"
        "${wren_SOURCE_DIR}/src/optional/*.c"
    )
    
    add_library(wren STATIC ${WREN_SOURCES})
    target_include_directories(wren PUBLIC 
        "${wren_SOURCE_DIR}/src/include"
    )
    target_include_directories(wren PRIVATE
        "${wren_SOURCE_DIR}/src/vm"
        "${wren_SOURCE_DIR}/src/optional"
    )
    
    # Wren configuration
    target_compile_definitions(wren PRIVATE 
        WREN_OPT_RANDOM=1
        WREN_OPT_META=1
    )
endif()

# Fetch AngelScript - lightweight, C++-like scripting language
option(ENABLE_ANGELSCRIPT "Enable AngelScript language support" ON)
if(ENABLE_ANGELSCRIPT)
    FetchContent_Declare(
        angelscript
        GIT_REPOSITORY https://github.com/anjo76/angelscript.git
        GIT_TAG v2.38.0
    )
    FetchContent_GetProperties(angelscript)
    if(NOT angelscript_populated)
        FetchContent_Populate(angelscript)
        
        # Build AngelScript as a static library
        file(GLOB ANGELSCRIPT_SOURCES
            "${angelscript_SOURCE_DIR}/sdk/angelscript/source/*.cpp"
        )
        
        if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
            list(APPEND ANGELSCRIPT_SOURCES "${angelscript_SOURCE_DIR}/sdk/angelscript/source/as_callfunc_x64_msvc_asm.asm")
        endif()
        
        add_library(angelscript STATIC ${ANGELSCRIPT_SOURCES})
        target_include_directories(angelscript PUBLIC 
            "${angelscript_SOURCE_DIR}/sdk/angelscript/include"
        )
        
        # AngelScript configuration
        if(MSVC)
            target_compile_definitions(angelscript PRIVATE 
                AS_WINDOWS
            )
        endif()
        
        # Disable exceptions by default (AngelScript handles errors internally)
        set_target_properties(angelscript PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
        )
    endif()
    set(ANGELSCRIPT_AVAILABLE TRUE)
else()
    set(ANGELSCRIPT_AVAILABLE FALSE)
endif()

# Fetch QuickJS - Lightweight JavaScript/TypeScript engine for scripting support
option(ENABLE_QUICKJS "Enable QuickJS (JavaScript/TypeScript support)" ON)
if(ENABLE_QUICKJS)
    FetchContent_Declare(
        quickjs
        GIT_REPOSITORY https://github.com/bellard/quickjs.git
        GIT_TAG master
    )
    FetchContent_GetProperties(quickjs)
    if(NOT quickjs_populated)
        FetchContent_Populate(quickjs)
        file(REMOVE "${quickjs_SOURCE_DIR}/version")
        
        # Build QuickJS as a static library
        file(GLOB QUICKJS_SOURCES
            "${quickjs_SOURCE_DIR}/quickjs.c"
            "${quickjs_SOURCE_DIR}/libregexp.c"
            "${quickjs_SOURCE_DIR}/libunicode.c"
            "${quickjs_SOURCE_DIR}/libbf.c"
            "${quickjs_SOURCE_DIR}/cutils.c"
        )
        
        add_library(quickjs STATIC ${QUICKJS_SOURCES})
        target_include_directories(quickjs PUBLIC 
            "${quickjs_SOURCE_DIR}"
        )
        
        # QuickJS configuration for Windows/MSVC
        if(MSVC)
            target_compile_definitions(quickjs PRIVATE 
                _CRT_SECURE_NO_WARNINGS
                _WINSOCK_DEPRECATED_NO_WARNINGS
            )
            # Fix "error C2099: initializer is not a constant" in Release builds
            # by disabling math intrinsics for quickjs.c
            target_compile_options(quickjs PRIVATE /Oi-)
        endif()
        
        # Enable TypeScript support via QuickJS transpiler
        target_compile_definitions(quickjs PUBLIC 
            CONFIG_VERSION="2024.01"
        )
        
        set_target_properties(quickjs PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
        )
    endif()
    file(REMOVE "${quickjs_SOURCE_DIR}/version")
    set(QUICKJS_AVAILABLE TRUE)
    add_compile_definitions(HAS_QUICKJS)
else()
    set(QUICKJS_AVAILABLE FALSE)
endif()

# Fetch Squirrel - lightweight, embeddable scripting language
option(ENABLE_SQUIRREL "Enable Squirrel scripting language support" ON)
if(ENABLE_SQUIRREL)
    FetchContent_Declare(
        squirrel
        GIT_REPOSITORY https://github.com/albertodemichelis/squirrel.git
        GIT_TAG master
    )
    FetchContent_GetProperties(squirrel)
    if(NOT squirrel_populated)
        FetchContent_Populate(squirrel)
        
        # Build Squirrel VM library
        file(GLOB SQUIRREL_VM_SOURCES
            "${squirrel_SOURCE_DIR}/squirrel/*.cpp"
        )
        
        # Build Squirrel standard library
        file(GLOB SQUIRREL_STDLIB_SOURCES
            "${squirrel_SOURCE_DIR}/sqstdlib/*.cpp"
        )
        
        add_library(squirrel STATIC 
            ${SQUIRREL_VM_SOURCES}
            ${SQUIRREL_STDLIB_SOURCES}
        )
        
        target_include_directories(squirrel PUBLIC 
            "${squirrel_SOURCE_DIR}/include"
            "${squirrel_SOURCE_DIR}/squirrel"
            "${squirrel_SOURCE_DIR}/sqstdlib"
        )
        
        # Squirrel configuration
        target_compile_definitions(squirrel PRIVATE 
            SQUIRREL_VERSION="3.2"
        )
        
        if(MSVC)
            target_compile_definitions(squirrel PRIVATE 
                _CRT_SECURE_NO_WARNINGS
            )
        endif()
        
        set_target_properties(squirrel PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
        )
    endif()
    set(SQUIRREL_AVAILABLE TRUE)
    add_compile_definitions(HAS_SQUIRREL)
else()
    set(SQUIRREL_AVAILABLE FALSE)
endif()

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python3 for DLL copying
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch WASM3 - WebAssembly interpreter
FetchContent_Declare(
    wasm3
    GIT_REPOSITORY https://github.com/wasm3/wasm3.git
    GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(wasm3)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find Mono
find_package(Mono)
if(MONO_FOUND)
    message(STATUS "Found Mono: ${MONO_LIBRARIES}")
    set(HAS_MONO TRUE)
else()
    message(STATUS "Mono not found, C# scripting will be disabled")
    set(HAS_MONO FALSE)
endif()
# ImGui sources are now handled via FetchContent target 'imgui'


# Engine Library
add_library(GameEngineLib STATIC
    src/Logger.cpp
    src/Window.cpp
    src/Application.cpp
    src/Shader.cpp
    src/Texture.cpp
    src/Mesh.cpp
    src/Renderer.cpp
    src/RenderBackend.cpp
    src/OpenGLBackend.cpp
    src/GPUScheduler.cpp
    src/EngineConfig.cpp
    src/Camera.cpp
    src/Quat.cpp
    src/Text.cpp
    src/GLExtensions.cpp
    src/ImGuiManager.cpp
    src/EditorMenuBar.cpp
    src/EditorHierarchy.cpp
    src/FuzzyMatcher.cpp
    src/QuickOpen.cpp
    src/IconRegistry.cpp
    src/ColorScheme.cpp
    src/InspectorLayout.cpp
    src/EditorPropertyPanel.cpp
    src/EditorDockingManager.cpp
    src/GizmoToolsPanel.cpp
    src/NVRHIBackend.cpp
    src/Skybox.cpp
    src/ShadowMap.cpp
    src/TextureManager.cpp
    src/MaterialLibrary.cpp
    src/GameObject.cpp
    src/Model.cpp
    src/ModelLoader.cpp
    src/GBuffer.cpp
    src/PostProcessing.cpp
    src/CubemapShadow.cpp
    src/CascadedShadowMap.cpp
    src/Frustum.cpp
    src/SSAO.cpp
    src/SSR.cpp
    src/TAA.cpp
    src/LightProbe.cpp
    src/ReflectionProbe.cpp
    src/GLTFLoader.cpp
    src/GLTFExtensions.cpp
    src/PreviewRenderer.cpp
    src/Material.cpp
    src/Decal.cpp
    # Animation system
    src/Animator.cpp
    src/Animation.cpp
    src/AnimationStateMachine.cpp
    src/BlendTree.cpp
    src/BlendCurve.cpp
    src/Bone.cpp
    src/BoneMask.cpp
    src/ECS/EntityManager.cpp
    src/ECS/Systems.cpp
    src/ECS/GameSystems.cpp
    src/ECS/Components.cpp
    src/AssetHotReloadManager.cpp
    src/Profiler.cpp
    src/PerformanceMonitor.cpp
    src/GpuProfiler.cpp
    src/Tools/PerformanceProfiler.cpp
    src/TelemetryServer.cpp
    src/IK.cpp
    src/FileWatcher.cpp
    src/BlendTreeEditor.cpp
    # Geometry & Hull Algorithms
    src/QuickHull.cpp
    src/GiftWrapping.cpp
    src/IncrementalHull.cpp
    src/DivideAndConquerHull.cpp
)

# Main Executable
add_executable(GameEngine src/main.cpp)

# Force tinygltf to not include STB implementation again
if(TARGET tinygltf)
    target_compile_definitions(tinygltf PRIVATE TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION)
endif()

set(SOURCES ${SOURCES}
    src/CollisionShape.cpp
    src/CollisionDemo.cpp
    src/ParticleTrail.cpp
    src/ParticleEmitter.cpp
    src/ParticleSystem.cpp
    src/VolumetricFog.cpp
    src/Water.cpp
    src/PlanarReflection.cpp
    src/WaterSprayEmitter.cpp
    src/Terrain.cpp
    src/Vegetation.cpp
    src/TerrainBrush.cpp
    src/Sprite.cpp
    src/SSBO.cpp
    src/AudioSystem.cpp
    src/AudioSource.cpp
    # Global Illumination system
    src/GlobalIllumination.cpp
    src/VoxelGrid.cpp
    src/LightPropagationVolume.cpp
    src/ProbeGrid.cpp
    src/ProbeBaker.cpp
    # Advanced audio features
    src/AudioMixer.cpp
    src/AudioSpatializer.cpp
    src/AudioOcclusion.cpp
    src/ImpactAudioSystem.cpp
    src/ContinuousAudioSystem.cpp
    src/ClothAudioSystem.cpp
    # Resource streaming and virtual filesystem
    src/VirtualFileSystem.cpp
    src/ResourceStreamingManager.cpp
    src/AssetPackage.cpp
    # Profiling and telemetry
    src/Profiler.cpp
    src/TelemetryServer.cpp
    # Physics system
    src/PhysicsSystem.cpp
    src/RigidBody.cpp
    src/KinematicController.cpp
    # Scene serialization and prefabs
    src/SceneSerializer.cpp
    src/Prefab.cpp
    # Asset Browser
    src/AssetBrowser.cpp
    src/AssetThumbnailGenerator.cpp
    # Gizmos
    src/Gizmo.cpp
    src/TranslationGizmo.cpp
    src/RotationGizmo.cpp
    src/ScaleGizmo.cpp
    src/GizmoManager.cpp
    # Scripting System
    src/ScriptSystem.cpp
    src/LuaJitScriptSystem.cpp
    src/PythonScriptSystem.cpp
    src/CSharpScriptSystem.cpp
    src/CustomScriptSystem.cpp
    src/VirtualMachine.cpp
    src/ScriptComponent.cpp
    src/WrenScriptSystem.cpp
    src/WrenScriptComponent.cpp
    src/GoScriptSystem.cpp
    src/GoRuntime.cpp
    src/GDScriptSystem.cpp
    src/AngelScriptSystem.cpp
    src/TypeScriptScriptSystem.cpp
    src/RustScriptSystem.cpp
    src/SquirrelScriptSystem.cpp
    src/Wasm/WasmScriptSystem.cpp
    src/Wasm/WasmRuntime.cpp
    src/Wasm/WasmModule.cpp
    src/Wasm/WasmInstance.cpp
    src/Wasm/WasmHelper.cpp
    src/Wasm/WasmEngineBindings.cpp
    src/ScriptDebugger.cpp
    src/ScriptDebuggerUI.cpp
    src/ScriptingProfilerUI.cpp
    src/ScriptLanguageRegistry.cpp
    # Asset pipeline - incremental builds, hashing, and conversion
    src/AssetHash.cpp
    src/AssetDatabase.cpp
    src/AssetConverter.cpp
    src/AssetPipeline.cpp
    # Networking layer - integrated into main engine
    game-engine-multiplayer/src/Server.cpp
    game-engine-multiplayer/src/Client.cpp
    game-engine-multiplayer/src/Message.cpp
    game-engine-multiplayer/src/Peer.cpp
    game-engine-multiplayer/src/serialization/Serializer.cpp
    # Hybrid deferred+forward rendering with GPU culling
    src/RenderPass.cpp
    src/HybridRenderer.cpp
    src/GPUCullingSystem.cpp
    # Adaptive Quality System
    src/PerformanceMonitor.cpp
    src/AdaptiveQualityController.cpp
    src/AdaptiveQualityIntegration.cpp
    # Multi-Threading Support
    src/ThreadPool.cpp
    src/ParallelSoftBodyManager.cpp
    src/Job.cpp
    src/JobScheduler.cpp
    src/WorkStealingThreadPool.cpp
    src/ThreadAffinity.cpp
    # Scene Partitioning
    src/PhysXScenePartition.cpp
    src/ScenePartitionManager.cpp
    src/ParallelPhysicsSystem.cpp
    src/AsyncCollisionDetector.cpp
    src/GpuProfiler.cpp
    src/GpuBatchManager.cpp
    src/NetworkManager.cpp
    src/NetworkManagerASIO.cpp
    src/NetworkManagerReliability.cpp
    src/NetworkManagerBandwidth.cpp
    src/CloudProvider.cpp
    src/ProtobufSerializer.cpp
    src/SoftBodyLOD.cpp
    src/SoftBodyLODManager.cpp
    src/PhysicsCollisionShape.cpp
    src/TetrahedralMeshSimplifier.cpp
)

# Add RDMA sources if available
if(HAS_RDMA)
    list(APPEND SOURCES
        src/RdmaManager.cpp
        src/DistributedBatchManagerRdma.cpp
    )
endif()

list(APPEND SOURCES ${ADDITIONAL_SOURCES}) # Includes generated protobuf sources



# Add the engine sources
target_sources(GameEngineLib PRIVATE ${SOURCES})


# Force tinygltf to not include STB implementation again again
if(TARGET tinygltf)
    target_compile_definitions(tinygltf PUBLIC TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION)
endif()
add_compile_definitions(TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION)
add_compile_definitions(TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION)


# Find PhysX
if(PHYSICS_BACKEND STREQUAL "PHYSX")
    find_package(PhysX REQUIRED)
endif()

# Find LZ4 for state serialization compression
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LZ4 liblz4)
endif()

if(NOT LZ4_FOUND)
    message(STATUS "LZ4 not found via pkg-config, will use fallback RLE compression")
    set(HAS_LZ4 FALSE)
else()
    message(STATUS "Found LZ4: ${LZ4_LIBRARIES}")
    set(HAS_LZ4 TRUE)
endif()

# Find Protocol Buffers for network serialization
find_package(Protobuf)
if(Protobuf_FOUND)
    message(STATUS "Found Protocol Buffers: ${Protobuf_VERSION}")
    set(HAS_PROTOBUF TRUE)
    
    # Generate C++ code from .proto files
    set(PROTO_FILES
        ${CMAKE_SOURCE_DIR}/proto/distributed_physics.proto
    )
    
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
    
    # Add generated files to sources
    list(APPEND ADDITIONAL_SOURCES ${PROTO_SRCS})
    
    message(STATUS "Generated Protocol Buffers sources: ${PROTO_SRCS}")
else()
    message(STATUS "Protocol Buffers not found, using manual serialization")
    set(HAS_PROTOBUF FALSE)
endif()

# Find ASIO for async networking
find_path(ASIO_INCLUDE_DIR asio.hpp)
if(ASIO_INCLUDE_DIR)
    message(STATUS "Found standalone ASIO: ${ASIO_INCLUDE_DIR}")
    set(HAS_ASIO TRUE)
    set(ASIO_STANDALONE TRUE)
else()
    # Try Boost.ASIO
    find_package(Boost COMPONENTS system)
    if(Boost_FOUND)
        message(STATUS "Found Boost.ASIO: ${Boost_VERSION}")
        set(HAS_ASIO TRUE)
        set(ASIO_STANDALONE FALSE)
    else()
        message(STATUS "Boost.ASIO not found, fetching standalone ASIO...")
        FetchContent_Declare(
            asio
            GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
            GIT_TAG asio-1-30-2
        )
        FetchContent_MakeAvailable(asio)
        set(ASIO_INCLUDE_DIR ${asio_SOURCE_DIR}/asio/include)
        set(HAS_ASIO TRUE)
        set(ASIO_STANDALONE TRUE)
    endif()
endif()

# Vulkan SDK (Optional - for Vulkan backend support)
option(ENABLE_VULKAN "Enable Vulkan backend support" ON)
if(ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Found Vulkan: ${Vulkan_VERSION}")
        set(HAS_VULKAN TRUE)
        add_compile_definitions(ENABLE_VULKAN)
        target_sources(GameEngineLib PRIVATE
            src/VulkanBackend.cpp
            src/VulkanShaderCompiler.cpp
            src/VulkanDebugUtils.cpp
        )
    else()
        message(STATUS "Vulkan SDK not found - Vulkan backend disabled (OpenGL will be used)")
        set(HAS_VULKAN FALSE)
        set(ENABLE_VULKAN OFF)
    endif()
else()
    message(STATUS "Vulkan support disabled by user")
    set(HAS_VULKAN FALSE)
endif()

# OpenGL
# Add PhysX backend sources if using PhysX
if(PHYSICS_BACKEND STREQUAL "PHYSX")
    target_sources(GameEngineLib PRIVATE
        src/PhysXBackend.cpp
        src/PhysXRigidBody.cpp
        src/PhysXFluidVolume.cpp
        src/PhysXShape.cpp
        src/PhysXCharacterController.cpp
        src/PhysXCloth.cpp
        src/PhysXSoftBody.cpp
        src/TetrahedralMeshGenerator.cpp
        src/SoftBodyTearSystem.cpp
        src/TetrahedralMeshSplitter.cpp
        src/ProceduralTearGenerator.cpp
        src/TearPropagationSystem.cpp
        src/PartialTearSystem.cpp
        src/CrackRenderer.cpp
        src/SoftBodyPieceManager.cpp
        src/TearResistanceMap.cpp
        src/SoftBodyTearPattern.cpp
        src/PhysXRagdoll.cpp
        src/PhysXRope.cpp
        src/PhysXCustomConstraint.cpp
        src/StraightTearPattern.cpp
        src/CurvedTearPattern.cpp
        src/RadialTearPattern.cpp
        src/ClothMeshGenerator.cpp
        src/ClothMeshSplitter.cpp
        src/ClothMeshSynchronizer.cpp
        src/ClothMeshSimplifier.cpp
        src/ClothPiece.cpp
        src/ClothPieceManager.cpp
        src/AsyncClothFactory.cpp
        src/ClothLOD.cpp
        src/ClothLODManager.cpp
        src/ClothTearPattern.cpp
        src/ClothTearPatternLibrary.cpp
        src/AnisotropicMaterial.cpp
        src/SoftBodySerializer.cpp
        src/SoftBodyDeformationRecorder.cpp
        src/SoftBodyEditor.cpp
        src/StressVisualizer.cpp
        src/SoftBodyPreset.cpp
        src/SoftBodyPresetLibrary.cpp
        src/ManualTearTrigger.cpp
        src/TearHistory.cpp
        src/TearPreview.cpp
        src/VertexPicker.cpp
        src/VertexHighlighter.cpp
        src/FractureLine.cpp
        src/FractureLineGizmo.cpp
        src/FractureLineToPattern.cpp
        src/TearPatternGizmo.cpp
        src/FractureLinePatternLibrary.cpp
        # Fluid simulation (PBD)
        src/PBDFluidSolver.cpp
        src/FluidEmitter.cpp
        src/FluidSimulation.cpp
        src/FluidRenderer.cpp
        src/FoamParticleSystem.cpp
        src/FluidSimulationEditor.cpp
        src/FluidDebugGizmo.cpp
        # Distributed Physics
        src/DistributedBatchManager.cpp
        src/DistributedBatchManagerFaultTolerance.cpp
        src/DistributedBatchManagerFailover.cpp
        src/DistributedBatchManagerSync.cpp
        src/DistributedBatchManagerResults.cpp
        src/DistributedBatchManagerRegistration.cpp
        src/DistributedBatchManagerHierarchical.cpp
        src/StateSerializer.cpp
    )
    
    if(CUDAToolkit_FOUND)
        target_sources(GameEngine PRIVATE src/PBDFluidSolverGPU.cu)
        set_source_files_properties(src/PBDFluidSolverGPU.cu PROPERTIES LANGUAGE CUDA)
    endif()
elseif(PHYSICS_BACKEND STREQUAL "BOX2D")
    target_sources(GameEngineLib PRIVATE
        src/Box2DBackend.cpp
        src/Box2DShape.cpp
        src/Box2DRigidBody.cpp
    )
else()
    # Add Bullet backend wrapper
    target_sources(GameEngineLib PRIVATE
        src/BulletBackend.cpp
    )
endif()


# Configure include directories based on physics backend
# Common include directories for all build configurations
set(COMMON_INCLUDES
    include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends

    game-engine-multiplayer/include
    ${miniaudio_SOURCE_DIR}
    ${assimp_SOURCE_DIR}/include
    ${nvrhi_SOURCE_DIR}/include
    ${nvrhi_SOURCE_DIR}/thirdparty/DirectX-Headers/include
    ${LUA_INCLUDE_DIR}
    ${wren_SOURCE_DIR}/src/include
    ${enet_SOURCE_DIR}/include
)

# Add QuickJS includes if enabled
if(QUICKJS_AVAILABLE)
    list(APPEND COMMON_INCLUDES ${quickjs_SOURCE_DIR})
endif()

# Add Squirrel includes if enabled
if(SQUIRREL_AVAILABLE)
    list(APPEND COMMON_INCLUDES
        ${squirrel_SOURCE_DIR}/include
        ${squirrel_SOURCE_DIR}/squirrel
        ${squirrel_SOURCE_DIR}/sqstdlib
    )
endif()

if(PHYSICS_BACKEND STREQUAL "PHYSX")
    list(APPEND COMMON_INCLUDES
        ${PHYSX_ROOT_DIR}/include
        ${PXSHARED_ROOT_DIR}/include
    )
    if(CUDAToolkit_FOUND)
        list(APPEND COMMON_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    endif()
elseif(PHYSICS_BACKEND STREQUAL "BOX2D")
    list(APPEND COMMON_INCLUDES ${box2d_SOURCE_DIR}/include)
else()
    list(APPEND COMMON_INCLUDES ${bullet3_SOURCE_DIR}/src)
endif()

if(MONO_FOUND)
    list(APPEND COMMON_INCLUDES ${MONO_INCLUDE_DIRS})
endif()

# Add WASM3 includes
list(APPEND COMMON_INCLUDES ${wasm3_SOURCE_DIR}/source)

target_include_directories(GameEngineLib PUBLIC ${COMMON_INCLUDES})

# Configure link libraries based on physics backend
# Common link libraries for all build configurations
set(COMMON_LIBS
    imgui
    glfw

    opengl32
    nlohmann_json
    tinygltf
    assimp
    ws2_32
    winmm
    ${LUA_LIBRARY}
    pybind11::embed
    glm::glm
    enet
    nvrhi
    nvrhi_d3d11
    nvrhi_d3d12
    nvrhi_vk
    wren
    spdlog::spdlog
    $<$<BOOL:${ANGELSCRIPT_AVAILABLE}>:angelscript>
    $<$<BOOL:${QUICKJS_AVAILABLE}>:quickjs>
    $<$<BOOL:${SQUIRREL_AVAILABLE}>:squirrel>
    m3
    uvwasi_a
    uv_a
)

if(PHYSICS_BACKEND STREQUAL "PHYSX")
    list(APPEND COMMON_LIBS
        PhysX
        PhysXCommon
        PhysXFoundation
        PhysXExtensions
        PhysXCharacterKinematic
        PhysXVehicle
    )
elseif(PHYSICS_BACKEND STREQUAL "BOX2D")
    list(APPEND COMMON_LIBS box2d)
else()
    list(APPEND COMMON_LIBS BulletDynamics BulletCollision LinearMath)
endif()

if(HAS_VULKAN)
    list(APPEND COMMON_LIBS Vulkan::Vulkan)
    target_compile_definitions(GameEngine PRIVATE VULKAN_ENABLED)
endif()

target_link_libraries(GameEngineLib PUBLIC ${COMMON_LIBS})

# Link GameEngine executable to GameEngineLib
target_link_libraries(GameEngine PRIVATE GameEngineLib)

# Global library configurations (CUDA, LZ4, Protobuf, ASIO)
# These should be applied regardless of the physics backend

# Link CUDA runtime if available
if(CUDAToolkit_FOUND)
    target_compile_definitions(GameEngineLib PUBLIC HAS_CUDA_TOOLKIT)
    target_include_directories(GameEngineLib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(GameEngineLib PUBLIC CUDA::cudart)
endif()

# Link LZ4 if available
if(HAS_LZ4)
    target_compile_definitions(GameEngineLib PUBLIC HAS_LZ4)
    target_include_directories(GameEngineLib PUBLIC ${LZ4_INCLUDE_DIRS})
    target_link_libraries(GameEngineLib PUBLIC ${LZ4_LIBRARIES})
endif()

# Link Protocol Buffers if available
if(HAS_PROTOBUF)
    target_compile_definitions(GameEngineLib PUBLIC HAS_PROTOBUF)
    target_include_directories(GameEngineLib PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_include_directories(GameEngineLib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})  # For generated headers
    target_link_libraries(GameEngineLib PUBLIC ${Protobuf_LIBRARIES})
endif()

# Link ASIO if available
if(HAS_ASIO)
    target_compile_definitions(GameEngineLib PUBLIC HAS_ASIO)
    if(ASIO_STANDALONE)
        target_compile_definitions(GameEngineLib PUBLIC ASIO_STANDALONE)
        target_include_directories(GameEngineLib PUBLIC ${ASIO_INCLUDE_DIR})
    else()
        target_include_directories(GameEngineLib PUBLIC ${Boost_INCLUDE_DIRS})
        target_link_libraries(GameEngineLib PUBLIC ${Boost_LIBRARIES})
    endif()
    # Disable deprecated warnings for ASIO
    if(MSVC)
        target_compile_definitions(GameEngineLib PUBLIC _WIN32_WINNT=0x0601)  # Windows 7+
    endif()
endif()

# Optional Mono support
if(HAS_MONO)
    target_compile_definitions(GameEngineLib PUBLIC HAS_MONO)
    target_link_libraries(GameEngineLib PUBLIC ${MONO_LIBRARIES})
endif()

# Copy assets to build directory
if(Python3_FOUND)
    get_filename_component(PYTHON_RUNTIME_DIR "${Python3_EXECUTABLE}" DIRECTORY)
endif()

add_custom_command(TARGET GameEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:GameEngine>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:GameEngine>/shaders
    # Copy Python runtime DLL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PYTHON_RUNTIME_DIR}/python314.dll"
    $<TARGET_FILE_DIR:GameEngine>/python314.dll
)

option(BUILD_TESTS "Build unit tests" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Disable warnings in googletest
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Core library tests
    add_executable(tests
        tests/main.cpp
        tests/test_math.cpp
        tests/test_transform.cpp
        tests/test_gameobject.cpp
        tests/test_material.cpp
        tests/test_shader.cpp
        tests/game-engine-multiplayer/test_message.cpp
        tests/game-engine-multiplayer/test_serializer.cpp
        tests/test_scripting.cpp
        tests/test_python_scripting.cpp
        tests/test_csharp_scripting.cpp
        tests/test_vm.cpp
        tests/test_quickhull.cpp
        tests/test_giftwrapping.cpp
        tests/test_incremental.cpp
        tests/test_divideandconquer.cpp
    )
    
    target_include_directories(tests PRIVATE 
        include 
        include/imgui 
        game-engine-multiplayer/include 
        ${miniaudio_SOURCE_DIR}
        ${LUA_INCLUDE_DIR}
    )
    
    target_link_libraries(tests PRIVATE 
        GameEngineLib
        gtest 
        gtest_main 
    )
    
    # Apply sanitizers to tests
    if(USE_ASAN AND NOT MSVC)
        target_compile_options(tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(tests PRIVATE -fsanitize=address)
    endif()
    
    if(USE_UBSAN AND NOT MSVC)
        target_compile_options(tests PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(tests PRIVATE -fsanitize=undefined)
    endif()
    
    enable_testing()
    
    # Add individual test targets
    add_test(NAME math_tests COMMAND tests --gtest_filter=MathTest.*)
    add_test(NAME transform_tests COMMAND tests --gtest_filter=TransformTest.*)
    add_test(NAME gameobject_tests COMMAND tests --gtest_filter=GameObjectTest.*)
    add_test(NAME material_tests COMMAND tests --gtest_filter=MaterialTest.*)
    add_test(NAME network_tests COMMAND tests --gtest_filter=NetworkTest.*)
    add_test(NAME serialization_tests COMMAND tests --gtest_filter=SerializationTest.*)
    add_test(NAME quickhull_tests COMMAND tests --gtest_filter=QuickHullTest.*)
    add_test(NAME giftwrapping_tests COMMAND tests --gtest_filter=GiftWrappingTest.*)
    add_test(NAME incremental_tests COMMAND tests --gtest_filter=IncrementalHullTest.*)
    add_test(NAME divideandconquer_tests COMMAND tests --gtest_filter=DivideAndConquerTest.*)
    
    # Run all tests
    add_test(NAME all_tests COMMAND tests)
endif()

# ============================================================================
# INSTALLATION & PACKAGING CONFIGURATION
# ============================================================================

# Install executable
install(TARGETS GameEngine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install assets directory
install(DIRECTORY assets/
    DESTINATION bin/assets
    PATTERN ".git*" EXCLUDE
)

# Install shaders directory
install(DIRECTORY shaders/
    DESTINATION bin/shaders
    PATTERN ".git*" EXCLUDE
)

# Install documentation
install(FILES 
    README.md
    LICENSE
    CHANGELOG.md
    GETTING_STARTED.md
    CI_QUICK_START.md
    IMPLEMENTATION_SUMMARY.md
    ARCHITECTURE_DIAGRAM.md
    CHECKLIST.md
    FILE_MANIFEST.md
    docs/TESTING_CI_GUIDE.md
    docs/AnimationEvents.md
    docs/particle_collision_guide.md
    docs/particle_trails_guide.md
    game-engine-multiplayer/README.md
    DESTINATION share/doc/GameEngine
    OPTIONAL
)

# Install include headers
install(DIRECTORY include/
    DESTINATION include/GameEngine
    PATTERN "*.h"
    PATTERN "imgui" EXCLUDE
)

# CPack configuration for distribution
set(CPACK_PACKAGE_NAME "GameEngine")
set(CPACK_PACKAGE_VENDOR "Development Team")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Game Engine with Deferred Rendering and Advanced Post-Processing")
set(CPACK_PACKAGE_DESCRIPTION "A C++20 OpenGL 3.3+ game engine with deferred rendering, advanced visual effects, and multiplayer networking support.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/YOUR_USERNAME/game-engine")
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/assets/icon.png")

# Installation directory
set(CPACK_INSTALL_PREFIX "/opt/GameEngine")

# Windows-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_PACKAGE_NAME "Game Engine")
    set(CPACK_NSIS_DISPLAY_NAME "Game Engine ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_INSTALL_ROOT "C:\\Program Files")
    set(CPACK_NSIS_MENU_LINKS
        "bin/GameEngine.exe" "Game Engine"
        "https://github.com/YOUR_USERNAME/game-engine" "Project Repository"
    )
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/YOUR_USERNAME/game-engine")
    set(CPACK_NSIS_CONTACT "your-email@example.com")
    set(CPACK_NSIS_CREATE_ADMINISTRATORS_GROUP TRUE)
endif()

# Linux-specific packaging
if(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    
    # DEB package settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Development Team <email@example.com>")
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION}")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libglfw3, libgl1-mesa0")
    
    # RPM package settings
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    set(CPACK_RPM_PACKAGE_REQUIRES "glfw-devel opengl-devel")
    set(CPACK_RPM_PACKAGE_URL "${CPACK_PACKAGE_HOMEPAGE_URL}")
endif()

# macOS-specific packaging
if(APPLE)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_VOLUME_NAME "Game Engine ${CPACK_PACKAGE_VERSION}")
    set(CPACK_DMG_FORMAT "UDBZ")
endif()

# Common packaging settings
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_VERBOSE ON)

# Include CPack
include(CPack)
