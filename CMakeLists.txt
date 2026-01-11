cmake_minimum_required(VERSION 3.10)

project(GameEngine VERSION 0.1.0 LANGUAGES CXX C)

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW) 
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(NOMINMAX)
add_compile_definitions(WIN32_LEAN_AND_MEAN)

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
FetchContent_GetProperties(json)
if(NOT json_populated)
    FetchContent_Populate(json)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${json_SOURCE_DIR}/include)
endif()

# Fetch tinygltf
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.8.13
)
FetchContent_MakeAvailable(tinygltf)

# Fetch miniaudio
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.21
)
FetchContent_GetProperties(miniaudio)
if(NOT miniaudio_populated)
    FetchContent_Populate(miniaudio)
endif()

# Fetch ENet
# Fetch ENet - DISABLED
# FetchContent_Declare(
#     enet
#     GIT_REPOSITORY https://github.com/lsalzman/enet.git
#     GIT_TAG v7.19
# )
# FetchContent_MakeAvailable(enet)

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
option(PHYSICS_BACKEND "Physics backend to use: BULLET or PHYSX" "BULLET")

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
        GIT_TAG 5.4.1
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

# Fetch Lua
FetchContent_Declare(
    lua
    GIT_REPOSITORY https://github.com/lua/lua.git
    GIT_TAG v5.4.6
)
FetchContent_GetProperties(lua)
if(NOT lua_populated)
    FetchContent_Populate(lua)
    
    # Lua CMake is often non-existent or minimal in the official repo, 
    # so we define a simple library here if needed, or rely on a fork.
    # Actually, the official Lua repo is just source. 
    # We will create a target manually for it.
    
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
endif()

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find Mono
# find_package(Mono REQUIRED)
set(IMGUI_SOURCES
    src/imgui/imgui.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_opengl3.cpp
)

add_executable(GameEngine 
    src/main.cpp
    src/Window.cpp
    src/Application.cpp
    src/Shader.cpp
    src/Texture.cpp
    src/Mesh.cpp
    src/Renderer.cpp
    src/Camera.cpp
    src/Quat.cpp
    src/Text.cpp
    src/GLExtensions.cpp
    src/ImGuiManager.cpp
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
    # Animation system - DISABLED
    # src/Animator.cpp
    # src/Animation.cpp
    # src/AnimationStateMachine.cpp
    # src/BlendTree.cpp
    # src/BlendTreeEditor.cpp
    src/CollisionShape.cpp
    # src/CollisionDemo.cpp
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
    # Global Illumination system
    # src/GlobalIllumination.cpp
    # src/VoxelGrid.cpp
    # src/LightPropagationVolume.cpp
    # src/ProbeGrid.cpp
    # src/ProbeBaker.cpp
    # Advanced audio features - DISABLED
    # src/AudioMixer.cpp
    # src/AudioSpatializer.cpp
    # src/AudioOcclusion.cpp
    # Resource streaming and virtual filesystem - DISABLED
    # src/VirtualFileSystem.cpp
    # src/ResourceStreamingManager.cpp
    # src/AssetPackage.cpp
    # Profiling and telemetry - DISABLED due to missing dependencies
    # src/Profiler.cpp
    # src/TelemetryServer.cpp
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
    src/PythonScriptSystem.cpp
    # src/CSharpScriptSystem.cpp
    src/CustomScriptSystem.cpp
    src/VirtualMachine.cpp
    src/ScriptComponent.cpp
    # Asset pipeline - incremental builds, hashing, and conversion - DISABLED
    # src/AssetHash.cpp
    # src/AssetDatabase.cpp
    # src/AssetConverter.cpp
    # src/AssetPipeline.cpp
    # Networking layer - integrated into main engine (DISABLED)
    # game-engine-multiplayer/src/NetworkManager.cpp
    # game-engine-multiplayer/src/Server.cpp
    # game-engine-multiplayer/src/Client.cpp
    # game-engine-multiplayer/src/Message.cpp
    # game-engine-multiplayer/src/Peer.cpp
    # game-engine-multiplayer/src/serialization/Serializer.cpp
    # Hybrid deferred+forward rendering with GPU culling - DISABLED
    # src/RenderPass.cpp
    # src/HybridRenderer.cpp
    # src/GPUCullingSystem.cpp
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
    src/DistributedBatchManager.cpp
    src/DistributedBatchManagerFaultTolerance.cpp
    src/DistributedBatchManagerFailover.cpp
    src/DistributedBatchManagerSync.cpp
    src/DistributedBatchManagerResults.cpp
    src/DistributedBatchManagerRegistration.cpp
    src/DistributedBatchManagerHierarchical.cpp
    src/CloudProvider.cpp
    src/StateSerializer.cpp
    src/ProtobufSerializer.cpp
)

# Add RDMA sources if available
if(HAS_RDMA)
    list(APPEND SOURCES
        src/RdmaManager.cpp
        src/DistributedBatchManagerRdma.cpp
    )
endif()

set(SOURCES ${SOURCES}
    ${IMGUI_SOURCES}
    ${ADDITIONAL_SOURCES}  # Includes generated protobuf sources
)

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
        message(STATUS "ASIO not found, using basic sockets")
        set(HAS_ASIO FALSE)
    endif()
endif()

# OpenGL
# Add PhysX backend sources if using PhysX
if(PHYSICS_BACKEND STREQUAL "PHYSX")
    target_sources(GameEngine PRIVATE
        src/PhysXBackend.cpp
        src/PhysXRigidBody.cpp
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
        src/QuickHull.cpp
        src/SoftBodyCollisionCache.cpp
        src/AnisotropicMaterial.cpp
        src/SoftBodyLOD.cpp
        src/SoftBodyLODManager.cpp
        src/TetrahedralMeshSimplifier.cpp
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
    )
    
    if(CUDAToolkit_FOUND)
        target_sources(GameEngine PRIVATE src/PBDFluidSolverGPU.cu)
        set_source_files_properties(src/PBDFluidSolverGPU.cu PROPERTIES LANGUAGE CUDA)
    endif()
else()
    # Add Bullet backend wrapper
    target_sources(GameEngine PRIVATE
        src/BulletBackend.cpp
    )
endif()


# Configure include directories based on physics backend
if(PHYSICS_BACKEND STREQUAL "PHYSX")
    target_include_directories(GameEngine PUBLIC 
        include 
        include/imgui 
        game-engine-multiplayer/include 
        ${miniaudio_SOURCE_DIR} 
        ${assimp_SOURCE_DIR}/include 
        ${MONO_INCLUDE_DIRS}
        ${PHYSX_ROOT_DIR}/include
        ${PXSHARED_ROOT_DIR}/include
    )
    
    # Add CUDA includes if available
    if(CUDAToolkit_FOUND)
        target_include_directories(GameEngine PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
else()
    target_include_directories(GameEngine PUBLIC 
        include 
        include/imgui 
        game-engine-multiplayer/include 
        ${miniaudio_SOURCE_DIR} 
        ${bullet3_SOURCE_DIR}/src 
        ${assimp_SOURCE_DIR}/include 
        ${MONO_INCLUDE_DIRS}
    )
endif()

# Configure link libraries based on physics backend
if(PHYSICS_BACKEND STREQUAL "PHYSX")
    target_link_libraries(GameEngine PRIVATE 
        glfw 
        opengl32 
        nlohmann_json 
        tinygltf 
        assimp 
        ws2_32 
        winmm 
        lua 
        pybind11::embed 
        glm::glm
        PhysX
        PhysXCommon
        PhysXFoundation
        PhysXExtensions
        PhysXCharacterKinematic
    )
    
    # Link CUDA runtime if available
    if(CUDAToolkit_FOUND)
        target_compile_definitions(GameEngine PRIVATE HAS_CUDA_TOOLKIT)
        target_include_directories(GameEngine PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(GameEngine PRIVATE CUDA::cudart)
    endif()

    # Link LZ4 if available
    if(HAS_LZ4)
        target_compile_definitions(GameEngine PRIVATE HAS_LZ4)
        target_include_directories(GameEngine PRIVATE ${LZ4_INCLUDE_DIRS})
        target_link_libraries(GameEngine PRIVATE ${LZ4_LIBRARIES})
    endif()
    
    # Link Protocol Buffers if available
    if(HAS_PROTOBUF)
        target_compile_definitions(GameEngine PRIVATE HAS_PROTOBUF)
        target_include_directories(GameEngine PRIVATE ${Protobuf_INCLUDE_DIRS})
        target_include_directories(GameEngine PRIVATE ${CMAKE_CURRENT_BINARY_DIR})  # For generated headers
        target_link_libraries(GameEngine PRIVATE ${Protobuf_LIBRARIES})
    endif()
    
    # Link ASIO if available
    if(HAS_ASIO)
        target_compile_definitions(GameEngine PRIVATE HAS_ASIO)
        if(ASIO_STANDALONE)
            target_compile_definitions(GameEngine PRIVATE ASIO_STANDALONE)
            target_include_directories(GameEngine PRIVATE ${ASIO_INCLUDE_DIR})
        else()
            target_include_directories(GameEngine PRIVATE ${Boost_INCLUDE_DIRS})
            target_link_libraries(GameEngine PRIVATE ${Boost_LIBRARIES})
        endif()
        # Disable deprecated warnings for ASIO
        if(MSVC)
            target_compile_definitions(GameEngine PRIVATE _WIN32_WINNT=0x0601)  # Windows 7+
        endif()
    endif()
else()
    target_link_libraries(GameEngine PRIVATE 
        glfw 
        opengl32 
        nlohmann_json 
        tinygltf 
        assimp 
        ws2_32 
        winmm 
        BulletDynamics 
        BulletCollision 
        LinearMath 
        lua 
        pybind11::embed 
        glm::glm
    )
endif()

# Copy assets to build directory
add_custom_command(TARGET GameEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:GameEngine>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:GameEngine>/shaders
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
        tests/game-engine-multiplayer/test_message.cpp
        tests/game-engine-multiplayer/test_serializer.cpp
        tests/test_scripting.cpp
        tests/test_python_scripting.cpp
        tests/test_csharp_scripting.cpp
        tests/test_vm.cpp
        tests/test_quickhull.cpp
        src/QuickHull.cpp
        tests/test_giftwrapping.cpp
        src/GiftWrapping.cpp
        tests/test_incremental.cpp
        src/IncrementalHull.cpp
        tests/test_divideandconquer.cpp
        src/DivideAndConquerHull.cpp
    )
    
    target_include_directories(tests PRIVATE 
        include 
        include/imgui 
        game-engine-multiplayer/include 
        ${miniaudio_SOURCE_DIR}
    )
    
    target_link_libraries(tests PRIVATE 
        gtest 
        gtest_main 
        nlohmann_json
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
