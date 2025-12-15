cmake_minimum_required(VERSION 3.10)

project(GameEngine VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Fetch GLFW
include(FetchContent)

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
FetchContent_Declare(
    enet
    GIT_REPOSITORY https://github.com/lsalzman/enet.git
    GIT_TAG v7.19
)
FetchContent_MakeAvailable(enet)

# Fetch Bullet3D Physics Engine
FetchContent_Declare(
    bullet3
    GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
    GIT_TAG 3.24
)
FetchContent_GetProperties(bullet3)
if(NOT bullet3_populated)
    FetchContent_Populate(bullet3)
    # Use Bullet2 API
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
    set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
    set(BUILD_CPU_DEMOS OFF CACHE BOOL "" FORCE)
    set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)
    set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${bullet3_SOURCE_DIR}/src bullet3_build EXCLUDE_FROM_ALL)
endif()

# ImGui sources
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
    src/Text.cpp
    src/GLExtensions.cpp
    src/ImGuiManager.cpp
    src/Skybox.cpp
    src/ShadowMap.cpp
    src/TextureManager.cpp
    src/MaterialLibrary.cpp
    src/GameObject.cpp
    src/Model.cpp
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
    src/PreviewRenderer.cpp
    src/Material.cpp
    src/Animator.cpp
    src/Animation.cpp
    src/AnimationStateMachine.cpp
    src/BlendTree.cpp
    src/BlendTreeEditor.cpp
    src/CollisionShape.cpp
    src/CollisionDemo.cpp
    src/ParticleTrail.cpp
    src/ParticleEmitter.cpp
    src/ParticleSystem.cpp
    src/VolumetricFog.cpp
    src/Sprite.cpp
    src/SSBO.cpp
    src/AudioSystem.cpp
    src/AudioSource.cpp
    src/AudioMixer.cpp
    src/AudioSpatializer.cpp
    src/AudioOcclusion.cpp
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
    # Networking layer - integrated into main engine
    game-engine-multiplayer/src/NetworkManager.cpp
    game-engine-multiplayer/src/Server.cpp
    game-engine-multiplayer/src/Client.cpp
    game-engine-multiplayer/src/Message.cpp
    game-engine-multiplayer/src/Peer.cpp
    game-engine-multiplayer/src/serialization/Serializer.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(GameEngine PUBLIC include include/imgui game-engine-multiplayer/include ${miniaudio_SOURCE_DIR} ${bullet3_SOURCE_DIR}/src)

target_link_libraries(GameEngine PRIVATE glfw opengl32 nlohmann_json tinygltf enet ws2_32 winmm BulletDynamics BulletCollision LinearMath)

# Copy assets to build directory
add_custom_command(TARGET GameEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:GameEngine>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:GameEngine>/shaders
)

option(BUILD_TESTS "Build unit tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.14.0
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
