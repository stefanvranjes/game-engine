cmake_minimum_required(VERSION 3.10)

project(GameEngine VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/FS)
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
    src/ParticleEmitter.cpp
    src/ParticleSystem.cpp
    src/Sprite.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(GameEngine PUBLIC include include/imgui)

target_link_libraries(GameEngine PRIVATE glfw opengl32 nlohmann_json tinygltf)

# Copy assets to build directory
add_custom_command(TARGET GameEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:GameEngine>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:GameEngine>/shaders
)
