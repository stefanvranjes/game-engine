cmake_minimum_required(VERSION 3.10)

project(GameEngine VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(GameEngine 
    src/main.cpp
    src/Window.cpp
    src/Application.cpp
    src/Shader.cpp
    src/Renderer.cpp
    src/GLExtensions.cpp
    src/Texture.cpp
    src/Camera.cpp
    src/Mesh.cpp
)

target_include_directories(GameEngine PUBLIC include)

target_link_libraries(GameEngine PRIVATE glfw opengl32)

# Copy assets to build directory
add_custom_command(TARGET GameEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:GameEngine>/assets
)
