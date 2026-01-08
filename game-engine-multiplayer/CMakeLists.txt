cmake_minimum_required(VERSION 3.10)

project(GameEngineMultiplayer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build unit tests" ON)

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)  # Main engine headers

# Find ENet
# Try to find system ENet first, otherwise use bundled
find_package(enet QUIET)
if(NOT enet_FOUND)
    # For Windows, we may need to specify paths
    if(WIN32)
        set(ENET_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/enet/include" CACHE PATH "ENet include directory")
        set(ENET_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../external/enet/lib/enet.lib" CACHE FILEPATH "ENet library")
    endif()
endif()

# Source files
set(MULTIPLAYER_SOURCES
    src/Message.cpp
    src/NetworkManager.cpp
    src/Peer.cpp
    src/NetworkTransform.cpp
    src/ReplicationManager.cpp
    src/ClientPrediction.cpp
    src/RPC.cpp
    src/MultiplayerManager.cpp
)

# Header files (for IDE visibility)
set(MULTIPLAYER_HEADERS
    include/Message.hpp
    include/NetworkManager.hpp
    include/Protocol.hpp
    include/Peer.hpp
    include/NetEntityId.hpp
    include/NetworkTransform.hpp
    include/ReplicationManager.hpp
    include/ClientPrediction.hpp
    include/RPC.hpp
    include/MultiplayerManager.hpp
)

# Create the library
add_library(GameEngineMultiplayer STATIC ${MULTIPLAYER_SOURCES} ${MULTIPLAYER_HEADERS})

# Include directories for target
target_include_directories(GameEngineMultiplayer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link ENet
if(enet_FOUND)
    target_link_libraries(GameEngineMultiplayer PUBLIC enet::enet)
elseif(EXISTS ${ENET_LIBRARY})
    target_include_directories(GameEngineMultiplayer PUBLIC ${ENET_INCLUDE_DIR})
    target_link_libraries(GameEngineMultiplayer PUBLIC ${ENET_LIBRARY})
else()
    message(STATUS "ENet not found - networking will not be available")
endif()

# Platform-specific socket libraries
if(WIN32)
    target_link_libraries(GameEngineMultiplayer PUBLIC ws2_32 winmm)
elseif(UNIX)
    # Linux/macOS don't need additional socket libraries
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(GameEngineMultiplayer PRIVATE /W4)
else()
    target_compile_options(GameEngineMultiplayer PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Main executable (for standalone testing)
add_executable(MultiplayerTest src/main.cpp)
target_link_libraries(MultiplayerTest PRIVATE GameEngineMultiplayer)

# Add examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS GameEngineMultiplayer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/multiplayer)