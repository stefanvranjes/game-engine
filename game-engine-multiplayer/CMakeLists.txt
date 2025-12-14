cmake_minimum_required(VERSION 3.10)

project(GameEngineMultiplayer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/NetworkManager.cpp
    src/Server.cpp
    src/Client.cpp
    src/Message.cpp
    src/Peer.cpp
    src/serialization/Serializer.cpp
)

# Create the executable
add_executable(GameEngineMultiplayer ${SOURCES})

# Link libraries if needed (e.g., networking libraries)
# target_link_libraries(GameEngineMultiplayer <libraries>) 

# Add examples
add_subdirectory(examples)

# Add tests
add_subdirectory(tests)