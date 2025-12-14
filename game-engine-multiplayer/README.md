# Game Engine Multiplayer Layer

This project implements a basic multiplayer layer for the Game Engine. It provides the necessary components to create a server-client architecture, allowing multiple clients to connect and communicate with each other.

## Project Structure

- **src/**: Contains the source files for the multiplayer layer.
  - **main.cpp**: Entry point of the application, initializes network components.
  - **NetworkManager.cpp**: Manages overall network operations, connections, and message routing.
  - **Server.cpp**: Handles client connections and broadcasts messages to connected clients.
  - **Client.cpp**: Connects to the server and manages message sending and receiving.
  - **Message.cpp**: Defines the structure of messages exchanged between clients and the server.
  - **Peer.cpp**: Represents a connected client or server, managing its state and communication.
  - **serialization/**: Contains serialization utilities.
    - **Serializer.cpp**: Provides functionality to serialize and deserialize messages.

- **include/**: Contains header files for the multiplayer layer.
  - **NetworkManager.hpp**: Declares the NetworkManager class.
  - **Server.hpp**: Declares the Server class.
  - **Client.hpp**: Declares the Client class.
  - **Message.hpp**: Declares the Message class.
  - **Peer.hpp**: Declares the Peer class.
  - **Protocol.hpp**: Defines the communication protocol.

- **examples/**: Contains example implementations.
  - **dedicated_server/**: Example of a dedicated server.
    - **main.cpp**: Demonstrates how to set up and run the server.
  - **local_client/**: Example of a local client.
    - **main.cpp**: Demonstrates how to connect to the server.

- **tests/**: Contains unit tests for the network components.
  - **NetworkTests.cpp**: Tests functionality of the network components.

- **CMakeLists.txt**: Configuration file for building the project with CMake.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd game-engine-multiplayer
   ```

3. Create a build directory and navigate into it:
   ```
   mkdir build
   cd build
   ```

4. Run CMake to configure the project:
   ```
   cmake ..
   ```

5. Build the project:
   ```
   cmake --build .
   ```

## Usage Examples

- To run the dedicated server:
  ```
  ./examples/dedicated_server/main
  ```

- To run the local client:
  ```
  ./examples/local_client/main
  ```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.