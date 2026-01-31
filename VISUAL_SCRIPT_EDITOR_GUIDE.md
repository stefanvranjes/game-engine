# Visual Script Editor Documentation

## Overview

The Visual Script Editor is a comprehensive node-based visual programming system for the game engine. It allows developers to create complex game logic without writing code, supporting:

- **Behavior Trees**: AI decision-making and hierarchical task planning
- **State Machines**: Character/object state management with transitions
- **Logic Graphs**: Visual programming with boolean, mathematical, and comparison operations
- **Animation Graphs**: Character animation blending and state transitions

## Key Features

### 1. Node-Based Graph System
- Modular node architecture with input/output ports
- Support for data types (execution, bool, float, Vec3, string, etc.)
- Dynamic node creation and connection
- Port validation and type checking

### 2. Hot-Reload Compilation
- Real-time code generation from visual graphs
- Automatic C++ code emission
- In-engine compilation with error reporting
- Live script reloading without engine restart

### 3. Visual Editor Interface
- ImGui-based editor with pan/zoom canvas
- Node library with categorized node types
- Properties panel for node configuration
- Real-time validation and error reporting
- Undo/Redo support

### 4. Graph Types

#### Behavior Trees
```
Root
  └─ Sequence
      ├─ Selector
      │   ├─ Leaf (Action)
      │   └─ Leaf (Action)
      └─ Parallel
          ├─ Leaf (Action)
          └─ Leaf (Action)
```

#### State Machines
```
[Idle] ──condition──> [Walk] ──condition──> [Run]
  ↑                                           │
  └───────────────────────condition───────────┘
```

#### Logic Graphs
```
[Input A] ──┐
            ├─> [And] ──> [Compare] ──> [Output]
[Input B] ──┘
```

## Architecture

### Core Components

1. **VisualNode** (`VisualScriptEditor/VisualNode.h`)
   - Base class for all node types
   - Manages ports, properties, and execution
   - Serializable and customizable

2. **VisualGraph** (`VisualScriptEditor/VisualGraph.h`)
   - Container for nodes and connections
   - Graph type management (BehaviorTree, StateMachine, etc.)
   - Validation and execution logic
   - Blackboard for shared node data

3. **VisualScriptEditor** (`VisualScriptEditor/VisualScriptEditor.h`)
   - Main editor UI using ImGui
   - Graph manipulation and node management
   - Code generation and compilation
   - Undo/Redo stack management

4. **CodeGenerator** (`VisualScriptEditor/CodeGenerator.h`)
   - Generates C++ code from visual graphs
   - Specialized generators for each graph type
   - Header and source file generation

5. **NodeRegistry** (`VisualScriptEditor/NodeRegistry.h`)
   - Central registry of available node types
   - Node factory with fluent API
   - Category organization

6. **VisualScriptHotReload** (`VisualScriptEditor/VisualScriptHotReload.h`)
   - Compilation and dynamic loading
   - File watching and auto-reload
   - Blackboard management
   - Compilation statistics

## Node Types

### Behavior Tree Nodes

| Node | Purpose | Behavior |
|------|---------|----------|
| **Sequence** | Execute children in order | Success if all succeed, Failure if any fail |
| **Selector** | Try children until success | Success if any succeed, Failure if all fail |
| **Parallel** | Execute all children simultaneously | Success if all succeed within time limit |
| **Decorator** | Modify child behavior | Inverts, repeats, or limits child execution |
| **Leaf** | Actual action/task | Returns Success, Failure, or Running |

### State Machine Nodes

| Node | Purpose |
|------|---------|
| **State** | Represents a state with Entry/Exit logic |
| **Transition** | Defines state change with condition |
| **Initial State** | Entry point for the state machine |

### Logic Nodes

| Node | Purpose | Operation |
|------|---------|-----------|
| **And** | Boolean AND | A && B |
| **Or** | Boolean OR | A \|\| B |
| **Not** | Boolean NOT | !A |
| **Compare** | Value comparison | A == B, A < B, etc. |
| **Branch** | Conditional branching | If condition, execute path |
| **Math** | Mathematical operations | +, -, *, /, sin, cos, sqrt, etc. |

### Utility Nodes

| Node | Purpose |
|------|---------|
| **Print** | Debug logging |
| **Variable** | Access variable values |
| **Get** | Read property |
| **Set** | Write property |

## Usage Guide

### Creating a Behavior Tree

```cpp
#include "VisualScriptEditor/VisualScriptEditor.h"

// Create editor and new graph
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("EnemyAI", GraphType::BehaviorTree);

auto graph = editor->GetCurrentGraph();

// Create nodes
auto root = graph->CreateNode(NodeType::BehaviorRoot, "Root");
auto selector = graph->CreateNode(NodeType::Selector, "Selector");
auto sequence = graph->CreateNode(NodeType::Sequence, "Action Sequence");

// Connect nodes
graph->Connect(root->GetId(), 0, selector->GetId(), 0);
graph->Connect(selector->GetId(), 0, sequence->GetId(), 0);

// Set root node
graph->SetRootNode(root->GetId());

// Save graph
editor->SaveGraph("graphs/EnemyAI.json");

// Generate code
editor->GenerateCode();
std::string code = editor->GetGeneratedCode();
```

### Creating a State Machine

```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("PlayerController", GraphType::StateMachine);

auto graph = std::dynamic_pointer_cast<StateMachineGraph>(
    editor->GetCurrentGraph());

// Create states
auto idle = graph->CreateNode(NodeType::StateNode, "Idle");
auto walk = graph->CreateNode(NodeType::StateNode, "Walk");
auto run = graph->CreateNode(NodeType::StateNode, "Run");

// Configure states
idle->SetProperty("onEnter", "PlayIdleAnimation");
walk->SetProperty("onEnter", "PlayWalkAnimation");
run->SetProperty("onEnter", "PlayRunAnimation");

// Create and connect transitions
auto idleToWalk = graph->CreateNode(NodeType::Transition, "IdleToWalk");
idleToWalk->SetProperty("condition", "inputSpeed > 0.5");

graph->Connect(idle->GetId(), 0, walk->GetId(), 0);
graph->Connect(walk->GetId(), 0, run->GetId(), 0);

// Set initial state
graph->SetCurrentState(idle->GetId());

editor->SaveGraph("graphs/PlayerController.json");
```

### Using Hot-Reload

```cpp
#include "VisualScriptEditor/VisualScriptHotReload.h"

// Initialize hot-reload system
auto hotReload = std::make_unique<VisualScriptHotReload>();
hotReload->Initialize("generated_scripts");

// Set callbacks
hotReload->SetOnCompileComplete([](bool success, const std::string& msg) {
    std::cout << "Compile: " << (success ? "SUCCESS" : "FAILED") << " - " << msg << std::endl;
});

// Create and compile a graph
auto graph = std::make_shared<BehaviorTreeGraph>("MyBehavior");
// ... add nodes ...

if (hotReload->CompileAndLoad(graph)) {
    std::cout << "Script ready for execution" << std::endl;
    
    // Get statistics
    const auto& stats = hotReload->GetStats();
    std::cout << "Compiled " << stats.totalGraphsCompiled << " graphs" << std::endl;
}

// Execute a script function
hotReload->ExecuteScriptFunction("MyBehavior", "Execute");
```

### Integrating with Application

```cpp
// In Application::Init()
m_VisualScriptEditor = std::make_unique<VisualScriptEditor>();
m_VisualScriptHotReload = std::make_unique<VisualScriptHotReload>();
m_VisualScriptHotReload->Initialize("scripts/generated");

// In Application::RenderEditorUI()
m_VisualScriptEditor->RenderUI();

// In Application::Update()
// Handle editor input and update graph execution
if (m_VisualScriptEditor->GetCurrentGraph()) {
    m_VisualScriptEditor->GetCurrentGraph()->Execute();
}
```

## File Format

Graphs are serialized to JSON for easy version control and editing:

```json
{
  "name": "EnemyAI",
  "type": 0,
  "version": 1,
  "rootNodeId": 1,
  "nodes": [
    {
      "id": 1,
      "type": 0,
      "name": "Root",
      "position": {"x": 100, "y": 100},
      "properties": {},
      "inputPorts": [],
      "outputPorts": [
        {
          "id": 0,
          "name": "Execute",
          "dataType": "execution",
          "isConnected": true
        }
      ]
    }
  ],
  "connections": [
    {
      "sourceNodeId": 1,
      "sourcePortId": 0,
      "targetNodeId": 2,
      "targetPortId": 0,
      "label": ""
    }
  ],
  "blackboard": {}
}
```

## Code Generation

The system generates clean, optimized C++ code:

```cpp
#pragma once

#ifndef GENERATED_ENEMYAI_H
#define GENERATED_ENEMYAI_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace VisualScripts {

class EnemyAIGraph {
public:
    EnemyAIGraph();
    ~EnemyAIGraph();

    void Initialize();
    void Execute();
    void Reset();

    void SetBlackboardValue(const std::string& key, const json& value);
    json GetBlackboardValue(const std::string& key) const;

private:
    std::map<std::string, json> m_Blackboard;
};

}  // namespace VisualScripts

#endif
```

## Validation

The editor provides real-time validation:

- **Type checking**: Ensures port connections have compatible types
- **Connection validation**: Prevents invalid connections
- **Graph structure validation**: Ensures graph is executable
- **Circular reference detection**: Prevents infinite loops

## Performance Considerations

1. **Node Pool**: Reuse nodes where possible
2. **Lazy Evaluation**: Compute only when needed
3. **GPU Optimization**: Complex graphs can use GPU compute
4. **Caching**: Cache compiled scripts for reuse
5. **LOD System**: Reduce graph complexity for distant objects

## Best Practices

1. **Keep graphs organized**: Use meaningful node names and positions
2. **Modular design**: Break complex logic into smaller graphs
3. **Blackboard usage**: Use blackboard for shared state between nodes
4. **Error handling**: Always check validation results
5. **Version control**: Save graphs as JSON for easy tracking

## Troubleshooting

### Compilation Errors
- Check error messages in the validation panel
- Ensure all input ports are connected
- Verify data type compatibility

### Hot-Reload Issues
- Ensure output directory exists
- Check file permissions
- Verify CMake is available

### Performance Issues
- Reduce node count
- Enable GPU compute for large graphs
- Profile with PerformanceMonitor

## Future Enhancements

- [ ] Visual debugging with breakpoints
- [ ] Graph templates and presets
- [ ] Collaboration support
- [ ] Advanced animation graph features
- [ ] Profiling integration
- [ ] Visual script marketplace
- [ ] Blueprint-style visual feedback
- [ ] Multi-graph composition

## References

- [Node Registry](VisualScriptEditor/NodeRegistry.h)
- [Code Examples](VisualScriptEditor/VisualScriptEditorExample.h)
- [Architecture](../../../HOT_RELOAD_ARCHITECTURE_DIAGRAM.md)
