# Visual Script Editor - Quick Reference

## Quick Start

### 1. Create a New Graph
```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("MyGraph", GraphType::BehaviorTree);
```

### 2. Add Nodes
```cpp
auto node = editor->GetCurrentGraph()->CreateNode(NodeType::Sequence, "MySequence");
```

### 3. Connect Nodes
```cpp
graph->Connect(sourceNodeId, sourcePortId, targetNodeId, targetPortId);
```

### 4. Generate Code
```cpp
editor->GenerateCode();
std::string code = editor->GetGeneratedCode();
```

### 5. Compile and Reload
```cpp
hotReload->CompileAndLoad(graph);
```

## Node Library

### Behavior Trees
- `NodeType::BehaviorRoot` - Tree root node
- `NodeType::Sequence` - Execute all children in order
- `NodeType::Selector` - Execute until one succeeds
- `NodeType::Parallel` - Execute all simultaneously
- `NodeType::Decorator` - Modify child behavior

### State Machines
- `NodeType::StateNode` - Represents a state
- `NodeType::Transition` - State transition with condition
- `NodeType::InitialState` - Entry point

### Logic
- `NodeType::Logic_And` - AND operation
- `NodeType::Logic_Or` - OR operation
- `NodeType::Logic_Not` - NOT operation
- `NodeType::Logic_Compare` - Compare values
- `NodeType::Logic_Branch` - Conditional branch
- `NodeType::Logic_Math` - Math operations

### Utility
- `NodeType::Utility_Print` - Debug print
- `NodeType::Utility_Variable` - Read variable
- `NodeType::Utility_Getter` - Get property
- `NodeType::Utility_Setter` - Set property

## File Locations

| Component | Header | Implementation |
|-----------|--------|-----------------|
| Node | `include/VisualScriptEditor/VisualNode.h` | `src/VisualScriptEditor/VisualNode.cpp` |
| Graph | `include/VisualScriptEditor/VisualGraph.h` | `src/VisualScriptEditor/VisualGraph.cpp` |
| Editor | `include/VisualScriptEditor/VisualScriptEditor.h` | `src/VisualScriptEditor/VisualScriptEditor.cpp` |
| Code Gen | `include/VisualScriptEditor/CodeGenerator.h` | `src/VisualScriptEditor/CodeGenerator.cpp` |
| Registry | `include/VisualScriptEditor/NodeRegistry.h` | `src/VisualScriptEditor/NodeRegistry.cpp` |
| Hot-Reload | `include/VisualScriptEditor/VisualScriptHotReload.h` | `src/VisualScriptEditor/VisualScriptHotReload.cpp` |
| Examples | `include/VisualScriptEditor/VisualScriptEditorExample.h` | (Header only) |

## Common Tasks

### Create a Behavior Tree for AI
```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("AIBehavior", GraphType::BehaviorTree);

auto graph = editor->GetCurrentGraph();
auto root = graph->CreateNode(NodeType::BehaviorRoot, "Root");
auto selector = graph->CreateNode(NodeType::Selector, "Decision");
auto sequence = graph->CreateNode(NodeType::Sequence, "Attack");

graph->Connect(root->GetId(), 0, selector->GetId(), 0);
graph->Connect(selector->GetId(), 0, sequence->GetId(), 0);
graph->SetRootNode(root->GetId());

editor->GenerateCode();
editor->SaveGraph("graphs/AI.json");
```

### Create a State Machine
```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("PlayerStates", GraphType::StateMachine);

auto graph = editor->GetCurrentGraph();
auto idle = graph->CreateNode(NodeType::StateNode, "Idle");
auto walk = graph->CreateNode(NodeType::StateNode, "Walk");
auto transition = graph->CreateNode(NodeType::Transition, "Move");

graph->Connect(idle->GetId(), 0, walk->GetId(), 0);
std::dynamic_pointer_cast<StateMachineGraph>(graph)->SetCurrentState(idle->GetId());

editor->SaveGraph("graphs/States.json");
```

### Use Hot-Reload
```cpp
auto hotReload = std::make_unique<VisualScriptHotReload>();
hotReload->Initialize("scripts/generated");

auto graph = std::make_shared<BehaviorTreeGraph>("Dynamic");
// Add nodes...

hotReload->CompileAndLoad(graph);
hotReload->ExecuteScriptFunction("Dynamic", "Execute");
```

### Load Existing Graph
```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->OpenGraph("graphs/AI.json");

auto graph = editor->GetCurrentGraph();
// Modify or execute...
```

### Validate Graph
```cpp
editor->ValidateGraph();
if (!editor->GetValidationErrors().empty()) {
    for (const auto& error : editor->GetValidationErrors()) {
        std::cerr << error << std::endl;
    }
}
```

### Access Blackboard (Shared Memory)
```cpp
auto graph = editor->GetCurrentGraph();

// Write values
graph->SetBlackboardValue("health", 100.0f);
graph->SetBlackboardValue("is_alive", true);

// Read values
auto health = graph->GetBlackboardValue("health");
auto isAlive = graph->GetBlackboardValue("is_alive");
```

## Key Classes

### VisualNode
```cpp
std::shared_ptr<VisualNode> node = graph->CreateNode(NodeType::Sequence, "Name");
uint32_t portId = node->AddInputPort("In", "execution");
node->SetProperty("key", "value");
```

### VisualGraph
```cpp
auto graph = std::make_shared<BehaviorTreeGraph>("Name");
auto node = graph->CreateNode(NodeType::Sequence, "Name");
graph->Connect(sourceId, sourcePort, targetId, targetPort);
graph->SetRootNode(rootId);
```

### VisualScriptEditor
```cpp
auto editor = std::make_unique<VisualScriptEditor>();
editor->NewGraph("Name", GraphType::BehaviorTree);
editor->RenderUI();
editor->GenerateCode();
editor->SaveGraph("path.json");
```

### VisualScriptHotReload
```cpp
auto hotReload = std::make_unique<VisualScriptHotReload>();
hotReload->Initialize("output_dir");
hotReload->CompileAndLoad(graph);
hotReload->ExecuteScriptFunction("name", "function");
```

### NodeRegistry
```cpp
auto& registry = NodeRegistry::Get();
auto node = registry.CreateNode(NodeType::Sequence, nodeId);
auto node2 = registry.CreateNodeByName("And", nodeId);
```

### CodeGenerator
```cpp
auto generator = CodeGeneratorFactory::CreateGenerator(GraphType::BehaviorTree);
std::string header = generator->GenerateHeader(graph);
std::string source = generator->GenerateCode(graph);
```

## Port Data Types

Standard port data types:
- `"execution"` - Control flow
- `"bool"` - Boolean values
- `"int"` - Integer values
- `"float"` - Floating point
- `"string"` - Text
- `"Vec2"` - 2D vector
- `"Vec3"` - 3D vector
- `"Transform"` - Transform matrix

## Event Handling

### On Compile Complete
```cpp
hotReload->SetOnCompileComplete([](bool success, const std::string& msg) {
    std::cout << "Compile: " << (success ? "OK" : "ERROR") << " - " << msg << std::endl;
});
```

### On Reload Complete
```cpp
hotReload->SetOnReloadComplete([](bool success) {
    std::cout << "Reload: " << (success ? "OK" : "ERROR") << std::endl;
});
```

## Editor Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+S | Save graph |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Delete | Delete selected node |
| Middle Mouse + Drag | Pan canvas |
| Mouse Wheel | Zoom canvas |
| Right Click | Context menu |

## Performance Tips

1. **Keep graphs small**: Break large logic into multiple graphs
2. **Minimize connections**: Fewer connections = faster execution
3. **Use LOD**: Simplify graphs for distant objects
4. **Cache compiled scripts**: Reuse compiled graphs
5. **Profile regularly**: Check performance stats in VisualScriptHotReload

## Integration Steps

1. Add to `CMakeLists.txt`:
```cmake
add_subdirectory(src/VisualScriptEditor)
```

2. In `Application::Init()`:
```cpp
m_VisualScriptEditor = std::make_unique<VisualScriptEditor>();
m_VisualScriptHotReload = std::make_unique<VisualScriptHotReload>();
m_VisualScriptHotReload->Initialize("scripts/generated");
```

3. In `Application::RenderEditorUI()`:
```cpp
m_VisualScriptEditor->RenderUI();
```

4. In `Application::Update()`:
```cpp
if (auto graph = m_VisualScriptEditor->GetCurrentGraph()) {
    graph->Execute();
}
```

## Debugging

### Enable Debug Visualization
```cpp
editor->SetShowNodeIds(true);
```

### Check Validation Errors
```cpp
editor->ValidateGraph();
for (const auto& error : editor->GetValidationErrors()) {
    std::cerr << error << std::endl;
}
```

### Get Compilation Errors
```cpp
if (hotReload->HasCompilationError()) {
    std::cerr << hotReload->GetLastCompilationError() << std::endl;
}
```

### View Generated Code
```cpp
editor->GenerateCode();
std::string code = editor->GetGeneratedCode();
std::cout << code << std::endl;
```

## Examples

See [VisualScriptEditorExample.h](include/VisualScriptEditor/VisualScriptEditorExample.h) for complete examples:

- `DemoBehaviorTree()` - Create an AI behavior tree
- `DemoStateMachine()` - Create a state machine
- `DemoLogicGraph()` - Create a logic graph
- `DemoHotReload()` - Demonstrate hot-reload
- `DemoCodeGeneration()` - Generate code
- `DemoSerialization()` - Save/load graphs
- `DemoBlackboard()` - Use shared memory
- `DemoNodeRegistry()` - Work with node registry
- `DemoAnimationGraph()` - Create animation graphs

## Additional Resources

- [Full Documentation](VISUAL_SCRIPT_EDITOR_GUIDE.md)
- [Hot-Reload Documentation](HOT_RELOAD_ARCHITECTURE_DIAGRAM.md)
- [Asset Pipeline Guide](ASSET_PIPELINE_GUIDE.md)
- [Architecture Overview](.github/copilot-instructions.md)
