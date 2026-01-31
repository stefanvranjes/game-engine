#pragma once

#include "VisualScriptEditor/VisualScriptEditor.h"
#include "VisualScriptEditor/VisualScriptHotReload.h"
#include <memory>

/**
 * @brief Example demonstrating the Visual Script Editor System
 * 
 * This example shows how to:
 * 1. Create and manage visual graphs (behavior trees, state machines, logic graphs)
 * 2. Use the visual editor UI to design logic visually
 * 3. Generate C++ code from visual graphs
 * 4. Compile and reload scripts with hot-reload
 * 5. Execute visual scripts at runtime
 */
class VisualScriptEditorExample {
public:
    static void DemoBehaviorTree() {
        // Create a new behavior tree graph
        auto editor = std::make_unique<VisualScriptEditor>();
        editor->NewGraph("AIBehavior", GraphType::BehaviorTree);
        
        auto graph = editor->GetCurrentGraph();
        
        // Create nodes
        auto rootNode = graph->CreateNode(NodeType::BehaviorRoot, "Root");
        auto sequenceNode = graph->CreateNode(NodeType::Sequence, "Sequence");
        auto selectorNode = graph->CreateNode(NodeType::Selector, "Selector");
        
        // Connect nodes
        graph->Connect(rootNode->GetId(), 0, sequenceNode->GetId(), 0);
        graph->Connect(sequenceNode->GetId(), 0, selectorNode->GetId(), 0);
        
        // Set root
        graph->SetRootNode(rootNode->GetId());
        
        // Generate code
        editor->GenerateCode();
        
        // Save the graph
        editor->SaveGraph("graphs/AIBehavior.json");
    }

    static void DemoStateMachine() {
        // Create a new state machine
        auto editor = std::make_unique<VisualScriptEditor>();
        editor->NewGraph("PlayerController", GraphType::StateMachine);
        
        auto graph = std::dynamic_pointer_cast<StateMachineGraph>(editor->GetCurrentGraph());
        
        // Create state nodes
        auto idleState = graph->CreateNode(NodeType::StateNode, "Idle");
        auto walkState = graph->CreateNode(NodeType::StateNode, "Walk");
        auto runState = graph->CreateNode(NodeType::StateNode, "Run");
        
        // Create transitions
        auto idleToWalk = graph->CreateNode(NodeType::Transition, "IdleToWalk");
        auto walkToRun = graph->CreateNode(NodeType::Transition, "WalkToRun");
        
        // Connect transitions
        graph->Connect(idleState->GetId(), 0, walkState->GetId(), 0);
        graph->Connect(walkState->GetId(), 0, runState->GetId(), 0);
        
        // Set initial state
        graph->SetCurrentState(idleState->GetId());
        
        // Generate and save
        editor->GenerateCode();
        editor->SaveGraph("graphs/PlayerController.json");
    }

    static void DemoLogicGraph() {
        // Create a logic graph for decision making
        auto editor = std::make_unique<VisualScriptEditor>();
        editor->NewGraph("TargetSelection", GraphType::LogicGraph);
        
        auto graph = editor->GetCurrentGraph();
        
        // Create nodes
        auto compareNode = graph->CreateNode(NodeType::Logic_Compare, "Distance Check");
        auto andNode = graph->CreateNode(NodeType::Logic_And, "And");
        auto branchNode = graph->CreateNode(NodeType::Logic_Branch, "Branch");
        
        // Connect logic flow
        graph->Connect(compareNode->GetId(), 0, andNode->GetId(), 0);
        graph->Connect(andNode->GetId(), 0, branchNode->GetId(), 0);
        
        graph->SetRootNode(compareNode->GetId());
        
        // Generate and save
        editor->GenerateCode();
        editor->SaveGraph("graphs/TargetSelection.json");
    }

    static void DemoHotReload() {
        // Setup hot-reload system
        auto hotReload = std::make_unique<VisualScriptHotReload>();
        hotReload->Initialize("scripts/generated");
        
        // Setup callbacks
        hotReload->SetOnCompileComplete([](bool success, const std::string& message) {
            std::cout << (success ? "Compilation successful" : "Compilation failed") << ": " << message << std::endl;
        });
        
        hotReload->SetOnReloadComplete([](bool success) {
            std::cout << (success ? "Reload successful" : "Reload failed") << std::endl;
        });
        
        // Create a graph
        auto graph = std::make_shared<BehaviorTreeGraph>("DynamicBehavior");
        
        // Create some nodes
        auto root = graph->CreateNode(NodeType::BehaviorRoot, "Root");
        auto seq = graph->CreateNode(NodeType::Sequence, "Sequence");
        graph->Connect(root->GetId(), 0, seq->GetId(), 0);
        graph->SetRootNode(root->GetId());
        
        // Compile and load the graph
        if (hotReload->CompileAndLoad(graph)) {
            std::cout << "Script compiled and loaded successfully" << std::endl;
            
            // Get compilation stats
            const auto& stats = hotReload->GetStats();
            std::cout << "Total graphs compiled: " << stats.totalGraphsCompiled << std::endl;
            std::cout << "Total scripts loaded: " << stats.totalScriptsLoaded << std::endl;
            std::cout << "Last compile time: " << stats.lastCompileTime << "s" << std::endl;
        }
    }

    static void DemoCodeGeneration() {
        // Create a graph
        auto graph = std::make_shared<BehaviorTreeGraph>("TestBehavior");
        
        // Create nodes
        auto root = graph->CreateNode(NodeType::BehaviorRoot, "Root");
        auto selector = graph->CreateNode(NodeType::Selector, "Selector");
        auto sequence = graph->CreateNode(NodeType::Sequence, "Sequence");
        
        // Set up connections
        graph->Connect(root->GetId(), 0, selector->GetId(), 0);
        graph->Connect(selector->GetId(), 0, sequence->GetId(), 0);
        graph->SetRootNode(root->GetId());
        
        // Generate code using the code generator
        auto generator = CodeGeneratorFactory::CreateGenerator(graph->GetType());
        std::string headerCode = generator->GenerateHeader(graph);
        std::string sourceCode = generator->GenerateCode(graph);
        
        std::cout << "Generated Header:\n" << headerCode << "\n\n";
        std::cout << "Generated Source:\n" << sourceCode << "\n";
    }

    static void DemoSerialization() {
        // Create a graph
        auto graph = std::make_shared<LogicGraph>("SerializationTest");
        
        // Add some nodes
        auto andNode = graph->CreateNode(NodeType::Logic_And, "And");
        auto orNode = graph->CreateNode(NodeType::Logic_Or, "Or");
        
        // Connect
        graph->Connect(andNode->GetId(), 0, orNode->GetId(), 0);
        
        // Serialize to JSON
        json serialized = graph->Serialize();
        std::cout << "Serialized graph:\n" << serialized.dump(2) << "\n";
        
        // Deserialize from JSON
        auto newGraph = std::make_shared<LogicGraph>("Deserialized");
        if (newGraph->Deserialize(serialized)) {
            std::cout << "Successfully deserialized graph\n";
            std::cout << "Deserialized graph has " << newGraph->GetNodeCount() << " nodes\n";
        }
    }

    static void DemoBlackboard() {
        // Create a graph with blackboard (shared memory for nodes)
        auto graph = std::make_shared<BehaviorTreeGraph>("BlackboardExample");
        
        // Set some blackboard values
        graph->SetBlackboardValue("player_health", 100.0f);
        graph->SetBlackboardValue("enemy_distance", 25.5f);
        graph->SetBlackboardValue("is_alerted", true);
        
        // Create nodes that would read from the blackboard
        auto compareNode = graph->CreateNode(NodeType::Logic_Compare, "Health Check");
        compareNode->SetProperty("variable", "player_health");
        compareNode->SetProperty("threshold", 50.0f);
        
        // Retrieve blackboard values
        auto health = graph->GetBlackboardValue("player_health");
        auto distance = graph->GetBlackboardValue("enemy_distance");
        
        std::cout << "Blackboard contents:\n";
        std::cout << "  Player health: " << health.get<float>() << "\n";
        std::cout << "  Enemy distance: " << distance.get<float>() << "\n";
    }

    static void DemoNodeRegistry() {
        auto& registry = NodeRegistry::Get();
        
        // Get all categories
        auto categories = registry.GetCategories();
        std::cout << "Available node categories:\n";
        for (const auto& cat : categories) {
            std::cout << "  - " << cat << "\n";
        }
        
        // Create a node by name
        auto logicAnd = registry.CreateNodeByName("And", 100);
        if (logicAnd) {
            std::cout << "\nCreated node: " << logicAnd->GetName() << "\n";
            std::cout << "  Type: " << static_cast<int>(logicAnd->GetNodeType()) << "\n";
        }
        
        // Create a node by type
        auto sequence = registry.CreateNode(NodeType::Sequence, 101);
        std::cout << "Created node: " << sequence->GetName() << "\n";
    }

    static void DemoAnimationGraph() {
        // Create an animation graph for character controller
        auto editor = std::make_unique<VisualScriptEditor>();
        editor->NewGraph("CharacterAnimations", GraphType::AnimationGraph);
        
        auto graph = std::dynamic_pointer_cast<AnimationGraph>(editor->GetCurrentGraph());
        
        // Create animation states
        auto idle = graph->CreateNode(NodeType::StateNode, "Idle");
        auto walk = graph->CreateNode(NodeType::StateNode, "Walk");
        auto run = graph->CreateNode(NodeType::StateNode, "Run");
        auto jump = graph->CreateNode(NodeType::StateNode, "Jump");
        
        // Set properties on animation states
        idle->SetProperty("animation", "idle");
        idle->SetProperty("speed", 1.0f);
        
        walk->SetProperty("animation", "walk");
        walk->SetProperty("speed", 1.5f);
        
        run->SetProperty("animation", "run");
        run->SetProperty("speed", 2.0f);
        
        jump->SetProperty("animation", "jump");
        jump->SetProperty("speed", 1.2f);
        
        // Generate code
        editor->GenerateCode();
        editor->SaveGraph("graphs/CharacterAnimations.json");
    }
};
