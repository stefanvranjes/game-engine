#include "../include/VisualScriptEditor/CodeGenerator.h"
#include <sstream>
#include <algorithm>
#include <cctype>

std::string CodeGenerator::EscapeName(const std::string& name) const {
    std::string result = name;
    std::replace_if(result.begin(), result.end(), 
        [](char c) { return !std::isalnum(c) && c != '_'; }, '_');
    if (!result.empty() && std::isdigit(result[0])) {
        result = "_" + result;
    }
    return result;
}

std::string CodeGenerator::GetNodeClassName(NodeType type) const {
    switch (type) {
        case NodeType::Sequence: return "SequenceNode";
        case NodeType::Selector: return "SelectorNode";
        case NodeType::Parallel: return "ParallelNode";
        case NodeType::Decorator: return "DecoratorNode";
        case NodeType::Leaf: return "LeafNode";
        case NodeType::StateNode: return "StateNode";
        case NodeType::Logic_And: return "AndNode";
        case NodeType::Logic_Or: return "OrNode";
        case NodeType::Logic_Not: return "NotNode";
        case NodeType::Logic_Compare: return "CompareNode";
        default: return "CustomNode";
    }
}

std::string CodeGenerator::GenerateHeader(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    std::string guardName = "GENERATED_" + EscapeName(graph->GetName()) + "_H";
    std::transform(guardName.begin(), guardName.end(), guardName.begin(), ::toupper);

    ss << "#pragma once\n\n";
    ss << "#ifndef " << guardName << "\n";
    ss << "#define " << guardName << "\n\n";
    
    ss << "#include <string>\n";
    ss << "#include <vector>\n";
    ss << "#include <map>\n";
    ss << "#include <nlohmann/json.hpp>\n\n";

    ss << "using json = nlohmann::json;\n\n";

    ss << "namespace VisualScripts {\n\n";
    ss << "class " << EscapeName(graph->GetName()) << "Graph {\n";
    ss << "public:\n";
    ss << "    " << EscapeName(graph->GetName()) << "Graph();\n";
    ss << "    ~" << EscapeName(graph->GetName()) << "Graph();\n\n";
    ss << "    void Initialize();\n";
    ss << "    void Execute();\n";
    ss << "    void Reset();\n\n";
    ss << "    void SetBlackboardValue(const std::string& key, const json& value);\n";
    ss << "    json GetBlackboardValue(const std::string& key) const;\n\n";
    ss << "private:\n";
    ss << "    std::map<std::string, json> m_Blackboard;\n";
    ss << "};\n\n";
    ss << "}  // namespace VisualScripts\n\n";
    ss << "#endif\n";

    return ss.str();
}

// BehaviorTreeCodeGenerator implementation
std::string BehaviorTreeCodeGenerator::GenerateCode(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    
    ss << "#include \"" << EscapeName(graph->GetName()) << ".h\"\n";
    ss << "#include <iostream>\n\n";
    ss << "namespace VisualScripts {\n\n";

    // Constructor
    ss << EscapeName(graph->GetName()) << "Graph::" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    // Destructor
    ss << EscapeName(graph->GetName()) << "Graph::~" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    // Initialize
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Initialize() {\n";
    ss << "    // Initialize blackboard and nodes\n";
    ss << "}\n\n";

    // Execute - traverse the tree
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Execute() {\n";
    if (auto rootNode = graph->GetNode(graph->GetRootNode())) {
        ss << "    // Execute root node: " << rootNode->GetName() << "\n";
        ss << GenerateTreeStructure(graph, graph->GetRootNode(), 1);
    }
    ss << "}\n\n";

    // Reset
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Reset() {\n";
    ss << "    m_Blackboard.clear();\n";
    ss << "}\n\n";

    // Blackboard methods
    ss << "void " << EscapeName(graph->GetName()) << "Graph::SetBlackboardValue(const std::string& key, const json& value) {\n";
    ss << "    m_Blackboard[key] = value;\n";
    ss << "}\n\n";

    ss << "json " << EscapeName(graph->GetName()) << "Graph::GetBlackboardValue(const std::string& key) const {\n";
    ss << "    auto it = m_Blackboard.find(key);\n";
    ss << "    return it != m_Blackboard.end() ? it->second : json();\n";
    ss << "}\n\n";

    ss << "}  // namespace VisualScripts\n";

    return ss.str();
}

std::string BehaviorTreeCodeGenerator::GenerateTreeStructure(const std::shared_ptr<VisualGraph>& graph, uint32_t nodeId, int indent) {
    std::stringstream ss;
    std::string indentStr(indent * 4, ' ');

    auto node = graph->GetNode(nodeId);
    if (!node) return "";

    ss << indentStr << "// Execute node: " << node->GetName() << " (ID: " << nodeId << ")\n";

    // Get connected output nodes
    for (const auto& port : node->GetOutputPorts()) {
        auto connections = graph->GetConnectionsFrom(nodeId, port.id);
        for (const auto& conn : connections) {
            ss << GenerateTreeStructure(graph, conn.targetNodeId, indent);
        }
    }

    return ss.str();
}

std::string BehaviorTreeCodeGenerator::GenerateNodeClass(const std::shared_ptr<VisualNode>& node, int indent) {
    std::stringstream ss;
    std::string indentStr(indent * 4, ' ');

    ss << indentStr << "class " << EscapeName(node->GetName()) << "Node {\n";
    ss << indentStr << "public:\n";
    ss << indentStr << "    enum class Status { Success, Failure, Running };\n";
    ss << indentStr << "    Status Execute();\n";
    ss << indentStr << "};\n";

    return ss.str();
}

std::string BehaviorTreeCodeGenerator::GenerateTreeExecution(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "    // Tree execution logic\n";
    return ss.str();
}

// StateMachineCodeGenerator implementation
std::string StateMachineCodeGenerator::GenerateCode(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;

    ss << "#include \"" << EscapeName(graph->GetName()) << ".h\"\n";
    ss << "#include <iostream>\n\n";
    ss << "namespace VisualScripts {\n\n";

    // Generate state enum
    ss << GenerateStateEnums(graph);
    ss << "\n";

    // Constructor
    ss << EscapeName(graph->GetName()) << "Graph::" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    // Destructor
    ss << EscapeName(graph->GetName()) << "Graph::~" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    // Initialize
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Initialize() {\n";
    ss << "    // Initialize state handlers\n";
    ss << GenerateStateHandlers(graph);
    ss << "}\n\n";

    // Execute
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Execute() {\n";
    ss << "    // Execute current state and check transitions\n";
    ss << GenerateTransitionLogic(graph);
    ss << "}\n\n";

    // Reset
    ss << "void " << EscapeName(graph->GetName()) << "Graph::Reset() {\n";
    ss << "    m_Blackboard.clear();\n";
    ss << "}\n\n";

    ss << "}  // namespace VisualScripts\n";

    return ss.str();
}

std::string StateMachineCodeGenerator::GenerateStateEnums(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "enum class State {\n";
    
    int stateCount = 0;
    for (const auto& [id, node] : graph->GetAllNodes()) {
        if (node->GetNodeType() == NodeType::StateNode) {
            ss << "    " << EscapeName(node->GetName()) << ",\n";
            stateCount++;
        }
    }
    
    if (stateCount == 0) {
        ss << "    None\n";
    }
    ss << "};\n";
    return ss.str();
}

std::string StateMachineCodeGenerator::GenerateStateHandlers(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    for (const auto& [id, node] : graph->GetAllNodes()) {
        if (node->GetNodeType() == NodeType::StateNode) {
            ss << "    // State handler for: " << node->GetName() << "\n";
        }
    }
    return ss.str();
}

std::string StateMachineCodeGenerator::GenerateTransitionLogic(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "    // Check transitions and update state\n";
    for (const auto& conn : graph->GetAllConnections()) {
        auto sourceNode = graph->GetNode(conn.sourceNodeId);
        auto targetNode = graph->GetNode(conn.targetNodeId);
        if (sourceNode && targetNode && sourceNode->GetNodeType() == NodeType::StateNode) {
            ss << "    // Transition from " << sourceNode->GetName() << " to " << targetNode->GetName() << "\n";
        }
    }
    return ss.str();
}

// LogicGraphCodeGenerator implementation
std::string LogicGraphCodeGenerator::GenerateCode(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;

    ss << "#include \"" << EscapeName(graph->GetName()) << ".h\"\n";
    ss << "#include <iostream>\n\n";
    ss << "namespace VisualScripts {\n\n";

    ss << EscapeName(graph->GetName()) << "Graph::" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    ss << EscapeName(graph->GetName()) << "Graph::~" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    ss << "void " << EscapeName(graph->GetName()) << "Graph::Initialize() {\n";
    ss << "}\n\n";

    ss << "void " << EscapeName(graph->GetName()) << "Graph::Execute() {\n";
    ss << GenerateLogicFunction(graph);
    ss << "}\n\n";

    ss << "void " << EscapeName(graph->GetName()) << "Graph::Reset() {\n";
    ss << "    m_Blackboard.clear();\n";
    ss << "}\n\n";

    ss << "}  // namespace VisualScripts\n";

    return ss.str();
}

std::string LogicGraphCodeGenerator::GenerateLogicFunction(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "    // Logic evaluation\n";
    for (const auto& [id, node] : graph->GetAllNodes()) {
        ss << GenerateNodeEvaluation(node, 1);
    }
    return ss.str();
}

std::string LogicGraphCodeGenerator::GenerateNodeEvaluation(const std::shared_ptr<VisualNode>& node, int indent) {
    std::stringstream ss;
    std::string indentStr(indent * 4, ' ');
    ss << indentStr << "// Evaluate: " << node->GetName() << "\n";
    return ss.str();
}

// AnimationGraphCodeGenerator implementation
std::string AnimationGraphCodeGenerator::GenerateCode(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;

    ss << "#include \"" << EscapeName(graph->GetName()) << ".h\"\n";
    ss << "#include <iostream>\n\n";
    ss << "namespace VisualScripts {\n\n";

    ss << EscapeName(graph->GetName()) << "Graph::" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    ss << EscapeName(graph->GetName()) << "Graph::~" << EscapeName(graph->GetName()) << "Graph() {\n";
    ss << "}\n\n";

    ss << "void " << EscapeName(graph->GetName()) << "Graph::Initialize() {\n";
    ss << GenerateAnimationStates(graph);
    ss << "}\n\n";

    ss << "void " << EscapeName(graph->GetName()) << "Graph::Execute() {\n";
    ss << GenerateBlending(graph);
    ss << "}\n\n";

    ss << "}  // namespace VisualScripts\n";

    return ss.str();
}

std::string AnimationGraphCodeGenerator::GenerateAnimationStates(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "    // Setup animation states\n";
    return ss.str();
}

std::string AnimationGraphCodeGenerator::GenerateBlending(const std::shared_ptr<VisualGraph>& graph) {
    std::stringstream ss;
    ss << "    // Perform animation blending\n";
    return ss.str();
}

// CodeGeneratorFactory implementation
std::unique_ptr<CodeGenerator> CodeGeneratorFactory::CreateGenerator(GraphType type) {
    switch (type) {
        case GraphType::BehaviorTree:
            return std::make_unique<BehaviorTreeCodeGenerator>();
        case GraphType::StateMachine:
            return std::make_unique<StateMachineCodeGenerator>();
        case GraphType::LogicGraph:
            return std::make_unique<LogicGraphCodeGenerator>();
        case GraphType::AnimationGraph:
            return std::make_unique<AnimationGraphCodeGenerator>();
        default:
            return nullptr;
    }
}
