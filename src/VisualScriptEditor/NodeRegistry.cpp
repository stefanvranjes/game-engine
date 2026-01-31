#include "../include/VisualScriptEditor/NodeRegistry.h"

NodeRegistry* NodeRegistry::s_Instance = nullptr;

NodeRegistry& NodeRegistry::Get() {
    if (!s_Instance) {
        s_Instance = new NodeRegistry();
    }
    return *s_Instance;
}

NodeRegistry::NodeRegistry() {
    InitializeBuiltInNodes();
}

void NodeRegistry::InitializeBuiltInNodes() {
    // Behavior Tree Nodes
    RegisterNode({
        NodeType::Sequence, "Sequence", "Behavior Trees", 
        "Execute child nodes in sequence",
        [](uint32_t id) { return std::make_shared<SequenceNode>(id); },
        0xFF66BB6A, 120.0f, 80.0f
    });

    RegisterNode({
        NodeType::Selector, "Selector", "Behavior Trees",
        "Execute child nodes until one succeeds",
        [](uint32_t id) { return std::make_shared<SelectorNode>(id); },
        0xFF81C784, 120.0f, 80.0f
    });

    RegisterNode({
        NodeType::Parallel, "Parallel", "Behavior Trees",
        "Execute all child nodes in parallel",
        [](uint32_t id) { return std::make_shared<ParallelNode>(id); },
        0xFFA5D6A7, 120.0f, 80.0f
    });

    RegisterNode({
        NodeType::Decorator, "Decorator", "Behavior Trees",
        "Modify the behavior of a child node",
        [](uint32_t id) { return std::make_shared<DecoratorNode>(id, "None"); },
        0xFFFFC107, 120.0f, 80.0f
    });

    // State Machine Nodes
    RegisterNode({
        NodeType::StateNode, "State", "State Machines",
        "Represents a state in a state machine",
        [](uint32_t id) { return std::make_shared<StateNode>(id, "State"); },
        0xFF2196F3, 120.0f, 60.0f
    });

    RegisterNode({
        NodeType::Transition, "Transition", "State Machines",
        "Represents a transition between states",
        [](uint32_t id) { return std::make_shared<TransitionNode>(id, 0, 0); },
        0xFF64B5F6, 120.0f, 40.0f
    });

    // Logic Nodes
    RegisterNode({
        NodeType::Logic_And, "And", "Logic",
        "Logical AND operation",
        [](uint32_t id) { return std::make_shared<LogicNode>(id, NodeType::Logic_And, "And"); },
        0xFF9C27B0, 80.0f, 60.0f
    });

    RegisterNode({
        NodeType::Logic_Or, "Or", "Logic",
        "Logical OR operation",
        [](uint32_t id) { return std::make_shared<LogicNode>(id, NodeType::Logic_Or, "Or"); },
        0xFF9C27B0, 80.0f, 60.0f
    });

    RegisterNode({
        NodeType::Logic_Not, "Not", "Logic",
        "Logical NOT operation",
        [](uint32_t id) { return std::make_shared<LogicNode>(id, NodeType::Logic_Not, "Not"); },
        0xFF9C27B0, 80.0f, 60.0f
    });

    RegisterNode({
        NodeType::Logic_Compare, "Compare", "Logic",
        "Compare two values",
        [](uint32_t id) { return std::make_shared<LogicNode>(id, NodeType::Logic_Compare, "Compare"); },
        0xFF9C27B0, 100.0f, 80.0f
    });

    RegisterNode({
        NodeType::Logic_Branch, "Branch", "Logic",
        "Branch based on condition",
        [](uint32_t id) { return std::make_shared<LogicNode>(id, NodeType::Logic_Branch, "Branch"); },
        0xFF9C27B0, 100.0f, 80.0f
    });

    RegisterNode({
        NodeType::Logic_Math, "Math", "Logic",
        "Perform mathematical operations",
        [](uint32_t id) { return std::make_shared<MathNode>(id, MathNode::Operation::Add); },
        0xFFF59E0B, 100.0f, 80.0f
    });

    // Utility Nodes
    RegisterNode({
        NodeType::Utility_Print, "Print", "Utility",
        "Print a message",
        [](uint32_t id) { return std::make_shared<PrintNode>(id); },
        0xFF00BCD4, 100.0f, 80.0f
    });

    RegisterNode({
        NodeType::Utility_Variable, "Variable", "Utility",
        "Read a variable value",
        [](uint32_t id) { return std::make_shared<VariableNode>(id, "Variable", "float"); },
        0xFFFF5722, 100.0f, 80.0f
    });

    RegisterNode({
        NodeType::Utility_Getter, "Get", "Utility",
        "Get a property value",
        [](uint32_t id) { return std::make_shared<VisualNode>(id, NodeType::Utility_Getter, "Get"); },
        0xFFFF5722, 80.0f, 60.0f
    });

    RegisterNode({
        NodeType::Utility_Setter, "Set", "Utility",
        "Set a property value",
        [](uint32_t id) { return std::make_shared<VisualNode>(id, NodeType::Utility_Setter, "Set"); },
        0xFFFF5722, 80.0f, 60.0f
    });
}

void NodeRegistry::RegisterNode(const NodeDefinition& definition) {
    m_Definitions[definition.type] = definition;
    m_DefinitionsByCategory[definition.category].push_back(definition);
}

const NodeDefinition* NodeRegistry::GetNodeDefinition(NodeType type) const {
    auto it = m_Definitions.find(type);
    return it != m_Definitions.end() ? &it->second : nullptr;
}

const NodeDefinition* NodeRegistry::GetNodeDefinitionByName(const std::string& name) const {
    for (const auto& [type, def] : m_Definitions) {
        if (def.name == name) {
            return &def;
        }
    }
    return nullptr;
}

std::shared_ptr<VisualNode> NodeRegistry::CreateNode(NodeType type, uint32_t id) {
    auto def = GetNodeDefinition(type);
    if (def && def->factory) {
        return def->factory(id);
    }
    // Fallback to basic node
    return std::make_shared<VisualNode>(id, type, "Unknown");
}

std::shared_ptr<VisualNode> NodeRegistry::CreateNodeByName(const std::string& name, uint32_t id) {
    auto def = GetNodeDefinitionByName(name);
    if (def) {
        return CreateNode(def->type, id);
    }
    return nullptr;
}

const std::map<std::string, std::vector<NodeDefinition>>& NodeRegistry::GetDefinitionsByCategory(const std::string& category) const {
    static std::map<std::string, std::vector<NodeDefinition>> empty;
    auto it = m_DefinitionsByCategory.find(category);
    if (it != m_DefinitionsByCategory.end()) {
        return m_DefinitionsByCategory;
    }
    return empty;
}

std::vector<std::string> NodeRegistry::GetCategories() const {
    std::vector<std::string> categories;
    for (const auto& [cat, defs] : m_DefinitionsByCategory) {
        categories.push_back(cat);
    }
    return categories;
}

// NodeBuilder implementation
NodeBuilder& NodeBuilder::WithName(const std::string& name) {
    m_Name = name;
    return *this;
}

NodeBuilder& NodeBuilder::WithCategory(const std::string& category) {
    m_Category = category;
    return *this;
}

NodeBuilder& NodeBuilder::WithDescription(const std::string& desc) {
    m_Description = desc;
    return *this;
}

NodeBuilder& NodeBuilder::WithColor(uint32_t color) {
    m_Color = color;
    return *this;
}

NodeBuilder& NodeBuilder::WithSize(float width, float height) {
    m_Width = width;
    m_Height = height;
    return *this;
}

NodeBuilder& NodeBuilder::WithInputPort(const std::string& name, const std::string& type) {
    m_InputPorts.push_back({name, type});
    return *this;
}

NodeBuilder& NodeBuilder::WithOutputPort(const std::string& name, const std::string& type) {
    m_OutputPorts.push_back({name, type});
    return *this;
}

NodeDefinition NodeBuilder::Build(NodeType type, std::function<std::shared_ptr<VisualNode>(uint32_t)> factory) {
    NodeDefinition def;
    def.type = type;
    def.name = m_Name;
    def.category = m_Category;
    def.description = m_Description;
    def.factory = factory;
    def.color = m_Color;
    def.width = m_Width;
    def.height = m_Height;
    return def;
}
