#include "../include/VisualScriptEditor/VisualNode.h"

VisualNode::VisualNode(uint32_t id, NodeType type, const std::string& name)
    : m_Id(id), m_NodeType(type), m_Name(name), m_Position(0, 0) {
}

uint32_t VisualNode::AddInputPort(const std::string& name, const std::string& dataType) {
    Port port;
    port.name = name;
    port.type = PortType::Input;
    port.dataType = dataType;
    port.id = m_NextPortId++;
    m_InputPorts.push_back(port);
    return port.id;
}

uint32_t VisualNode::AddOutputPort(const std::string& name, const std::string& dataType) {
    Port port;
    port.name = name;
    port.type = PortType::Output;
    port.dataType = dataType;
    port.id = m_NextPortId++;
    m_OutputPorts.push_back(port);
    return port.id;
}

const Port* VisualNode::FindPort(uint32_t portId) const {
    for (const auto& port : m_InputPorts) {
        if (port.id == portId) return &port;
    }
    for (const auto& port : m_OutputPorts) {
        if (port.id == portId) return &port;
    }
    return nullptr;
}

const Port* VisualNode::FindPortByName(const std::string& name) const {
    for (const auto& port : m_InputPorts) {
        if (port.name == name) return &port;
    }
    for (const auto& port : m_OutputPorts) {
        if (port.name == name) return &port;
    }
    return nullptr;
}

json VisualNode::Serialize() const {
    json obj;
    obj["id"] = m_Id;
    obj["type"] = static_cast<int>(m_NodeType);
    obj["name"] = m_Name;
    obj["position"] = {{"x", m_Position.x}, {"y", m_Position.y}};
    
    obj["properties"] = m_Properties;
    
    json inputPorts = json::array();
    for (const auto& port : m_InputPorts) {
        inputPorts.push_back({
            {"id", port.id},
            {"name", port.name},
            {"dataType", port.dataType},
            {"isConnected", port.isConnected}
        });
    }
    obj["inputPorts"] = inputPorts;
    
    json outputPorts = json::array();
    for (const auto& port : m_OutputPorts) {
        outputPorts.push_back({
            {"id", port.id},
            {"name", port.name},
            {"dataType", port.dataType},
            {"isConnected", port.isConnected}
        });
    }
    obj["outputPorts"] = outputPorts;
    
    return obj;
}

bool VisualNode::Deserialize(const json& data) {
    try {
        if (data.contains("position")) {
            m_Position = {data["position"]["x"], data["position"]["y"]};
        }
        if (data.contains("name")) {
            m_Name = data["name"];
        }
        if (data.contains("properties")) {
            m_Properties = data["properties"].get<std::map<std::string, json>>();
        }
        return true;
    } catch (...) {
        return false;
    }
}

// Behavior Tree Node implementations
BehaviorTreeNode::BehaviorTreeNode(uint32_t id, const std::string& name)
    : VisualNode(id, NodeType::BehaviorRoot, name) {
    AddOutputPort("Execute", "execution");
}

SequenceNode::SequenceNode(uint32_t id)
    : VisualNode(id, NodeType::Sequence, "Sequence") {
    AddInputPort("In", "execution");
    AddOutputPort("Success", "execution");
    AddOutputPort("Failure", "execution");
}

SelectorNode::SelectorNode(uint32_t id)
    : VisualNode(id, NodeType::Selector, "Selector") {
    AddInputPort("In", "execution");
    AddOutputPort("Success", "execution");
    AddOutputPort("Failure", "execution");
}

ParallelNode::ParallelNode(uint32_t id)
    : VisualNode(id, NodeType::Parallel, "Parallel") {
    AddInputPort("In", "execution");
    AddOutputPort("Success", "execution");
    AddOutputPort("Failure", "execution");
}

DecoratorNode::DecoratorNode(uint32_t id, const std::string& decoratorType)
    : VisualNode(id, NodeType::Decorator, "Decorator") {
    SetProperty("decoratorType", decoratorType);
    AddInputPort("In", "execution");
    AddOutputPort("Out", "execution");
}

StateNode::StateNode(uint32_t id, const std::string& stateName)
    : VisualNode(id, NodeType::StateNode, stateName) {
    AddInputPort("Enter", "execution");
    AddOutputPort("Exit", "execution");
}

TransitionNode::TransitionNode(uint32_t id, uint32_t fromState, uint32_t toState)
    : VisualNode(id, NodeType::Transition, "Transition"),
      m_FromState(fromState), m_ToState(toState) {
    AddInputPort("Condition", "bool");
    AddOutputPort("Execute", "execution");
}

LogicNode::LogicNode(uint32_t id, NodeType type, const std::string& name)
    : VisualNode(id, type, name) {
    AddInputPort("A", "bool");
    AddInputPort("B", "bool");
    AddOutputPort("Result", "bool");
}

MathNode::MathNode(uint32_t id, Operation op)
    : VisualNode(id, NodeType::Logic_Math, "Math"),
      m_Operation(op) {
    AddInputPort("X", "float");
    AddInputPort("Y", "float");
    AddOutputPort("Result", "float");
}

VariableNode::VariableNode(uint32_t id, const std::string& varName, const std::string& varType)
    : VisualNode(id, NodeType::Utility_Variable, varName) {
    SetProperty("variableName", varName);
    SetProperty("variableType", varType);
    AddOutputPort("Value", varType);
}

PrintNode::PrintNode(uint32_t id)
    : VisualNode(id, NodeType::Utility_Print, "Print") {
    AddInputPort("Exec", "execution");
    AddInputPort("Message", "string");
    AddOutputPort("Exec", "execution");
}

void PrintNode::Execute() {
    auto msg = GetProperty("message");
    if (msg.is_string()) {
        // In a real implementation, this would print to the engine's log
    }
}
