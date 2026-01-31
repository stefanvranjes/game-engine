#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <nlohmann/json.hpp>
#include "../Math/Vec.h"

using json = nlohmann::json;

enum class NodeType {
    Undefined,
    // Behavior Tree Nodes
    BehaviorRoot,
    Sequence,
    Selector,
    Parallel,
    Decorator,
    Leaf,
    // State Machine Nodes
    StateNode,
    Transition,
    InitialState,
    // Logic Nodes
    Logic_And,
    Logic_Or,
    Logic_Not,
    Logic_Compare,
    Logic_Branch,
    Logic_Math,
    // Utility Nodes
    Utility_Print,
    Utility_Variable,
    Utility_Getter,
    Utility_Setter,
    Custom
};

enum class PortType {
    Execution,   // For control flow
    Input,       // Data input
    Output,      // Data output
    Condition    // For transitions
};

struct Port {
    std::string name;
    PortType type;
    std::string dataType;  // e.g., "float", "bool", "Vec3", "string"
    uint32_t id;
    bool isConnected = false;
};

struct Connection {
    uint32_t sourceNodeId;
    uint32_t sourcePortId;
    uint32_t targetNodeId;
    uint32_t targetPortId;
    std::string label;
};

class VisualNode {
public:
    VisualNode(uint32_t id, NodeType type, const std::string& name);
    virtual ~VisualNode() = default;

    // Core properties
    uint32_t GetId() const { return m_Id; }
    NodeType GetNodeType() const { return m_NodeType; }
    const std::string& GetName() const { return m_Name; }
    void SetName(const std::string& name) { m_Name = name; }

    // Position in editor
    Vec2 GetPosition() const { return m_Position; }
    void SetPosition(const Vec2& pos) { m_Position = pos; }

    // Ports (connections)
    const std::vector<Port>& GetInputPorts() const { return m_InputPorts; }
    const std::vector<Port>& GetOutputPorts() const { return m_OutputPorts; }
    
    uint32_t AddInputPort(const std::string& name, const std::string& dataType = "execution");
    uint32_t AddOutputPort(const std::string& name, const std::string& dataType = "execution");
    
    const Port* FindPort(uint32_t portId) const;
    const Port* FindPortByName(const std::string& name) const;

    // Node-specific data (for properties, parameters, etc.)
    void SetProperty(const std::string& key, const json& value) { m_Properties[key] = value; }
    json GetProperty(const std::string& key) const {
        auto it = m_Properties.find(key);
        return it != m_Properties.end() ? it->second : json();
    }
    const std::map<std::string, json>& GetAllProperties() const { return m_Properties; }

    // Serialization
    virtual json Serialize() const;
    virtual bool Deserialize(const json& data);

    // Execution (for runtime evaluation)
    virtual void Execute() {}

    // Color for editor visualization
    virtual uint32_t GetColor() const { return 0xFF888888; }
    virtual float GetWidth() const { return 150.0f; }
    virtual float GetHeight() const { return 80.0f; }

protected:
    uint32_t m_Id;
    NodeType m_NodeType;
    std::string m_Name;
    Vec2 m_Position;
    std::vector<Port> m_InputPorts;
    std::vector<Port> m_OutputPorts;
    std::map<std::string, json> m_Properties;
    uint32_t m_NextPortId = 0;
};

// Specific node types

class BehaviorTreeNode : public VisualNode {
public:
    BehaviorTreeNode(uint32_t id, const std::string& name);
    uint32_t GetColor() const override { return 0xFF4CAF50; }  // Green
};

class SequenceNode : public VisualNode {
public:
    SequenceNode(uint32_t id);
    uint32_t GetColor() const override { return 0xFF66BB6A; }  // Light green
};

class SelectorNode : public VisualNode {
public:
    SelectorNode(uint32_t id);
    uint32_t GetColor() const override { return 0xFF81C784; }  // Light green variant
};

class ParallelNode : public VisualNode {
public:
    ParallelNode(uint32_t id);
    uint32_t GetColor() const override { return 0xFFA5D6A7; }  // Very light green
};

class DecoratorNode : public VisualNode {
public:
    DecoratorNode(uint32_t id, const std::string& decoratorType);
    uint32_t GetColor() const override { return 0xFFFFC107; }  // Amber
};

class StateNode : public VisualNode {
public:
    StateNode(uint32_t id, const std::string& stateName);
    uint32_t GetColor() const override { return 0xFF2196F3; }  // Blue
    float GetWidth() const override { return 120.0f; }
    float GetHeight() const override { return 60.0f; }
};

class TransitionNode : public VisualNode {
public:
    TransitionNode(uint32_t id, uint32_t fromState, uint32_t toState);
    uint32_t GetColor() const override { return 0xFF64B5F6; }  // Light blue
    uint32_t GetFromState() const { return m_FromState; }
    uint32_t GetToState() const { return m_ToState; }

private:
    uint32_t m_FromState;
    uint32_t m_ToState;
};

class LogicNode : public VisualNode {
public:
    LogicNode(uint32_t id, NodeType type, const std::string& name);
    uint32_t GetColor() const override { return 0xFF9C27B0; }  // Purple
};

class MathNode : public VisualNode {
public:
    enum class Operation {
        Add, Subtract, Multiply, Divide,
        Sin, Cos, Tan, Sqrt, Power, Abs
    };

    MathNode(uint32_t id, Operation op);
    uint32_t GetColor() const override { return 0xFFF59E0B; }  // Orange

private:
    Operation m_Operation;
};

class VariableNode : public VisualNode {
public:
    VariableNode(uint32_t id, const std::string& varName, const std::string& varType);
    uint32_t GetColor() const override { return 0xFFFF5722; }  // Deep orange
};

class PrintNode : public VisualNode {
public:
    PrintNode(uint32_t id);
    uint32_t GetColor() const override { return 0xFF00BCD4; }  // Cyan
    void Execute() override;
};
