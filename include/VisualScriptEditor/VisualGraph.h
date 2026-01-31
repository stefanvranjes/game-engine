#pragma once

#include "VisualNode.h"
#include <memory>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class GraphType {
    BehaviorTree,
    StateMachine,
    LogicGraph,
    AnimationGraph
};

class VisualGraph {
public:
    VisualGraph(const std::string& name, GraphType type);
    virtual ~VisualGraph() = default;

    // Graph properties
    const std::string& GetName() const { return m_Name; }
    void SetName(const std::string& name) { m_Name = name; }
    
    GraphType GetType() const { return m_Type; }
    uint32_t GetVersion() const { return m_Version; }

    // Node management
    std::shared_ptr<VisualNode> CreateNode(NodeType type, const std::string& name = "");
    void RemoveNode(uint32_t nodeId);
    std::shared_ptr<VisualNode> GetNode(uint32_t nodeId) const;
    const std::map<uint32_t, std::shared_ptr<VisualNode>>& GetAllNodes() const { return m_Nodes; }

    // Connection management
    uint32_t Connect(uint32_t sourceNodeId, uint32_t sourcePortId,
                     uint32_t targetNodeId, uint32_t targetPortId,
                     const std::string& label = "");
    bool Disconnect(uint32_t sourceNodeId, uint32_t sourcePortId,
                    uint32_t targetNodeId, uint32_t targetPortId);
    bool DisconnectPort(uint32_t nodeId, uint32_t portId);

    const std::vector<Connection>& GetAllConnections() const { return m_Connections; }
    std::vector<Connection> GetConnectionsFrom(uint32_t nodeId, uint32_t portId) const;
    std::vector<Connection> GetConnectionsTo(uint32_t nodeId, uint32_t portId) const;

    // Root node (for behavior trees)
    void SetRootNode(uint32_t nodeId) { m_RootNodeId = nodeId; }
    uint32_t GetRootNode() const { return m_RootNodeId; }

    // Validation
    bool IsValid() const;
    std::vector<std::string> GetValidationErrors() const;

    // Serialization
    virtual json Serialize() const;
    virtual bool Deserialize(const json& data);

    // Execution context
    std::map<std::string, json>& GetBlackboard() { return m_Blackboard; }
    const std::map<std::string, json>& GetBlackboardConst() const { return m_Blackboard; }
    void SetBlackboardValue(const std::string& key, const json& value) { m_Blackboard[key] = value; }
    json GetBlackboardValue(const std::string& key) const {
        auto it = m_Blackboard.find(key);
        return it != m_Blackboard.end() ? it->second : json();
    }

    // Execution
    virtual void Execute() {}
    void Reset();

    // Statistics
    size_t GetNodeCount() const { return m_Nodes.size(); }
    size_t GetConnectionCount() const { return m_Connections.size(); }

protected:
    std::string m_Name;
    GraphType m_Type;
    uint32_t m_Version = 1;
    
    std::map<uint32_t, std::shared_ptr<VisualNode>> m_Nodes;
    std::vector<Connection> m_Connections;
    uint32_t m_NextNodeId = 1;
    uint32_t m_NextConnectionId = 0;
    
    uint32_t m_RootNodeId = 0;  // For behavior trees
    
    // Runtime state
    std::map<std::string, json> m_Blackboard;
};

// Specialized graph types

class BehaviorTreeGraph : public VisualGraph {
public:
    BehaviorTreeGraph(const std::string& name);
    
    bool IsValid() const override;
    void Execute() override;

private:
    struct NodeStatus {
        enum class Status { Running, Success, Failure } status = Status::Running;
        float timeStarted = 0.0f;
    };
    std::map<uint32_t, NodeStatus> m_NodeStatuses;
};

class StateMachineGraph : public VisualGraph {
public:
    StateMachineGraph(const std::string& name);
    
    bool IsValid() const override;
    void Execute() override;
    
    uint32_t GetCurrentState() const { return m_CurrentStateId; }
    void SetCurrentState(uint32_t stateId);
    bool CanTransition(uint32_t fromState, uint32_t toState) const;

private:
    uint32_t m_CurrentStateId = 0;
};

class LogicGraph : public VisualGraph {
public:
    LogicGraph(const std::string& name);
    
    bool IsValid() const override;
    void Execute() override;
};

class AnimationGraph : public VisualGraph {
public:
    AnimationGraph(const std::string& name);
    
    bool IsValid() const override;
    void Execute() override;
};
