#include "../include/VisualScriptEditor/VisualGraph.h"
#include "../include/VisualScriptEditor/NodeRegistry.h"

VisualGraph::VisualGraph(const std::string& name, GraphType type)
    : m_Name(name), m_Type(type) {
}

std::shared_ptr<VisualNode> VisualGraph::CreateNode(NodeType type, const std::string& name) {
    auto node = NodeRegistry::Get().CreateNode(type, m_NextNodeId++);
    if (!name.empty()) {
        node->SetName(name);
    }
    m_Nodes[node->GetId()] = node;
    return node;
}

void VisualGraph::RemoveNode(uint32_t nodeId) {
    // Remove all connections involving this node
    auto it = m_Connections.begin();
    while (it != m_Connections.end()) {
        if (it->sourceNodeId == nodeId || it->targetNodeId == nodeId) {
            it = m_Connections.erase(it);
        } else {
            ++it;
        }
    }
    
    // Remove the node
    m_Nodes.erase(nodeId);
}

std::shared_ptr<VisualNode> VisualGraph::GetNode(uint32_t nodeId) const {
    auto it = m_Nodes.find(nodeId);
    return it != m_Nodes.end() ? it->second : nullptr;
}

uint32_t VisualGraph::Connect(uint32_t sourceNodeId, uint32_t sourcePortId,
                              uint32_t targetNodeId, uint32_t targetPortId,
                              const std::string& label) {
    auto sourceNode = GetNode(sourceNodeId);
    auto targetNode = GetNode(targetNodeId);
    
    if (!sourceNode || !targetNode) {
        return 0;
    }

    Connection conn;
    conn.sourceNodeId = sourceNodeId;
    conn.sourcePortId = sourcePortId;
    conn.targetNodeId = targetNodeId;
    conn.targetPortId = targetPortId;
    conn.label = label;

    m_Connections.push_back(conn);
    
    // Mark ports as connected
    if (auto port = sourceNode->FindPort(sourcePortId)) {
        const_cast<Port*>(port)->isConnected = true;
    }
    if (auto port = targetNode->FindPort(targetPortId)) {
        const_cast<Port*>(port)->isConnected = true;
    }

    return m_NextConnectionId++;
}

bool VisualGraph::Disconnect(uint32_t sourceNodeId, uint32_t sourcePortId,
                             uint32_t targetNodeId, uint32_t targetPortId) {
    auto it = std::find_if(m_Connections.begin(), m_Connections.end(),
        [=](const Connection& c) {
            return c.sourceNodeId == sourceNodeId && c.sourcePortId == sourcePortId &&
                   c.targetNodeId == targetNodeId && c.targetPortId == targetPortId;
        });

    if (it != m_Connections.end()) {
        m_Connections.erase(it);
        
        // Check if any other connections use these ports
        auto sourceConnected = std::any_of(m_Connections.begin(), m_Connections.end(),
            [=](const Connection& c) { return (c.sourceNodeId == sourceNodeId && c.sourcePortId == sourcePortId) ||
                                              (c.targetNodeId == sourceNodeId && c.targetPortId == sourcePortId); });
        auto targetConnected = std::any_of(m_Connections.begin(), m_Connections.end(),
            [=](const Connection& c) { return (c.sourceNodeId == targetNodeId && c.sourcePortId == targetPortId) ||
                                              (c.targetNodeId == targetNodeId && c.targetPortId == targetPortId); });

        auto sourceNode = GetNode(sourceNodeId);
        auto targetNode = GetNode(targetNodeId);
        
        if (sourceNode && !sourceConnected) {
            if (auto port = sourceNode->FindPort(sourcePortId)) {
                const_cast<Port*>(port)->isConnected = false;
            }
        }
        if (targetNode && !targetConnected) {
            if (auto port = targetNode->FindPort(targetPortId)) {
                const_cast<Port*>(port)->isConnected = false;
            }
        }

        return true;
    }
    return false;
}

bool VisualGraph::DisconnectPort(uint32_t nodeId, uint32_t portId) {
    std::vector<Connection> toRemove;
    
    for (auto& conn : m_Connections) {
        if ((conn.sourceNodeId == nodeId && conn.sourcePortId == portId) ||
            (conn.targetNodeId == nodeId && conn.targetPortId == portId)) {
            toRemove.push_back(conn);
        }
    }

    for (const auto& conn : toRemove) {
        Disconnect(conn.sourceNodeId, conn.sourcePortId, conn.targetNodeId, conn.targetPortId);
    }

    return !toRemove.empty();
}

std::vector<Connection> VisualGraph::GetConnectionsFrom(uint32_t nodeId, uint32_t portId) const {
    std::vector<Connection> result;
    for (const auto& conn : m_Connections) {
        if (conn.sourceNodeId == nodeId && conn.sourcePortId == portId) {
            result.push_back(conn);
        }
    }
    return result;
}

std::vector<Connection> VisualGraph::GetConnectionsTo(uint32_t nodeId, uint32_t portId) const {
    std::vector<Connection> result;
    for (const auto& conn : m_Connections) {
        if (conn.targetNodeId == nodeId && conn.targetPortId == portId) {
            result.push_back(conn);
        }
    }
    return result;
}

bool VisualGraph::IsValid() const {
    // Base implementation - can be overridden by subclasses
    return true;
}

std::vector<std::string> VisualGraph::GetValidationErrors() const {
    return {};
}

json VisualGraph::Serialize() const {
    json obj;
    obj["name"] = m_Name;
    obj["type"] = static_cast<int>(m_Type);
    obj["version"] = m_Version;
    obj["rootNodeId"] = m_RootNodeId;

    // Serialize nodes
    json nodes = json::array();
    for (const auto& [id, node] : m_Nodes) {
        nodes.push_back(node->Serialize());
    }
    obj["nodes"] = nodes;

    // Serialize connections
    json connections = json::array();
    for (const auto& conn : m_Connections) {
        connections.push_back({
            {"sourceNodeId", conn.sourceNodeId},
            {"sourcePortId", conn.sourcePortId},
            {"targetNodeId", conn.targetNodeId},
            {"targetPortId", conn.targetPortId},
            {"label", conn.label}
        });
    }
    obj["connections"] = connections;

    // Serialize blackboard
    obj["blackboard"] = m_Blackboard;

    return obj;
}

bool VisualGraph::Deserialize(const json& data) {
    try {
        if (data.contains("name")) {
            m_Name = data["name"];
        }
        if (data.contains("version")) {
            m_Version = data["version"];
        }
        if (data.contains("rootNodeId")) {
            m_RootNodeId = data["rootNodeId"];
        }

        // Deserialize nodes
        if (data.contains("nodes")) {
            m_Nodes.clear();
            for (const auto& nodeData : data["nodes"]) {
                NodeType type = static_cast<NodeType>(nodeData["type"].get<int>());
                auto node = NodeRegistry::Get().CreateNode(type, nodeData["id"]);
                if (node) {
                    node->Deserialize(nodeData);
                    m_Nodes[node->GetId()] = node;
                }
            }
        }

        // Deserialize connections
        if (data.contains("connections")) {
            m_Connections.clear();
            for (const auto& connData : data["connections"]) {
                Connection conn;
                conn.sourceNodeId = connData["sourceNodeId"];
                conn.sourcePortId = connData["sourcePortId"];
                conn.targetNodeId = connData["targetNodeId"];
                conn.targetPortId = connData["targetPortId"];
                conn.label = connData["label"];
                m_Connections.push_back(conn);
            }
        }

        // Deserialize blackboard
        if (data.contains("blackboard")) {
            m_Blackboard = data["blackboard"].get<std::map<std::string, json>>();
        }

        return true;
    } catch (...) {
        return false;
    }
}

void VisualGraph::Reset() {
    m_Blackboard.clear();
}

// BehaviorTreeGraph implementation
BehaviorTreeGraph::BehaviorTreeGraph(const std::string& name)
    : VisualGraph(name, GraphType::BehaviorTree) {
}

bool BehaviorTreeGraph::IsValid() const {
    return m_RootNodeId != 0 && m_Nodes.find(m_RootNodeId) != m_Nodes.end();
}

void BehaviorTreeGraph::Execute() {
    if (IsValid()) {
        auto rootNode = GetNode(m_RootNodeId);
        if (rootNode) {
            rootNode->Execute();
        }
    }
}

// StateMachineGraph implementation
StateMachineGraph::StateMachineGraph(const std::string& name)
    : VisualGraph(name, GraphType::StateMachine), m_CurrentStateId(0) {
}

bool StateMachineGraph::IsValid() const {
    return m_CurrentStateId == 0 || m_Nodes.find(m_CurrentStateId) != m_Nodes.end();
}

void StateMachineGraph::Execute() {
    if (IsValid() && m_CurrentStateId != 0) {
        auto currentState = GetNode(m_CurrentStateId);
        if (currentState) {
            currentState->Execute();
        }
    }
}

void StateMachineGraph::SetCurrentState(uint32_t stateId) {
    if (m_Nodes.find(stateId) != m_Nodes.end()) {
        m_CurrentStateId = stateId;
    }
}

bool StateMachineGraph::CanTransition(uint32_t fromState, uint32_t toState) const {
    for (const auto& conn : m_Connections) {
        if (conn.sourceNodeId == fromState && conn.targetNodeId == toState) {
            return true;
        }
    }
    return false;
}

// LogicGraph implementation
LogicGraph::LogicGraph(const std::string& name)
    : VisualGraph(name, GraphType::LogicGraph) {
}

bool LogicGraph::IsValid() const {
    return m_RootNodeId != 0 && m_Nodes.find(m_RootNodeId) != m_Nodes.end();
}

void LogicGraph::Execute() {
    if (IsValid()) {
        auto rootNode = GetNode(m_RootNodeId);
        if (rootNode) {
            rootNode->Execute();
        }
    }
}

// AnimationGraph implementation
AnimationGraph::AnimationGraph(const std::string& name)
    : VisualGraph(name, GraphType::AnimationGraph) {
}

bool AnimationGraph::IsValid() const {
    return m_RootNodeId != 0 && m_Nodes.find(m_RootNodeId) != m_Nodes.end();
}

void AnimationGraph::Execute() {
    if (IsValid()) {
        auto rootNode = GetNode(m_RootNodeId);
        if (rootNode) {
            rootNode->Execute();
        }
    }
}
