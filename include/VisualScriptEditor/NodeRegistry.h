#pragma once

#include "VisualNode.h"
#include <map>
#include <functional>
#include <memory>

struct NodeDefinition {
    NodeType type;
    std::string name;
    std::string category;
    std::string description;
    std::function<std::shared_ptr<VisualNode>(uint32_t)> factory;
    uint32_t color;
    float width;
    float height;
};

class NodeRegistry {
public:
    static NodeRegistry& Get();

    // Register a custom node type
    void RegisterNode(const NodeDefinition& definition);

    // Get node definition
    const NodeDefinition* GetNodeDefinition(NodeType type) const;
    const NodeDefinition* GetNodeDefinitionByName(const std::string& name) const;

    // Factory method
    std::shared_ptr<VisualNode> CreateNode(NodeType type, uint32_t id);
    std::shared_ptr<VisualNode> CreateNodeByName(const std::string& name, uint32_t id);

    // Get all definitions
    const std::map<NodeType, NodeDefinition>& GetAllDefinitions() const { return m_Definitions; }
    const std::map<std::string, NodeDefinition>& GetDefinitionsByCategory(const std::string& category) const;

    // Categories
    std::vector<std::string> GetCategories() const;

private:
    NodeRegistry();
    void InitializeBuiltInNodes();

    std::map<NodeType, NodeDefinition> m_Definitions;
    std::map<std::string, std::vector<NodeDefinition>> m_DefinitionsByCategory;

    static NodeRegistry* s_Instance;
};

// Node builder helper for fluent API
class NodeBuilder {
public:
    NodeBuilder& WithName(const std::string& name);
    NodeBuilder& WithCategory(const std::string& category);
    NodeBuilder& WithDescription(const std::string& desc);
    NodeBuilder& WithColor(uint32_t color);
    NodeBuilder& WithSize(float width, float height);
    NodeBuilder& WithInputPort(const std::string& name, const std::string& type = "execution");
    NodeBuilder& WithOutputPort(const std::string& name, const std::string& type = "execution");

    NodeDefinition Build(NodeType type, std::function<std::shared_ptr<VisualNode>(uint32_t)> factory);

private:
    std::string m_Name;
    std::string m_Category;
    std::string m_Description;
    uint32_t m_Color = 0xFF888888;
    float m_Width = 150.0f;
    float m_Height = 80.0f;
    std::vector<std::pair<std::string, std::string>> m_InputPorts;
    std::vector<std::pair<std::string, std::string>> m_OutputPorts;
};
