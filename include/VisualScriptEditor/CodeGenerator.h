#pragma once

#include "VisualGraph.h"
#include <string>
#include <sstream>
#include <memory>

class CodeGenerator {
public:
    virtual ~CodeGenerator() = default;

    // Generate C++ code from a graph
    virtual std::string GenerateCode(const std::shared_ptr<VisualGraph>& graph) = 0;

    // Generate header file content
    virtual std::string GenerateHeader(const std::shared_ptr<VisualGraph>& graph);

    // Get file extension (cpp, h, etc.)
    virtual std::string GetFileExtension() const { return ".cpp"; }

protected:
    std::string EscapeName(const std::string& name) const;
    std::string GetNodeClassName(NodeType type) const;
};

// Behavior Tree Code Generator
class BehaviorTreeCodeGenerator : public CodeGenerator {
public:
    std::string GenerateCode(const std::shared_ptr<VisualGraph>& graph) override;
    std::string GetFileExtension() const override { return ".cpp"; }

private:
    std::string GenerateNodeClass(const std::shared_ptr<VisualNode>& node, int indent = 0);
    std::string GenerateTreeStructure(const std::shared_ptr<VisualGraph>& graph, uint32_t nodeId, int indent);
    std::string GenerateTreeExecution(const std::shared_ptr<VisualGraph>& graph);
};

// State Machine Code Generator
class StateMachineCodeGenerator : public CodeGenerator {
public:
    std::string GenerateCode(const std::shared_ptr<VisualGraph>& graph) override;
    std::string GetFileExtension() const override { return ".cpp"; }

private:
    std::string GenerateStateEnums(const std::shared_ptr<VisualGraph>& graph);
    std::string GenerateStateHandlers(const std::shared_ptr<VisualGraph>& graph);
    std::string GenerateTransitionLogic(const std::shared_ptr<VisualGraph>& graph);
};

// Logic Graph Code Generator
class LogicGraphCodeGenerator : public CodeGenerator {
public:
    std::string GenerateCode(const std::shared_ptr<VisualGraph>& graph) override;
    std::string GetFileExtension() const override { return ".cpp"; }

private:
    std::string GenerateLogicFunction(const std::shared_ptr<VisualGraph>& graph);
    std::string GenerateNodeEvaluation(const std::shared_ptr<VisualNode>& node, int indent);
};

// Animation Graph Code Generator
class AnimationGraphCodeGenerator : public CodeGenerator {
public:
    std::string GenerateCode(const std::shared_ptr<VisualGraph>& graph) override;
    std::string GetFileExtension() const override { return ".cpp"; }

private:
    std::string GenerateAnimationStates(const std::shared_ptr<VisualGraph>& graph);
    std::string GenerateBlending(const std::shared_ptr<VisualGraph>& graph);
};

// Factory for creating appropriate code generator
class CodeGeneratorFactory {
public:
    static std::unique_ptr<CodeGenerator> CreateGenerator(GraphType type);
};
