#pragma once

#include "VisualGraph.h"
#include "../ImGuiManager.h"
#include <memory>
#include <string>
#include <vector>

class VisualScriptEditor {
public:
    VisualScriptEditor();
    ~VisualScriptEditor();

    // Graph management
    void NewGraph(const std::string& name, GraphType type);
    void OpenGraph(const std::string& filePath);
    void SaveGraph(const std::string& filePath = "");
    void CloseGraph();

    std::shared_ptr<VisualGraph> GetCurrentGraph() const { return m_CurrentGraph; }
    bool IsGraphModified() const { return m_IsModified; }
    void MarkGraphModified() { m_IsModified = true; }

    // Editor UI rendering
    void RenderUI();
    void RenderMenuBar();
    void RenderGraphCanvas();
    void RenderPropertiesPanel();
    void RenderNodeLibrary();
    void RenderStatusBar();

    // Node management
    uint32_t GetSelectedNodeId() const { return m_SelectedNodeId; }
    void SelectNode(uint32_t nodeId) { m_SelectedNodeId = nodeId; }
    void DeselectNode() { m_SelectedNodeId = 0; }

    // Connection management (for the user dragging connections)
    void StartConnection(uint32_t nodeId, uint32_t portId);
    void EndConnection(uint32_t nodeId, uint32_t portId);
    void CancelConnection();
    bool IsConnecting() const { return m_IsConnecting; }

    // Camera/viewport
    Vec2 GetCanvasOffset() const { return m_CanvasOffset; }
    float GetCanvasScale() const { return m_CanvasScale; }
    void SetCanvasOffset(const Vec2& offset) { m_CanvasOffset = offset; }
    void SetCanvasScale(float scale) { m_CanvasScale = scale; }

    // Code generation & compilation
    void GenerateCode();
    void CompileAndReload();
    std::string GetGeneratedCode() const { return m_GeneratedCode; }

    // Settings
    bool GetShowNodeIds() const { return m_ShowNodeIds; }
    void SetShowNodeIds(bool show) { m_ShowNodeIds = show; }

    bool GetAutoReload() const { return m_AutoReload; }
    void SetAutoReload(bool enabled) { m_AutoReload = enabled; }

    // Undo/Redo
    void Undo();
    void Redo();
    bool CanUndo() const;
    bool CanRedo() const;

    // Recent files
    const std::vector<std::string>& GetRecentFiles() const { return m_RecentFiles; }
    void AddRecentFile(const std::string& path);

    // Execution/debugging
    void ExecuteGraph();
    void PauseExecution();
    void StopExecution();
    bool IsExecuting() const { return m_IsExecuting; }

    // Validation
    void ValidateGraph();
    const std::vector<std::string>& GetValidationErrors() const { return m_ValidationErrors; }

private:
    void RenderNodeContextMenu();
    void RenderCanvasContextMenu();
    void RenderConnectionLine(const Vec2& from, const Vec2& to);
    void RenderNode(uint32_t nodeId);
    void RenderNodePort(const Port& port, const Vec2& nodePos, bool isInput);

    void DeleteNode(uint32_t nodeId);
    void DeleteConnection(const Connection& conn);

    Vec2 ConvertScreenToGraph(const Vec2& screenPos) const;
    Vec2 ConvertGraphToScreen(const Vec2& graphPos) const;

    // State
    std::shared_ptr<VisualGraph> m_CurrentGraph;
    std::string m_CurrentFilePath;
    bool m_IsModified = false;

    // Editor state
    uint32_t m_SelectedNodeId = 0;
    uint32_t m_HoveredNodeId = 0;
    uint32_t m_HoveredPortId = 0;

    // Connection drawing
    bool m_IsConnecting = false;
    uint32_t m_ConnectionSourceNodeId = 0;
    uint32_t m_ConnectionSourcePortId = 0;
    Vec2 m_ConnectionCurrentPos;

    // Canvas
    Vec2 m_CanvasOffset = {100, 100};
    float m_CanvasScale = 1.0f;
    bool m_IsPanning = false;
    Vec2 m_LastMousePos;

    // Settings
    bool m_ShowNodeIds = false;
    bool m_AutoReload = true;
    bool m_IsExecuting = false;

    // Generated code
    std::string m_GeneratedCode;

    // Recent files
    std::vector<std::string> m_RecentFiles;
    static const size_t MAX_RECENT_FILES = 10;

    // Undo/Redo
    std::vector<json> m_UndoStack;
    std::vector<json> m_RedoStack;
    static const size_t MAX_UNDO_STEPS = 50;

    // Validation
    std::vector<std::string> m_ValidationErrors;

    // UI State
    bool m_ShowNodeLibrary = true;
    bool m_ShowPropertiesPanel = true;
    bool m_ShowGeneratedCode = false;
    bool m_ShowValidationErrors = false;

    ImGui::ID m_NodeLibraryWindowId = 0;
    ImGui::ID m_PropertiesPanelWindowId = 0;
    ImGui::ID m_CanvasWindowId = 0;
};
