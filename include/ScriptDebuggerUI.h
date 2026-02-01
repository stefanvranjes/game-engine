#pragma once

#include "ScriptDebugger.h"
#include <string>
#include <vector>
#include <memory>
#include "imgui/imgui.h"

/**
 * @class ScriptDebuggerUI
 * @brief ImGui-based debugger UI for the script debugging system
 * 
 * Provides visual debugging interface with multiple panels:
 * - Main debugger window with controls
 * - Call stack inspector
 * - Variables watch panel
 * - Console output
 * - Breakpoint management
 * - Source code viewer with line annotations
 * - Performance metrics
 */
class ScriptDebuggerUI {
public:
    ScriptDebuggerUI();
    ~ScriptDebuggerUI();

    /**
     * @brief Initialize the debugger UI
     */
    void Init();

    /**
     * @brief Shutdown the debugger UI
     */
    void Shutdown();

    /**
     * @brief Update debugger UI (call each frame)
     */
    void Update(float deltaTime);

    /**
     * @brief Render all debugger panels
     * Should be called from RenderEditorUI()
     */
    void Render();

    /**
     * @brief Check if debugger window is open
     */
    bool IsOpen() const { return m_IsOpen; }

    /**
     * @brief Set debugger window visibility
     */
    void SetOpen(bool open) { m_IsOpen = open; }

    /**
     * @brief Show the debugger UI
     */
    void Show() { m_IsOpen = true; }

    /**
     * @brief Hide the debugger UI
     */
    void Hide() { m_IsOpen = false; }

    /**
     * @brief Toggle debugger visibility
     */
    void Toggle() { m_IsOpen = !m_IsOpen; }

    // UI configuration
    void SetFont(void* font);
    void SetThemeDarkMode(bool darkMode);

private:
    // Main windows
    void RenderMainWindow();
    void RenderCallStackWindow();
    void RenderVariablesWindow();
    void RenderWatchWindow();
    void RenderConsoleWindow();
    void RenderBreakpointsWindow();
    void RenderSourceCodeWindow();

    // UI helpers
    void RenderToolbar();
    void RenderExecutionStateIndicator();
    void RenderBreakpointList();
    void RenderCallStack();
    void RenderLocalVariables();
    void RenderGlobalVariables();
    void RenderWatchVariables();
    void RenderConsoleOutput();
    void RenderConsoleInput();
    void RenderSourceFile();

    // Variable inspection
    void RenderVariable(const DebugVariable& var, bool expanded = false);
    void RenderVariableNode(const DebugVariable& var, const std::string& id);

    // Event handlers
    void OnBreakpointClicked(uint32_t breakpointId);
    void OnBreakpointToggled(uint32_t breakpointId);
    void OnBreakpointRemoved(uint32_t breakpointId);
    void OnStackFrameSelected(uint32_t frameIndex);
    void OnExecutionStateChanged(ExecutionState state);

    // Internal utilities
    void UpdateSourceCodeCache();
    void HighlightCurrentLine();
    std::string FormatValue(const std::string& value);
    std::string TruncateText(const std::string& text, size_t maxLength);

    // Window states
    bool m_IsOpen = false;
    bool m_ShowCallStack = true;
    bool m_ShowVariables = true;
    bool m_ShowWatch = true;
    bool m_ShowConsole = true;
    bool m_ShowBreakpoints = true;
    bool m_ShowSourceCode = true;

    // Layout configuration
    float m_MainWindowWidth = 1000.0f;
    float m_MainWindowHeight = 600.0f;
    float m_CallStackHeight = 150.0f;
    float m_VariablesHeight = 250.0f;
    float m_ConsoleHeight = 200.0f;

    // UI state
    int m_SelectedCallStackFrame = 0;
    int m_SelectedWatchVariable = -1;
    std::string m_ConsoleInput;
    std::string m_CurrentlyViewedFile;
    std::vector<std::string> m_SourceCodeLines;
    std::vector<bool> m_ExpandedVariables;

    // Colors
    ImVec4 m_BreakpointColor = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);  // Red
    ImVec4 m_CurrentLineColor = ImVec4(1.0f, 1.0f, 0.0f, 0.3f);  // Yellow
    ImVec4 m_ExecutingColor = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);    // Green
    ImVec4 m_PausedColor = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);       // Orange
    ImVec4 m_ErrorColor = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);        // Light Red

    // Performance tracking
    double m_LastUpdateTime = 0.0;
    double m_UIRenderTime = 0.0;
    uint32_t m_FrameCount = 0;

    // Reference to debugger
    ScriptDebugger& m_Debugger;

    // ImGui element IDs for uniqueness
    uint32_t m_WindowIdCounter = 0;
};
