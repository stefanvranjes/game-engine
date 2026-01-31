#include "ScriptDebuggerUI.h"
#include <imgui/imgui.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

ScriptDebuggerUI::ScriptDebuggerUI()
    : m_Debugger(ScriptDebugger::GetInstance())
{
}

ScriptDebuggerUI::~ScriptDebuggerUI()
{
    Shutdown();
}

void ScriptDebuggerUI::Init()
{
    // Register debugger callbacks
    DebugCallbacks callbacks;
    
    callbacks.onBreakpointHit = [this](const Breakpoint& bp) {
        // Could display a notification here
    };

    callbacks.onStateChanged = [this](ExecutionState state) {
        OnExecutionStateChanged(state);
    };

    callbacks.onStackUpdated = [this](const std::vector<StackFrame>& stack) {
        // Update call stack display
    };

    m_Debugger.SetCallbacks(callbacks);

    m_IsOpen = false;
    m_SelectedCallStackFrame = 0;
    m_SelectedWatchVariable = -1;
    m_ConsoleInput.clear();
    m_SourceCodeLines.clear();
    m_ExpandedVariables.clear();
}

void ScriptDebuggerUI::Shutdown()
{
    // Clean up UI resources
    m_SourceCodeLines.clear();
    m_ConsoleInput.clear();
}

void ScriptDebuggerUI::Update(float deltaTime)
{
    m_LastUpdateTime = deltaTime;
    m_FrameCount++;
}

void ScriptDebuggerUI::Render()
{
    if (!m_IsOpen) {
        return;
    }

    RenderMainWindow();

    if (m_ShowCallStack) {
        RenderCallStackWindow();
    }

    if (m_ShowVariables) {
        RenderVariablesWindow();
    }

    if (m_ShowWatch) {
        RenderWatchWindow();
    }

    if (m_ShowConsole) {
        RenderConsoleWindow();
    }

    if (m_ShowBreakpoints) {
        RenderBreakpointsWindow();
    }

    if (m_ShowSourceCode) {
        RenderSourceCodeWindow();
    }
}

void ScriptDebuggerUI::RenderMainWindow()
{
    ImGui::SetNextWindowSize(ImVec2(m_MainWindowWidth, m_MainWindowHeight), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Script Debugger", &m_IsOpen, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }

    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Script...")) {
                // Would open file dialog
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Close", "Ctrl+W")) {
                m_IsOpen = false;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Call Stack", nullptr, &m_ShowCallStack);
            ImGui::MenuItem("Variables", nullptr, &m_ShowVariables);
            ImGui::MenuItem("Watch", nullptr, &m_ShowWatch);
            ImGui::MenuItem("Console", nullptr, &m_ShowConsole);
            ImGui::MenuItem("Breakpoints", nullptr, &m_ShowBreakpoints);
            ImGui::MenuItem("Source Code", nullptr, &m_ShowSourceCode);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Debug")) {
            ExecutionState state = m_Debugger.GetExecutionState();
            
            if (ImGui::MenuItem("Start", "F5", false, state == ExecutionState::Stopped)) {
                if (!m_Debugger.GetDebuggedFile().empty()) {
                    m_Debugger.StartDebugSession(m_Debugger.GetDebuggedFile());
                }
            }

            if (ImGui::MenuItem("Pause", "Ctrl+Alt+Break", false, state == ExecutionState::Running)) {
                m_Debugger.Pause();
            }

            if (ImGui::MenuItem("Continue", "F5", false, state == ExecutionState::Paused)) {
                m_Debugger.Resume();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Step Into", "F11", false, state == ExecutionState::Paused)) {
                m_Debugger.StepInto();
            }

            if (ImGui::MenuItem("Step Over", "F10", false, state == ExecutionState::Paused)) {
                m_Debugger.StepOver();
            }

            if (ImGui::MenuItem("Step Out", "Shift+F11", false, state == ExecutionState::Paused)) {
                m_Debugger.StepOut();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Stop", "Shift+F5")) {
                m_Debugger.StopDebugSession();
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    RenderToolbar();
    RenderExecutionStateIndicator();

    ImGui::Separator();

    // Main content area with tabs
    if (ImGui::BeginTabBar("DebugTabs")) {
        if (ImGui::BeginTabItem("Overview")) {
            ImGui::Text("Debugged File: %s", m_Debugger.GetDebuggedFile().c_str());
            ImGui::Text("Current Line: %u", m_Debugger.GetCurrentLine());

            const auto& callStack = m_Debugger.GetCallStack();
            ImGui::Text("Call Stack Depth: %zu", callStack.size());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Quick Stats:");
            ImGui::Text("  Breakpoints: %zu", m_Debugger.GetAllBreakpoints().size());
            ImGui::Text("  Watches: %zu", m_Debugger.GetWatchedVariables().size());

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Call Stack")) {
            RenderCallStack();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Variables")) {
            ImGui::BeginTabBar("VariablesTabs");

            if (ImGui::BeginTabItem("Local")) {
                RenderLocalVariables();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Global")) {
                RenderGlobalVariables();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void ScriptDebuggerUI::RenderToolbar()
{
    ExecutionState state = m_Debugger.GetExecutionState();

    if (ImGui::Button("Continue##toolbar", ImVec2(80, 0))) {
        m_Debugger.Resume();
    }
    ImGui::SameLine();

    if (ImGui::Button("Pause##toolbar", ImVec2(80, 0))) {
        m_Debugger.Pause();
    }
    ImGui::SameLine();

    if (ImGui::Button("Stop##toolbar", ImVec2(80, 0))) {
        m_Debugger.StopDebugSession();
    }
    ImGui::SameLine();

    ImGui::Separator();
    ImGui::SameLine();

    if (ImGui::Button("Step Into##toolbar", ImVec2(80, 0))) {
        m_Debugger.StepInto();
    }
    ImGui::SameLine();

    if (ImGui::Button("Step Over##toolbar", ImVec2(80, 0))) {
        m_Debugger.StepOver();
    }
    ImGui::SameLine();

    if (ImGui::Button("Step Out##toolbar", ImVec2(80, 0))) {
        m_Debugger.StepOut();
    }
}

void ScriptDebuggerUI::RenderExecutionStateIndicator()
{
    ExecutionState state = m_Debugger.GetExecutionState();

    std::string stateText;
    ImVec4 stateColor;

    switch (state) {
        case ExecutionState::Stopped:
            stateText = "STOPPED";
            stateColor = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
            break;
        case ExecutionState::Running:
            stateText = "RUNNING";
            stateColor = m_ExecutingColor;
            break;
        case ExecutionState::Paused:
            stateText = "PAUSED";
            stateColor = m_PausedColor;
            break;
        case ExecutionState::Stepping:
            stateText = "STEPPING";
            stateColor = m_PausedColor;
            break;
        case ExecutionState::SteppingOver:
            stateText = "STEP OVER";
            stateColor = m_PausedColor;
            break;
        case ExecutionState::SteppingOut:
            stateText = "STEP OUT";
            stateColor = m_PausedColor;
            break;
    }

    ImGui::TextColored(stateColor, "Status: %s", stateText.c_str());
}

void ScriptDebuggerUI::RenderCallStackWindow()
{
    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 350), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Call Stack##window", &m_ShowCallStack)) {
        ImGui::End();
        return;
    }

    RenderCallStack();
    ImGui::End();
}

void ScriptDebuggerUI::RenderCallStack()
{
    const auto& callStack = m_Debugger.GetCallStack();

    if (callStack.empty()) {
        ImGui::TextDisabled("(empty call stack)");
        return;
    }

    if (ImGui::BeginTable("CallStackTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Frame");
        ImGui::TableSetupColumn("Function");
        ImGui::TableSetupColumn("File");
        ImGui::TableSetupColumn("Line");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < callStack.size(); ++i) {
            const auto& frame = callStack[i];
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%zu", i);

            ImGui::TableSetColumnIndex(1);
            if (ImGui::Selectable(frame.functionName.c_str(), m_SelectedCallStackFrame == static_cast<int>(i))) {
                m_SelectedCallStackFrame = static_cast<int>(i);
                OnStackFrameSelected(static_cast<uint32_t>(i));
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::TextWrapped("%s", frame.filepath.c_str());

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%u", frame.line);
        }

        ImGui::EndTable();
    }
}

void ScriptDebuggerUI::RenderVariablesWindow()
{
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(450, 350), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Variables##window", &m_ShowVariables)) {
        ImGui::End();
        return;
    }

    ImGui::BeginTabBar("VarsTabBar");

    if (ImGui::BeginTabItem("Local")) {
        RenderLocalVariables();
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Global")) {
        RenderGlobalVariables();
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    ImGui::End();
}

void ScriptDebuggerUI::RenderLocalVariables()
{
    const auto& locals = m_Debugger.GetLocalVariables();

    if (locals.empty()) {
        ImGui::TextDisabled("(no local variables)");
        return;
    }

    for (const auto& var : locals) {
        RenderVariableNode(var, "local_" + var.name);
    }
}

void ScriptDebuggerUI::RenderGlobalVariables()
{
    const auto& globals = m_Debugger.GetGlobalVariables();

    if (globals.empty()) {
        ImGui::TextDisabled("(no global variables)");
        return;
    }

    for (const auto& var : globals) {
        RenderVariableNode(var, "global_" + var.name);
    }
}

void ScriptDebuggerUI::RenderVariableNode(const DebugVariable& var, const std::string& id)
{
    if (var.expandable && !var.children.empty()) {
        bool open = ImGui::TreeNode(id.c_str(), "%s: %s = %s",
            var.type.c_str(), var.name.c_str(), FormatValue(var.value).c_str());

        if (open) {
            for (const auto& child : var.children) {
                RenderVariableNode(child, id + "_" + child.name);
            }
            ImGui::TreePop();
        }
    } else {
        ImGui::BulletText("%s: %s = %s", var.type.c_str(), var.name.c_str(),
            FormatValue(var.value).c_str());
    }
}

void ScriptDebuggerUI::RenderWatchWindow()
{
    ImGui::SetNextWindowSize(ImVec2(400, 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(850, 350), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Watch##window", &m_ShowWatch)) {
        ImGui::End();
        return;
    }

    RenderWatchVariables();

    ImGui::Separator();

    // Add new watch
    static char watchExpr[256] = "";
    ImGui::InputText("Expression##watch", watchExpr, sizeof(watchExpr));
    ImGui::SameLine();

    if (ImGui::Button("Add##watch")) {
        if (strlen(watchExpr) > 0) {
            m_Debugger.AddWatch(watchExpr);
            memset(watchExpr, 0, sizeof(watchExpr));
        }
    }

    ImGui::End();
}

void ScriptDebuggerUI::RenderWatchVariables()
{
    const auto& watches = m_Debugger.GetWatchedVariables();

    if (watches.empty()) {
        ImGui::TextDisabled("(no watched variables)");
        return;
    }

    if (ImGui::BeginTable("WatchTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Expression", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& watch : watches) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", watch.name.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", watch.type.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::TextWrapped("%s", FormatValue(watch.value).c_str());
        }

        ImGui::EndTable();
    }
}

void ScriptDebuggerUI::RenderConsoleWindow()
{
    ImGui::SetNextWindowSize(ImVec2(1000, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 550), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Console##window", &m_ShowConsole)) {
        ImGui::End();
        return;
    }

    RenderConsoleOutput();

    ImGui::Separator();

    RenderConsoleInput();

    ImGui::End();
}

void ScriptDebuggerUI::RenderConsoleOutput()
{
    const auto& history = m_Debugger.GetConsoleHistory();

    if (ImGui::BeginChild("ConsoleOutput", ImVec2(0, -30), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        for (const auto& msg : history) {
            ImGui::TextWrapped("%s", msg.first.c_str());
        }

        // Auto-scroll to bottom
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1) {
            ImGui::SetScrollHereY(1.0f);
        }
    }
    ImGui::EndChild();
}

void ScriptDebuggerUI::RenderConsoleInput()
{
    ImGui::InputText("##ConsoleinputField", m_ConsoleInput.data(), m_ConsoleInput.capacity(),
        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();

    if (ImGui::Button("Execute##console")) {
        if (!m_ConsoleInput.empty()) {
            // Would execute the command
            m_ConsoleInput.clear();
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Clear##console")) {
        const_cast<std::vector<std::pair<std::string, float>>&>(
            m_Debugger.GetConsoleHistory()).clear();
    }
}

void ScriptDebuggerUI::RenderBreakpointsWindow()
{
    ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1050, 50), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Breakpoints##window", &m_ShowBreakpoints)) {
        ImGui::End();
        return;
    }

    RenderBreakpointList();
    ImGui::End();
}

void ScriptDebuggerUI::RenderBreakpointList()
{
    const auto& breakpoints = m_Debugger.GetAllBreakpoints();

    if (breakpoints.empty()) {
        ImGui::TextDisabled("(no breakpoints)");
        return;
    }

    if (ImGui::BeginTable("BreakpointsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("File");
        ImGui::TableSetupColumn("Line", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableHeadersRow();

        for (const auto& bp : breakpoints) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            bool enabled = bp.enabled;
            if (ImGui::Checkbox(("##bp_enabled_" + std::to_string(bp.id)).c_str(), &enabled)) {
                const_cast<Breakpoint&>(bp).enabled = enabled;
            }

            ImGui::TableSetColumnIndex(1);
            ImGui::TextWrapped("%s", bp.filepath.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%u", bp.line);

            ImGui::TableSetColumnIndex(3);
            if (bp.isLogpoint) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Log");
            } else if (bp.isConditional) {
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Cond");
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Break");
            }

            ImGui::TableSetColumnIndex(4);
            if (ImGui::Button(("Remove##bp_" + std::to_string(bp.id)).c_str(), ImVec2(60, 0))) {
                m_Debugger.RemoveBreakpoint(bp.id);
            }
        }

        ImGui::EndTable();
    }
}

void ScriptDebuggerUI::RenderSourceCodeWindow()
{
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 200), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Source Code##window", &m_ShowSourceCode)) {
        ImGui::End();
        return;
    }

    RenderSourceFile();
    ImGui::End();
}

void ScriptDebuggerUI::RenderSourceFile()
{
    const std::string& file = m_Debugger.GetDebuggedFile();
    uint32_t currentLine = m_Debugger.GetCurrentLine();

    if (file.empty()) {
        ImGui::TextDisabled("(no file being debugged)");
        return;
    }

    if (m_CurrentlyViewedFile != file) {
        UpdateSourceCodeCache();
        m_CurrentlyViewedFile = file;
    }

    if (ImGui::BeginChild("SourceCodeArea", ImVec2(0, -25), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        const auto& lines = m_SourceCodeLines;
        auto breakpointsInFile = m_Debugger.GetBreakpointsForFile(file);

        for (size_t i = 0; i < lines.size(); ++i) {
            uint32_t lineNum = static_cast<uint32_t>(i + 1);
            bool isCurrentLine = (lineNum == currentLine);
            bool hasBreakpoint = std::any_of(breakpointsInFile.begin(), breakpointsInFile.end(),
                [lineNum](const Breakpoint& bp) { return bp.line == lineNum; });

            // Line number and gutter
            if (hasBreakpoint) {
                ImGui::TextColored(m_BreakpointColor, "%4u", lineNum);
            } else {
                ImGui::Text("%4u", lineNum);
            }

            ImGui::SameLine();

            // Source code with highlight
            if (isCurrentLine) {
                ImGui::TextColored(m_CurrentLineColor, "%s", lines[i].c_str());
            } else {
                ImGui::Text("%s", lines[i].c_str());
            }

            // Clickable for breakpoints
            ImGui::SetItemAllowOverlap();
            ImGui::PushID(static_cast<int>(i));
            ImGui::InvisibleButton("##LineClickable", ImVec2(ImGui::GetContentRegionAvail().x, 0));
            if (ImGui::IsItemClicked()) {
                if (hasBreakpoint) {
                    auto it = std::find_if(breakpointsInFile.begin(), breakpointsInFile.end(),
                        [lineNum](const Breakpoint& bp) { return bp.line == lineNum; });
                    if (it != breakpointsInFile.end()) {
                        m_Debugger.RemoveBreakpoint(it->id);
                    }
                } else {
                    m_Debugger.AddBreakpoint(file, lineNum);
                }
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    ImGui::Text("File: %s", file.c_str());
}

void ScriptDebuggerUI::UpdateSourceCodeCache()
{
    m_SourceCodeLines.clear();

    const std::string& file = m_Debugger.GetDebuggedFile();
    if (file.empty()) {
        return;
    }

    std::ifstream infile(file);
    if (!infile.is_open()) {
        m_SourceCodeLines.push_back("<Could not open file: " + file + ">");
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        m_SourceCodeLines.push_back(line);
    }

    infile.close();
}

void ScriptDebuggerUI::OnBreakpointClicked(uint32_t breakpointId)
{
    // Handle breakpoint interaction
}

void ScriptDebuggerUI::OnBreakpointToggled(uint32_t breakpointId)
{
    // Handle breakpoint toggle
}

void ScriptDebuggerUI::OnBreakpointRemoved(uint32_t breakpointId)
{
    // Handle breakpoint removal
}

void ScriptDebuggerUI::OnStackFrameSelected(uint32_t frameIndex)
{
    m_SelectedCallStackFrame = static_cast<int>(frameIndex);
}

void ScriptDebuggerUI::OnExecutionStateChanged(ExecutionState state)
{
    // Handle state change
}

std::string ScriptDebuggerUI::FormatValue(const std::string& value)
{
    if (value.length() > 100) {
        return value.substr(0, 97) + "...";
    }
    return value;
}

std::string ScriptDebuggerUI::TruncateText(const std::string& text, size_t maxLength)
{
    if (text.length() > maxLength) {
        return text.substr(0, maxLength - 3) + "...";
    }
    return text;
}

void ScriptDebuggerUI::SetFont(void* font)
{
    // Would set the ImGui font
}

void ScriptDebuggerUI::SetThemeDarkMode(bool darkMode)
{
    if (darkMode) {
        ImGui::StyleColorsDark();
    } else {
        ImGui::StyleColorsLight();
    }
}
