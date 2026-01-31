#include "ScriptDebugger.h"
#include "IScriptSystem.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

void ScriptDebugger::Init()
{
    m_IsDebugging = false;
    m_ExecutionState = ExecutionState::Stopped;
    m_Breakpoints.clear();
    m_NextBreakpointId = 1;
    m_BreakpointsByFile.clear();
    m_Watches.clear();
    m_NextWatchId = 1;
    m_ConsoleHistory.clear();
}

void ScriptDebugger::Shutdown()
{
    StopDebugSession();
    m_Breakpoints.clear();
    m_BreakpointsByFile.clear();
    m_Watches.clear();
    m_ConsoleHistory.clear();
}

uint32_t ScriptDebugger::AddBreakpoint(const std::string& filepath, uint32_t line)
{
    return AddConditionalBreakpoint(filepath, line, "");
}

uint32_t ScriptDebugger::AddConditionalBreakpoint(const std::string& filepath, uint32_t line,
                                                   const std::string& condition)
{
    Breakpoint bp;
    bp.id = GenerateBreakpointId();
    bp.filepath = filepath;
    bp.line = line;
    bp.enabled = true;
    bp.isConditional = !condition.empty();
    bp.condition = condition;
    bp.isLogpoint = false;
    bp.hitCount = 0;
    bp.targetHitCount = 0;

    m_Breakpoints.push_back(bp);
    m_BreakpointsByFile[filepath].push_back(bp.id);

    return bp.id;
}

uint32_t ScriptDebugger::AddLogpoint(const std::string& filepath, uint32_t line,
                                     const std::string& message)
{
    Breakpoint bp;
    bp.id = GenerateBreakpointId();
    bp.filepath = filepath;
    bp.line = line;
    bp.enabled = true;
    bp.isConditional = false;
    bp.isLogpoint = true;
    bp.logMessage = message;
    bp.hitCount = 0;
    bp.targetHitCount = 0;

    m_Breakpoints.push_back(bp);
    m_BreakpointsByFile[filepath].push_back(bp.id);

    return bp.id;
}

void ScriptDebugger::RemoveBreakpoint(uint32_t id)
{
    // Remove from main list
    auto it = std::find_if(m_Breakpoints.begin(), m_Breakpoints.end(),
        [id](const Breakpoint& bp) { return bp.id == id; });

    if (it != m_Breakpoints.end()) {
        std::string filepath = it->filepath;
        m_Breakpoints.erase(it);

        // Remove from file map
        auto& bpList = m_BreakpointsByFile[filepath];
        bpList.erase(std::remove(bpList.begin(), bpList.end(), id), bpList.end());
        
        if (bpList.empty()) {
            m_BreakpointsByFile.erase(filepath);
        }
    }
}

void ScriptDebugger::SetBreakpointEnabled(uint32_t id, bool enabled)
{
    auto it = std::find_if(m_Breakpoints.begin(), m_Breakpoints.end(),
        [id](const Breakpoint& bp) { return bp.id == id; });

    if (it != m_Breakpoints.end()) {
        it->enabled = enabled;
    }
}

std::vector<Breakpoint> ScriptDebugger::GetBreakpointsForFile(const std::string& filepath) const
{
    std::vector<Breakpoint> result;
    auto it = m_BreakpointsByFile.find(filepath);
    
    if (it != m_BreakpointsByFile.end()) {
        for (uint32_t id : it->second) {
            auto bpIt = std::find_if(m_Breakpoints.begin(), m_Breakpoints.end(),
                [id](const Breakpoint& bp) { return bp.id == id; });
            
            if (bpIt != m_Breakpoints.end()) {
                result.push_back(*bpIt);
            }
        }
    }

    return result;
}

void ScriptDebugger::StartDebugSession(const std::string& filepath, IScriptSystem* scriptSystem)
{
    m_IsDebugging = true;
    m_DebuggedFile = filepath;
    m_CurrentLine = 0;
    m_ExecutionState = ExecutionState::Running;
    m_CallStack.clear();
    m_LocalVariables.clear();
    m_GlobalVariables.clear();

    if (scriptSystem) {
        AttachToScriptSystem(scriptSystem);
    }

    if (m_Callbacks.onExecutionStarted) {
        m_Callbacks.onExecutionStarted();
    }

    if (m_Callbacks.onStateChanged) {
        m_Callbacks.onStateChanged(ExecutionState::Running);
    }

    // Log to console
    m_ConsoleHistory.emplace_back("[DEBUG] Started debugging: " + filepath, 0.0f);
}

void ScriptDebugger::StopDebugSession()
{
    if (!m_IsDebugging) {
        return;
    }

    m_IsDebugging = false;
    m_ExecutionState = ExecutionState::Stopped;
    m_DebuggedFile.clear();
    m_CurrentLine = 0;
    m_CallStack.clear();
    m_LocalVariables.clear();
    m_GlobalVariables.clear();

    DetachFromScriptSystem();

    if (m_Callbacks.onExecutionStopped) {
        m_Callbacks.onExecutionStopped();
    }

    if (m_Callbacks.onStateChanged) {
        m_Callbacks.onStateChanged(ExecutionState::Stopped);
    }

    m_ConsoleHistory.emplace_back("[DEBUG] Debug session ended", 0.0f);
}

void ScriptDebugger::Pause()
{
    if (m_ExecutionState == ExecutionState::Running) {
        m_ExecutionState = ExecutionState::Paused;
        
        if (m_Callbacks.onStateChanged) {
            m_Callbacks.onStateChanged(ExecutionState::Paused);
        }

        m_ConsoleHistory.emplace_back("[DEBUG] Execution paused", 0.0f);
    }
}

void ScriptDebugger::Resume()
{
    if (m_ExecutionState == ExecutionState::Paused || m_ExecutionState == ExecutionState::Stepping) {
        m_ExecutionState = ExecutionState::Running;

        if (m_Callbacks.onStateChanged) {
            m_Callbacks.onStateChanged(ExecutionState::Running);
        }

        m_ConsoleHistory.emplace_back("[DEBUG] Execution resumed", 0.0f);
    }
}

void ScriptDebugger::StepInto()
{
    if (m_ExecutionState == ExecutionState::Paused) {
        m_ExecutionState = ExecutionState::Stepping;
        m_StepTargetFrame = m_CallStack.size();

        if (m_Callbacks.onStateChanged) {
            m_Callbacks.onStateChanged(ExecutionState::Stepping);
        }

        m_ConsoleHistory.emplace_back("[DEBUG] Step into", 0.0f);
    }
}

void ScriptDebugger::StepOver()
{
    if (m_ExecutionState == ExecutionState::Paused) {
        m_ExecutionState = ExecutionState::SteppingOver;
        m_StepTargetFrame = m_CallStack.empty() ? 0 : m_CallStack.size() - 1;

        if (m_Callbacks.onStateChanged) {
            m_Callbacks.onStateChanged(ExecutionState::SteppingOver);
        }

        m_ConsoleHistory.emplace_back("[DEBUG] Step over", 0.0f);
    }
}

void ScriptDebugger::StepOut()
{
    if (m_ExecutionState == ExecutionState::Paused) {
        m_ExecutionState = ExecutionState::SteppingOut;
        m_StepTargetFrame = m_CallStack.empty() ? 0 : m_CallStack.size() - 1;

        if (m_Callbacks.onStateChanged) {
            m_Callbacks.onStateChanged(ExecutionState::SteppingOut);
        }

        m_ConsoleHistory.emplace_back("[DEBUG] Step out", 0.0f);
    }
}

uint32_t ScriptDebugger::AddWatch(const std::string& expression)
{
    uint32_t watchId = m_NextWatchId++;
    m_Watches[watchId] = expression;

    DebugVariable var;
    var.name = expression;
    var.value = EvaluateExpression(expression);
    var.type = "watch";
    m_WatchedVariables.push_back(var);

    return watchId;
}

void ScriptDebugger::RemoveWatch(uint32_t watchId)
{
    auto it = m_Watches.find(watchId);
    if (it != m_Watches.end()) {
        m_Watches.erase(it);
        
        // Remove from watched variables list
        const std::string& expr = it->second;
        auto varIt = std::find_if(m_WatchedVariables.begin(), m_WatchedVariables.end(),
            [&expr](const DebugVariable& var) { return var.name == expr; });
        
        if (varIt != m_WatchedVariables.end()) {
            m_WatchedVariables.erase(varIt);
        }
    }
}

std::string ScriptDebugger::EvaluateExpression(const std::string& expression)
{
    // This would be implemented per script system
    // For now, return a placeholder
    if (expression.empty()) {
        return "undefined";
    }

    // Try to find in local variables
    for (const auto& var : m_LocalVariables) {
        if (var.name == expression) {
            return var.value;
        }
    }

    // Try to find in global variables
    for (const auto& var : m_GlobalVariables) {
        if (var.name == expression) {
            return var.value;
        }
    }

    return "<error: variable not found>";
}

void ScriptDebugger::Update(float deltaTime)
{
    if (!m_IsDebugging) {
        return;
    }

    // Update watched variables
    for (auto& var : m_WatchedVariables) {
        var.value = EvaluateExpression(var.name);
    }

    // Decrease console message lifetimes
    for (auto& msg : m_ConsoleHistory) {
        msg.second += deltaTime;
    }

    // Keep console history reasonable size
    const size_t MAX_CONSOLE_LINES = 1000;
    if (m_ConsoleHistory.size() > MAX_CONSOLE_LINES) {
        m_ConsoleHistory.erase(m_ConsoleHistory.begin(),
            m_ConsoleHistory.begin() + (m_ConsoleHistory.size() - MAX_CONSOLE_LINES));
    }
}

void ScriptDebugger::AttachToScriptSystem(IScriptSystem* scriptSystem)
{
    m_AttachedScriptSystem = scriptSystem;
    // Script system-specific hook-up would go here
}

void ScriptDebugger::DetachFromScriptSystem()
{
    m_AttachedScriptSystem = nullptr;
}

void ScriptDebugger::InternalBreak(const Breakpoint& bp)
{
    m_ExecutionState = ExecutionState::Paused;
    m_CurrentLine = bp.line;

    UpdateCallStack();
    UpdateVariables();

    if (m_Callbacks.onBreakpointHit) {
        m_Callbacks.onBreakpointHit(bp);
    }

    if (m_Callbacks.onStateChanged) {
        m_Callbacks.onStateChanged(ExecutionState::Paused);
    }

    if (m_Callbacks.onStackUpdated) {
        m_Callbacks.onStackUpdated(m_CallStack);
    }

    std::string msg = "[BREAKPOINT] Hit at " + bp.filepath + ":" + std::to_string(bp.line);
    m_ConsoleHistory.emplace_back(msg, 0.0f);
}

void ScriptDebugger::UpdateCallStack()
{
    // This would be populated by the script system
    // For now, just ensure we have at least one frame
    if (m_CallStack.empty()) {
        StackFrame frame;
        frame.frameIndex = 0;
        frame.functionName = "<main>";
        frame.filepath = m_DebuggedFile;
        frame.line = m_CurrentLine;
        frame.column = 0;
        m_CallStack.push_back(frame);
    }
}

void ScriptDebugger::UpdateVariables()
{
    // This would be populated by the script system
    m_LocalVariables.clear();
    m_GlobalVariables.clear();
}

uint32_t ScriptDebugger::GenerateBreakpointId()
{
    return m_NextBreakpointId++;
}
