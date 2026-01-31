#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <functional>
#include <cstdint>

/**
 * @class ScriptDebugger
 * @brief Core script debugging system supporting multiple script languages
 * 
 * Provides comprehensive debugging capabilities:
 * - Breakpoint management (line, conditional, logpoint)
 * - Single-stepping through code
 * - Call stack inspection
 * - Variable watching and inspection
 * - Script execution state tracking
 * - Execution profiling
 * - Error/exception handling
 */

// Forward declarations
class IScriptSystem;

/**
 * @struct Breakpoint
 * @brief Represents a debugger breakpoint
 */
struct Breakpoint {
    uint32_t id;                    // Unique breakpoint ID
    std::string filepath;           // File path
    uint32_t line;                  // Line number (1-based)
    bool enabled;                   // Whether breakpoint is active
    bool isConditional;             // If true, only break when condition is true
    std::string condition;          // Lua/AngelScript expression to evaluate
    bool isLogpoint;                // If true, print message instead of breaking
    std::string logMessage;         // Message to print when hit
    uint32_t hitCount;              // Number of times this breakpoint was hit
    uint32_t targetHitCount;        // Break only after N hits (0 = always)

    Breakpoint()
        : id(0), line(0), enabled(true), isConditional(false),
          isLogpoint(false), hitCount(0), targetHitCount(0) {}
};

/**
 * @struct StackFrame
 * @brief Represents a function call in the call stack
 */
struct StackFrame {
    uint32_t frameIndex;            // Frame position (0 = current)
    std::string functionName;       // Name of function
    std::string filepath;           // Source file
    uint32_t line;                  // Current line
    uint32_t column;                // Current column
    std::map<std::string, std::string> localVariables;  // name -> value
    
    StackFrame()
        : frameIndex(0), line(0), column(0) {}
};

/**
 * @struct DebugVariable
 * @brief Represents a variable being watched/inspected
 */
struct DebugVariable {
    std::string name;               // Variable name
    std::string value;              // Current value
    std::string type;               // Variable type
    std::string scope;              // local, global, instance, etc.
    bool expandable;                // Can be expanded to show children
    std::vector<DebugVariable> children;  // For complex types
    
    DebugVariable() : expandable(false) {}
};

/**
 * @enum ExecutionState
 * @brief Current execution state of debugged script
 */
enum class ExecutionState {
    Stopped,        // Not running
    Running,        // Executing normally
    Paused,         // Hit breakpoint or manually paused
    Stepping,       // Single stepping
    SteppingOver,   // Step over (skip functions)
    SteppingOut,    // Step out of current function
};

/**
 * @struct DebugCallbacks
 * @brief Callbacks for debugger events
 */
struct DebugCallbacks {
    std::function<void(const Breakpoint&)> onBreakpointHit;
    std::function<void(const std::string&)> onExceptionThrown;
    std::function<void()> onExecutionStarted;
    std::function<void()> onExecutionStopped;
    std::function<void(ExecutionState)> onStateChanged;
    std::function<void(const std::vector<StackFrame>&)> onStackUpdated;
    std::function<void(const std::string&)> onConsoleOutput;
};

class ScriptDebugger {
public:
    static ScriptDebugger& GetInstance() {
        static ScriptDebugger instance;
        return instance;
    }

    // Initialization
    void Init();
    void Shutdown();

    // Breakpoint management
    /**
     * @brief Add a line breakpoint
     * @param filepath Source file path
     * @param line Line number (1-based)
     * @return Breakpoint ID
     */
    uint32_t AddBreakpoint(const std::string& filepath, uint32_t line);

    /**
     * @brief Add a conditional breakpoint
     * @param filepath Source file path
     * @param line Line number (1-based)
     * @param condition Expression that must be true to break
     * @return Breakpoint ID
     */
    uint32_t AddConditionalBreakpoint(const std::string& filepath, uint32_t line, 
                                      const std::string& condition);

    /**
     * @brief Add a logpoint (prints instead of breaking)
     * @param filepath Source file path
     * @param line Line number (1-based)
     * @param message Message to log
     * @return Breakpoint ID
     */
    uint32_t AddLogpoint(const std::string& filepath, uint32_t line, 
                         const std::string& message);

    /**
     * @brief Remove a breakpoint
     * @param id Breakpoint ID
     */
    void RemoveBreakpoint(uint32_t id);

    /**
     * @brief Enable/disable a breakpoint without removing it
     * @param id Breakpoint ID
     * @param enabled Whether to enable
     */
    void SetBreakpointEnabled(uint32_t id, bool enabled);

    /**
     * @brief Get all breakpoints for a file
     */
    std::vector<Breakpoint> GetBreakpointsForFile(const std::string& filepath) const;

    /**
     * @brief Get all breakpoints
     */
    const std::vector<Breakpoint>& GetAllBreakpoints() const { return m_Breakpoints; }

    // Execution control
    /**
     * @brief Start debugging a script
     * @param filepath Path to script file
     * @param scriptSystem The script system to use (null = auto-detect)
     */
    void StartDebugSession(const std::string& filepath, IScriptSystem* scriptSystem = nullptr);

    /**
     * @brief Stop debugging
     */
    void StopDebugSession();

    /**
     * @brief Pause execution at current location
     */
    void Pause();

    /**
     * @brief Resume execution (continue)
     */
    void Resume();

    /**
     * @brief Single-step into next statement
     */
    void StepInto();

    /**
     * @brief Single-step over current statement
     */
    void StepOver();

    /**
     * @brief Step out of current function
     */
    void StepOut();

    // State inspection
    /**
     * @brief Get current execution state
     */
    ExecutionState GetExecutionState() const { return m_ExecutionState; }

    /**
     * @brief Check if debugger is active
     */
    bool IsDebugging() const { return m_IsDebugging; }

    /**
     * @brief Get current call stack
     */
    const std::vector<StackFrame>& GetCallStack() const { return m_CallStack; }

    /**
     * @brief Get variables in current scope
     */
    const std::vector<DebugVariable>& GetLocalVariables() const { return m_LocalVariables; }

    /**
     * @brief Get global variables
     */
    const std::vector<DebugVariable>& GetGlobalVariables() const { return m_GlobalVariables; }

    /**
     * @brief Watch a variable expression
     * @return Watch ID
     */
    uint32_t AddWatch(const std::string& expression);

    /**
     * @brief Remove a watch
     */
    void RemoveWatch(uint32_t watchId);

    /**
     * @brief Get all watched variables
     */
    const std::vector<DebugVariable>& GetWatchedVariables() const { return m_WatchedVariables; }

    /**
     * @brief Evaluate an expression in current context
     */
    std::string EvaluateExpression(const std::string& expression);

    // Callbacks
    void SetCallbacks(const DebugCallbacks& callbacks) { m_Callbacks = callbacks; }

    // Console output
    const std::vector<std::pair<std::string, float>>& GetConsoleHistory() const 
    { 
        return m_ConsoleHistory; 
    }

    void ClearConsoleHistory() { m_ConsoleHistory.clear(); }

    // Update (call each frame)
    void Update(float deltaTime);

    // Get script file being debugged
    const std::string& GetDebuggedFile() const { return m_DebuggedFile; }
    uint32_t GetCurrentLine() const { return m_CurrentLine; }

    // Attach to script system for callback hooks
    void AttachToScriptSystem(IScriptSystem* scriptSystem);
    void DetachFromScriptSystem();

private:
    ScriptDebugger() = default;
    ~ScriptDebugger() = default;
    ScriptDebugger(const ScriptDebugger&) = delete;
    ScriptDebugger& operator=(const ScriptDebugger&) = delete;

    void InternalBreak(const Breakpoint& bp);
    void UpdateCallStack();
    void UpdateVariables();
    uint32_t GenerateBreakpointId();

    // State
    bool m_IsDebugging = false;
    ExecutionState m_ExecutionState = ExecutionState::Stopped;
    IScriptSystem* m_AttachedScriptSystem = nullptr;

    // Breakpoints
    std::vector<Breakpoint> m_Breakpoints;
    uint32_t m_NextBreakpointId = 1;
    std::map<std::string, std::vector<uint32_t>> m_BreakpointsByFile;

    // Current debug session
    std::string m_DebuggedFile;
    uint32_t m_CurrentLine = 0;
    std::vector<StackFrame> m_CallStack;
    std::vector<DebugVariable> m_LocalVariables;
    std::vector<DebugVariable> m_GlobalVariables;
    std::vector<DebugVariable> m_WatchedVariables;

    // Watches
    std::map<uint32_t, std::string> m_Watches;
    uint32_t m_NextWatchId = 1;

    // Console history
    std::vector<std::pair<std::string, float>> m_ConsoleHistory;

    // Callbacks
    DebugCallbacks m_Callbacks;

    // Stepping state
    uint32_t m_StepTargetFrame = 0;
};
