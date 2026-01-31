# Script Debugger Integration Examples

## Integration with AngelScript

### Hooking Breakpoints in AngelScriptSystem

```cpp
// In AngelScriptSystem.h, add friend class:
friend class ScriptDebugger;

// Add method for debugging:
void OnScriptDebugEvent(const std::string& file, uint32_t line, 
                        const std::string& functionName);

// In AngelScriptSystem.cpp, implement hook:

void AngelScriptSystem::OnScriptDebugEvent(const std::string& file, uint32_t line,
                                           const std::string& functionName) {
    auto& debugger = ScriptDebugger::GetInstance();
    auto bps = debugger.GetBreakpointsForFile(file);
    
    for (const auto& bp : bps) {
        if (bp.line == line && bp.enabled) {
            // Check if conditional breakpoint
            if (bp.isConditional) {
                std::string condResult = debugger.EvaluateExpression(bp.condition);
                if (condResult != "true") {
                    continue; // Skip this breakpoint
                }
            }
            
            // Handle logpoint
            if (bp.isLogpoint) {
                debugger.GetConsoleHistory().emplace_back(bp.logMessage, 0.0f);
                continue; // Don't break, just log
            }
            
            // Track hit count
            const_cast<Breakpoint&>(bp).hitCount++;
            if (bp.targetHitCount > 0 && bp.hitCount < bp.targetHitCount) {
                continue; // Haven't reached target hit count yet
            }
            
            // Break execution
            debugger.InternalBreak(bp);
            
            // Update call stack with AngelScript context
            m_CurrentContext->GetCallStack(debugger);
            
            // Pause execution
            break;
        }
    }
}

// Call from main script execution loop:
void AngelScriptSystem::ExecuteScript(const std::string& scriptCode) {
    // ... setup ...
    
    // Enable line-by-line callback if debugging
    if (debugger.IsDebugging()) {
        // Hook up debugging callback
        ctx->SetLineCallback(asCALL_CDECL, this, 
            asMETHODPR(AngelScriptSystem, OnScriptDebugEvent, 
                      (const std::string&, uint32_t, const std::string&), void));
    }
    
    r = ctx->Execute();
    
    // ... cleanup ...
}
```

### Extracting Call Stack from AngelScript

```cpp
void AngelScriptSystem::PopulateCallStack(asIScriptContext* ctx, ScriptDebugger& dbg) {
    std::vector<StackFrame> stack;
    
    int stackSize = ctx->GetCallstackSize();
    for (int i = 0; i < stackSize; ++i) {
        const asIScriptFunction* func = ctx->GetFunction(i);
        int lineNum = ctx->GetLineNumber(i, 0, nullptr);
        
        StackFrame frame;
        frame.frameIndex = i;
        frame.functionName = func->GetName();
        frame.filepath = func->GetScriptSectionName();
        frame.line = lineNum;
        frame.column = 0;
        
        // Get local variables for this frame
        int varCount = ctx->GetVarCount(i);
        for (int v = 0; v < varCount; ++v) {
            const char* varName = nullptr;
            int typeId = ctx->GetVarTypeId(i, v);
            ctx->GetVar(i, v, (void*)&varName);
            
            DebugVariable var;
            var.name = varName ? varName : "unknown";
            var.type = "var";
            var.value = "...";
            var.scope = "local";
            
            frame.localVariables[var.name] = var.value;
        }
        
        stack.push_back(frame);
    }
    
    // Update debugger with new stack
    dbg.GetCallStack() = stack;
}
```

## Integration with Lua/LuaJIT

### Lua Debug Hook

```cpp
// In LuaJitScriptSystem.cpp

void LuaJitScriptSystem::SetupDebugHook(lua_State* L) {
    auto& debugger = ScriptDebugger::GetInstance();
    
    if (!debugger.IsDebugging()) {
        return;
    }
    
    // Set line hook
    lua_sethook(L, [](lua_State* L, lua_Debug* ar) {
        auto& dbg = ScriptDebugger::GetInstance();
        auto& sys = LuaJitScriptSystem::GetInstance();
        
        if (ar->event == LUA_HOOKLINE) {
            // Get current file and line
            lua_getinfo(L, "S", ar);
            std::string file = ar->source;
            int line = ar->currentline;
            
            // Check breakpoints
            auto bps = dbg.GetBreakpointsForFile(file);
            for (const auto& bp : bps) {
                if (bp.line == line && bp.enabled) {
                    dbg.InternalBreak(bp);
                    
                    // Update Lua-specific state
                    sys.UpdateCallStackFromLua(L, dbg);
                    sys.UpdateVariablesFromLua(L, dbg);
                }
            }
        }
    }, LUA_MASKLINE, 0);
}

// Extract variables from Lua
void LuaJitScriptSystem::UpdateVariablesFromLua(lua_State* L, ScriptDebugger& dbg) {
    std::vector<DebugVariable> locals;
    
    lua_Debug ar;
    int level = 0;
    while (lua_getstack(L, level, &ar)) {
        if (level == 0) { // Current frame only
            int i = 1;
            const char* name = lua_getlocal(L, &ar, i);
            while (name != nullptr) {
                DebugVariable var;
                var.name = name;
                
                // Get value
                if (lua_isnumber(L, -1)) {
                    var.type = "number";
                    var.value = std::to_string(lua_tonumber(L, -1));
                } else if (lua_isstring(L, -1)) {
                    var.type = "string";
                    var.value = lua_tostring(L, -1);
                } else if (lua_istable(L, -1)) {
                    var.type = "table";
                    var.value = "<table>";
                    var.expandable = true;
                } else if (lua_isfunction(L, -1)) {
                    var.type = "function";
                    var.value = "<function>";
                } else {
                    var.type = "unknown";
                    var.value = "?";
                }
                
                var.scope = "local";
                locals.push_back(var);
                
                lua_pop(L, 1);
                i++;
                name = lua_getlocal(L, &ar, i);
            }
        }
        level++;
    }
    
    dbg.GetLocalVariables() = locals;
}

// Extract call stack
void LuaJitScriptSystem::UpdateCallStackFromLua(lua_State* L, ScriptDebugger& dbg) {
    std::vector<StackFrame> stack;
    
    lua_Debug ar;
    int level = 0;
    while (lua_getstack(L, level, &ar)) {
        lua_getinfo(L, "Sl", &ar);
        
        StackFrame frame;
        frame.frameIndex = level;
        frame.functionName = ar.name ? ar.name : "[unknown]";
        frame.filepath = ar.source;
        frame.line = ar.currentline;
        frame.column = 0;
        
        stack.push_back(frame);
        level++;
    }
    
    dbg.GetCallStack() = stack;
}
```

## Integration with Python

### Python sys.settrace Hook

```cpp
// In PythonScriptSystem.cpp

static ScriptDebugger* g_DebuggerInstance = nullptr;

static int PythonDebugCallback(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
    if (!g_DebuggerInstance || !g_DebuggerInstance->IsDebugging()) {
        return 0;
    }
    
    if (what == PyTrace_LINE) {
        const char* filename = PyUnicode_AsUTF8(frame->f_code->co_filename);
        int lineno = PyFrame_GetLineNumber(frame);
        
        auto bps = g_DebuggerInstance->GetBreakpointsForFile(filename);
        for (const auto& bp : bps) {
            if (bp.line == lineno && bp.enabled) {
                g_DebuggerInstance->InternalBreak(bp);
            }
        }
    }
    
    return 0;
}

void PythonScriptSystem::SetupDebugHook() {
    auto& debugger = ScriptDebugger::GetInstance();
    g_DebuggerInstance = &debugger;
    
    PyEval_SetTrace(PythonDebugCallback, nullptr);
}

void PythonScriptSystem::UpdateVariablesFromPython(ScriptDebugger& dbg) {
    PyObject* frame = PyEval_GetFrame();
    if (!frame) return;
    
    std::vector<DebugVariable> locals;
    
    // Get locals
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(frame->f_locals, &pos, &key, &value)) {
        DebugVariable var;
        var.name = PyUnicode_AsUTF8(key);
        var.scope = "local";
        
        // Get type and value
        PyObject* typeObj = PyObject_Type(value);
        var.type = PyUnicode_AsUTF8(typeObj->tp_name);
        
        PyObject* reprObj = PyObject_Repr(value);
        var.value = PyUnicode_AsUTF8(reprObj);
        
        locals.push_back(var);
    }
    
    dbg.GetLocalVariables() = locals;
}
```

## Integration with Custom Languages

### Generic Script System Integration Template

```cpp
template<typename ScriptSystemType>
class DebuggerIntegration {
public:
    static void Setup(ScriptSystemType* scriptSystem, ScriptDebugger& debugger) {
        scriptSystem->RegisterDebugCallback([&debugger](const std::string& file, 
                                                        uint32_t line,
                                                        const std::string& funcName) {
            OnExecutionLine(file, line, funcName, debugger, scriptSystem);
        });
    }
    
private:
    static void OnExecutionLine(const std::string& file, uint32_t line,
                               const std::string& funcName,
                               ScriptDebugger& debugger,
                               ScriptSystemType* scriptSystem) {
        // Check breakpoints
        auto bps = debugger.GetBreakpointsForFile(file);
        for (const auto& bp : bps) {
            if (bp.line == line && bp.enabled) {
                // Handle conditional
                if (bp.isConditional) {
                    // Evaluate condition
                    // (script system specific)
                }
                
                // Handle logpoint
                if (bp.isLogpoint) {
                    // Log message
                    // (script system specific)
                    continue;
                }
                
                // Break
                debugger.InternalBreak(bp);
                
                // Update state
                UpdateState(scriptSystem, debugger);
                break;
            }
        }
    }
    
    static void UpdateState(ScriptSystemType* scriptSystem, ScriptDebugger& debugger) {
        // Update call stack
        std::vector<StackFrame> stack = scriptSystem->GetCallStack();
        debugger.GetCallStack() = stack;
        
        // Update variables
        std::vector<DebugVariable> vars = scriptSystem->GetLocalVariables();
        debugger.GetLocalVariables() = vars;
    }
};

// Usage:
// DebuggerIntegration<AngelScriptSystem>::Setup(&angelScript, debugger);
```

## Example Debug Session Script

### sample_debug_script.as
```angelscript
// Simple test script for debugger
class Player {
    float health;
    Vec3 position;
    
    Player(float h, Vec3 pos) {
        health = h;
        position = pos;
    }
    
    void TakeDamage(float damage) {        // Breakpoint here (line 12)
        health -= damage;                   // Step into this
        if (health <= 0) {
            OnDeath();
        }
    }
    
    void OnDeath() {
        print("Player died!");
    }
}

void UpdateGame(float dt) {                 // Breakpoint here (line 24)
    Player player(100.0f, Vec3(0, 0, 0));
    player.TakeDamage(25.0f);              // Step over this
    
    if (player.health > 0) {
        print("Player alive with " + player.health + " health");
    }
}
```

### Debug Steps
1. Set breakpoint at line 24 (UpdateGame)
2. Set conditional breakpoint at line 12 with condition "damage > 20"
3. Set logpoint at line 18 with message "Player death triggered"
4. Run script with F5
5. When paused at line 24:
   - Step over TakeDamage call with F10
   - Check player.health in Watch
   - Step into OnDeath with F11
6. Continue with F5

## Testing Integration

### Unit Test Example

```cpp
#include <gtest/gtest.h>
#include "ScriptDebugger.h"
#include "AngelScriptSystem.h"

class DebuggerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        debugger.Init();
        scriptSystem.Init();
        debugger.AttachToScriptSystem(&scriptSystem);
    }
    
    void TearDown() override {
        debugger.Shutdown();
        scriptSystem.Shutdown();
    }
    
    ScriptDebugger& debugger = ScriptDebugger::GetInstance();
    AngelScriptSystem scriptSystem;
};

TEST_F(DebuggerIntegrationTest, BreakpointPausesExecution) {
    debugger.StartDebugSession("test_script.as");
    debugger.AddBreakpoint("test_script.as", 10);
    
    // This would execute the script
    // debugger.ResumeExecution();
    
    // Wait for breakpoint
    // std::this_thread::sleep_until(timeout);
    
    // EXPECT_EQ(debugger.GetExecutionState(), ExecutionState::Paused);
    // EXPECT_EQ(debugger.GetCurrentLine(), 10);
}

TEST_F(DebuggerIntegrationTest, StepIntoEntersFunction) {
    debugger.StartDebugSession("test_script.as");
    debugger.AddBreakpoint("test_script.as", 20);
    
    // Resume to breakpoint
    // debugger.ResumeExecution();
    // wait...
    
    // Step into function call
    // debugger.StepInto();
    // wait...
    
    // EXPECT_EQ(debugger.GetCallStack().size(), 2);
}

TEST_F(DebuggerIntegrationTest, WatchExpressionUpdates) {
    uint32_t watchId = debugger.AddWatch("player.health");
    auto watches = debugger.GetWatchedVariables();
    
    EXPECT_EQ(watches.size(), 1);
    EXPECT_EQ(watches[0].name, "player.health");
}
```

## Async Debugging Pattern

For games with background script threads:

```cpp
class AsyncScriptDebugger {
private:
    std::mutex m_DebugMutex;
    std::queue<DebugEvent> m_DebugEvents;
    
public:
    void QueueBreakpoint(const Breakpoint& bp) {
        std::lock_guard<std::mutex> lock(m_DebugMutex);
        m_DebugEvents.push(DebugEvent{EventType::Breakpoint, bp});
    }
    
    void ProcessDebugQueue() {
        std::lock_guard<std::mutex> lock(m_DebugMutex);
        while (!m_DebugEvents.empty()) {
            auto event = m_DebugEvents.front();
            m_DebugEvents.pop();
            
            if (event.type == EventType::Breakpoint) {
                auto& debugger = ScriptDebugger::GetInstance();
                debugger.InternalBreak(event.breakpoint);
            }
        }
    }
};
```

See the full API documentation for more integration examples.
