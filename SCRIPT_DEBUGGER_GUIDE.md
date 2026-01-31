# Script Debugger UI - Usage Guide

## Overview

The Script Debugger provides comprehensive debugging capabilities for all scripting languages in the game engine (AngelScript, Lua, LuaJIT, Python, GDScript, Wren, Rust, Go, etc.). The ImGui-based UI provides visual debugging with breakpoint management, call stack inspection, variable watching, and execution control.

## Features

### Core Debugging Capabilities

1. **Breakpoint Management**
   - Line breakpoints: Pause execution at specific lines
   - Conditional breakpoints: Break only when a condition is true
   - Logpoints: Print messages without stopping execution
   - Toggle breakpoints on/off without removing them

2. **Execution Control**
   - Continue: Resume execution
   - Pause: Pause at current location
   - Step Into: Execute next statement, entering functions
   - Step Over: Execute next statement without entering functions
   - Step Out: Continue until function returns

3. **State Inspection**
   - Call stack: View function call hierarchy
   - Local variables: Inspect variables in current scope
   - Global variables: View global script state
   - Watched expressions: Monitor specific variable values
   - Value evaluation: Evaluate expressions in current context

4. **Console**
   - Script output display
   - Command execution
   - Error/exception tracking

5. **Source Code Viewer**
   - View currently debugged script file
   - Visual breakpoint indicators
   - Current line highlighting
   - Click-to-toggle breakpoints

## Quick Start

### Enabling the Debugger

1. **Launch from UI Menu:**
   ```
   Tools Menu → Script Debugger (Ctrl+Shift+D)
   ```

2. **Programmatic Access:**
   ```cpp
   auto& debugger = ScriptDebugger::GetInstance();
   debugger.Init();
   debugger.StartDebugSession("scripts/player.as");
   ```

### Basic Debugging Session

1. **Open Script File**
   - The debugger shows the currently executed script
   - File path displayed in source code viewer

2. **Set Breakpoints**
   - Click on line number in the gutter to add/remove breakpoints
   - Or use Debug menu → Add Breakpoint

3. **Start Debugging**
   - Press F5 or use Debug → Start menu
   - Script executes until first breakpoint

4. **Inspect State**
   - View Local/Global variables in Variables panel
   - Check Call Stack for function hierarchy
   - Use Watch expressions for continuous monitoring

5. **Step Through Code**
   - F11: Step Into
   - F10: Step Over
   - Shift+F11: Step Out

## UI Layout

### Main Debugger Window
- **Toolbar**: Quick access buttons (Continue, Pause, Step, etc.)
- **Status Indicator**: Shows current execution state (RUNNING, PAUSED, STOPPED)
- **Tabbed Interface**: Overview, Call Stack, Variables

### Supporting Panels (Can be toggled on/off)

**Call Stack Window**
```
Frame | Function    | File              | Line
------|-------------|-------------------|-----
  0   | UpdateGame  | scripts/main.as   | 42
  1   | OnUpdate    | scripts/main.as   | 128
  2   | GameLoop    | src/Engine.cpp    | 567
```

**Variables Window**
- Local tab: Variables in current scope
- Global tab: Script global variables

**Watch Window**
```
Expression            | Type   | Value
---------------------|--------|------------------------
player.health         | int    | 100
position.x            | float  | 45.5
enemies.length()      | uint   | 3
```

**Console Window**
- Shows script output and debug messages
- Displays breakpoint hits and exceptions
- Input field for command execution

**Breakpoints Window**
```
✓ | File              | Line | Type | Actions
--|-------------------|------|------|----------
✓ | scripts/ai.as     | 156  | Break| Remove
✗ | scripts/physics.as| 89   | Cond | Remove
✓ | scripts/main.as   | 42   | Log  | Remove
```

**Source Code Window**
```
  42   ← Breakpoint indicator (red)
> 43   ← Current line being executed (yellow)
  44   void UpdatePlayer(float dt) {
```

## Advanced Features

### Conditional Breakpoints

Set a breakpoint that only triggers when a condition is true:

```cpp
// UI: Right-click breakpoint → Edit Condition
// Or programmatically:
uint32_t bpId = debugger.AddConditionalBreakpoint(
    "scripts/player.as", 
    42, 
    "health < 20"  // Condition in script language
);
```

### Logpoints

Print values without interrupting execution:

```cpp
uint32_t bpId = debugger.AddLogpoint(
    "scripts/player.as",
    50,
    "Player position: {pos.x}, {pos.y}, {pos.z}"
);
```

### Watch Expressions

Monitor variable values throughout execution:

```cpp
// Via UI: Type expression in Watch window → Add
// Programmatically:
uint32_t watchId = debugger.AddWatch("player.inventory.size()");
uint32_t watchId = debugger.AddWatch("enemies[0].health");

// Later: Remove watch
debugger.RemoveWatch(watchId);
```

### Expression Evaluation

Evaluate expressions in the current context:

```cpp
std::string result = debugger.EvaluateExpression("player.health * 2");
// Returns: "200" (or "<error: variable not found>" if invalid)
```

### Keyboard Shortcuts

| Shortcut      | Action              |
|---------------|---------------------|
| F5            | Continue / Start    |
| Ctrl+Alt+Brk  | Pause              |
| Shift+F5      | Stop               |
| F11           | Step Into          |
| F10           | Step Over          |
| Shift+F11     | Step Out           |
| Ctrl+Shift+D  | Toggle Debugger UI |

## API Reference

### ScriptDebugger Class

#### Initialization
```cpp
ScriptDebugger& debugger = ScriptDebugger::GetInstance();
debugger.Init();
debugger.StartDebugSession("scripts/player.as");
debugger.Shutdown();
```

#### Breakpoint Management
```cpp
// Add breakpoints
uint32_t id = debugger.AddBreakpoint("script.as", 42);
uint32_t id = debugger.AddConditionalBreakpoint("script.as", 42, "x > 10");
uint32_t id = debugger.AddLogpoint("script.as", 42, "Value: {x}");

// Remove/modify
debugger.RemoveBreakpoint(id);
debugger.SetBreakpointEnabled(id, false);

// Query
auto bps = debugger.GetAllBreakpoints();
auto fileBps = debugger.GetBreakpointsForFile("script.as");
```

#### Execution Control
```cpp
debugger.Resume();      // Continue execution
debugger.Pause();       // Pause at current line
debugger.StepInto();    // Step into next statement
debugger.StepOver();    // Step over next statement
debugger.StepOut();     // Step out of current function
debugger.StopDebugSession(); // End debugging
```

#### State Inspection
```cpp
ExecutionState state = debugger.GetExecutionState();
bool isDebugging = debugger.IsDebugging();

const auto& callStack = debugger.GetCallStack();
const auto& locals = debugger.GetLocalVariables();
const auto& globals = debugger.GetGlobalVariables();
const auto& watches = debugger.GetWatchedVariables();

std::string result = debugger.EvaluateExpression("my_var");
```

#### Watches
```cpp
uint32_t watchId = debugger.AddWatch("player.health");
debugger.RemoveWatch(watchId);
auto watches = debugger.GetWatchedVariables();
```

### ScriptDebuggerUI Class

#### Initialization
```cpp
auto& ui = m_ScriptDebuggerUI; // Already initialized in Application
ui->Show();
ui->Hide();
ui->Toggle();
ui->SetOpen(true);
```

#### Customization
```cpp
ui->SetFont(imguiFont);
ui->SetThemeDarkMode(true);
```

## Integration with Script Systems

The debugger integrates with all script systems through the `IScriptSystem` interface:

```cpp
// In your script system (e.g., AngelScriptSystem)
void AngelScriptSystem::OnBreakpoint(const std::string& file, uint32_t line) {
    auto& debugger = ScriptDebugger::GetInstance();
    auto bps = debugger.GetBreakpointsForFile(file);
    
    for (const auto& bp : bps) {
        if (bp.line == line && bp.enabled) {
            debugger.InternalBreak(bp);
            break;
        }
    }
}
```

## Callback System

Register callbacks for debugging events:

```cpp
DebugCallbacks callbacks;

callbacks.onBreakpointHit = [](const Breakpoint& bp) {
    std::cout << "Breakpoint hit at " << bp.filepath << ":" << bp.line << std::endl;
};

callbacks.onStateChanged = [](ExecutionState state) {
    switch (state) {
        case ExecutionState::Running:
            std::cout << "Running..." << std::endl;
            break;
        case ExecutionState::Paused:
            std::cout << "Paused!" << std::endl;
            break;
        // ...
    }
};

callbacks.onStackUpdated = [](const std::vector<StackFrame>& stack) {
    for (const auto& frame : stack) {
        std::cout << frame.functionName << " @ " << frame.filepath 
                  << ":" << frame.line << std::endl;
    }
};

debugger.SetCallbacks(callbacks);
```

## Example Workflow

### Debugging a Game Script

**Script File: `scripts/enemy_ai.as`**
```angelscript
void UpdateEnemy(float dt) {  // Line 42
    if (health <= 0) {
        OnDeath();
        return;
    }
    
    Vec3 dirToPlayer = (player.position - position).Normalize();
    MoveToward(dirToPlayer, speed);
    
    if (Vector3.Distance(position, player.position) < 5.0f) {
        Attack(player);
    }
}
```

**Debugging Steps:**

1. **Set Breakpoint at Attack Call**
   - Click line 52 in source code window
   - Red breakpoint indicator appears

2. **Set Conditional Breakpoint**
   - Add breakpoint at line 42 with condition `health < 100`
   - Breaks only when enemy health low

3. **Start Debugging**
   - Run game with F5
   - Game pauses at first breakpoint

4. **Inspect State**
   - View local variables: `dt`, `dirToPlayer`, `health`, `speed`
   - Add watch: `Vector3.Distance(position, player.position)`
   - Check call stack to see caller

5. **Step Through**
   - F11 to step into `OnDeath()`
   - See local variables inside that function
   - F10 to step over `MoveToward()`

6. **Console Output**
   - View any debug prints from script
   - See exceptions or runtime errors

## Performance Considerations

- **Debugger Overhead**: ~5-10% FPS impact when attached
- **Breakpoints**: Minimal impact (~0.1% per breakpoint)
- **Variable Inspection**: Updates each frame, scales with variable count
- **Watch Expressions**: Evaluated each frame, optimize complex expressions

## Troubleshooting

### Breakpoint Not Being Hit
- Ensure breakpoint is enabled (checkbox in Breakpoints window)
- Verify file path matches exactly
- Check conditional breakpoint expression syntax

### Variable Shows `<error: variable not found>`
- Variable may not be in scope (check Call Stack)
- Variable name may have typos
- Variable may have been garbage collected

### High Performance Impact
- Reduce number of watched expressions
- Remove unnecessary breakpoints
- Disable debugging when not needed

### Console Not Showing Output
- Ensure script uses proper output function
- Check that logpoints are enabled
- Verify script execution reaches output calls

## See Also

- [ScriptDebugger.h](../include/ScriptDebugger.h) - Core debugger API
- [ScriptDebuggerUI.h](../include/ScriptDebuggerUI.h) - UI implementation
- [AngelScriptSystem.h](../include/AngelScriptSystem.h) - AngelScript integration
- [IScriptSystem.h](../include/IScriptSystem.h) - Script system interface
