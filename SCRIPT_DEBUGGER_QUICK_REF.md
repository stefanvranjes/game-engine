# Script Debugger UI - Quick Reference Card

## Opening the Debugger

```
Tools Menu → Script Debugger
OR
Ctrl+Shift+D (Keyboard Shortcut)
```

## Main Controls

### Execution Buttons
| Button | Shortcut | Action |
|--------|----------|--------|
| Continue | F5 | Resume execution |
| Pause | Ctrl+Alt+Brk | Pause at current line |
| Stop | Shift+F5 | Stop debugging |
| Step Into | F11 | Enter next function |
| Step Over | F10 | Skip function bodies |
| Step Out | Shift+F11 | Exit current function |

## Breakpoints

### Set Breakpoint
- **Click** on line number in source code viewer's gutter (left side)
- Red circle appears on that line
- Script pauses when line executes

### Remove Breakpoint
- **Click** on red breakpoint circle to remove
- Or right-click → Delete in Breakpoints window

### Conditional Breakpoint
```cpp
// In C++ code:
uint32_t id = debugger.AddConditionalBreakpoint(
    "script.as",
    42,
    "health < 20"  // Only breaks when true
);
```

### Logpoint
```cpp
// Prints without breaking:
uint32_t id = debugger.AddLogpoint(
    "script.as",
    50,
    "Player at {x}, {y}"
);
```

### Enable/Disable
- Checkbox in Breakpoints window
- Disabled breakpoints don't trigger but remain set

## Inspecting State

### Local Variables
- **Call Stack** window → Select frame → Local tab in Variables window
- Shows all variables in current function scope

### Global Variables
- **Variables** window → **Global** tab
- Shows module/script-level variables

### Watched Variables
- **Watch** window
- Enter expression → Click **Add**
- Updates automatically each frame

### Call Stack
- **Call Stack** window shows function call hierarchy
- Frame 0 is current function
- Frame N is caller of frame N-1

## Common Workflows

### Debugging a Bug in Script

1. Open debugger: **Ctrl+Shift+D**
2. Set breakpoint at suspected location (click line number)
3. Run game with **F5**
4. When paused:
   - View **Local Variables** to check values
   - Step through with **F10** (step over) or **F11** (step into)
   - Add watches for key variables
   - Check **Call Stack** to see who called this function

### Tracking Variable Changes

1. Open debugger
2. Click **Watch** panel
3. Type variable name/expression
4. Click **Add**
5. Watch updates every frame showing current value
6. Stops updating if variable goes out of scope

### Finding Where Exception Occurs

1. Open debugger
2. Set breakpoint near suspected error location
3. Step through code with **F11** or **F10**
4. Watch local variables for invalid values
5. Exception/error appears in **Console** panel

## Keyboard Shortcuts

```
F5              Continue / Start
Ctrl+Alt+Brk    Pause
Shift+F5        Stop
F11             Step Into
F10             Step Over
Shift+F11       Step Out
Ctrl+Shift+D    Toggle Debugger
```

## Status Colors

| Color | Meaning |
|-------|---------|
| 🟢 Green | Script running normally |
| 🟡 Orange | Paused at breakpoint |
| 🔴 Red | Error/exception occurred |
| ⚪ Gray | Not running/stopped |
| 🟡 Yellow | Current line being executed |
| 🔴 Red circle | Breakpoint marker |

## Window Layout

```
┌─ Main Debugger Window ─────────────────┐
│ [Continue] [Pause] [Stop] [Step>] [F11]│  ← Toolbar
│ Status: PAUSED                         │
├────────────────────────────────────────┤
│ Overview | Call Stack | Variables      │  ← Tabs
├────────────────────────────────────────┤
│ Debugged File: scripts/player.as       │
│ Current Line: 42                       │
│ Call Depth: 3                          │
└────────────────────────────────────────┘

┌─ Call Stack ───┐  ┌─ Variables ────┐  ┌─ Watch ────────┐
│ Frame Function │  │ Local | Global │  │ Expr = Value   │
├────────────────┤  ├────────────────┤  ├────────────────┤
│ 0 UpdateGame   │  │ x = 10         │  │ health = 100   │
│ 1 OnUpdate     │  │ y = 20         │  │ pos.x = 45.5   │
│ 2 GameLoop     │  │ health = 100   │  │ enemies = 3    │
└────────────────┘  └────────────────┘  └────────────────┘

┌─ Breakpoints ──────┐  ┌─ Source Code ─────────────┐
│ Line Type Enabled  │  │  42  ← Breakpoint        │
├────────────────────┤  │> 43    void Update() {   │
│ 42   Break   ✓     │  │  44      x += dt;        │
│ 89   Cond    ✓     │  │  45      if (x > 10) {   │
│ 156  Log     ✗     │  │  46        OnBound();     │
└────────────────────┘  │  47      }               │
                        └────────────────────────┘

┌─ Console ───────────────────────────┐
│ [DEBUG] Started debugging script    │
│ [BREAKPOINT] Hit at script.as:42    │
│ Player position: (10, 20, 30)       │
│ ► [Input command here]              │
└─────────────────────────────────────┘
```

## Tips & Tricks

### Finding Performance Bottlenecks
1. Add watches for function entry/exit timing
2. Check which functions take longest
3. Step over expensive operations to skip them
4. Profile with breakpoint hit counts

### Debugging State Corruption
1. Set conditional breakpoints before state changes
2. Watch the variable immediately before/after
3. Step through assignment statements
4. Check call stack to see who modified it

### Understanding Complex Data
1. Expand variables in Watch window (if object type)
2. Use Step Into to see helper function behavior
3. Add watches for intermediate calculations
4. Print intermediate values with logpoints

### Isolating Script-Engine Communication
1. Set breakpoints at script-to-C++ boundaries
2. Step through and verify parameter passing
3. Check return values in debugger
4. Add watches for bridge data structures

## Common Issues

### "Variable Not Found"
- Variable may be out of scope
- Try different call stack frame
- Check variable name spelling
- Variable may have been garbage collected

### Breakpoint Won't Break
- Verify breakpoint is enabled (checkbox)
- Check file path matches exactly
- Verify script actually reaches that line
- Check conditional breakpoint expression

### High Performance Impact
- Too many breakpoints or watches
- Complex watch expressions
- Large console history
→ Disable/remove unused debugging aids

### Can't Step Into Function
- Function may be C++ (not scriptable)
- Use Step Over instead (F10)
- Function may be optimized away
- Check that function has debugging symbols

## Script System Specific

### AngelScript
- Expressions: C++-like syntax
- Conditions: `health < 20 && mana > 0`
- Watches: Access members with `.` or `->`

### Lua/LuaJIT
- Expressions: Lua syntax
- Conditions: `health < 20 and mana > 0`
- Watches: Access tables with `[key]` or `.key`

### Python
- Expressions: Python syntax
- Conditions: `health < 20 and mana > 0`
- Watches: Access attributes with `.`

## Further Reading

See full documentation:
- [SCRIPT_DEBUGGER_GUIDE.md](SCRIPT_DEBUGGER_GUIDE.md) - Complete user guide
- [SCRIPT_DEBUGGER_IMPLEMENTATION.md](SCRIPT_DEBUGGER_IMPLEMENTATION.md) - Technical details
- [include/ScriptDebugger.h](include/ScriptDebugger.h) - API reference
- [include/ScriptDebuggerUI.h](include/ScriptDebuggerUI.h) - UI API reference
