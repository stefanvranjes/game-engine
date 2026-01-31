# Script Debugger UI - Documentation Index

## 📚 Complete Documentation Set

### Start Here 👇

#### **SCRIPT_DEBUGGER_QUICK_REF.md** 
🎯 **READ THIS FIRST** if you just want to use the debugger
- Quick start (5 minutes)
- Keyboard shortcuts
- Common workflows
- Troubleshooting
- **Best for**: End users wanting quick answers

---

## Documentation by Use Case

### For End Users (Debugging Scripts)
1. **SCRIPT_DEBUGGER_QUICK_REF.md** - Keyboard shortcuts and quick start
2. **SCRIPT_DEBUGGER_GUIDE.md** - Complete user guide with examples

### For Developers (Integrating with Script Systems)
1. **SCRIPT_DEBUGGER_EXAMPLES.md** - Code examples for:
   - AngelScript integration
   - Lua/LuaJIT integration
   - Python integration
   - Custom language template
2. **SCRIPT_DEBUGGER_IMPLEMENTATION.md** - Technical deep dive

### For Architects (System Design)
1. **SCRIPT_DEBUGGER_ARCHITECTURE.md** - Visual diagrams
2. **SCRIPT_DEBUGGER_DELIVERY.md** - Executive summary
3. **SCRIPT_DEBUGGER_IMPLEMENTATION.md** - Technical details

### For Build/DevOps (Build Integration)
1. **SCRIPT_DEBUGGER_CMAKE.md** - CMake integration guide
2. **SCRIPT_DEBUGGER_MANIFEST.md** - File listing and status

---

## Documentation Files

### 📖 User-Facing Documentation

#### SCRIPT_DEBUGGER_QUICK_REF.md (350 lines)
**Quick lookup guide for common tasks**

Contents:
- Opening the debugger
- Main controls and shortcuts
- Breakpoint operations
- Variable inspection
- Watch expressions
- Common workflows
- Tips & tricks
- Troubleshooting

**When to Read**: You need to debug something NOW
**Time to Read**: 5-10 minutes
**Reading Level**: Beginner

---

#### SCRIPT_DEBUGGER_GUIDE.md (380 lines)
**Comprehensive user guide with detailed explanations**

Contents:
- Feature overview
- Quick start (step-by-step)
- UI layout explanation
- Advanced features
  - Conditional breakpoints
  - Logpoints
  - Watch expressions
  - Expression evaluation
- Keyboard shortcuts
- API reference (for programmatic use)
- Example workflows
- Performance considerations
- Troubleshooting

**When to Read**: You want to understand all features deeply
**Time to Read**: 30-45 minutes
**Reading Level**: Intermediate

**Key Sections**:
- [Quick Start](SCRIPT_DEBUGGER_GUIDE.md#quick-start)
- [UI Layout](SCRIPT_DEBUGGER_GUIDE.md#ui-layout)
- [Advanced Features](SCRIPT_DEBUGGER_GUIDE.md#advanced-features)
- [Troubleshooting](SCRIPT_DEBUGGER_GUIDE.md#troubleshooting)

---

### 👨‍💻 Developer-Facing Documentation

#### SCRIPT_DEBUGGER_EXAMPLES.md (400 lines)
**Practical code examples for integration**

Contents:
- AngelScript integration with:
  - Hooking breakpoints
  - Extracting call stack
  - Populating variables
- Lua/LuaJIT integration with:
  - Debug hook setup
  - Variable extraction
  - Call stack extraction
- Python integration with:
  - sys.settrace hook
  - Frame inspection
- Generic template for custom languages
- Example test script
- Unit test patterns
- Async debugging pattern

**When to Read**: You're integrating debugger with a script system
**Time to Read**: 45 minutes - 1 hour
**Reading Level**: Advanced

**Key Sections**:
- [AngelScript Integration](SCRIPT_DEBUGGER_EXAMPLES.md#integration-with-angelscript)
- [Lua Integration](SCRIPT_DEBUGGER_EXAMPLES.md#integration-with-lualuajit)
- [Python Integration](SCRIPT_DEBUGGER_EXAMPLES.md#integration-with-python)
- [Generic Template](SCRIPT_DEBUGGER_EXAMPLES.md#integration-with-custom-languages)

---

#### SCRIPT_DEBUGGER_IMPLEMENTATION.md (350 lines)
**Technical implementation details**

Contents:
- File descriptions
- Data structure definitions
- UI component descriptions
- Multi-language support design
- Memory usage analysis
- Thread safety notes
- Extension points
- Known limitations
- Future enhancements
- Performance benchmarks
- Testing strategies
- Unit test examples
- Integration test examples

**When to Read**: You need to understand how it works internally
**Time to Read**: 1-2 hours
**Reading Level**: Advanced

**Key Sections**:
- [Data Structures](SCRIPT_DEBUGGER_IMPLEMENTATION.md#data-structures)
- [UI Components](SCRIPT_DEBUGGER_IMPLEMENTATION.md#ui-components)
- [Multi-Language Support](SCRIPT_DEBUGGER_IMPLEMENTATION.md#multi-language-support)
- [Extension Points](SCRIPT_DEBUGGER_IMPLEMENTATION.md#extension-points)
- [Performance Benchmarks](SCRIPT_DEBUGGER_IMPLEMENTATION.md#performance-benchmarks)

---

### 🏗️ Architecture & System Documentation

#### SCRIPT_DEBUGGER_ARCHITECTURE.md (450 lines)
**Visual diagrams and system architecture**

Contents:
- System architecture diagram
- State machine diagram
- UI layout diagram
- Data flow diagram
- Breakpoint types visualization
- Debugging workflow sequence
- Variable inspection example
- Performance impact graph
- File organization

**When to Read**: You want to understand the overall design visually
**Time to Read**: 20-30 minutes
**Reading Level**: Intermediate

**Key Diagrams**:
- System Architecture
- State Machine
- Data Flow
- UI Layout
- Performance Graph

---

#### SCRIPT_DEBUGGER_DELIVERY.md (250 lines)
**Executive summary of the complete delivery**

Contents:
- Overview
- Deliverables listing
- Features implemented (with checkmarks)
- Architecture summary
- Integration points
- Usage quick start
- Performance impact metrics
- Testing verification
- Documentation quality
- Multi-language support status
- Code statistics
- Verification results

**When to Read**: You need an overview of what was delivered
**Time to Read**: 10-15 minutes
**Reading Level**: Beginner

---

### 📋 Reference & Manifest Documentation

#### SCRIPT_DEBUGGER_MANIFEST.md (350 lines)
**Complete file listing and status**

Contents:
- File-by-file description
- Purpose and contents of each file
- File summary table
- Feature checklist
- Compilation status
- Usage instructions
- Documentation access guide
- Performance metrics
- Support matrix for different languages
- Next steps

**When to Read**: You need to know what files exist and what they do
**Time to Read**: 15-20 minutes
**Reading Level**: Beginner

**Key Tables**:
- File Summary
- Feature Checklist
- Support Matrix

---

#### SCRIPT_DEBUGGER_CMAKE.md (280 lines)
**Build system integration guide**

Contents:
- Automatic inclusion explanation
- Manual integration snippets
- Conditional compilation options
- Feature flag configuration
- Compiler requirements
- Runtime configuration
- Dependency specification
- Performance optimization
- Installation configuration
- Cross-platform support
- Troubleshooting
- Example CMakeLists

**When to Read**: You're setting up the build system
**Time to Read**: 20-30 minutes
**Reading Level**: Intermediate

---

## Reading Paths

### Path 1: "I Want to Debug Scripts" (Beginner)
```
1. SCRIPT_DEBUGGER_QUICK_REF.md (5 min)
2. SCRIPT_DEBUGGER_GUIDE.md → [Quick Start] (10 min)
3. Try it out!
```
**Total Time**: 15 minutes

---

### Path 2: "I Need to Integrate This" (Developer)
```
1. SCRIPT_DEBUGGER_QUICK_REF.md (5 min)
2. SCRIPT_DEBUGGER_DELIVERY.md (10 min)
3. SCRIPT_DEBUGGER_EXAMPLES.md (45 min)
4. Header files for API details (30 min)
5. Implement integration
```
**Total Time**: 90 minutes + coding

---

### Path 3: "I Need Full Understanding" (Architect)
```
1. SCRIPT_DEBUGGER_DELIVERY.md (10 min)
2. SCRIPT_DEBUGGER_ARCHITECTURE.md (20 min)
3. SCRIPT_DEBUGGER_IMPLEMENTATION.md (60 min)
4. SCRIPT_DEBUGGER_GUIDE.md (40 min)
5. SCRIPT_DEBUGGER_EXAMPLES.md (40 min)
```
**Total Time**: 170 minutes

---

### Path 4: "I'm Building This" (DevOps/Build)
```
1. SCRIPT_DEBUGGER_CMAKE.md (20 min)
2. SCRIPT_DEBUGGER_MANIFEST.md (15 min)
3. Verify compilation
```
**Total Time**: 35 minutes

---

## API Quick Reference

### Core Singleton Access
```cpp
auto& debugger = ScriptDebugger::GetInstance();
```

### Common Operations
```cpp
// Add breakpoint
uint32_t id = debugger.AddBreakpoint("script.as", 42);

// Add conditional breakpoint
uint32_t id = debugger.AddConditionalBreakpoint("script.as", 42, "health < 20");

// Add watch
uint32_t watchId = debugger.AddWatch("player.health");

// Execution control
debugger.Resume();
debugger.Pause();
debugger.StepInto();
debugger.StepOver();
debugger.StepOut();

// Query state
ExecutionState state = debugger.GetExecutionState();
const auto& callStack = debugger.GetCallStack();
const auto& locals = debugger.GetLocalVariables();
```

### UI Operations
```cpp
auto& ui = m_ScriptDebuggerUI;
ui->Show();
ui->Hide();
ui->Toggle();
ui->Render();
```

---

## Source Code Navigation

### Header Files
- **include/ScriptDebugger.h** - Core API (320 lines)
  - `class ScriptDebugger` - Main debugger
  - `struct Breakpoint` - Breakpoint definition
  - `struct StackFrame` - Call stack entry
  - `struct DebugVariable` - Variable definition
  - `enum ExecutionState` - State enum
  - `struct DebugCallbacks` - Event callbacks

- **include/ScriptDebuggerUI.h** - UI API (180 lines)
  - `class ScriptDebuggerUI` - ImGui frontend
  - Window management methods
  - Panel render methods
  - State management

### Implementation Files
- **src/ScriptDebugger.cpp** - Core logic (380 lines)
  - Breakpoint management
  - Execution state machine
  - Variable tracking

- **src/ScriptDebuggerUI.cpp** - ImGui rendering (900 lines)
  - All panel rendering
  - Interactive elements
  - ImGui integration

### Integration Points
- **include/Application.h** - Application integration (2 additions)
- **src/Application.cpp** - Initialization & rendering (4 additions)

---

## Key Concepts

### Execution States
- **Stopped** - Not debugging
- **Running** - Script executing normally
- **Paused** - Hit breakpoint
- **Stepping** - Single stepping active
- **SteppingOver** - Step over function
- **SteppingOut** - Step out of function

### Breakpoint Types
- **Line** - Break always on this line
- **Conditional** - Break if expression is true
- **Logpoint** - Print message without breaking

### Debug Panels
- **Call Stack** - Function call hierarchy
- **Variables** - Local and global variables
- **Watch** - User-specified expressions
- **Console** - Output and input
- **Breakpoints** - Breakpoint management
- **Source Code** - Current file being debugged

---

## Troubleshooting Guide

### Common Issues

**Q: How do I open the debugger?**
A: Tools Menu → Script Debugger, or Ctrl+Shift+D

**Q: Breakpoint won't break?**
A: Make sure it's enabled (checkbox), file path matches, script reaches that line

**Q: Variable shows "not found"?**
A: Variable might be out of scope. Check Call Stack to select right frame.

**Q: How do I watch a variable?**
A: Type expression in Watch panel, click Add

**Q: Performance is slow?**
A: Remove unnecessary breakpoints/watches, disable debugging when not needed

---

## Document Sizes & Reading Times

| Document | Lines | Est. Read Time | Audience |
|----------|-------|---|---|
| QUICK_REF | 350 | 5-10 min | End users |
| GUIDE | 380 | 30-45 min | Users |
| EXAMPLES | 400 | 45-60 min | Developers |
| IMPLEMENTATION | 350 | 60-90 min | Developers |
| ARCHITECTURE | 450 | 20-30 min | Architects |
| DELIVERY | 250 | 10-15 min | Managers |
| MANIFEST | 350 | 15-20 min | Leads |
| CMAKE | 280 | 20-30 min | Build engineers |

---

## Getting Help

1. **For quick answers** → SCRIPT_DEBUGGER_QUICK_REF.md
2. **For features** → SCRIPT_DEBUGGER_GUIDE.md
3. **For integration** → SCRIPT_DEBUGGER_EXAMPLES.md
4. **For internals** → SCRIPT_DEBUGGER_IMPLEMENTATION.md
5. **For overview** → SCRIPT_DEBUGGER_DELIVERY.md

---

## Next Steps

1. ✅ Read the Quick Reference (5 minutes)
2. ✅ Try using the debugger (10 minutes)
3. ✅ Read relevant section (15-60 minutes)
4. ✅ Integrate if needed (1-4 hours)

---

**Total Documentation**: 8 comprehensive guides
**Total Lines**: 4,600+ 
**Status**: ✅ COMPLETE

All documentation is up-to-date and ready for use.
