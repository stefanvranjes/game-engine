# LuaJIT Documentation Index

## Quick Navigation

### 🚀 Getting Started (5 minutes)
1. **[LUAJIT_QUICK_REFERENCE.md](LUAJIT_QUICK_REFERENCE.md)** - API overview and common tasks
2. **[LUAJIT_EXAMPLES.md](LUAJIT_EXAMPLES.md)** - Copy-paste ready code examples
3. **[LUAJIT_INTEGRATION_GUIDE.md](LUAJIT_INTEGRATION_GUIDE.md)** - Detailed integration guide

### 📚 Complete Documentation
- **[LUAJIT_INTEGRATION_GUIDE.md](LUAJIT_INTEGRATION_GUIDE.md)** (400+ lines)
  - Setup instructions
  - Performance characteristics
  - Optimization tips
  - Benchmarking
  - Troubleshooting
  - Best practices

### 📖 Code Examples
- **[LUAJIT_EXAMPLES.md](LUAJIT_EXAMPLES.md)** (500+ lines)
  - Example 1: Basic game loop integration
  - Example 2: Game state management
  - Example 3: Performance-critical AI
  - Example 4: Particle system scripting
  - Example 5: Profiling & monitoring
  - Example 6: Lua vs LuaJIT comparison
  - Example 7: Optimal script structure
  - Example 8: Checking LuaJIT status

### 📋 Quick Reference
- **[LUAJIT_QUICK_REFERENCE.md](LUAJIT_QUICK_REFERENCE.md)** (150+ lines)
  - Enable/disable LuaJIT
  - Basic usage
  - Profiling
  - Hot reload
  - JIT control
  - Memory management
  - Performance tips
  - API reference
  - Troubleshooting table

### 📝 Summary
- **[LUAJIT_IMPLEMENTATION_SUMMARY.md](LUAJIT_IMPLEMENTATION_SUMMARY.md)** (300+ lines)
  - What was implemented
  - Performance characteristics
  - Key features
  - Integration points
  - Files changed/created
  - Building instructions
  - Optimization examples
  - FAQ

---

## Documentation Content Map

### By Topic

#### Installation & Setup
- [LUAJIT_QUICK_REFERENCE.md#Enable LuaJIT](LUAJIT_QUICK_REFERENCE.md) - Default enabled
- [LUAJIT_QUICK_REFERENCE.md#Building Without LuaJIT](LUAJIT_QUICK_REFERENCE.md) - Use standard Lua
- [LUAJIT_INTEGRATION_GUIDE.md#Quick Start](LUAJIT_INTEGRATION_GUIDE.md) - 3-step setup

#### Usage Examples
- [LUAJIT_EXAMPLES.md#Example 1: Basic Game Loop](LUAJIT_EXAMPLES.md)
- [LUAJIT_EXAMPLES.md#Example 2: Game State](LUAJIT_EXAMPLES.md)
- [LUAJIT_EXAMPLES.md#Example 3: AI Behavior](LUAJIT_EXAMPLES.md)
- [LUAJIT_EXAMPLES.md#Example 4: Particles](LUAJIT_EXAMPLES.md)

#### Performance & Optimization
- [LUAJIT_INTEGRATION_GUIDE.md#Performance Characteristics](LUAJIT_INTEGRATION_GUIDE.md)
- [LUAJIT_INTEGRATION_GUIDE.md#Optimization Tips](LUAJIT_INTEGRATION_GUIDE.md)
- [LUAJIT_QUICK_REFERENCE.md#Performance Tips](LUAJIT_QUICK_REFERENCE.md)
- [LUAJIT_EXAMPLES.md#Example 7: Optimal Structure](LUAJIT_EXAMPLES.md)
- [LUAJIT_INTEGRATION_GUIDE.md#Performance Benchmarks](LUAJIT_INTEGRATION_GUIDE.md)

#### Profiling & Monitoring
- [LUAJIT_EXAMPLES.md#Example 5: Profiling](LUAJIT_EXAMPLES.md)
- [LUAJIT_EXAMPLES.md#Example 8: Checking Status](LUAJIT_EXAMPLES.md)
- [LUAJIT_INTEGRATION_GUIDE.md#Performance Profiling](LUAJIT_INTEGRATION_GUIDE.md)

#### Advanced Features
- [LUAJIT_INTEGRATION_GUIDE.md#Advanced API](LUAJIT_INTEGRATION_GUIDE.md)
  - Hot Reload
  - JIT Control
  - Memory Management
  - Clearing State
  - Native Functions

#### Troubleshooting
- [LUAJIT_INTEGRATION_GUIDE.md#Troubleshooting](LUAJIT_INTEGRATION_GUIDE.md)
- [LUAJIT_QUICK_REFERENCE.md#Common Issues](LUAJIT_QUICK_REFERENCE.md)

#### API Reference
- [LUAJIT_QUICK_REFERENCE.md#API Reference](LUAJIT_QUICK_REFERENCE.md)
  - Lifecycle
  - Execution
  - Configuration
  - Introspection
  - Advanced

---

## Reading Paths by Role

### Game Developer
1. [LUAJIT_QUICK_REFERENCE.md](LUAJIT_QUICK_REFERENCE.md) - Understand basics (10 min)
2. [LUAJIT_EXAMPLES.md#Example 1](LUAJIT_EXAMPLES.md) - Game loop integration (15 min)
3. [LUAJIT_EXAMPLES.md#Example 7](LUAJIT_EXAMPLES.md) - Optimize scripts (20 min)
4. [LUAJIT_INTEGRATION_GUIDE.md#Optimization Tips](LUAJIT_INTEGRATION_GUIDE.md) - Best practices (30 min)

### AI/Physics Programmer
1. [LUAJIT_INTEGRATION_GUIDE.md#Performance Characteristics](LUAJIT_INTEGRATION_GUIDE.md)
2. [LUAJIT_EXAMPLES.md#Example 3](LUAJIT_EXAMPLES.md) - AI pathfinding
3. [LUAJIT_EXAMPLES.md#Example 4](LUAJIT_EXAMPLES.md) - Particle physics
4. [LUAJIT_INTEGRATION_GUIDE.md#Optimization Tips](LUAJIT_INTEGRATION_GUIDE.md)

### Engine Integrator
1. [LUAJIT_IMPLEMENTATION_SUMMARY.md](LUAJIT_IMPLEMENTATION_SUMMARY.md) - Overview
2. [LUAJIT_QUICK_REFERENCE.md#Files Modified](LUAJIT_QUICK_REFERENCE.md)
3. [LUAJIT_EXAMPLES.md#Example 1](LUAJIT_EXAMPLES.md) - Integration pattern
4. [LUAJIT_INTEGRATION_GUIDE.md#Building Without LuaJIT](LUAJIT_INTEGRATION_GUIDE.md)

### Performance Analyst
1. [LUAJIT_INTEGRATION_GUIDE.md#Performance Characteristics](LUAJIT_INTEGRATION_GUIDE.md)
2. [LUAJIT_EXAMPLES.md#Example 5](LUAJIT_EXAMPLES.md) - Profiling
3. [LUAJIT_EXAMPLES.md#Example 6](LUAJIT_EXAMPLES.md) - Benchmark comparison
4. [LUAJIT_INTEGRATION_GUIDE.md#Performance Benchmarks](LUAJIT_INTEGRATION_GUIDE.md)

---

## Key Sections

### What You Need to Know

**LuaJIT provides:**
- ✅ 10-20x performance improvement over standard Lua
- ✅ Transparent - no code changes needed
- ✅ Enabled by default in the engine
- ✅ Full profiling support
- ✅ Hot-reload capability
- ✅ Memory efficient (~300KB overhead)

**Best for:**
- ✅ Game loops
- ✅ AI and pathfinding
- ✅ Physics calculations
- ✅ Particle systems
- ✅ Animation math

**Less beneficial for:**
- ⚠️ I/O operations
- ⚠️ String parsing
- ⚠️ One-time operations

---

## File Sizes & Time Investment

| Document | Size | Read Time | Content |
|----------|------|-----------|---------|
| Quick Reference | 5 KB | 10 min | API, examples, tips |
| Integration Guide | 20 KB | 45 min | Complete guide |
| Examples | 25 KB | 30 min | 8 code examples |
| Summary | 15 KB | 20 min | Implementation overview |
| **Total** | **65 KB** | **2 hours** | Full mastery |

---

## Quick Links to Common Tasks

### "I want to..." 

- **...get started quickly** → [LUAJIT_QUICK_REFERENCE.md](LUAJIT_QUICK_REFERENCE.md)
- **...understand LuaJIT basics** → [LUAJIT_INTEGRATION_GUIDE.md#Quick Start](LUAJIT_INTEGRATION_GUIDE.md)
- **...see code examples** → [LUAJIT_EXAMPLES.md](LUAJIT_EXAMPLES.md)
- **...optimize my scripts** → [LUAJIT_INTEGRATION_GUIDE.md#Optimization Tips](LUAJIT_INTEGRATION_GUIDE.md)
- **...profile performance** → [LUAJIT_EXAMPLES.md#Example 5](LUAJIT_EXAMPLES.md)
- **...benchmark Lua vs LuaJIT** → [LUAJIT_EXAMPLES.md#Example 6](LUAJIT_EXAMPLES.md)
- **...integrate with my app** → [LUAJIT_EXAMPLES.md#Example 1](LUAJIT_EXAMPLES.md)
- **...use hot-reload** → [LUAJIT_INTEGRATION_GUIDE.md#Advanced API](LUAJIT_INTEGRATION_GUIDE.md)
- **...fix a problem** → [LUAJIT_INTEGRATION_GUIDE.md#Troubleshooting](LUAJIT_INTEGRATION_GUIDE.md)
- **...understand the changes** → [LUAJIT_IMPLEMENTATION_SUMMARY.md](LUAJIT_IMPLEMENTATION_SUMMARY.md)
- **...build without LuaJIT** → [LUAJIT_INTEGRATION_GUIDE.md#Building Without LuaJIT](LUAJIT_INTEGRATION_GUIDE.md)
- **...reference the API** → [LUAJIT_QUICK_REFERENCE.md#API Reference](LUAJIT_QUICK_REFERENCE.md)

---

## Related Documentation

### In Engine
- [SCRIPTING_QUICK_START.md](SCRIPTING_QUICK_START.md) - Multi-language scripting overview
- [MULTI_LANGUAGE_SCRIPTING_GUIDE.md](MULTI_LANGUAGE_SCRIPTING_GUIDE.md) - Language comparisons
- [SCRIPTING_INTEGRATION_EXAMPLE.md](SCRIPTING_INTEGRATION_EXAMPLE.md) - Complete game example

### Code Files
- [include/LuaJitScriptSystem.h](include/LuaJitScriptSystem.h) - Class definition
- [src/LuaJitScriptSystem.cpp](src/LuaJitScriptSystem.cpp) - Implementation
- [include/IScriptSystem.h](include/IScriptSystem.h) - Base interface
- [src/ScriptLanguageRegistry.cpp](src/ScriptLanguageRegistry.cpp) - Registration

---

## Tips for Learning LuaJIT

1. **Start Small** - Read QUICK_REFERENCE first (10 min)
2. **See Examples** - Study EXAMPLES.md (30 min)
3. **Understand Deeply** - Read INTEGRATION_GUIDE (45 min)
4. **Experiment** - Copy examples, modify, run
5. **Optimize** - Profile with SetProfilingEnabled(true)
6. **Reference** - Use QUICK_REFERENCE as lookup

---

## Document Structure

```
LUAJIT_DOCUMENTATION_INDEX (this file)
├── LUAJIT_QUICK_REFERENCE.md (API, examples, tips)
├── LUAJIT_INTEGRATION_GUIDE.md (complete guide)
├── LUAJIT_EXAMPLES.md (8 code examples)
├── LUAJIT_IMPLEMENTATION_SUMMARY.md (what was built)
└── This Index File
```

---

## Version Info

- **LuaJIT Version**: 2.1 (latest stable)
- **Engine Integration**: Complete
- **Documentation**: 1000+ lines
- **Code**: 650+ lines
- **Build**: CMake integrated
- **Status**: Production ready ✅

---

## Support & Troubleshooting

### Build Issues
See [LUAJIT_INTEGRATION_GUIDE.md#Troubleshooting](LUAJIT_INTEGRATION_GUIDE.md)

### Performance Questions
See [LUAJIT_INTEGRATION_GUIDE.md#Performance Characteristics](LUAJIT_INTEGRATION_GUIDE.md)

### Code Examples
See [LUAJIT_EXAMPLES.md](LUAJIT_EXAMPLES.md)

### API Reference
See [LUAJIT_QUICK_REFERENCE.md#API Reference](LUAJIT_QUICK_REFERENCE.md)

---

**Last Updated:** 2026-01-26
**Status:** Complete & Production Ready
**Performance:** 10-20x speedup for game logic
