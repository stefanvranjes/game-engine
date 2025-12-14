# Contributing to Game Engine

Welcome! We're excited that you want to contribute to the Game Engine. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)

---

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're committed to providing a welcoming and inspiring community for all.

---

## Getting Started

### Prerequisites

- **C++20 compatible compiler** (MSVC, Clang, or GCC)
- **CMake 3.20+**
- **GLFW 3.3.8** (auto-fetched by CMake)
- **OpenGL 3.3+** compatible GPU
- **Git**
- **Doxygen** (for generating API documentation)

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/game-engine.git
   cd game-engine
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Build the engine:**
   ```powershell
   # Windows (PowerShell)
   .\build.ps1
   
   # Windows (Batch)
   build.bat
   
   # Linux/macOS
   mkdir -p build && cd build
   cmake .. && cmake --build .
   ```

4. **Run tests:**
   ```powershell
   .\build.ps1 test
   ```

---

## Development Workflow

### 1. Pick an Issue

- Check the [Issues](https://github.com/yourusername/game-engine/issues) tab
- Look for issues labeled `good-first-issue` or `help-wanted`
- Comment on the issue to express interest before starting work
- Discuss approach with maintainers for complex features

### 2. Create Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/short-descriptive-name
```

**Branch naming conventions:**
- `feature/add-new-system` - New functionality
- `fix/bug-description` - Bug fixes
- `docs/documentation-update` - Documentation
- `refactor/component-name` - Code refactoring
- `perf/optimization-area` - Performance improvements

### 3. Make Changes

- Follow the [Coding Standards](#coding-standards)
- Write clear, descriptive commit messages
- Keep commits atomic and logical
- Reference issue numbers: `git commit -m "Fix #123: Description"`

### 4. Build and Test Locally

```bash
# Build with tests
.\build.ps1 test

# Or individually:
cmake --build build --config Debug
ctest --output-on-failure
```

### 5. Run Code Quality Checks

Ensure your code passes:
- Unit tests (via `ctest`)
- Static analysis (Clang-Tidy)
- Compilation without warnings

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with a descriptive title and detailed description.

---

## Coding Standards

### C++ Style Guide

#### File Organization

```cpp
// Header guards (use #pragma once)
#pragma once

#include <standard-library>
#include <third-party>

#include "internal-headers"

// Forward declarations
class GameObject;
class Renderer;

// Namespace
namespace GameEngine {

/**
 * @class MyClass
 * @brief Brief description of the class.
 * 
 * Detailed description of the class functionality,
 * usage patterns, and important notes.
 */
class MyClass {
public:
    MyClass();
    ~MyClass();
    
    // Public API methods
    void PublicMethod();
    
private:
    // Private implementation
    void PrivateMethod();
    int m_member;
};

} // namespace GameEngine
```

#### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `ParticleSystem`, `GameObject` |
| Methods | camelCase | `update()`, `getRenderData()` |
| Member Variables | m_camelCase | `m_position`, `m_renderer` |
| Static Members | s_camelCase | `s_globalInstance` |
| Constants | UPPER_SNAKE_CASE | `MAX_PARTICLES`, `DEFAULT_SIZE` |
| Local Variables | snake_case | `particle_count`, `is_active` |
| Macros | UPPER_SNAKE_CASE | `SAFE_DELETE`, `ASSERT_VALID` |

#### Comments and Documentation

Use Doxygen-style comments for public APIs:

```cpp
/**
 * @brief Updates particle positions based on physics.
 * 
 * Applies gravity, wind forces, and collision responses
 * to all active particles in this emitter.
 * 
 * @param deltaTime Time step in seconds
 * @return Number of particles updated
 * 
 * @note GPU compute path requires OpenGL 4.3+
 * @warning This function modifies internal state
 */
int Update(float deltaTime);

/// @private
void InternalHelperFunction();
```

#### Smart Pointers

```cpp
// Exclusive ownership
std::unique_ptr<Renderer> m_renderer;

// Shared ownership
std::shared_ptr<Texture> m_texture;

// Reference counting in callbacks
class MyObject : public std::enable_shared_from_this<MyObject> {
public:
    void RegisterCallback() {
        // Safe to pass shared_ptr in callback
        someSystem->setCallback([self = shared_from_this()]() {
            self->handleEvent();
        });
    }
};
```

#### Error Handling

```cpp
// Assertions for programmer errors
assert(position != nullptr);

// Exceptions for runtime errors
try {
    GLuint shader = LoadShader(path);
} catch (const std::exception& e) {
    SPDLOG_ERROR("Failed to load shader: {}", e.what());
    return false;
}

// Return codes for recoverable failures
bool LoadModel(const std::string& path) {
    if (!FileExists(path)) {
        SPDLOG_WARN("Model file not found: {}", path);
        return false;
    }
    return true;
}
```

---

## Documentation

### API Documentation

API documentation is generated using Doxygen. When adding public APIs:

1. **Add Doxygen comments** to header files
2. **Generate docs locally:**
   ```bash
   doxygen Doxyfile
   # Output: api-docs/html/index.html
   ```
3. **Use standard tags:**
   - `@brief` - One line summary
   - `@param` - Parameter description
   - `@return` - Return value
   - `@throws` - Exceptions
   - `@note` - Important notes
   - `@warning` - Warnings
   - `@example` - Usage examples
   - `@see` - Related items
   - `@deprecated` - For deprecated APIs

### Markdown Documentation

Create guides in the `docs/` folder for non-API documentation:
- System architecture
- Integration guides
- Usage examples
- Performance tips
- Troubleshooting

---

## Testing

### Writing Tests

Tests use GoogleTest (GTest). Add tests to `tests/test_*.cpp`:

```cpp
#include <gtest/gtest.h>
#include "ParticleSystem.h"

using namespace GameEngine;

class ParticleSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        particle_system = std::make_unique<ParticleSystem>();
    }
    
    void TearDown() override {
        particle_system.reset();
    }
    
    std::unique_ptr<ParticleSystem> particle_system;
};

TEST_F(ParticleSystemTest, InitializesWithCorrectCapacity) {
    EXPECT_EQ(particle_system->GetActiveParticleCount(), 0);
}

TEST_F(ParticleSystemTest, EmitterProducesParticles) {
    auto emitter = particle_system->CreateEmitter();
    emitter->Emit(100);
    EXPECT_EQ(particle_system->GetActiveParticleCount(), 100);
}
```

### Test Coverage

- Aim for >80% code coverage on new code
- Test both happy path and error cases
- Include edge cases and boundary conditions
- Use parameterized tests for variations

### Running Tests

```bash
# Run all tests
.\build.ps1 test

# Run specific test
ctest -R ParticleSystemTest -V

# With sanitizers (Linux/macOS)
cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
cmake --build build
ctest --output-on-failure
```

---

## Submitting Changes

### Pull Request Guidelines

1. **Title**: Clear, descriptive title (e.g., "Add GPU particle compute path")
2. **Description**: Include:
   - What changes were made
   - Why these changes were needed
   - How to test the changes
   - Any breaking changes
   - Closes #123 (if applicable)

3. **Example PR Description:**
   ```
   ## Summary
   Implements GPU compute shader path for particle updates to improve 
   performance for large emitters (10k+ particles).
   
   ## Changes
   - Add ParticleComputeShader class
   - Implement particle position update in GLSL compute shader
   - Add GPU path selection logic in ParticleSystem::Update()
   - Add unit tests for compute shader
   
   ## Testing
   - [x] All existing tests pass
   - [x] Added 8 new tests for GPU compute path
   - [x] Tested with 100k particles
   - [x] Performance improved 3x over CPU path
   
   ## Breaking Changes
   - None
   
   ## Closes
   #234
   ```

4. **Checklist** (in PR template):
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] No compiler warnings
   - [ ] Commit messages are clear

---

## Code Review Process

### What Reviewers Look For

1. **Correctness**: Does the code work as intended?
2. **Design**: Is the architecture sound and maintainable?
3. **Style**: Does it follow conventions?
4. **Tests**: Is there adequate test coverage?
5. **Documentation**: Are APIs documented?
6. **Performance**: Any regressions or inefficiencies?
7. **Security**: Are there any vulnerabilities?

### Responding to Feedback

- Be open to suggestions
- Explain your reasoning if you disagree
- Make requested changes in new commits (don't force-push during review)
- Reply to each comment
- Re-request review after addressing feedback

### Merging

A PR can be merged when:
- ✅ All CI checks pass
- ✅ At least 2 approvals from maintainers
- ✅ All conversations resolved
- ✅ Branch is up-to-date with main/develop

---

## System-Specific Guidelines

### Adding a New Rendering System

1. Create `include/YourSystem.h` with public API
2. Create `src/YourSystem.cpp` with implementation
3. Add Doxygen comments to header
4. Update [Renderer.h](include/Renderer.h) if integrating with main pipeline
5. Add unit tests in `tests/test_yoursystem.cpp`
6. Create integration guide in `docs/YOURSYSTEM_GUIDE.md`
7. Update [Renderer.cpp](src/Renderer.cpp) render pipeline if needed

### Adding a New Component Type

1. Create component class inheriting from component interface
2. Add serialization support (JSON via nlohmann/json)
3. Add to GameObject's component manager
4. Create tests for component lifecycle
5. Document in `docs/COMPONENTS.md`

### Multiplayer/Networking Contributions

1. Extend [Message.hpp](game-engine-multiplayer/include/Message.hpp) for new message types
2. Update [NetworkManager.hpp](game-engine-multiplayer/include/NetworkManager.hpp) if needed
3. Add server/client handling
4. Create integration tests
5. Document in [game-engine-multiplayer/README.md](game-engine-multiplayer/README.md)

---

## Performance Considerations

- Profile before and after changes with [Profiler.h](include/Profiler.h)
- Avoid allocations in hot loops
- Use object pooling for frequently created/destroyed objects
- Consider SIMD for batch operations
- Document performance implications in comments
- Add benchmarks for critical paths

---

## Useful Resources

- [API Documentation](./api-docs/html/index.html) - Generated Doxygen docs
- [Architecture Diagram](./ARCHITECTURE_DIAGRAM.md) - System overview
- [Testing Guide](./docs/TESTING_CI_GUIDE.md) - Comprehensive testing documentation
- [Physics Engine](./docs/PHYSICS_ENGINE_README.md) - Physics system guide
- [GitHub Actions CI](./docs/TESTING_CI_GUIDE.md#ci-pipeline) - CI workflow details

---

## Questions?

- Open an issue for questions
- Check existing issues/PRs for similar discussions
- Ask in code reviews
- Contact maintainers: [contact info]

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🚀
