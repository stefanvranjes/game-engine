# Documentation Guide

This guide explains how to generate and maintain API documentation for the Game Engine.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Doxygen Setup](#doxygen-setup)
3. [Writing Documentation](#writing-documentation)
4. [Generating Documentation](#generating-documentation)
5. [Documentation Standards](#documentation-standards)
6. [Maintaining Documentation](#maintaining-documentation)

---

## Quick Start

### Generate API Docs (2 minutes)

```bash
# Prerequisites: Doxygen installed
# Download: https://www.doxygen.nl/download.html

# Generate documentation
doxygen Doxyfile

# Open in browser
api-docs/html/index.html
```

### Or on Windows

```powershell
# Using PowerShell
doxygen Doxyfile
Start-Process "api-docs\html\index.html"
```

---

## Doxygen Setup

### Installation

**Windows:**
- Download installer from https://www.doxygen.nl/download.html
- Run installer, add to PATH
- Verify: `doxygen --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install doxygen graphviz
```

**macOS:**
```bash
brew install doxygen graphviz
```

### Configuration

The [Doxyfile](../Doxyfile) in the project root contains all Doxygen settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| INPUT | include/ src/ game-engine-multiplayer/ | Source code to document |
| OUTPUT_DIRECTORY | ./api-docs | Output location |
| RECURSIVE | YES | Process subdirectories |
| GENERATE_HTML | YES | Generate HTML output |
| GENERATE_LATEX | YES | Generate PDF-ready LaTeX |
| GENERATE_XML | YES | Generate XML for tools |
| HAVE_DOT | YES | Use Graphviz for diagrams |
| EXTRACT_ALL | NO | Document only public APIs |

**Customize these in Doxyfile as needed for your project.**

---

## Writing Documentation

### Doxygen Comment Styles

Use Javadoc style (recommended):

```cpp
/**
 * @brief Brief one-line description.
 * 
 * Longer detailed description explaining what this does,
 * when to use it, and any important implementation details.
 */
```

Or Qt style:

```cpp
/*!
 * @brief Brief description
 * 
 * Detailed description
 */
```

### Class Documentation

```cpp
/**
 * @class ParticleSystem
 * @brief Manages particle emission, simulation, and rendering.
 * 
 * The ParticleSystem coordinates multiple ParticleEmitters,
 * supports both CPU and GPU particle updates, and handles
 * depth-sorting and rendering of particles.
 * 
 * Features:
 * - CPU and GPU compute paths with adaptive selection
 * - Physics: gravity, wind, attractors, collisions
 * - Trail rendering with persistent history
 * - Global particle budget tracking
 * 
 * Usage:
 * @code
 * auto particle_system = std::make_unique<ParticleSystem>();
 * auto emitter = particle_system->CreateEmitter("explosion");
 * emitter->SetEmissionRate(100);
 * emitter->Emit(50);
 * 
 * // Update in game loop
 * particle_system->Update(deltaTime);
 * @endcode
 * 
 * @note Requires OpenGL 4.3+ for GPU compute path
 * @warning Particle budget is global across all emitters
 * 
 * @see ParticleEmitter
 * @see ParticlePhysics
 */
class ParticleSystem {
    // ...
};
```

### Method Documentation

```cpp
/**
 * @brief Updates particles based on physics and time.
 * 
 * Applies gravity, wind forces, and collision responses to all
 * active particles. For emitters with GPU compute enabled,
 * this is handled by compute shaders.
 * 
 * @param deltaTime Time step in seconds (typically 1/60 for 60 FPS)
 * @return Number of particles updated in this frame
 * 
 * @pre deltaTime must be > 0
 * @post All active particles are updated
 * 
 * @throws std::runtime_error if GPU compute shader compilation fails
 * 
 * @note Call once per frame from Application::Update()
 * @warning Modifies internal particle data
 * 
 * @example
 * @code
 * float deltaTime = 0.016f; // 60 FPS
 * int updated = particle_system->Update(deltaTime);
 * SPDLOG_INFO("Updated {} particles", updated);
 * @endcode
 * 
 * @see ParticlePhysics::ApplyForces()
 */
int Update(float deltaTime);
```

### Member Variable Documentation

```cpp
class GameObject {
private:
    /// Transform hierarchy (position, rotation, scale)
    std::unique_ptr<Transform> m_transform;
    
    /// Per-object material with PBR parameters
    std::shared_ptr<Material> m_material;
    
    /// @brief Active components by type ID
    /// @see ComponentRegistry
    std::unordered_map<std::string, std::shared_ptr<Component>> m_components;
};
```

### Common Doxygen Tags

| Tag | Purpose | Example |
|-----|---------|---------|
| `@brief` | One-line summary | `@brief Updates particle positions` |
| `@param` | Parameter documentation | `@param deltaTime Time step in seconds` |
| `@return` | Return value | `@return Number of particles updated` |
| `@throws` | Exception documentation | `@throws std::runtime_error` |
| `@note` | Important note | `@note Requires OpenGL 4.3+` |
| `@warning` | Warning message | `@warning Modifies internal state` |
| `@example` / `@code` | Usage example | `@code ... @endcode` |
| `@see` | Related items | `@see ParticleEmitter` |
| `@deprecated` | For old APIs | `@deprecated Use NewMethod() instead` |
| `@pre` / `@post` | Preconditions/postconditions | `@pre value >= 0` |
| `@private` | Hide from docs | `/// @private` |
| `@internal` | Internal implementation | `/// @internal` |

---

## Generating Documentation

### Command Line

```bash
# Generate with default settings
doxygen Doxyfile

# Generate specific output
doxygen -g MyDoxyfile           # Create new config
doxygen -d QUIET                # Suppress output
doxygen -d Platform=Windows     # Platform-specific
```

### Automated Generation

Add to build system (CMakeLists.txt):

```cmake
# Find Doxygen
find_package(Doxygen OPTIONAL_COMPONENTS dot mscgen dia)

if(DOXYGEN_FOUND)
    # Configure Doxyfile
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/api-docs)
    set(DOXYGEN_INPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/include)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX YES)
    set(DOXYGEN_USE_MATHJAX YES)
    
    # Add doxygen target
    doxygen_add_docs(doxygen
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/docs/API_OVERVIEW.md
        ALL
        COMMENT "Generate API documentation with Doxygen"
    )
endif()
```

Then build docs:

```bash
cmake --build build --target doxygen
```

### GitHub Actions CI/CD

Add workflow (`.github/workflows/docs.yml`):

```yaml
name: Generate API Docs

on:
  push:
    branches: [main, develop]
    paths:
      - 'include/**'
      - 'src/**'
      - 'Doxyfile'

jobs:
  doxygen:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Doxygen
        run: sudo apt-get install -y doxygen graphviz
      
      - name: Generate Docs
        run: doxygen Doxyfile
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./api-docs/html
```

---

## Documentation Standards

### Header File Documentation

Every public header should have:

```cpp
#pragma once

/**
 * @file ParticleSystem.h
 * @brief Particle emission, simulation, and rendering system.
 * 
 * Provides ParticleSystem and ParticleEmitter classes for managing
 * particle effects with CPU/GPU compute paths.
 * 
 * @author Your Name
 * @version 1.0
 * @date 2025
 * 
 * @see ParticleEmitter for emitter-specific configuration
 * @see ParticlePhysics for physics parameters
 */

#include <vector>
// ...
```

### Namespace Documentation

```cpp
/**
 * @namespace GameEngine
 * @brief Main namespace for the Game Engine.
 * 
 * All public API classes and functions reside in this namespace.
 */
namespace GameEngine {
    // ...
}
```

### Enum Documentation

```cpp
/**
 * @enum MessageType
 * @brief Network message types for multiplayer communication.
 */
enum class MessageType {
    Chat,      ///< Player chat message
    Join,      ///< Player joined game
    Leave,     ///< Player left game
    Ping,      ///< Keep-alive ping
    State,     ///< Game state update
};
```

### Structure Documentation

```cpp
/**
 * @struct ParticleData
 * @brief Per-particle attribute storage.
 * 
 * Stores position, velocity, lifetime, and other per-particle data
 * for efficient batch processing.
 */
struct ParticleData {
    glm::vec3 position;    ///< World position
    glm::vec3 velocity;    ///< Velocity vector
    float lifetime;        ///< Remaining lifetime in seconds
    float maxLifetime;     ///< Initial lifetime
};
```

### Documentation Checklist

- [ ] All public classes documented
- [ ] All public methods documented with `@brief`
- [ ] All parameters documented with `@param`
- [ ] Return values documented with `@return`
- [ ] Exceptions documented with `@throws`
- [ ] Usage examples provided with `@code ... @endcode`
- [ ] Related items referenced with `@see`
- [ ] Important notes/warnings included
- [ ] Deprecated APIs marked
- [ ] Private implementation marked `@private`

---

## Maintaining Documentation

### Update Checklist

When changing code:

1. **Before committing:**
   - Update Doxygen comments in header files
   - Add `@deprecated` tags for removed APIs
   - Update usage examples if API changed
   - Check for broken `@see` references

2. **Example commit message:**
   ```
   feat: Add GPU particle compute system
   
   - Add ParticleComputeShader class
   - Update ParticleEmitter with GPU path
   - Add comprehensive Doxygen documentation
   - Include usage examples in API docs
   
   Closes #123
   ```

### Documentation Review

In code reviews, check:

- [ ] Doxygen comments are complete
- [ ] Parameter names match actual code
- [ ] Examples compile and are correct
- [ ] No broken cross-references
- [ ] API stability clearly stated
- [ ] Breaking changes documented

### Keeping Docs in Sync

**Problem:** Docs get out of date with code changes.

**Solution:** 

1. **Documentation-as-Code**
   - Keep docs in version control ✓
   - Review docs in pull requests ✓
   - Track doc changes in git history ✓

2. **Automated Checks**
   - Enforce Doxygen generation in CI
   - Fail builds if undocumented API warnings
   - Check for missing `@param` tags

3. **Regular Audits**
   - Monthly review of major systems
   - Update deprecated API lists
   - Refresh examples

---

## Hosting Documentation

### Option 1: GitHub Pages (Recommended)

```yaml
# In .github/workflows/docs.yml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./api-docs/html
```

Then docs auto-deploy to `https://yourusername.github.io/game-engine/`

### Option 2: Read the Docs

1. Create `readthedocs.yml` in project root
2. Connect GitHub repo to Read the Docs
3. Docs auto-build on commits
4. Available at `https://game-engine.readthedocs.io/`

### Option 3: Self-Hosted

```bash
# Copy api-docs/html to web server
scp -r api-docs/html user@server:/var/www/docs/
```

---

## Troubleshooting

### Issue: Doxygen not finding files

**Solution:**
- Check INPUT paths in Doxyfile
- Verify files have `.h` or `.cpp` extensions
- Ensure RECURSIVE = YES for subdirectories

### Issue: Comments not appearing in docs

**Solution:**
- Use correct comment syntax (/** ... */ or ///)
- Add `@brief` tag
- Set EXTRACT_ALL = YES if needed

### Issue: Broken links/references

**Solution:**
- Use exact class/function names
- Check for typos in `@see` tags
- Verify referenced items are documented

### Issue: No class diagrams

**Solution:**
- Install Graphviz: `apt-get install graphviz` (Linux)
- Set HAVE_DOT = YES in Doxyfile
- Ensure dot executable is in PATH

---

## Resources

- **Official Doxygen Manual:** https://www.doxygen.nl/manual/
- **Doxygen Commands:** https://www.doxygen.nl/manual/commands.html
- **Markdown in Doxygen:** https://www.doxygen.nl/manual/markdown.html
- **Example Projects:** https://www.doxygen.nl/manual/starting.html

---

## Contributing to Documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute documentation improvements.

