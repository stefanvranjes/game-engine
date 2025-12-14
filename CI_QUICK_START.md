# CI/CD Infrastructure Setup Guide

This document provides a quick start guide for the comprehensive testing and continuous integration setup.

## Quick Start

### 1. Build with Tests (Recommended)

**Windows (Command Prompt):**
```bash
build_with_options.bat test
```

**Windows (PowerShell):**
```powershell
.\build.ps1 test
```

**Linux/macOS:**
```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j4
ctest --output-on-failure
```

### 2. Run All Tests

```bash
# From build directory
ctest --output-on-failure --verbose

# Or directly
.\build\bin\tests.exe
```

### 3. Run Specific Test Category

```bash
ctest -R math --output-on-failure
ctest -R network --output-on-failure
```

---

## What's Included

### ✅ Unit Testing (GoogleTest)
- **Location**: `tests/` directory
- **Tests**: Math, Transform, GameObject, Material, Shader, Networking
- **Framework**: Google Test v1.14.0
- **Coverage**: 40+ test cases

### ✅ Static Analysis (Clang-Tidy)
- **Configuration**: `.clang-tidy`
- **Checks**: Readability, Performance, Modernization, Bug Detection
- **Enable**: `cmake .. -DENABLE_CLANG_TIDY=ON`

### ✅ Runtime Sanitizers
- **AddressSanitizer (ASAN)**: Memory error detection
- **UndefinedBehaviorSanitizer (UBSAN)**: Undefined behavior detection
- **MemorySanitizer (MSAN)**: Uninitialized memory detection
- **ThreadSanitizer (TSAN)**: Data race detection
- **Enable**: `cmake .. -DUSE_ASAN=ON -DUSE_UBSAN=ON`

### ✅ GitHub Actions CI Pipeline
- **Workflow**: `.github/workflows/ci-pipeline.yml`
- **Jobs**: 
  - Windows Debug/Release builds
  - Linux unit tests
  - Clang-Tidy analysis
  - All 4 sanitizers
  - Code quality checks
- **Triggers**: Push to main/develop, Pull Requests

---

## Build Commands

### Release Build
```bash
build_with_options.bat release
# or
.\build.ps1 release
```

### Debug Build with Tests
```bash
build_with_options.bat debug test
# or
.\build.ps1 debug test
```

### Full Build (Release + Tests + Analysis)
```bash
build_with_options.bat all
# or
.\build.ps1 all
```

### Clean Build
```bash
build_with_options.bat clean all
# or
.\build.ps1 clean all
```

---

## Test Execution

### Run All Tests
```bash
# Method 1: Direct execution
.\build\bin\tests.exe

# Method 2: Via CTest
cd build && ctest --output-on-failure
```

### Run Specific Tests
```bash
# All math tests
.\build\bin\tests.exe --gtest_filter=MathTest.*

# Specific test
.\build\bin\tests.exe --gtest_filter=MathTest.Vector3Addition

# All tests except network
.\build\bin\tests.exe --gtest_filter=-NetworkTest.*
```

### Test Output Options
```bash
# Verbose output
.\build\bin\tests.exe --gtest_verbose

# List all tests
.\build\bin\tests.exe --gtest_list_tests

# Print time for each test
.\build\bin\tests.exe --gtest_print_time=1

# Shuffle test order (detect dependencies)
.\build\bin\tests.exe --gtest_shuffle
```

---

## Sanitizer Usage

### Address Sanitizer (Memory Errors)
Detects: buffer overflows, use-after-free, memory leaks

```bash
# Linux/macOS
cmake .. -DUSE_ASAN=ON -DBUILD_TESTS=ON
cmake --build . -j4
ctest --output-on-failure
```

### Undefined Behavior Sanitizer
Detects: integer overflow, null pointer deref, array bounds

```bash
# Linux/macOS
cmake .. -DUSE_UBSAN=ON -DBUILD_TESTS=ON
cmake --build . -j4
ctest --output-on-failure
```

### Combined Sanitizers
```bash
# Linux/macOS
cmake .. -DUSE_ASAN=ON -DUSE_UBSAN=ON -DBUILD_TESTS=ON
cmake --build . -j4
ctest --output-on-failure
```

### Debug Sanitizer Output
```bash
# More verbose output
export ASAN_OPTIONS=verbosity=2:halt_on_error=1
./build/bin/tests
```

---

## Static Analysis (Clang-Tidy)

### Enable in Build
```bash
cmake .. -DENABLE_CLANG_TIDY=ON -DBUILD_TESTS=ON
cmake --build . 2>&1 | tee build.log
grep "error:" build.log  # Check for errors
```

### Manual Execution
```bash
# Single file
clang-tidy src/Renderer.cpp -- -Iinclude -std=c++20

# All files
find src include -name "*.cpp" | xargs clang-tidy
```

### Suppress Warnings
```cpp
// NOLINT(check-name)
int magic_number = 42;

// NOLINTNEXTLINE(check-name)
void problematic_function() { }
```

---

## GitHub Actions CI

### Workflow Overview

The CI pipeline automatically runs on:
- **Triggers**: Push to main/develop, Pull Requests
- **Platforms**: Windows, Linux
- **Jobs**:
  1. Windows builds (Debug + Release)
  2. Linux unit tests
  3. Clang-Tidy analysis
  4. Address Sanitizer
  5. UB Sanitizer
  6. Memory Sanitizer
  7. Thread Sanitizer
  8. Code quality checks
  9. Debug build verification

### Monitor CI Status

1. Go to GitHub repository
2. Click "Actions" tab
3. View workflow runs and job details
4. Click on failed jobs for logs

### Add CI Badge to README

```markdown
[![CI Pipeline](https://github.com/YOUR_USERNAME/game-engine/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/game-engine/actions/workflows/ci-pipeline.yml)
```

---

## File Structure

```
game-engine/
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml          # GitHub Actions CI
├── tests/
│   ├── main.cpp                     # Test entry point
│   ├── test_math.cpp               # Math tests
│   ├── test_transform.cpp          # Transform tests
│   ├── test_gameobject.cpp         # GameObject tests
│   ├── test_material.cpp           # Material tests
│   ├── test_shader.cpp             # Shader tests
│   └── game-engine-multiplayer/
│       ├── test_message.cpp        # Network tests
│       └── test_serializer.cpp     # Serialization tests
├── docs/
│   └── TESTING_CI_GUIDE.md         # Detailed testing guide
├── .clang-tidy                     # Clang-Tidy configuration
├── CMakeLists.txt                  # CMake configuration
├── build.ps1                       # PowerShell build script
├── build_with_options.bat          # Batch build script
└── build.bat                       # Original build script
```

---

## Common Tasks

### Adding a New Test

1. Create file in `tests/test_mycomponent.cpp`
2. Write tests using GoogleTest
3. Add to CMakeLists.txt test sources
4. Build: `cmake --build build`
5. Run: `ctest --output-on-failure`

### Fixing Clang-Tidy Warnings

1. Build with clang-tidy: `cmake .. -DENABLE_CLANG_TIDY=ON`
2. Review warnings in build output
3. Fix issues or suppress with `// NOLINT`
4. Rebuild to verify

### Debugging Sanitizer Issues

1. Enable sanitizer: `cmake .. -DUSE_ASAN=ON`
2. Build and run tests
3. Review sanitizer output in test results
4. Set environment variables for more details: `ASAN_OPTIONS=...`
5. Use debugger if available

### Running Tests Locally Before Push

```bash
# Full quality check
.\build.ps1 clean all
ctest --output-on-failure --verbose
```

---

## Troubleshooting

### "CMake not found"
- Windows: Install from [cmake.org](https://cmake.org/download/)
- Linux: `sudo apt-get install cmake`
- macOS: `brew install cmake`

### Tests won't compile
- Ensure C++20 support: Check compiler version
- Update CMake: `cmake --version` should be 3.10+
- Delete build directory and rebuild

### Sanitizer not available
- ASAN/UBSAN require Clang or GCC with sanitizer support
- Windows: Use WSL or Linux VM
- macOS: Update Xcode

### GitHub Actions failing
1. Check workflow file syntax: `.github/workflows/ci-pipeline.yml`
2. Review job logs in GitHub Actions UI
3. Reproduce locally: `cmake .. -DUSE_ASAN=ON -DBUILD_TESTS=ON`

---

## Performance Benchmarks

Typical build times:

| Build Type | Windows (MSVC) | Linux (GCC/Clang) |
|-----------|---|---|
| Release (no tests) | ~30s | ~25s |
| Debug (no tests) | ~40s | ~35s |
| Release + Tests | ~45s | ~40s |
| With ASAN | N/A | ~50s |
| With UBSAN | N/A | ~50s |
| Clang-Tidy | N/A | ~60s |

---

## Next Steps

1. **Review** [TESTING_CI_GUIDE.md](../docs/TESTING_CI_GUIDE.md) for detailed testing documentation
2. **Run** tests locally: `.\build.ps1 test`
3. **Add** more test cases for critical components
4. **Monitor** GitHub Actions for CI results
5. **Enable** sanitizers in development workflow

## Resources

- [GoogleTest Documentation](https://github.com/google/googletest)
- [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/)
- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [CMake Documentation](https://cmake.org/cmake/help/latest/)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Last Updated**: December 2024
**Maintainer**: Game Engine Development Team
