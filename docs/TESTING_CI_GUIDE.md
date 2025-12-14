# Testing & CI/CD Infrastructure Guide

## Overview

This guide covers the unit testing, integration testing, sanitizers, and CI/CD pipeline infrastructure for the Game Engine project.

## Table of Contents

1. [GoogleTest Framework](#googletest-framework)
2. [Running Tests](#running-tests)
3. [Sanitizers](#sanitizers)
4. [Clang-Tidy Static Analysis](#clang-tidy-static-analysis)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Writing New Tests](#writing-new-tests)

---

## GoogleTest Framework

The project uses **Google Test (gtest)** for unit and integration testing. GoogleTest is automatically fetched via CMake and integrated into the build system.

### Test Organization

Tests are organized in the `tests/` directory:

```
tests/
├── main.cpp                          # GoogleTest entry point
├── test_math.cpp                     # Math utilities tests
├── test_transform.cpp                # Transform system tests
├── test_gameobject.cpp               # GameObject tests
├── test_material.cpp                 # Material system tests
├── test_shader.cpp                   # Shader tests
└── game-engine-multiplayer/
    ├── test_message.cpp              # Network message tests
    └── test_serializer.cpp           # Serialization tests
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interaction between components
- **Network Tests**: Test multiplayer/networking functionality

---

## Running Tests

### Build with Tests

```bash
# Configure and build with tests enabled (default)
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release

# Or directly with the provided script
.\build.bat
```

### Execute Tests

```bash
# Run all tests
.\build\bin\tests

# Run tests with verbose output
.\build\bin\tests --gtest_verbose

# Run specific test suite
.\build\bin\tests --gtest_filter=MathTest.*

# Run specific test
.\build\bin\tests --gtest_filter=MathTest.Vector3Addition

# List all available tests
.\build\bin\tests --gtest_list_tests

# Run tests with output on failure
.\build\bin\tests --gtest_print_time=1
```

### CTest Integration

```bash
cd build

# Run all CTest tests
ctest --output-on-failure

# Run specific test category
ctest -R math --output-on-failure

# Run tests with verbose output
ctest --verbose

# Run tests in parallel
ctest -j4 --output-on-failure
```

### Test Output Examples

**Successful test run:**
```
[==========] Running 42 tests from 7 test suites.
[----------] Global test environment set-up.
[----------] 10 tests from MathTest
[ RUN      ] MathTest.Vector3Addition
[       OK ] MathTest.Vector3Addition (0 ms)
...
[==========] 42 tests from 7 test suites ran. (150 ms total)
[  PASSED  ] 42 tests.
```

**Test failure:**
```
[----------] 5 tests from TransformTest
[ RUN      ] TransformTest.SetPosition
../tests/test_transform.cpp:32: Failure
Value of: pos.x
  Actual: 4.0
Expected: 5.0
[  FAILED  ] TransformTest.SetPosition (2 ms)
```

---

## Sanitizers

Sanitizers detect memory errors, undefined behavior, and threading issues at runtime.

### Address Sanitizer (ASAN)

Detects memory errors (buffer overflows, use-after-free, leaks).

**Enable:**
```bash
cd build
cmake .. -DUSE_ASAN=ON -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

**Common detections:**
- Heap buffer overflow
- Use-after-free
- Memory leaks
- Double free

### Undefined Behavior Sanitizer (UBSAN)

Detects undefined behavior like:
- Integer overflow
- Division by zero
- Null pointer dereference
- Out-of-bounds array access

**Enable:**
```bash
cd build
cmake .. -DUSE_UBSAN=ON -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

### Memory Sanitizer (MSAN)

Detects uninitialized memory access (only on Linux with clang).

**Enable:**
```bash
cd build
cmake .. -DUSE_MSAN=ON -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

### Thread Sanitizer (TSAN)

Detects data races and threading issues.

**Enable:**
```bash
cd build
cmake .. -DUSE_TSAN=ON -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

### Combined Sanitizers

Enable multiple sanitizers (except MSAN which conflicts with others):

```bash
cd build
cmake .. -DUSE_ASAN=ON -DUSE_UBSAN=ON -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

**Example sanitizer output:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on unknown address 0x61200004eff4
  READ of size 4 at 0x61200004eff4 thread T0
  at test.cpp:45:10
```

---

## Clang-Tidy Static Analysis

**Clang-Tidy** performs static code analysis to find bugs, style issues, and modernize code.

### Configuration

The `.clang-tidy` file in the root directory configures which checks to run:

```yaml
Checks: >
  readability-*,           # Readability issues
  performance-*,           # Performance problems
  modernize-*,             # C++ modernization
  bugprone-*,              # Common bugs
  portability-*            # Portability issues
```

### Run Clang-Tidy

**Manual execution:**
```bash
# Install clang-tools
# Ubuntu: sudo apt-get install clang-tools
# Windows: choco install llvm

# Run on specific file
clang-tidy src/Renderer.cpp -- -Iinclude -std=c++20

# Run on all files
find src include -name "*.cpp" -o -name "*.h" | xargs clang-tidy
```

**Via CMake:**
```bash
cd build
cmake .. -DENABLE_CLANG_TIDY=ON -DBUILD_TESTS=ON
cmake --build . # Will run clang-tidy automatically
```

### Common Clang-Tidy Checks

| Check | Description |
|-------|-------------|
| `readability-identifier-length` | Variable names too short |
| `performance-unnecessary-value-param` | Pass by reference instead of copy |
| `modernize-use-auto` | Use auto for type deduction |
| `bugprone-integer-division` | Integer division where floating-point expected |
| `portability-simd-intrinsics` | Platform-specific SIMD calls |

### Suppressing Checks

Add comments to suppress specific checks:

```cpp
// NOLINT(readability-magic-numbers)
float value = 3.14159;

// NOLINTNEXTLINE(modernize-use-auto)
int x = getValue();
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive CI pipeline in `.github/workflows/ci-pipeline.yml` that runs automatically on push and pull requests.

### Pipeline Jobs

#### 1. **Windows Build** (`build-windows`)
- Runs on `windows-latest`
- Tests both Debug and Release configurations
- Builds with Visual Studio 2022

```yaml
runs-on: windows-latest
strategy:
  matrix:
    config: [Debug, Release]
```

#### 2. **Clang-Tidy Analysis** (`clang-tidy`)
- Runs static analysis checks
- Fails if warnings are found
- Runs on Ubuntu (clang-tools available)

#### 3. **Sanitizers** (`sanitizers`)
- Runs with Address Sanitizer (ASAN)
- Runs with Undefined Behavior Sanitizer (UBSAN)
- Runs with Memory Sanitizer (MSAN)
- Runs with Thread Sanitizer (TSAN)

Each sanitizer variant builds and tests independently.

#### 4. **Unit Tests** (`unit-tests`)
- Comprehensive test suite execution
- Generates coverage reports
- Runs on Ubuntu

#### 5. **Code Quality** (`code-quality`)
- Clang-format linting (checking only, not modifying)
- Ensures consistent code style

#### 6. **Windows Debug Build** (`windows-debug-build`)
- Dedicated debug build job
- Ensures debug symbols are preserved
- Tests with detailed debug information

### Pipeline Triggers

The pipeline runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`

### Viewing Results

1. Go to GitHub repository → Actions tab
2. Click on specific workflow run
3. View job logs and failure details
4. Each job shows build, test, and analysis output

### Pipeline Status Badge

Add to README.md:
```markdown
[![CI Pipeline](https://github.com/YOUR_REPO/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/YOUR_REPO/actions/workflows/ci-pipeline.yml)
```

---

## Writing New Tests

### Test File Template

```cpp
#include <gtest/gtest.h>
#include "YourHeader.h"

class YourTestClass : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code run before each test
    }

    void TearDown() override {
        // Cleanup code run after each test
    }
    
    // Shared test fixtures
    YourClass testObject;
};

TEST_F(YourTestClass, TestName) {
    // Arrange
    int expected = 5;
    
    // Act
    int result = testObject.GetValue();
    
    // Assert
    EXPECT_EQ(result, expected);
}
```

### Google Test Assertions

```cpp
// Equality
EXPECT_EQ(a, b);      // Non-fatal
ASSERT_EQ(a, b);      // Fatal

// Inequality
EXPECT_NE(a, b);
ASSERT_NE(a, b);

// Comparison
EXPECT_LT(a, b);      // Less than
EXPECT_LE(a, b);      // Less or equal
EXPECT_GT(a, b);      // Greater than
EXPECT_GE(a, b);      // Greater or equal

// Boolean
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Floating point (with epsilon)
EXPECT_NEAR(a, b, epsilon);
EXPECT_FLOAT_EQ(a, b);
EXPECT_DOUBLE_EQ(a, b);

// Exceptions
EXPECT_THROW(statement, ExceptionType);
EXPECT_NO_THROW(statement);

// String comparison
EXPECT_STREQ(str1, str2);
EXPECT_STRNE(str1, str2);
```

### Example Test Suite

```cpp
#include <gtest/gtest.h>
#include "MyClass.h"

class MyClassTest : public ::testing::Test {
protected:
    MyClass obj;
};

// Test basic functionality
TEST_F(MyClassTest, BasicOperation) {
    EXPECT_EQ(obj.getValue(), 0);
}

// Test with setup
TEST_F(MyClassTest, OperationAfterInit) {
    obj.initialize(42);
    EXPECT_EQ(obj.getValue(), 42);
}

// Test edge cases
TEST_F(MyClassTest, EdgeCasesHandled) {
    EXPECT_NO_THROW(obj.handleNullptr(nullptr));
    EXPECT_THROW(obj.throwOnError(), std::runtime_error);
}

// Parametrized test
class MyClassParametrized : public ::testing::TestWithParam<int> {};

TEST_P(MyClassParametrized, MultipleValues) {
    MyClass obj;
    obj.setValue(GetParam());
    EXPECT_EQ(obj.getValue(), GetParam());
}

INSTANTIATE_TEST_SUITE_P(Values, MyClassParametrized,
    ::testing::Values(0, 1, 42, -1, 100)
);
```

### Adding New Test Files

1. Create test file in `tests/` directory: `test_mycomponent.cpp`
2. Add test executable source to `CMakeLists.txt`:
   ```cmake
   add_executable(tests
       tests/main.cpp
       tests/test_mycomponent.cpp  # Add here
   )
   ```
3. Add test case to CTest:
   ```cmake
   add_test(NAME mycomponent_tests COMMAND tests --gtest_filter=MyComponentTest.*)
   ```
4. Rebuild and run: `cmake --build . && ctest --output-on-failure`

---

## Best Practices

### Test Quality

- ✅ Write tests for critical functionality
- ✅ Test edge cases and error conditions
- ✅ Use descriptive test names
- ✅ Keep tests focused and independent
- ✅ Use setup/teardown for common code
- ✅ Test behavior, not implementation

### Performance

- ❌ Don't test OpenGL rendering directly (requires context)
- ✅ Mock or stub external dependencies
- ✅ Use lightweight test data
- ✅ Avoid expensive I/O in tests
- ✅ Test performance-critical code with benchmarks

### Maintenance

- 📝 Document complex test logic
- 🔄 Update tests when behavior changes
- 🧹 Remove disabled tests
- 📊 Monitor test coverage trends
- ✨ Keep test code clean and readable

---

## Troubleshooting

### Tests Won't Build

```bash
# Ensure GoogleTest is fetched
rm -rf build
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON

# Check compiler requirements (C++20)
cmake .. -DCMAKE_CXX_STANDARD=20
```

### Sanitizer Output Confusing

```bash
# Get more details
export ASAN_OPTIONS=verbosity=2:halt_on_error=1
./build/bin/tests
```

### Clang-Tidy Warnings

```bash
# Check specific file
clang-tidy src/MyFile.cpp -- -Iinclude -std=c++20

# Generate clang-tidy compile database
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Platform-Specific Issues

- **Windows**: Use `ctest --build-config Debug|Release`
- **Linux**: May need to install `libglfw3-dev libgl1-mesa-dev`
- **macOS**: Check Xcode command line tools: `xcode-select --install`

---

## Resources

- [Google Test Documentation](https://github.com/google/googletest)
- [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/)
- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [CTest Documentation](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
