# Packaging Quick Reference

## 30-Second Setup

```bash
# Choose your preset
cmake --preset windows-msvc-release      # Windows
cmake --preset linux-gcc-release         # Linux
cmake --preset macos-clang-release       # macOS

# Build
cmake --build --preset <preset-name>

# Install
cmake --install build

# Package
cpack
```

---

## Build Presets

### Windows Presets

| Preset | Purpose | Command |
|--------|---------|---------|
| `windows-msvc-release` | MSVC Release | `cmake --preset windows-msvc-release` |
| `windows-msvc-debug` | MSVC Debug | `cmake --preset windows-msvc-debug` |
| `windows-clang-release` | Clang Release | `cmake --preset windows-clang-release` |
| `windows-sanitizer-asan` | AddressSanitizer | `cmake --preset windows-sanitizer-asan` |
| `windows-sanitizer-ubsan` | UB Sanitizer | `cmake --preset windows-sanitizer-ubsan` |
| `windows-clang-tidy` | Static Analysis | `cmake --preset windows-clang-tidy` |

### Linux Presets

| Preset | Purpose | Command |
|--------|---------|---------|
| `linux-gcc-release` | GCC Release | `cmake --preset linux-gcc-release` |
| `linux-gcc-debug` | GCC Debug | `cmake --preset linux-gcc-debug` |
| `linux-clang-release` | Clang Release | `cmake --preset linux-clang-release` |
| `linux-sanitizer-asan` | AddressSanitizer | `cmake --preset linux-sanitizer-asan` |
| `linux-sanitizer-ubsan` | UB Sanitizer | `cmake --preset linux-sanitizer-ubsan` |
| `linux-sanitizer-tsan` | Thread Sanitizer | `cmake --preset linux-sanitizer-tsan` |
| `linux-sanitizer-msan` | Memory Sanitizer | `cmake --preset linux-sanitizer-msan` |
| `linux-clang-tidy` | Static Analysis | `cmake --preset linux-clang-tidy` |

### macOS Presets

| Preset | Purpose | Command |
|--------|---------|---------|
| `macos-clang-release` | Clang Release | `cmake --preset macos-clang-release` |
| `macos-clang-debug` | Clang Debug | `cmake --preset macos-clang-debug` |

---

## Common Commands

### Configure & Build

```bash
# One-step build
cmake --preset windows-msvc-release && cmake --build --preset windows-msvc-release

# Separate steps
cmake --preset linux-gcc-release
cmake --build build/linux-release

# With parallel jobs
cmake --build build -j4
```

### Testing

```bash
# Run all tests
ctest --preset windows-release

# Run specific test
ctest -R "MathTest.*" --preset windows-release

# Verbose output
ctest --preset windows-release --verbose
```

### Installation

```bash
# Default location
cmake --install build

# Custom location
cmake --install build --prefix /custom/path

# Specific component
cmake --install build --component executable
```

### Packaging

```bash
# Create portable bundle
.\scripts\package.ps1 portable          # Windows PowerShell
./scripts/create-portable-bundle.sh     # Linux

# Create installers
cpack                                    # All formats
cpack -G NSIS                           # Windows installer
cpack -G DEB                            # Debian package
cpack -G RPM                            # RPM package
cpack -G TGZ                            # Tarball
cpack -G DragNDrop                      # macOS DMG
```

---

## Package Types

### Windows

| Type | Command | Output | Use Case |
|------|---------|--------|----------|
| **NSIS** | `cpack -G NSIS` | `.exe` installer | System installation |
| **ZIP** | `cpack -G ZIP` | `.zip` archive | Portable, USB |
| **Portable** | `.\scripts\package.ps1 portable` | Directory + launchers | USB drives, cloud |

**Example:**
```powershell
.\scripts\package.ps1 all     # Creates all three
```

### Linux

| Type | Command | Output | Use Case |
|------|---------|--------|----------|
| **DEB** | `cpack -G DEB` | `.deb` package | Ubuntu/Debian |
| **RPM** | `cpack -G RPM` | `.rpm` package | Fedora/RHEL |
| **TGZ** | `cpack -G TGZ` | `.tar.gz` archive | Any Linux |
| **Portable** | `./scripts/create-portable-bundle.sh` | `.tar.xz` + bundle | Any Linux |

**Example:**
```bash
cd build && cpack -G "DEB;RPM;TGZ"
```

### macOS

| Type | Command | Output | Use Case |
|------|---------|--------|----------|
| **DMG** | `cpack -G DragNDrop` | `.dmg` image | App Store style |
| **TGZ** | `cpack -G TGZ` | `.tar.gz` archive | Universal |

---

## Preset Selection Guide

### I want to...

**Build for release**
```bash
cmake --preset <platform>-release
```

**Build for development**
```bash
cmake --preset <platform>-debug
```

**Check for memory leaks**
```bash
cmake --preset <platform>-sanitizer-asan
```

**Find undefined behavior**
```bash
cmake --preset <platform>-sanitizer-ubsan
```

**Create portable bundle**
```bash
cmake --preset <platform>-release
cmake --build --preset <platform>-release
.\scripts\package.ps1 portable    # Windows
./scripts/create-portable-bundle.sh  # Linux
```

**Distribute to users**
```bash
# Windows
.\scripts\package.ps1 all

# Linux
cd build && cpack -G "DEB;RPM;TGZ"

# macOS
cd build && cpack -G DragNDrop
```

**Analyze code quality**
```bash
cmake --preset <platform>-clang-tidy
cmake --build --preset <platform>-clang-tidy
```

---

## Workflow Examples

### Basic Release Build

```powershell
# Windows PowerShell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
cmake --install build/release --prefix "C:\Program Files\GameEngine"
```

### Linux Distribution

```bash
# Configure and build
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release

# Create packages for distribution
cd build/linux-release
cpack -G DEB        # For Ubuntu/Debian
cpack -G RPM        # For Fedora/RHEL
cpack -G TGZ        # For any Linux

# Create portable bundle
cd ../..
./scripts/create-portable-bundle.sh
```

### Development with Testing

```bash
# Debug build with tests
cmake --preset linux-gcc-debug
cmake --build --preset linux-gcc-debug
ctest --preset linux-debug --verbose

# With sanitizers
cmake --preset linux-sanitizer-asan
cmake --build --preset linux-sanitizer-asan
ctest --preset linux-debug
```

### Complete Release Workflow

```bash
# 1. Configure release build
cmake --preset windows-msvc-release

# 2. Build
cmake --build --preset windows-msvc-release

# 3. Test thoroughly
ctest --preset windows-release --verbose

# 4. Install
cmake --install build/release

# 5. Create packages
.\scripts\package.ps1 all

# 6. Generate checksums
cd build\packages
certutil -hashfile GameEngine-*.exe SHA256 > checksums.sha256
certutil -hashfile GameEngine-*.zip SHA256 >> checksums.sha256

# 7. Create GitHub release with packages
# Upload to: https://github.com/YOUR_REPO/releases/new
```

---

## File Locations

After building:

```
game-engine/
├── build/
│   ├── Debug/
│   │   └── GameEngine.exe
│   ├── Release/
│   │   └── GameEngine.exe
│   ├── install/
│   │   └── (installed files)
│   └── packages/
│       ├── GameEngine-0.1.0-Windows-x86_64.exe
│       ├── GameEngine-0.1.0-Windows-x86_64.zip
│       └── GameEngine-portable/
│
├── build-linux/
│   └── packages/
│       ├── *.deb
│       ├── *.rpm
│       └── *.tar.gz
```

---

## Verification

### Check Built Binary

```bash
# Windows
dir build\Release\GameEngine.exe

# Linux
ls -lh build/linux-release/bin/GameEngine

# macOS
ls -lh build/macos-release/bin/GameEngine
```

### Verify Package Contents

```bash
# List DEB contents
dpkg -L /path/to/*.deb

# List RPM contents
rpm -ql /path/to/*.rpm

# Extract and check portable
tar -tzf GameEngine-*.tar.xz
```

### Test Installation

```bash
# Windows
msiexec /i GameEngine-installer.exe /quiet

# Linux
sudo dpkg -i GameEngine-*.deb

# Verify
which GameEngine
GameEngine --help
```

---

## Troubleshooting

### "Preset not found"
```bash
cmake --list-presets     # See available presets
cmake --preset <name>    # Use exact name
```

### "Build failed"
```bash
# Verbose output
cmake --build . --verbose

# Check dependencies
cmake --preset <name>    # Will show errors
```

### "Package creation failed"
```bash
# Verbose CPack
cd build && cpack -V

# Check generator available
cpack --help | grep -i generator
```

### "Installation failed - permission denied"
```bash
# Use sudo for system paths
sudo cmake --install build --prefix /usr/local

# Or use user path
cmake --install build --prefix ~/.local
```

### "Portable bundle not working"
```bash
# Check dependencies
ldd ./GameEngine                    # Linux
otool -L ./GameEngine              # macOS
dumpbin /imports GameEngine.exe     # Windows
```

---

## Tips & Tricks

💡 **List all presets** - `cmake --list-presets`

💡 **Build without reconfigure** - `cmake --build --preset <name>`

💡 **Parallel builds** - Add `-j4` to cmake --build command

💡 **Clean build** - `rm -rf build && cmake --preset ...`

💡 **See what installs** - `cmake --install build --verbose`

💡 **Create source package** - `cpack --config CPackSourceConfig.cmake`

💡 **Custom install path** - Use `-DCMAKE_INSTALL_PREFIX=/path`

💡 **Skip tests in package** - `cmake --preset <name> -DBUILD_TESTS=OFF`

---

## Resources

- [Full Packaging Guide](BUILD_PRESETS_PACKAGING.md)
- [CMakePresets Docs](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [CPack Docs](https://cmake.org/cmake/help/latest/manual/cpack.1.html)

---

**Quick Links**
- [Build Presets & Packaging](BUILD_PRESETS_PACKAGING.md) - Full guide
- [Getting Started](../GETTING_STARTED.md) - Setup guide
- [Testing Guide](TESTING_CI_GUIDE.md) - Test with presets

---

**Last Updated**: December 2024  
**Presets**: 18 available  
**Package Types**: 7 supported (Windows, Linux, macOS)  
**Status**: ✅ Ready to use
