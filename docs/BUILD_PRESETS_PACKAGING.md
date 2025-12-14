# Build Presets & Packaging Guide

## Table of Contents

1. [Build Presets Overview](#build-presets-overview)
2. [CMakePresets.json](#cmakepresetsjson)
3. [Installation Configuration](#installation-configuration)
4. [Packaging Options](#packaging-options)
5. [Portable Bundles](#portable-bundles)
6. [Distribution Methods](#distribution-methods)
7. [Quick Start Commands](#quick-start-commands)
8. [Advanced Topics](#advanced-topics)

---

## Build Presets Overview

**CMakePresets** provide standardized build configurations that ensure consistent builds across different machines and CI systems.

### Key Benefits

✅ **Consistency** - Same settings across all developers and CI
✅ **Simplicity** - No need to remember complex CMake flags
✅ **Cross-platform** - Platform-specific configurations built-in
✅ **IDE Integration** - VS Code, Visual Studio, CLion support
✅ **Documentation** - Presets self-document build options

---

## CMakePresets.json

The project includes 18 pre-configured build presets across Windows, Linux, and macOS.

### Preset Categories

#### Windows (7 presets)

**Release Builds:**
- `windows-msvc-release` - Visual Studio 2022 Release build
- `windows-clang-release` - Clang Release build for Windows

**Debug Builds:**
- `windows-msvc-debug` - Visual Studio 2022 Debug build

**Analysis:**
- `windows-sanitizer-asan` - AddressSanitizer configuration
- `windows-sanitizer-ubsan` - UndefinedBehavior Sanitizer
- `windows-clang-tidy` - Clang-Tidy static analysis

#### Linux (7 presets)

**Release Builds:**
- `linux-gcc-release` - GCC Release build
- `linux-clang-release` - Clang Release build

**Debug Builds:**
- `linux-gcc-debug` - GCC Debug build

**Sanitizers:**
- `linux-sanitizer-asan` - AddressSanitizer
- `linux-sanitizer-ubsan` - UndefinedBehavior Sanitizer
- `linux-sanitizer-tsan` - Thread Sanitizer
- `linux-sanitizer-msan` - Memory Sanitizer (Clang only)

**Analysis:**
- `linux-clang-tidy` - Clang-Tidy static analysis

#### macOS (4 presets)

**Release & Debug:**
- `macos-clang-release` - Apple Clang Release
- `macos-clang-debug` - Apple Clang Debug

### Using Build Presets

#### In Visual Studio Code

1. Install CMake Tools extension
2. Press `Ctrl+Shift+P` → "CMake: Select a kit"
3. Select your compiler
4. Press `Ctrl+Shift+P` → "CMake: Select Configure Preset"
5. Choose preset (e.g., `windows-msvc-release`)
6. Build with `Ctrl+Shift+P` → "CMake: Build"

#### Command Line

**Configure:**
```bash
cmake --preset windows-msvc-release
```

**Build:**
```bash
cmake --build --preset windows-msvc-release
```

**Test:**
```bash
ctest --preset windows-release
```

#### Visual Studio 2022

1. Open project in Visual Studio
2. Select preset from "Project" menu
3. Build/Run normally

---

## Installation Configuration

The project includes comprehensive install targets configured in CMakeLists.txt.

### Install Targets

**Executable:**
```cmake
install(TARGETS GameEngine
    RUNTIME DESTINATION bin
)
```

**Assets:**
```cmake
install(DIRECTORY assets/
    DESTINATION bin/assets
)
```

**Shaders:**
```cmake
install(DIRECTORY shaders/
    DESTINATION bin/shaders
)
```

**Documentation:**
```cmake
install(FILES README.md GETTING_STARTED.md ...
    DESTINATION share/doc/GameEngine
)
```

**Headers (for SDK):**
```cmake
install(DIRECTORY include/
    DESTINATION include/GameEngine
)
```

### Install Prefixes

Different presets use different install prefixes:

| Preset | Prefix |
|--------|--------|
| Windows MSVC | `${sourceDir}/install/release` |
| Linux GCC | `/usr/local/GameEngine` |
| macOS Clang | `/usr/local/GameEngine` |

### Custom Install Location

```bash
# Override install prefix
cmake --preset windows-msvc-release \
    -DCMAKE_INSTALL_PREFIX="C:/GameEngine"

cmake --build . --target install
```

---

## Packaging Options

### Windows Packaging

#### NSIS Installer (Native Installation)

CPack automatically generates NSIS installers:

```bash
cd build
cpack -G NSIS
```

**Creates:**
- `GameEngine-0.1.0-Windows-x86_64.exe` (~200MB)
- Installs to Program Files
- Adds Start Menu shortcuts
- Includes uninstaller

**Configuration in CMakeLists.txt:**
```cmake
set(CPACK_NSIS_PACKAGE_NAME "Game Engine")
set(CPACK_NSIS_INSTALL_ROOT "C:\\Program Files")
set(CPACK_NSIS_CREATE_ADMINISTRATORS_GROUP TRUE)
```

#### ZIP Portable Bundle

```bash
cd build
cpack -G ZIP
```

**Creates:**
- `GameEngine-0.1.0-Windows-x86_64.zip` (~150MB)
- No installation required
- Works from USB drives
- Keep folder structure intact

### Linux Packaging

#### DEB Package (Debian/Ubuntu)

```bash
cd build
cpack -G DEB
```

**Creates:**
- `GameEngine-0.1.0-Linux-x86_64.deb`
- Install: `sudo dpkg -i GameEngine-*.deb`
- Manages dependencies automatically

#### RPM Package (RedHat/CentOS/Fedora)

```bash
cd build
cpack -G RPM
```

**Creates:**
- `GameEngine-0.1.0-Linux-x86_64.rpm`
- Install: `sudo rpm -i GameEngine-*.rpm`

#### TGZ Archive

```bash
cd build
cpack -G TGZ
```

**Creates:**
- `GameEngine-0.1.0-Linux-x86_64.tar.gz`
- Portable, works on any Linux
- Extract: `tar -xzf GameEngine-*.tar.gz`

### macOS Packaging

#### DMG Installer (Disk Image)

```bash
cd build
cpack -G DragNDrop
```

**Creates:**
- `GameEngine-0.1.0-Darwin-x86_64.dmg`
- Familiar Mac installation experience
- Drag to Applications folder

#### TGZ Archive

```bash
cd build
cpack -G TGZ
```

---

## Portable Bundles

Portable bundles are self-contained applications that work on any system without installation.

### Creating Portable Bundles

#### Linux Portable Bundle

```bash
./scripts/create-portable-bundle.sh
```

**Creates:**
- Self-contained directory with all dependencies
- Relocatable to any location
- Works on USB drives
- Tarball archive: `GameEngine-0.1.0-linux-x86_64-portable.tar.xz`

**Script features:**
- Analyzes dependencies with `ldd`
- Copies all required libraries
- Creates launcher script
- Generates checksums

**Usage:**
```bash
tar -xJf GameEngine-0.1.0-linux-x86_64-portable.tar.xz
./GameEngine-portable/GameEngine
```

#### Windows Portable Bundle

**PowerShell:**
```powershell
.\scripts\package.ps1 portable
```

**Batch:**
```cmd
.\scripts\package.bat portable
```

**Creates:**
- `GameEngine-portable/` directory
- `GameEngine.bat` launcher
- `GameEngine-silent.vbs` silent launcher
- All assets and shaders included

**Usage:**
```
GameEngine-portable\GameEngine.bat
```

### Portable Bundle Features

✅ **No Installation** - Extract and run
✅ **Relocatable** - Move folder anywhere
✅ **USB Friendly** - Works from USB drives
✅ **Self-Contained** - All dependencies included
✅ **Fast Distribution** - Small download size
✅ **No Admin Rights** - User-level execution

---

## Distribution Methods

### Method 1: GitHub Releases

Create GitHub releases with different package types:

```bash
# Tag version
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Upload packages as release assets
# - GameEngine-0.1.0-Windows-x86_64.exe
# - GameEngine-0.1.0-Windows-x86_64.zip
# - GameEngine-0.1.0-Linux-x86_64.deb
# - GameEngine-0.1.0-Linux-x86_64.rpm
# - GameEngine-0.1.0-Darwin-x86_64.dmg
```

### Method 2: Package Managers

#### Linux - Submit to Package Repositories

- **Ubuntu/Debian**: Ubuntu Personal Package Archives (PPA)
- **Fedora**: Fedora Package Collection
- **AUR**: Arch User Repository

#### Windows - Chocolatey Package

```powershell
# Create package
choco pack

# Submit to Chocolatey Community Repository
choco apikey -k YOUR_API_KEY -source https://push.chocolatey.org/

# Push package
choco push GameEngine.0.1.0.nupkg --source https://push.chocolatey.org/
```

### Method 3: Direct Download

Host packages on your website:
- Portable bundles (most compatible)
- Platform-specific installers
- Checksums for verification

### Method 4: Container Images

Docker image for containerized deployment:

```dockerfile
FROM ubuntu:22.04
RUN apt-get install -y libglfw3 libgl1-mesa0
COPY GameEngine /opt/GameEngine
ENTRYPOINT ["/opt/GameEngine/bin/GameEngine"]
```

---

## Quick Start Commands

### Build with Preset

```bash
# Configure with preset
cmake --preset windows-msvc-release

# Build
cmake --build --preset windows-msvc-release

# Test
ctest --preset windows-release

# Install
cmake --install build/release
```

### Create Package

**Windows - PowerShell:**
```powershell
# Portable bundle
.\scripts\package.ps1 portable

# All packages
.\scripts\package.ps1 all
```

**Windows - Batch:**
```cmd
# Portable bundle
.\scripts\package.bat portable

# All packages
.\scripts\package.bat all
```

**Linux:**
```bash
# Portable bundle
./scripts/create-portable-bundle.sh

# CPack packages
cd build
cpack -G "DEB;RPM;TGZ"
```

### Full Workflow

```bash
# 1. Configure
cmake --preset linux-gcc-release

# 2. Build
cmake --build --preset linux-gcc-release

# 3. Install
cmake --install build/linux-release

# 4. Package
cd build/linux-release
cpack -G DEB
cpack -G RPM
cpack -G TGZ

# 5. Create portable bundle
./scripts/create-portable-bundle.sh
```

---

## Advanced Topics

### Custom Presets

Add custom presets to CMakePresets.json:

```json
{
  "name": "my-custom-preset",
  "displayName": "My Custom Configuration",
  "inherits": "windows-msvc-release",
  "cacheVariables": {
    "MY_OPTION": "ON",
    "CMAKE_INSTALL_PREFIX": "C:/MyCustomPath"
  }
}
```

### Custom Install Locations

Override installation paths:

```bash
cmake --preset linux-gcc-release \
    -DCMAKE_INSTALL_PREFIX=/custom/path

cmake --install build
```

### Conditional Installation

Install only specific components:

```bash
cmake --install build --component executable
cmake --install build --component documentation
```

### Post-Install Scripts

Add custom post-installation scripts:

```cmake
install(SCRIPT post_install.cmake)
```

### Package Component Selection

```bash
cd build
cpack -D CPACK_COMPONENTS_ALL="executable;documentation"
```

### CPack Configuration

Edit `CMakeLists.txt` to customize:

```cmake
# Change package name
set(CPACK_PACKAGE_NAME "MyGameEngine")

# Change version
set(CPACK_PACKAGE_VERSION "1.0.0")

# Add custom maintainer
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "email@example.com")

# Custom installation directory
set(CPACK_INSTALL_PREFIX "/opt/mygame")
```

### Generating Checksums

After packaging:

```bash
cd build
sha256sum GameEngine-*.* > SHA256SUMS
md5sum GameEngine-*.* > MD5SUMS
```

For verification:
```bash
sha256sum -c SHA256SUMS
```

### Platform Detection

Presets automatically detect platform:

```json
"condition": {
  "type": "equals",
  "lhs": "${hostSystemName}",
  "rhs": "Windows"
}
```

Supported values:
- `Windows` - Windows systems
- `Linux` - Linux systems
- `Darwin` - macOS systems

---

## Troubleshooting

### Preset Not Found

```bash
# List available presets
cmake --list-presets

# Use full path
cmake --preset windows-msvc-release
```

### NSIS Not Found

Install NSIS from: https://nsis.sourceforge.io/

Or skip NSIS:
```bash
cpack -G ZIP
```

### Dependencies Not Included

Check portable bundle dependencies:
```bash
# Linux
ldd ./build/bin/GameEngine
```

Add missing libraries manually to bundle.

### Installation Permissions

For system-wide installation:
```bash
sudo cmake --install build --prefix /usr/local
```

### Installer Size Too Large

Strip symbols to reduce size:
```cmake
set(CPACK_STRIP_FILES TRUE)
```

---

## Best Practices

✅ **Use Presets** - Don't memorize CMake flags
✅ **Test Each Package Type** - Some users may use different installers
✅ **Provide Checksums** - Users can verify downloads
✅ **Document Requirements** - System dependencies, graphics drivers, etc.
✅ **Include README** - Quick start guide in packages
✅ **Version Everything** - Track version across packages
✅ **Test Installation** - Install on fresh system to verify
✅ **Sign Packages** - Use cryptographic signatures for security

---

## Version Info

Current Version: `0.1.0`

Update in three places:
1. `CMakeLists.txt` - `project(GameEngine VERSION 0.1.0)`
2. `CMakePresets.json` - Individual preset versions
3. Package metadata

Semantic Versioning:
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

---

## Resources

- [CMakePresets.json Docs](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [CPack Documentation](https://cmake.org/cmake/help/latest/manual/cpack.1.html)
- [NSIS Installer](https://nsis.sourceforge.io/)
- [Packaging Best Practices](https://cmake.org/cmake/help/latest/manual/cpack-generators.7.html)

---

## Support

For packaging issues:
1. Check CMakeLists.txt install() targets
2. Verify CMakePresets.json syntax
3. Run cpack with verbose flag: `cpack -V`
4. Check GitHub releases for examples

---

**Last Updated**: December 2024  
**Status**: ✅ Ready for Production  
**Tested On**: Windows, Linux, macOS
