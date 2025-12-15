# Asset Pipeline Build Script

The `build_with_assets.bat` script automates the build process with integrated asset pipeline processing.

## Usage

### Basic Build (Incremental)
```batch
build_with_assets.bat
```
- Builds game engine in Debug mode
- Processes only changed assets
- Fast iteration for development

### Full Rebuild
```batch
build_with_assets.bat full-rebuild
```
- Full CMake configuration
- Reprocesses all assets
- Use when assets structure changes

### Clean Build
```batch
build_with_assets.bat clean
```
- Removes all processed assets
- Clears asset database
- Forces full asset reprocessing

### Compressed Build (Pre-ship)
```batch
build_with_assets.bat compress
```
- Enables texture/mesh compression
- Maximum quality optimization
- Longer processing time
- Smaller output size

### Combined Options
```batch
build_with_assets.bat clean full-rebuild compress
```
- Maximum processing: clean, rebuild, optimize
- Use before shipping

## Build Stages

1. **Directory Validation**
   - Checks assets/ exists
   - Creates build/ if needed

2. **Clean (if requested)**
   - Removes processed assets
   - Clears database
   - Starts fresh

3. **CMake Configuration**
   - Generates Visual Studio solution
   - Configures build system

4. **Engine Build**
   - Compiles C++ source files
   - Links game engine executable
   - Parallel build with all cores

5. **Asset Processing**
   - Scans asset directory
   - Detects changed assets
   - Converts formats
   - Validates integrity
   - Builds database

6. **Game Launch** (optional)
   - Runs executable if not no-run flag

## Environment Variables

The script uses these internal variables:

- `BUILD_DIR` = `build/`
- `ASSETS_DIR` = `assets/`
- `ASSETS_PROCESSED` = `assets/.processed/`
- `DATABASE_FILE` = `assets/.database.json`

## Output Structure

After successful build:

```
game-engine/
├── build/
│   ├── Debug/
│   │   └── GameEngine.exe
│   └── CMakeCache.txt
├── assets/
│   ├── textures/
│   ├── models/
│   ├── shaders/
│   ├── .processed/          ← Converted assets
│   │   ├── textures/
│   │   ├── models/
│   │   └── shaders/
│   └── .database.json       ← Asset metadata
└── shaders/
```

## Return Codes

- `0` = Success
- `1` = Build failed
- `1` = CMake configuration error
- `1` = Asset processing error

## Examples

### Development Workflow
```batch
REM Initial setup
build_with_assets.bat clean

REM Iterate (incremental)
build_with_assets.bat

REM After asset structure changes
build_with_assets.bat full-rebuild
```

### Before Release
```batch
REM Optimize for shipping
build_with_assets.bat clean compress

REM Verify no errors
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)
```

### Continuous Integration
```batch
REM On every commit
build_with_assets.bat clean full-rebuild

REM Validate
build_with_assets.bat no-run
```

## Troubleshooting

### Build Fails After Changes
```batch
build_with_assets.bat clean
build_with_assets.bat full-rebuild
```

### Asset Format Issues
```batch
REM Full reprocessing with validation
build_with_assets.bat clean
build_with_assets.bat compress
```

### Out of Memory
Edit the script to reduce threads:
```batch
REM In build_with_assets.bat, modify pipeline config
REM config.maxThreads = 2;  (instead of 4)
```

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Build with Asset Pipeline
  run: build_with_assets.bat clean full-rebuild
```

### Azure Pipelines
```yaml
- script: build_with_assets.bat clean full-rebuild
  displayName: 'Build with Asset Pipeline'
```

### Jenkins
```groovy
stage('Build') {
    steps {
        bat 'build_with_assets.bat clean full-rebuild'
    }
}
```

## Performance Notes

- **Incremental builds**: 2-10 seconds
- **Full rebuild (small project)**: 30-60 seconds
- **Compressed build**: 1-5 minutes
- **Thread count**: Auto-detected from CPU cores

## Next Steps

After successful build:
1. Run `GameEngine.exe` to verify
2. Check `assets/.database.json` for asset metadata
3. Review `assets/.processed/` for converted assets
4. Commit database file to version control

See [ASSET_PIPELINE_GUIDE.md](ASSET_PIPELINE_GUIDE.md) for detailed documentation.
