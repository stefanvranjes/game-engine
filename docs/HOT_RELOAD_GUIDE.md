# Asset Hot-Reload System Documentation

## Overview

The Asset Hot-Reload system enables real-time reloading of shaders, textures, and other assets during development without restarting the engine. This dramatically speeds up iteration cycles when tuning visual effects, shaders, and textures.

## Features

- **Shader Hot-Reload**: Monitor shader files (`.glsl`, `.vert`, `.frag`, `.geom`, `.comp`) and automatically recompile when changes are detected
- **Texture Hot-Reload**: Watch texture directories and reload images (`.png`, `.jpg`, `.bmp`, `.hdr`, etc.) when modified
- **File Watcher**: Cross-platform file monitoring with directory recursion support
- **Editor Integration**: ImGui UI panel to monitor and control hot-reload status
- **Error Resilience**: Failed reloads preserve the previous working version
- **Real-time Feedback**: See changes instantly without engine restart

## Architecture

### Components

#### 1. FileWatcher ([FileWatcher.h](../include/FileWatcher.h))
Low-level file monitoring system that tracks file modifications.

**Key Methods:**
- `WatchFile(path, callback)` - Monitor a single file
- `WatchDirectory(directory, extension, callback)` - Recursively monitor a directory for specific file types
- `Update(pollIntervalMs)` - Poll for changes (call once per frame)
- `Unwatch(handle)` - Stop watching a file or directory

**Usage:**
```cpp
FileWatcher watcher;
auto handle = watcher.WatchFile("shaders/lighting.frag", [](const std::string& path) {
    std::cout << "File changed: " << path << std::endl;
});

// In game loop
watcher.Update(100);  // Check every 100ms

watcher.Unwatch(handle);  // Stop watching
```

#### 2. Enhanced TextureManager ([TextureManager.h](../include/TextureManager.h))
Extended to support automatic texture reloading.

**New Methods:**
- `SetHotReloadEnabled(bool enabled)` - Enable/disable hot-reload
- `WatchTextureDirectory(directory)` - Watch a directory for texture changes
- `SetOnTextureReloaded(callback)` - Register callback when textures reload

**Usage:**
```cpp
auto textureManager = renderer->GetTextureManager();
textureManager->SetHotReloadEnabled(true);
textureManager->WatchTextureDirectory("assets/textures/");
textureManager->SetOnTextureReloaded([](const std::string& path) {
    std::cout << "Texture reloaded: " << path << std::endl;
});
```

#### 3. AssetHotReloadManager ([AssetHotReloadManager.h](../include/AssetHotReloadManager.h))
High-level centralized manager for all asset hot-reload functionality.

**Key Methods:**
- `Initialize(renderer, textureManager)` - Set up the manager
- `SetEnabled(bool)` - Enable/disable hot-reload globally
- `WatchShaderDirectory(path)` - Monitor shader directory
- `WatchTextureDirectory(path)` - Monitor texture directory
- `Update()` - Call once per frame from application loop
- `SetOnAssetReloaded(callback)` - Register callback for all asset changes

**Usage:**
```cpp
// In Application::Init()
m_HotReloadManager = std::make_unique<AssetHotReloadManager>();
m_HotReloadManager->Initialize(m_Renderer.get(), m_Renderer->GetTextureManager());
m_HotReloadManager->SetEnabled(true);
m_HotReloadManager->WatchShaderDirectory("shaders/");
m_HotReloadManager->WatchTextureDirectory("assets/");

m_HotReloadManager->SetOnAssetReloaded([](const std::string& type, const std::string& path) {
    if (type == "shader") {
        std::cout << "Shader reloaded: " << path << std::endl;
    } else if (type == "texture") {
        std::cout << "Texture reloaded: " << path << std::endl;
    }
});

// In Application::Update()
m_HotReloadManager->Update();
```

#### 4. Enhanced Shader ([Shader.h](../include/Shader.h))
Existing `CheckForUpdates()` method now automatically reloads shaders when files change.

```cpp
// In Renderer::UpdateShaders()
void Renderer::UpdateShaders() {
    if (m_Shader) m_Shader->CheckForUpdates();
    if (m_DepthShader) m_DepthShader->CheckForUpdates();
    // ... other shaders
}
```

## Integration Points

### Application Class
The hot-reload system is integrated into the main `Application` class:

1. **Initialization** (Application::Init):
```cpp
m_HotReloadManager = std::make_unique<AssetHotReloadManager>();
m_HotReloadManager->Initialize(m_Renderer.get(), m_Renderer->GetTextureManager());
m_HotReloadManager->SetEnabled(true);
m_HotReloadManager->WatchShaderDirectory("shaders/");
m_HotReloadManager->WatchTextureDirectory("assets/");
```

2. **Update** (Application::Update):
```cpp
if (m_HotReloadManager) {
    m_HotReloadManager->Update();
}
```

3. **Editor UI** (Application::RenderEditorUI):
An "Asset Hot-Reload" ImGui panel displays:
- Enable/disable toggle
- Current status (ACTIVE/INACTIVE)
- Number of watched files
- Reload count statistics
- Information about watched directories

## Supported File Types

### Shaders
- `.glsl` - Generic OpenGL Shader Language
- `.vert` - Vertex shaders
- `.frag` - Fragment shaders
- `.geom` - Geometry shaders
- `.comp` - Compute shaders
- `.tese` / `.tesc` - Tessellation shaders

### Textures
- `.png` - PNG images
- `.jpg`, `.jpeg` - JPEG images
- `.bmp` - Bitmap images
- `.tga` - TGA images
- `.hdr` - HDR images
- `.exr` - OpenEXR images

## Usage Workflows

### Shader Development Workflow

1. **Edit a shader file** in your editor (e.g., `shaders/lighting.frag`)
2. **Save the file**
3. **See the change immediately** in the engine - no restart needed
4. **If compilation fails**, the previous working shader is preserved and an error message is printed

Example:
```glsl
// shaders/lighting.frag
uniform vec3 u_LightColor;
void main() {
    // Edit this...
    gl_FragColor = vec4(u_LightColor * 0.5, 1.0);  // Changes to 0.5
    // Save and see immediate feedback
}
```

### Texture Iteration Workflow

1. **Edit or replace texture files** in `assets/` or subdirectories
2. **Save the texture**
3. **Textures automatically reload** and appear updated in real-time
4. Objects using the texture display the new version without restart

Example:
```
assets/textures/metallic_base.png → Edit in Photoshop → Save
// Engine automatically reloads and displays updated texture
```

### Multiple Asset Workflow

Edit shaders, textures, and materials together:
1. Modify `shaders/pbr.frag`
2. Update `assets/textures/material.png`
3. Adjust properties in engine UI
4. All changes propagate instantly without restart

## Configuration

### Enable/Disable Hot-Reload

Via code:
```cpp
m_HotReloadManager->SetEnabled(true);  // Enable
m_HotReloadManager->SetEnabled(false); // Disable
```

Via ImGui Editor Panel:
- Check the "Enable Hot-Reload##main" checkbox in the "Asset Hot-Reload" panel

### Watch Custom Directories

```cpp
// Watch additional directories
m_HotReloadManager->WatchShaderDirectory("my_custom_shaders/");
m_HotReloadManager->WatchTextureDirectory("my_custom_assets/");
```

### Adjust Poll Interval

The default file poll interval is 100ms. To change:
```cpp
// In AssetHotReloadManager::Update()
m_FileWatcher->Update(200);  // Check every 200ms instead
```

## Error Handling

### Shader Compilation Failures

If a shader fails to compile:
1. Previous working shader is preserved
2. Error message is printed to console: "Shader reload failed! Keeping old shader."
3. Fix the shader and save again
4. The corrected version reloads automatically

Example:
```
>> Reloading shader...
>> Shader compilation failed (FRAGMENT):
>> 0:10(5): error: expected '(' syntax error
>> Shader reload failed! Keeping old shader.
// Fix the error and save...
>> Reloading shader...
>> Shader reloaded successfully!
```

### Texture Load Failures

If a texture fails to load:
1. Previous texture is preserved
2. Error logged to console
3. Attempts to reload on next modification

## Performance Considerations

### File Polling
- Default 100ms poll interval balances responsiveness and CPU usage
- Increase interval for slower systems
- Decrease for more responsive feedback

### Texture Loading
- Async loading prevents frame rate drops
- Texture uploads happen on main thread during `Update()`
- Large textures may take multiple frames to appear

### Shader Compilation
- Compilation happens on main thread
- Simple shaders compile in <10ms
- Complex shaders with includes may take longer
- No frame rate impact on most modern GPUs

## Best Practices

1. **Organize assets hierarchically**:
   ```
   assets/
   ├── textures/
   ├── materials/
   └── models/
   
   shaders/
   ├── lighting/
   ├── post_processing/
   └── common/
   ```

2. **Use meaningful file names**:
   - `metallic_roughness.png` instead of `tex1.png`
   - `pbr_lighting.frag` instead of `shader.frag`

3. **Test shader changes incrementally**:
   - Make small changes
   - Verify output
   - Save and check result
   - Iterate quickly

4. **Use external texture editors**:
   - Paint.NET, Photoshop, GIMP all support "save" operations
   - Changes are picked up automatically
   - No need to reload or restart engine

5. **Monitor the reload count**:
   - Check the "Reloads" statistic in the ImGui panel
   - Unexpected high counts indicate file system noise
   - Normal development might show 10-50 reloads per minute

## Debugging

### Check Watched Files

```cpp
// Get list of all watched files
auto watched = m_HotReloadManager->GetWatchedFileCount();
std::cout << "Watching " << watched << " files" << std::endl;
```

### Monitor Reload Events

Register a callback to track all reloads:
```cpp
m_HotReloadManager->SetOnAssetReloaded([](const std::string& type, const std::string& path) {
    std::cout << "[" << type << "] " << path << " reloaded" << std::endl;
});
```

### Check File Watcher Status

The ImGui panel shows:
- Current enable status
- Number of watched files
- Total reload count
- List of monitored directories

### Console Output

Look for messages like:
```
AssetHotReloadManager: Hot-reload ENABLED
AssetHotReloadManager: Watching shader directory: shaders/
AssetHotReloadManager: Watching texture directory: assets/
TextureManager: Watching directory for hot-reload: assets/
TextureManager: Reloading texture: assets/textures/material.png
AssetHotReloadManager: Shader changed, reloading all shaders
```

## Limitations

1. **File moves/renames**: Watched paths must exist when watching starts
2. **Network locations**: May have higher latency, increase poll interval
3. **Circular includes**: Shader includes are not tracked individually
4. **Asset references**: Materials must be relinked if texture paths change
5. **Binary formats**: Only supports text-based shaders and standard image formats

## Future Enhancements

Potential improvements for future versions:
- Material hot-reload (`.mat` files)
- Model/mesh hot-reload (`.gltf`, `.fbx`)
- Animation hot-reload
- Shader include dependency tracking
- Automatic material reassignment
- Hot-reload statistics visualization

## Related Classes

- [FileWatcher.h](../include/FileWatcher.h) - Low-level file monitoring
- [TextureManager.h](../include/TextureManager.h) - Texture asset management
- [Shader.h](../include/Shader.h) - Shader compilation and management
- [Renderer.h](../include/Renderer.h) - Main rendering system
- [Application.h](../include/Application.h) - Main application loop

## See Also

- [Renderer Documentation](./API_OVERVIEW.md) - Rendering pipeline overview
- [Asset Management Guide](./DOCUMENTATION_GUIDE.md) - General asset workflow
