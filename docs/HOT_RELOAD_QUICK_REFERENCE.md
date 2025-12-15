# Hot-Reload Quick Reference

## Enable Hot-Reload

```cpp
// Automatically enabled in Application::Init()
m_HotReloadManager->SetEnabled(true);
```

Or via ImGui panel:
- Open "Asset Hot-Reload" panel
- Toggle "Enable Hot-Reload##main"

## Edit Assets

**Shaders:**
1. Open shader file (`shaders/*.glsl`, `*.frag`, etc.)
2. Edit and save
3. Changes appear instantly

**Textures:**
1. Open image file in editor (`assets/textures/*.png`, etc.)
2. Edit and save
3. Texture updates automatically

## Monitor Status

### ImGui Panel
- **Enable Hot-Reload##main**: Toggle on/off
- **Status**: Shows ACTIVE or INACTIVE
- **Watched Files**: Number of files being monitored
- **Reloads**: Count of asset reloads
- **Watching**: Lists monitored directories

### Console Output
```
AssetHotReloadManager: Shader changed, reloading all shaders
TextureManager: Reloading texture: assets/textures/material.png
Shader reloaded successfully!
```

## Common Tasks

### Watch New Directory

```cpp
// In Application::Init() or runtime
m_HotReloadManager->WatchShaderDirectory("my_shaders/");
m_HotReloadManager->WatchTextureDirectory("my_assets/");
```

### Register Reload Callback

```cpp
m_HotReloadManager->SetOnAssetReloaded([](const std::string& type, const std::string& path) {
    std::cout << type << " reloaded: " << path << std::endl;
});
```

### Disable Hot-Reload

```cpp
m_HotReloadManager->SetEnabled(false);
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Changes not appearing | Check "Enable Hot-Reload##main" is toggled ON |
| Shader won't compile | Check console for errors, fix syntax, save again |
| Texture doesn't update | Ensure file is in watched directory, check file permissions |
| High CPU usage | Increase poll interval in `Update(500)` |
| Too many reloads | File system noise; ignore or increase poll interval |

## File Types Supported

| Type | Extensions |
|------|-----------|
| Shaders | `.glsl` `.vert` `.frag` `.geom` `.comp` `.tese` `.tesc` |
| Textures | `.png` `.jpg` `.jpeg` `.bmp` `.tga` `.hdr` `.exr` |

## Default Watched Directories

- `shaders/` - All shader types
- `assets/` - All texture types

## Key Classes

| Class | Purpose |
|-------|---------|
| `FileWatcher` | Low-level file monitoring |
| `TextureManager` | Texture asset loading and reloading |
| `AssetHotReloadManager` | Centralized hot-reload control |
| `Shader` | Shader compilation with hot-reload support |

## Update Frequency

- Call `m_HotReloadManager->Update()` once per frame in `Application::Update()`
- File system polling interval: 100ms (configurable)
- Texture uploads: Once per frame
- Shader compilation: On-demand when files change

## Performance Impact

- **Minimal**: ~1-2ms per frame for file polling
- **Shader compilation**: 5-50ms depending on complexity
- **Texture loading**: Async (no frame rate impact)
- **Disabled mode**: Zero overhead

## Development Tips

1. **Quick iteration**: Edit → Save → Instant feedback
2. **Test incrementally**: Make small changes, verify output
3. **Use external editors**: Photoshop, GIMP, Paint.NET work seamlessly
4. **Check console**: Errors and reloads logged automatically
5. **Monitor stats**: Use ImGui panel to track reload activity
