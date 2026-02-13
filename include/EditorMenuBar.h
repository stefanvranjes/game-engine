#pragma once

#include <string>
#include <functional>
#include <vector>

/**
 * @brief Main menu bar for the editor with File, Edit, View, Window, and Help menus
 * 
 * Provides a professional menu structure for accessing editor functionality.
 * All menu items call registered callbacks for specific operations.
 */
class EditorMenuBar {
public:
    EditorMenuBar();
    ~EditorMenuBar();

    /**
     * @brief Render the main menu bar
     */
    void Render();

    // File Menu Callbacks
    void SetOnNewScene(std::function<void()> callback) { m_OnNewScene = callback; }
    void SetOnOpenScene(std::function<void()> callback) { m_OnOpenScene = callback; }
    void SetOnSaveScene(std::function<void()> callback) { m_OnSaveScene = callback; }
    void SetOnSaveSceneAs(std::function<void()> callback) { m_OnSaveSceneAs = callback; }
    void SetOnRecentScene(std::function<void(const std::string&)> callback) { m_OnRecentScene = callback; }
    void SetOnExit(std::function<void()> callback) { m_OnExit = callback; }

    // Edit Menu Callbacks
    void SetOnUndo(std::function<void()> callback) { m_OnUndo = callback; }
    void SetOnRedo(std::function<void()> callback) { m_OnRedo = callback; }
    void SetOnDelete(std::function<void()> callback) { m_OnDelete = callback; }
    void SetOnSelectAll(std::function<void()> callback) { m_OnSelectAll = callback; }

    // View Menu Callbacks
    void SetOnToggleHierarchy(std::function<void(bool)> callback) { m_OnToggleHierarchy = callback; }
    void SetOnToggleInspector(std::function<void(bool)> callback) { m_OnToggleInspector = callback; }
    void SetOnToggleAssetBrowser(std::function<void(bool)> callback) { m_OnToggleAssetBrowser = callback; }
    void SetOnToggleProfiler(std::function<void(bool)> callback) { m_OnToggleProfiler = callback; }
    void SetOnResetLayout(std::function<void()> callback) { m_OnResetLayout = callback; }

    // Window Menu - auto-populated with available windows
    void RegisterWindow(const std::string& name, std::function<void(bool)> toggleCallback);
    void UnregisterWindow(const std::string& name);

    // Help Menu Callbacks
    void SetOnShowDocumentation(std::function<void()> callback) { m_OnShowDocumentation = callback; }
    void SetOnShowAbout(std::function<void()> callback) { m_OnShowAbout = callback; }
    void SetOnShowShortcuts(std::function<void()> callback) { m_OnShowShortcuts = callback; }

    // UI Visibility Toggles
    bool IsHierarchyVisible() const { return m_ShowHierarchy; }
    bool IsInspectorVisible() const { return m_ShowInspector; }
    bool IsAssetBrowserVisible() const { return m_ShowAssetBrowser; }
    bool IsProfilerVisible() const { return m_ShowProfiler; }

    void SetHierarchyVisible(bool visible) { m_ShowHierarchy = visible; }
    void SetInspectorVisible(bool visible) { m_ShowInspector = visible; }
    void SetAssetBrowserVisible(bool visible) { m_ShowAssetBrowser = visible; }
    void SetProfilerVisible(bool visible) { m_ShowProfiler = visible; }

    // Recent scenes management
    void AddRecentScene(const std::string& path);
    const std::vector<std::string>& GetRecentScenes() const { return m_RecentScenes; }

private:
    void RenderFileMenu();
    void RenderEditMenu();
    void RenderViewMenu();
    void RenderWindowMenu();
    void RenderHelpMenu();

    // File Menu Callbacks
    std::function<void()> m_OnNewScene;
    std::function<void()> m_OnOpenScene;
    std::function<void()> m_OnSaveScene;
    std::function<void()> m_OnSaveSceneAs;
    std::function<void(const std::string&)> m_OnRecentScene;
    std::function<void()> m_OnExit;

    // Edit Menu Callbacks
    std::function<void()> m_OnUndo;
    std::function<void()> m_OnRedo;
    std::function<void()> m_OnDelete;
    std::function<void()> m_OnSelectAll;

    // View Menu Callbacks
    std::function<void(bool)> m_OnToggleHierarchy;
    std::function<void(bool)> m_OnToggleInspector;
    std::function<void(bool)> m_OnToggleAssetBrowser;
    std::function<void(bool)> m_OnToggleProfiler;
    std::function<void()> m_OnResetLayout;

    // Help Menu Callbacks
    std::function<void()> m_OnShowDocumentation;
    std::function<void()> m_OnShowAbout;
    std::function<void()> m_OnShowShortcuts;

    // Panel visibility states
    bool m_ShowHierarchy = true;
    bool m_ShowInspector = true;
    bool m_ShowAssetBrowser = false;
    bool m_ShowProfiler = false;

    // Registered custom windows
    struct WindowEntry {
        std::string name;
        std::function<void(bool)> toggleCallback;
        bool visible = true;
    };
    std::vector<WindowEntry> m_RegisteredWindows;

    // Recent scenes (max 10)
    std::vector<std::string> m_RecentScenes;
    static const size_t MAX_RECENT = 10;

    // Keyboard shortcuts
    bool CheckShortcut(const char* keys);
};
