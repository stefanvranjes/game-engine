#include "EditorMenuBar.h"
#include "EditorDockingManager.h"
#include "Logger.h"
#include <imgui.h>
#include <algorithm>
#include <ctime>

EditorMenuBar::EditorMenuBar()
    : m_ShowHierarchy(true)
    , m_ShowInspector(true)
    , m_ShowAssetBrowser(false)
    , m_ShowProfiler(false)
{
}

EditorMenuBar::~EditorMenuBar() {
}

void EditorMenuBar::Render() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            RenderFileMenu();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            RenderEditMenu();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            RenderViewMenu();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            RenderWindowMenu();
            ImGui::EndMenu();
        }
        // Layout Menu
        if (ImGui::BeginMenu("Layout")) {
             RenderLayoutMenu();
             ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            RenderHelpMenu();
            ImGui::EndMenu();
        }

        // Display FPS in menu bar
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextDisabled("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::EndMainMenuBar();
    }
}

void EditorMenuBar::RenderFileMenu() {
    if (ImGui::MenuItem("New Scene", "Ctrl+N")) {
        if (m_OnNewScene) m_OnNewScene();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Open Scene...", "Ctrl+O")) {
        if (m_OnOpenScene) m_OnOpenScene();
    }

    if (ImGui::MenuItem("Save Scene", "Ctrl+S")) {
        if (m_OnSaveScene) m_OnSaveScene();
    }

    if (ImGui::MenuItem("Save Scene As...", "Ctrl+Shift+S")) {
        if (m_OnSaveSceneAs) m_OnSaveSceneAs();
    }

    ImGui::Separator();

    // Recent Scenes
    if (!m_RecentScenes.empty() && ImGui::BeginMenu("Recent Scenes")) {
        for (const auto& scene : m_RecentScenes) {
            if (ImGui::MenuItem(scene.c_str())) {
                if (m_OnRecentScene) m_OnRecentScene(scene);
            }
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Exit", "Alt+F4")) {
        if (m_OnExit) m_OnExit();
    }
}

void EditorMenuBar::RenderEditMenu() {
    if (ImGui::MenuItem("Undo", "Ctrl+Z")) {
        if (m_OnUndo) m_OnUndo();
    }

    if (ImGui::MenuItem("Redo", "Ctrl+Y")) {
        if (m_OnRedo) m_OnRedo();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Delete", "Del")) {
        if (m_OnDelete) m_OnDelete();
    }

    if (ImGui::MenuItem("Select All", "Ctrl+A")) {
        if (m_OnSelectAll) m_OnSelectAll();
    }
}

void EditorMenuBar::RenderViewMenu() {
    // Toggle panels
    if (ImGui::MenuItem("Scene Hierarchy", nullptr, m_ShowHierarchy)) {
        m_ShowHierarchy = !m_ShowHierarchy;
        if (m_OnToggleHierarchy) m_OnToggleHierarchy(m_ShowHierarchy);
    }

    if (ImGui::MenuItem("Properties", nullptr, m_ShowInspector)) {
        m_ShowInspector = !m_ShowInspector;
        if (m_OnToggleInspector) m_OnToggleInspector(m_ShowInspector);
    }

    if (ImGui::MenuItem("Asset Browser", nullptr, m_ShowAssetBrowser)) {
        m_ShowAssetBrowser = !m_ShowAssetBrowser;
        if (m_OnToggleAssetBrowser) m_OnToggleAssetBrowser(m_ShowAssetBrowser);
    }

    if (ImGui::MenuItem("Performance Profiler", nullptr, m_ShowProfiler)) {
        m_ShowProfiler = !m_ShowProfiler;
        if (m_OnToggleProfiler) m_OnToggleProfiler(m_ShowProfiler);
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Reset Layout")) {
        if (m_OnResetLayout) m_OnResetLayout();
    }
}

void EditorMenuBar::RenderWindowMenu() {
    for (auto& window : m_RegisteredWindows) {
        if (ImGui::MenuItem(window.name.c_str(), nullptr, window.visible)) {
            window.visible = !window.visible;
            window.toggleCallback(window.visible);
        }
    }
}

void EditorMenuBar::RenderLayoutMenu() {
    if (m_DockingManager) {
        m_DockingManager->RenderLayoutSelector();
    } else {
        ImGui::TextDisabled("Layout manager not connected");
    }
}

void EditorMenuBar::RenderHelpMenu() {
    if (ImGui::MenuItem("Documentation")) {
        if (m_OnShowDocumentation) m_OnShowDocumentation();
    }

    if (ImGui::MenuItem("Keyboard Shortcuts", "?")) {
        if (m_OnShowShortcuts) m_OnShowShortcuts();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("About GameEngine")) {
        if (m_OnShowAbout) m_OnShowAbout();
    }
}

void EditorMenuBar::RegisterWindow(const std::string& name, std::function<void(bool)> toggleCallback) {
    WindowEntry entry;
    entry.name = name;
    entry.toggleCallback = toggleCallback;
    entry.visible = true;
    m_RegisteredWindows.push_back(entry);
}

void EditorMenuBar::UnregisterWindow(const std::string& name) {
    auto it = std::find_if(m_RegisteredWindows.begin(), m_RegisteredWindows.end(),
        [&name](const WindowEntry& entry) { return entry.name == name; });
    if (it != m_RegisteredWindows.end()) {
        m_RegisteredWindows.erase(it);
    }
}

void EditorMenuBar::AddRecentScene(const std::string& path) {
    // Remove if already exists
    auto it = std::find(m_RecentScenes.begin(), m_RecentScenes.end(), path);
    if (it != m_RecentScenes.end()) {
        m_RecentScenes.erase(it);
    }

    // Add to front
    m_RecentScenes.insert(m_RecentScenes.begin(), path);

    // Keep only last 10
    if (m_RecentScenes.size() > MAX_RECENT) {
        m_RecentScenes.resize(MAX_RECENT);
    }
}

bool EditorMenuBar::CheckShortcut(const char* keys) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Simple keyboard shortcut checking
    // Format: "Ctrl+S", "Shift+Alt+D", etc.
    // This is a simplified implementation; production code would be more robust
    
    return false; // Simplified for now
}
