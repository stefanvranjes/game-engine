#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * @class EditorDockingManager
 * @brief Manages ImGui docking system and layout presets for the editor
 * 
 * Provides flexible window docking, layout save/load, and preset management.
 * Enables users to arrange editor panels (hierarchy, inspector, profiler, etc.)
 * in customizable layouts.
 * 
 * Usage:
 *   m_DockingManager = std::make_unique<EditorDockingManager>();
 *   m_DockingManager->Initialize();
 *   m_DockingManager->BeginDockspace();
 *   
 *   // Render windows inside dockspace...
 *   ImGui::SetNextWindowDockID(m_DockingManager->GetDockID("left"), ImGuiCond_FirstUseEver);
 *   ImGui::Begin("Hierarchy");
 *   // Render hierarchy
 *   ImGui::End();
 *   
 *   m_DockingManager->EndDockspace();
 */
class EditorDockingManager {
public:
    enum class LayoutPreset {
        GameDev,        ///< Optimized for game development (hierarchy left, viewport center, inspector right)
        Animation,      ///< Optimized for animation work (timeline bottom, hierarchy left, viewport center)
        Rendering,      ///< Optimized for rendering (shader editor left, viewport center, profiler right)
        Minimal,        ///< Minimal layout with only essential panels
        TallLeft,       ///< Tall panels on left, wide viewport on right
        Custom          ///< User-defined layout
    };

    EditorDockingManager();
    ~EditorDockingManager();

    /**
     * @brief Initialize docking system
     * Must be called once before using docking features
     */
    void Initialize();

    /**
     * @brief Begin dockspace layout
     * Call once per frame at the start of ImGui UI rendering
     * Returns the dockspace ID for reference
     */
    ImGuiID BeginDockspace();

    /**
     * @brief End dockspace
     * Call once per frame after rendering all docked windows
     */
    void EndDockspace();

    /**
     * @brief Get dock ID for a specific panel location
     * @param panelName Name of the panel (e.g., "hierarchy", "inspector", "viewport")
     * @return ImGui dock ID to use with ImGui::SetNextWindowDockID()
     */
    ImGuiID GetDockID(const std::string& panelName) const;

    /**
     * @brief Apply a layout preset
     * @param preset The layout configuration to apply
     */
    void ApplyLayoutPreset(LayoutPreset preset);

    /**
     * @brief Get the current layout preset
     */
    LayoutPreset GetCurrentPreset() const { return m_CurrentPreset; }

    /**
     * @brief Reset to default layout
     */
    void ResetLayout();

    /**
     * @brief Save current layout to INI file (ImGui native)
     */
    void SaveLayout();

    /**
     * @brief Load layout from INI file (ImGui native)
     */
    void LoadLayout();

    /**
     * @brief Check if docking is enabled
     */
    bool IsDockingEnabled() const { return m_DockingEnabled; }

    /**
     * @brief Toggle docking on/off
     */
    void SetDockingEnabled(bool enabled);

    /**
     * @brief Check if a specific dock area is visible
     */
    bool IsDockAreaVisible(const std::string& areaName) const;

    /**
     * @brief Render layout preset selector UI
     * Call this from your menu bar or tools panel
     */
    void RenderLayoutSelector();

private:
    /**
     * @brief Setup default GameDev layout
     */
    void SetupGameDevLayout();

    /**
     * @brief Setup Animation layout
     */
    void SetupAnimationLayout();

    /**
     * @brief Setup Rendering layout
     */
    void SetupRenderingLayout();

    /**
     * @brief Setup Minimal layout
     */
    void SetupMinimalLayout();

    /**
     * @brief Setup TallLeft layout
     */
    void SetupTallLeftLayout();

    ImGuiID m_DockspaceID;
    LayoutPreset m_CurrentPreset;
    bool m_DockingEnabled;
    bool m_FirstFrame;
    
    // Cache dock IDs
    std::unordered_map<std::string, ImGuiID> m_DockIDCache;
};
