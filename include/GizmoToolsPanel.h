#pragma once

#include "Gizmo.h"
#include <memory>
#include <functional>

class GizmoManager;

/**
 * @brief Advanced gizmo control panel for the editor
 * 
 * Provides UI controls for:
 * - Transform mode selection (Translate/Rotate/Scale)
 * - Snap to grid toggle with configurable snap values
 * - Local/World space toggle
 * - Gizmo size adjustment
 * - Quick access to gizmo settings
 */
class GizmoToolsPanel {
public:
    GizmoToolsPanel();
    ~GizmoToolsPanel();

    /**
     * @brief Render the gizmo tools panel using ImGui
     * @param gizmoManager Reference to the active GizmoManager
     */
    void Render(std::shared_ptr<GizmoManager> gizmoManager);

    /**
     * @brief Check if any settings have changed
     */
    bool HasChanges() const { return m_HasChanges; }

    /**
     * @brief Clear the changes flag
     */
    void ClearChanges() { m_HasChanges = false; }

    /**
     * @brief Reset all gizmo settings to defaults
     */
    void ResetSettings();

    /**
     * @brief Set whether the panel should show advanced options
     */
    void SetShowAdvancedOptions(bool show) { m_ShowAdvancedOptions = show; }
    bool AreAdvancedOptionsShown() const { return m_ShowAdvancedOptions; }

    /**
     * @brief Callback for gizmo mode changes
     */
    void SetOnGizmoModeChanged(std::function<void(GizmoType)> callback) {
        m_OnGizmoModeChanged = callback;
    }

private:
    // UI State
    bool m_HasChanges = false;
    bool m_ShowAdvancedOptions = true;
    
    // Snap settings (cached for UI)
    bool m_SnapEnabled = false;
    float m_TranslationSnapValue = 1.0f;
    float m_RotationSnapValue = 15.0f;
    float m_ScaleSnapValue = 0.5f;
    
    // Space and size settings
    bool m_UseLocalSpace = false;
    float m_GizmoSize = 1.0f;
    
    // UI helper methods
    void RenderModeButtons(std::shared_ptr<GizmoManager> gizmoManager);
    void RenderSnapSettings(std::shared_ptr<GizmoManager> gizmoManager);
    void RenderSpaceToggle(std::shared_ptr<GizmoManager> gizmoManager);
    void RenderSizeControl(std::shared_ptr<GizmoManager> gizmoManager);
    void RenderQuickSettings(std::shared_ptr<GizmoManager> gizmoManager);
    
    // Callback
    std::function<void(GizmoType)> m_OnGizmoModeChanged;
};
