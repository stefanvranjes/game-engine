#include "GizmoToolsPanel.h"
#include "GizmoManager.h"
#include <imgui.h>

GizmoToolsPanel::GizmoToolsPanel() = default;

GizmoToolsPanel::~GizmoToolsPanel() = default;

void GizmoToolsPanel::Render(std::shared_ptr<GizmoManager> gizmoManager) {
    if (!gizmoManager) {
        ImGui::TextDisabled("GizmoManager not available");
        return;
    }

    if (!ImGui::Begin("Gizmo Tools")) {
        ImGui::End();
        return;
    }

    ImGui::Text("Transform Tools");
    ImGui::Separator();
    
    // Render different sections
    RenderModeButtons(gizmoManager);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    RenderSpaceToggle(gizmoManager);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    RenderSizeControl(gizmoManager);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    RenderSnapSettings(gizmoManager);
    
    if (m_ShowAdvancedOptions) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        RenderQuickSettings(gizmoManager);
    }

    ImGui::End();
}

void GizmoToolsPanel::RenderModeButtons(std::shared_ptr<GizmoManager> gizmoManager) {
    ImGui::Text("Transform Mode:");
    ImGui::Indent();
    
    GizmoType currentMode = gizmoManager->GetGizmoType();
    
    // Translation button (W key)
    if (ImGui::Button("Translate##mode", ImVec2(-1, 0))) {
        gizmoManager->SetGizmoType(GizmoType::Translation);
        m_HasChanges = true;
        if (m_OnGizmoModeChanged) m_OnGizmoModeChanged(GizmoType::Translation);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch to Translation mode\nShortcut: W");
    }
    
    // Rotation button (E key)
    if (ImGui::Button("Rotate##mode", ImVec2(-1, 0))) {
        gizmoManager->SetGizmoType(GizmoType::Rotation);
        m_HasChanges = true;
        if (m_OnGizmoModeChanged) m_OnGizmoModeChanged(GizmoType::Rotation);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch to Rotation mode\nShortcut: E");
    }
    
    // Scale button (R key)
    if (ImGui::Button("Scale##mode", ImVec2(-1, 0))) {
        gizmoManager->SetGizmoType(GizmoType::Scale);
        m_HasChanges = true;
        if (m_OnGizmoModeChanged) m_OnGizmoModeChanged(GizmoType::Scale);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Switch to Scale mode\nShortcut: R");
    }
    
    // Show current mode
    ImGui::Spacing();
    const char* modeNames[] = { "None", "Translation", "Rotation", "Scale" };
    ImGui::Text("Current: %s", modeNames[static_cast<int>(currentMode)]);
    
    ImGui::Unindent();
}

void GizmoToolsPanel::RenderSpaceToggle(std::shared_ptr<GizmoManager> gizmoManager) {
    ImGui::Text("Transform Space:");
    ImGui::Indent();
    
    bool useLocal = gizmoManager->IsUsingLocalSpace();
    if (ImGui::Checkbox("Use Local Space##toggle", &useLocal)) {
        gizmoManager->SetUseLocalSpace(useLocal);
        m_UseLocalSpace = useLocal;
        m_HasChanges = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle between local object space and world space");
    }
    
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Local Space: Gizmo moves along object axes\nWorld Space: Gizmo moves along world axes");
    }
    
    ImGui::Text("Mode: %s", useLocal ? "Local" : "World");
    
    ImGui::Unindent();
}

void GizmoToolsPanel::RenderSizeControl(std::shared_ptr<GizmoManager> gizmoManager) {
    ImGui::Text("Gizmo Size:");
    ImGui::Indent();
    
    float size = gizmoManager->GetGizmoSize();
    if (ImGui::SliderFloat("##GizmoSize", &size, 0.1f, 5.0f, "%.2f")) {
        gizmoManager->SetGizmoSize(size);
        m_GizmoSize = size;
        m_HasChanges = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Adjust the overall scale of gizmos");
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Reset##size")) {
        gizmoManager->SetGizmoSize(1.0f);
        m_GizmoSize = 1.0f;
        m_HasChanges = true;
    }
    
    ImGui::Unindent();
}

void GizmoToolsPanel::RenderSnapSettings(std::shared_ptr<GizmoManager> gizmoManager) {
    ImGui::Text("Snap to Grid:");
    ImGui::Indent();
    
    bool snapEnabled = gizmoManager->IsSnappingEnabled();
    if (ImGui::Checkbox("Enable Snapping##toggle", &snapEnabled)) {
        gizmoManager->SetSnappingEnabled(snapEnabled);
        m_SnapEnabled = snapEnabled;
        m_HasChanges = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enable/disable snapping to grid");
    }
    
    if (snapEnabled) {
        ImGui::Spacing();
        
        // Translation snap
        float transSnap = gizmoManager->GetTranslationSnap();
        if (ImGui::DragFloat("Translation Snap##value", &transSnap, 0.1f, 0.01f, 100.0f)) {
            gizmoManager->SetTranslationSnap(transSnap);
            m_TranslationSnapValue = transSnap;
            m_HasChanges = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Grid size for translation snapping");
        }
        
        // Rotation snap
        float rotSnap = gizmoManager->GetRotationSnap();
        if (ImGui::DragFloat("Rotation Snap##value", &rotSnap, 0.5f, 0.1f, 360.0f)) {
            gizmoManager->SetRotationSnap(rotSnap);
            m_RotationSnapValue = rotSnap;
            m_HasChanges = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Angle increment for rotation snapping (degrees)");
        }
        
        // Scale snap
        float scaleSnap = gizmoManager->GetScaleSnap();
        if (ImGui::DragFloat("Scale Snap##value", &scaleSnap, 0.05f, 0.01f, 10.0f)) {
            gizmoManager->SetScaleSnap(scaleSnap);
            m_ScaleSnapValue = scaleSnap;
            m_HasChanges = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Scale increment for scale snapping");
        }
    }
    
    ImGui::Unindent();
}

void GizmoToolsPanel::RenderQuickSettings(std::shared_ptr<GizmoManager> gizmoManager) {
    ImGui::Text("Quick Settings:");
    ImGui::Indent();
    
    // Preset buttons
    if (ImGui::Button("Default Settings", ImVec2(-1, 0))) {
        ResetSettings();
        gizmoManager->ResetGizmoSettings();
        m_HasChanges = true;
    }
    
    ImGui::Spacing();
    
    // Precision presets
    ImGui::Text("Precision Presets:");
    
    if (ImGui::Button("Fine (0.1)", ImVec2(-1, 0))) {
        gizmoManager->SetTranslationSnap(0.1f);
        gizmoManager->SetRotationSnap(5.0f);
        gizmoManager->SetScaleSnap(0.1f);
        gizmoManager->SetSnappingEnabled(true);
        m_TranslationSnapValue = 0.1f;
        m_RotationSnapValue = 5.0f;
        m_ScaleSnapValue = 0.1f;
        m_SnapEnabled = true;
        m_HasChanges = true;
    }
    
    if (ImGui::Button("Medium (1.0)", ImVec2(-1, 0))) {
        gizmoManager->SetTranslationSnap(1.0f);
        gizmoManager->SetRotationSnap(15.0f);
        gizmoManager->SetScaleSnap(0.5f);
        gizmoManager->SetSnappingEnabled(true);
        m_TranslationSnapValue = 1.0f;
        m_RotationSnapValue = 15.0f;
        m_ScaleSnapValue = 0.5f;
        m_SnapEnabled = true;
        m_HasChanges = true;
    }
    
    if (ImGui::Button("Coarse (5.0)", ImVec2(-1, 0))) {
        gizmoManager->SetTranslationSnap(5.0f);
        gizmoManager->SetRotationSnap(45.0f);
        gizmoManager->SetScaleSnap(1.0f);
        gizmoManager->SetSnappingEnabled(true);
        m_TranslationSnapValue = 5.0f;
        m_RotationSnapValue = 45.0f;
        m_ScaleSnapValue = 1.0f;
        m_SnapEnabled = true;
        m_HasChanges = true;
    }
    
    ImGui::Unindent();
}

void GizmoToolsPanel::ResetSettings() {
    m_SnapEnabled = false;
    m_TranslationSnapValue = 1.0f;
    m_RotationSnapValue = 15.0f;
    m_ScaleSnapValue = 0.5f;
    m_UseLocalSpace = false;
    m_GizmoSize = 1.0f;
}
