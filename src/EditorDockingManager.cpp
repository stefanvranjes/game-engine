#include "EditorDockingManager.h"
#include <imgui_internal.h>
#include <iostream>

EditorDockingManager::EditorDockingManager()
    : m_DockspaceID(0),
      m_CurrentPreset(LayoutPreset::GameDev),
      m_DockingEnabled(true),
      m_FirstFrame(true) {
}

EditorDockingManager::~EditorDockingManager() = default;

void EditorDockingManager::Initialize() {
    // Enable docking in ImGui
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    #ifdef IMGUI_HAS_VIEWPORT
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    #endif

    std::cout << "EditorDockingManager initialized - Docking enabled" << std::endl;
}

ImGuiID EditorDockingManager::BeginDockspace() {
    if (!m_DockingEnabled) {
        return 0;
    }

    ImGuiID dockspace_id = ImGui::GetID("EditorDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    
    m_DockspaceID = dockspace_id;
    
    // Apply layout on first frame
    if (m_FirstFrame) {
        m_FirstFrame = false;
        ApplyLayoutPreset(m_CurrentPreset);
    }
    
    return dockspace_id;
}

void EditorDockingManager::EndDockspace() {
    // No explicit cleanup needed per frame
}

ImGuiID EditorDockingManager::GetDockID(const std::string& panelName) const {
    auto it = m_DockIDCache.find(panelName);
    if (it != m_DockIDCache.end()) {
        return it->second;
    }
    
    // Generate ID if not cached
    ImGuiID id = ImGui::GetID(panelName.c_str());
    return id;
}

void EditorDockingManager::ApplyLayoutPreset(LayoutPreset preset) {
    m_CurrentPreset = preset;
    
    switch (preset) {
        case LayoutPreset::GameDev:
            SetupGameDevLayout();
            break;
        case LayoutPreset::Animation:
            SetupAnimationLayout();
            break;
        case LayoutPreset::Rendering:
            SetupRenderingLayout();
            break;
        case LayoutPreset::Minimal:
            SetupMinimalLayout();
            break;
        case LayoutPreset::TallLeft:
            SetupTallLeftLayout();
            break;
        case LayoutPreset::Custom:
            // Custom layout is managed by user
            break;
    }
    
    std::cout << "Layout preset applied: " << static_cast<int>(preset) << std::endl;
}

void EditorDockingManager::ResetLayout() {
    // Clear ImGui dock context to reset layout
    if (ImGui::GetCurrentContext()) {
        ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    }
    
    m_FirstFrame = true;
    m_DockIDCache.clear();
    
    std::cout << "Layout reset" << std::endl;
}

void EditorDockingManager::SaveLayout() {
    // ImGui automatically saves layout to imgui.ini
    // This is called when the application closes
    std::cout << "Layout saved to imgui.ini" << std::endl;
}

void EditorDockingManager::LoadLayout() {
    // ImGui automatically loads layout from imgui.ini
    // This is called when the application starts
    std::cout << "Layout loaded from imgui.ini" << std::endl;
}

void EditorDockingManager::SetDockingEnabled(bool enabled) {
    m_DockingEnabled = enabled;
    if (enabled) {
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        std::cout << "Docking enabled" << std::endl;
    } else {
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags &= ~ImGuiConfigFlags_DockingEnable;
        std::cout << "Docking disabled" << std::endl;
    }
}

bool EditorDockingManager::IsDockAreaVisible(const std::string& areaName) const {
    // Query ImGui for dock node visibility
    // This is a simplified implementation
    return true; // Placeholder - always visible for now
}

void EditorDockingManager::RenderLayoutSelector() {
    ImGui::Separator();
    ImGui::Text("Layout Presets");
    
    if (ImGui::Button("Game Dev", ImVec2(-1, 0))) {
        ApplyLayoutPreset(LayoutPreset::GameDev);
    }
    
    if (ImGui::Button("Animation", ImVec2(-1, 0))) {
        ApplyLayoutPreset(LayoutPreset::Animation);
    }
    
    if (ImGui::Button("Rendering", ImVec2(-1, 0))) {
        ApplyLayoutPreset(LayoutPreset::Rendering);
    }
    
    if (ImGui::Button("Minimal", ImVec2(-1, 0))) {
        ApplyLayoutPreset(LayoutPreset::Minimal);
    }
    
    if (ImGui::Button("Tall Left", ImVec2(-1, 0))) {
        ApplyLayoutPreset(LayoutPreset::TallLeft);
    }
    
    if (ImGui::Button("Reset Layout", ImVec2(-1, 0))) {
        ResetLayout();
    }
    
    ImGui::Separator();
}

void EditorDockingManager::SetupGameDevLayout() {
    // GameDev: Hierarchy on left, Viewport center-right, Inspector right
    // Timeline at bottom, Tools floating
    
    if (!m_DockspaceID) return;
    
    ImGuiID dockspace_id = m_DockspaceID;
    
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    ImGuiID dock_main_id = dockspace_id;
    ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.20f, nullptr, &dock_main_id);
    ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.25f, nullptr, &dock_main_id);
    ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.25f, nullptr, &dock_main_id);
    
    ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_id_left);
    ImGui::DockBuilderDockWindow("Inspector", dock_id_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_bottom);
    ImGui::DockBuilderDockWindow("Performance Profiler", dock_id_bottom);
    ImGui::DockBuilderDockWindow("Tools", dock_id_right);
    
    ImGui::DockBuilderFinish(dockspace_id);
}

void EditorDockingManager::SetupAnimationLayout() {
    // Animation: Timeline at bottom, Hierarchy on left, Viewport center, Properties/Graph on right
    
    if (!m_DockspaceID) return;
    
    ImGuiID dockspace_id = m_DockspaceID;
    
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    ImGuiID dock_main_id = dockspace_id;
    ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.20f, nullptr, &dock_main_id);
    ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.25f, nullptr, &dock_main_id);
    ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.30f, nullptr, &dock_main_id);
    
    ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_id_left);
    ImGui::DockBuilderDockWindow("Inspector", dock_id_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
    ImGui::DockBuilderDockWindow("Animation Timeline", dock_id_bottom);
    ImGui::DockBuilderDockWindow("Performance Profiler", dock_id_bottom);
    
    ImGui::DockBuilderFinish(dockspace_id);
}

void EditorDockingManager::SetupRenderingLayout() {
    // Rendering: Shader editor on left, Viewport center, Profiler right with viewport at top
    
    if (!m_DockspaceID) return;
    
    ImGuiID dockspace_id = m_DockspaceID;
    
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    ImGuiID dock_main_id = dockspace_id;
    ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
    ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.25f, nullptr, &dock_main_id);
    ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.25f, nullptr, &dock_main_id);
    
    ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_id_left);
    ImGui::DockBuilderDockWindow("Performance Profiler", dock_id_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
    ImGui::DockBuilderDockWindow("Inspector", dock_id_right);
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_bottom);
    
    ImGui::DockBuilderFinish(dockspace_id);
}

void EditorDockingManager::SetupMinimalLayout() {
    // Minimal: Just viewport centered with optional Inspector on right
    
    if (!m_DockspaceID) return;
    
    ImGuiID dockspace_id = m_DockspaceID;
    
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    ImGuiID dock_main_id = dockspace_id;
    ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.20f, nullptr, &dock_main_id);
    
    ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
    ImGui::DockBuilderDockWindow("Inspector", dock_id_right);
    
    ImGui::DockBuilderFinish(dockspace_id);
}

void EditorDockingManager::SetupTallLeftLayout() {
    // TallLeft: Full-height panel on left (Hierarchy + Tools), large viewport right
    
    if (!m_DockspaceID) return;
    
    ImGuiID dockspace_id = m_DockspaceID;
    
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    ImGuiID dock_main_id = dockspace_id;
    ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
    ImGuiID dock_id_right_top = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.30f, nullptr, &dock_main_id);
    
    ImGui::DockBuilderDockWindow("Scene Hierarchy", dock_id_left);
    ImGui::DockBuilderDockWindow("Tools", dock_id_left);
    ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
    ImGui::DockBuilderDockWindow("Inspector", dock_id_right_top);
    ImGui::DockBuilderDockWindow("Performance Profiler", dock_id_right_top);
    
    ImGui::DockBuilderFinish(dockspace_id);
}
