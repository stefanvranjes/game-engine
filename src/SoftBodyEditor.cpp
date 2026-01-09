#include "SoftBodyEditor.h"
#include "SoftBodyEditorWidgets.h"
#include "PhysXSoftBody.h"
#include "SoftBodyLOD.h"
#include "StressVisualizer.h"
#include "SoftBodyPresetLibrary.h"
#include "ManualTearTrigger.h"
#include "TearHistory.h"
#include "TearPreview.h"
#include "VertexPicker.h"
#include "VertexHighlighter.h"
#include <imgui.h>
#include <iostream>

using namespace SoftBodyEditorWidgets;

SoftBodyEditor::SoftBodyEditor()
    : m_SelectedSoftBody(nullptr)
    , m_Visible(true)
    , m_CurrentTab(0)
    , m_SaveFilePath("softbody_config.json")
    , m_LoadFilePath("softbody_config.json")
    , m_CurrentPreset("None")
    , m_TearMode(false)
{
    m_StressVisualizer = std::make_unique<StressVisualizer>();
    m_PresetLibrary = std::make_unique<SoftBodyPresetLibrary>();
    m_ManualTearTrigger = std::make_unique<ManualTearTrigger>();
    m_TearHistory = std::make_unique<TearHistory>();
    m_TearPreview = std::make_unique<TearPreview>();
    m_VertexPicker = std::make_unique<VertexPicker>();
    m_VertexHighlighter = std::make_unique<VertexHighlighter>();
}

SoftBodyEditor::~SoftBodyEditor() {
}

void SoftBodyEditor::SetSelectedSoftBody(PhysXSoftBody* softBody) {
    m_SelectedSoftBody = softBody;
}

void SoftBodyEditor::Render(PhysXSoftBody* softBody) {
    if (!m_Visible) return;
    
    // Update selected soft body if provided
    if (softBody) {
        m_SelectedSoftBody = softBody;
    }
    
    ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Soft Body Editor", &m_Visible)) {
        ImGui::End();
        return;
    }
    
    if (!m_SelectedSoftBody) {
        RenderEmptyState();
        ImGui::End();
        return;
    }
    
    // Render tab bar
    RenderTabBar();
    
    // Render current tab content
    switch (m_CurrentTab) {
        case 0: RenderBasicPropertiesPanel(); break;
        case 1: RenderLODPanel(); break;
        case 2: RenderMaterialPanel(); break;
        case 3: RenderTearingPanel(); break;
        case 4: RenderVisualizationPanel(); break;
        case 5: RenderSerializationPanel(); break;
        case 6: RenderStatisticsPanel(); break;
        case 7: RenderPresetPanel(); break;
        case 8: RenderStressVisualizationPanel(); break;
    }
    
    ImGui::End();
}

void SoftBodyEditor::RenderEmptyState() {
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No soft body selected");
    ImGui::Separator();
    ImGui::TextWrapped("Select a GameObject with a soft body component to edit its properties.");
}

void SoftBodyEditor::RenderTabBar() {
    if (ImGui::BeginTabBar("SoftBodyTabs")) {
        if (ImGui::BeginTabItem("Physics")) {
            m_CurrentTab = 0;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LOD")) {
            m_CurrentTab = 1;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Material")) {
            m_CurrentTab = 2;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Tearing")) {
            m_CurrentTab = 3;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Visualization")) {
            m_CurrentTab = 4;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Serialization")) {
            m_CurrentTab = 5;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Statistics")) {
            m_CurrentTab = 6;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Presets")) {
            m_CurrentTab = 7;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Stress")) {
            m_CurrentTab = 8;
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

// ============================================================================
// Basic Properties Panel
// ============================================================================

void SoftBodyEditor::RenderBasicPropertiesPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    // Stiffness section
    if (ImGui::CollapsingHeader("Stiffness", ImGuiTreeNodeFlags_DefaultOpen)) {
        float volumeStiffness = m_SelectedSoftBody->GetVolumeStiffness();
        if (StiffnessSlider("Volume", volumeStiffness)) {
            m_SelectedSoftBody->SetVolumeStiffness(volumeStiffness);
        }
        ImGui::SameLine(); HelpMarker("Controls volume preservation (0=soft, 1=rigid)");
        
        float shapeStiffness = m_SelectedSoftBody->GetShapeStiffness();
        if (StiffnessSlider("Shape", shapeStiffness)) {
            m_SelectedSoftBody->SetShapeStiffness(shapeStiffness);
        }
        ImGui::SameLine(); HelpMarker("Controls shape matching (0=deformable, 1=rigid)");
        
        float deformationStiffness = m_SelectedSoftBody->GetDeformationStiffness();
        if (StiffnessSlider("Deformation", deformationStiffness)) {
            m_SelectedSoftBody->SetDeformationStiffness(deformationStiffness);
        }
        ImGui::SameLine(); HelpMarker("Controls resistance to deformation");
    }
    
    ImGui::Spacing();
    
    // Mass section
    if (ImGui::CollapsingHeader("Mass", ImGuiTreeNodeFlags_DefaultOpen)) {
        float totalMass = m_SelectedSoftBody->GetTotalMass();
        ImGui::Text("Total Mass:");
        ImGui::SameLine(150);
        ImGui::PushItemWidth(200);
        if (ImGui::DragFloat("##mass", &totalMass, 0.1f, 0.1f, 1000.0f, "%.2f kg")) {
            m_SelectedSoftBody->SetTotalMass(totalMass);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine(); HelpMarker("Total mass of the soft body");
    }
    
    ImGui::Spacing();
    
    // Collision section
    if (ImGui::CollapsingHeader("Collision", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool sceneCollision = true; // TODO: Add getter
        if (ImGui::Checkbox("Scene Collision", &sceneCollision)) {
            m_SelectedSoftBody->SetSceneCollision(sceneCollision);
        }
        ImGui::SameLine(); HelpMarker("Enable collision with scene objects");
        
        bool selfCollision = false; // TODO: Add getter
        if (ImGui::Checkbox("Self Collision", &selfCollision)) {
            m_SelectedSoftBody->SetSelfCollision(selfCollision);
        }
        ImGui::SameLine(); HelpMarker("Enable self-collision detection");
        
        float margin = 0.01f; // TODO: Add getter
        ImGui::Text("Margin:");
        ImGui::SameLine(150);
        ImGui::PushItemWidth(200);
        if (ImGui::SliderFloat("##margin", &margin, 0.001f, 0.1f, "%.3f")) {
            m_SelectedSoftBody->SetCollisionMargin(margin);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine(); HelpMarker("Collision detection margin");
    }
    
    ImGui::Spacing();
    
    // State section
    if (ImGui::CollapsingHeader("State", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool enabled = m_SelectedSoftBody->IsEnabled();
        if (ImGui::Checkbox("Simulation Enabled", &enabled)) {
            m_SelectedSoftBody->SetEnabled(enabled);
        }
        ImGui::SameLine(); HelpMarker("Enable/disable simulation updates");
    }
}

// ============================================================================
// LOD Panel
// ============================================================================

void SoftBodyEditor::RenderLODPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    // LOD enable/disable
    bool lodEnabled = m_SelectedSoftBody->IsLODEnabled();
    if (ImGui::Checkbox("Enable LOD System", &lodEnabled)) {
        m_SelectedSoftBody->SetLODEnabled(lodEnabled);
    }
    ImGui::SameLine(); HelpMarker("Enable automatic Level-of-Detail based on distance");
    
    if (!lodEnabled) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "LOD system is disabled");
        return;
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    const SoftBodyLODConfig* lodConfig = m_SelectedSoftBody->GetLODConfig();
    if (!lodConfig) {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "No LOD configuration");
        if (ImGui::Button("Create Default LOD Config")) {
            SoftBodyLODConfig defaultConfig = SoftBodyLODConfig::CreateDefault(
                m_SelectedSoftBody->GetVertexCount(),
                0 // TODO: Get tetrahedron count
            );
            m_SelectedSoftBody->SetLODConfig(defaultConfig);
        }
        return;
    }
    
    // Current LOD indicator
    int currentLOD = m_SelectedSoftBody->GetCurrentLOD();
    int maxLOD = lodConfig->GetLODCount() - 1;
    LODIndicator(currentLOD, maxLOD);
    
    ImGui::Spacing();
    
    // Force LOD selector
    ImGui::Text("Force LOD:");
    ImGui::SameLine(150);
    const char* lodOptions[] = {"Automatic", "LOD 0", "LOD 1", "LOD 2", "LOD 3"};
    static int forcedLOD = 0;
    if (ImGui::Combo("##forceLOD", &forcedLOD, lodOptions, IM_ARRAYSIZE(lodOptions))) {
        m_SelectedSoftBody->ForceLOD(forcedLOD - 1); // -1 for automatic
    }
    ImGui::SameLine(); HelpMarker("Override automatic LOD selection");
    
    ImGui::Spacing();
    
    // Camera position
    static Vec3 cameraPos(0, 10, 0);
    if (Vec3Input("Camera Position", cameraPos, 0.5f)) {
        m_SelectedSoftBody->SetCameraPosition(cameraPos);
    }
    ImGui::SameLine(); HelpMarker("Camera position for distance calculation");
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // LOD levels table
    ImGui::Text("LOD Levels:");
    
    if (ImGui::BeginTable("LODTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Level");
        ImGui::TableSetupColumn("Distance");
        ImGui::TableSetupColumn("Vertices");
        ImGui::TableSetupColumn("Update");
        ImGui::TableSetupColumn("State");
        ImGui::TableHeadersRow();
        
        for (int i = 0; i < lodConfig->GetLODCount(); ++i) {
            const SoftBodyLODLevel* level = lodConfig->GetLODLevel(i);
            if (!level) continue;
            
            ImGui::TableNextRow();
            
            // Level
            ImGui::TableNextColumn();
            if (i == currentLOD) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%d", i);
            } else {
                ImGui::Text("%d", i);
            }
            
            // Distance
            ImGui::TableNextColumn();
            ImGui::Text("%.1fm", level->minDistance);
            
            // Vertices
            ImGui::TableNextColumn();
            if (level->isFrozen) {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.8f, 1.0f), "Frozen");
            } else {
                ImGui::Text("%d", level->vertexCount);
            }
            
            // Update frequency
            ImGui::TableNextColumn();
            if (level->isFrozen) {
                ImGui::Text("-");
            } else {
                ImGui::Text("%dx", level->updateFrequency);
            }
            
            // State
            ImGui::TableNextColumn();
            if (level->hasMeshData) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Ready");
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "No Mesh");
            }
        }
        
        ImGui::EndTable();
    }
}

// ============================================================================
// Material Panel
// ============================================================================

void SoftBodyEditor::RenderMaterialPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    // Tear resistance
    if (ImGui::CollapsingHeader("Tear Resistance", ImGuiTreeNodeFlags_DefaultOpen)) {
        float tearThreshold = m_SelectedSoftBody->GetTearThreshold();
        ImGui::Text("Tear Threshold:");
        ImGui::SameLine(150);
        ImGui::PushItemWidth(200);
        if (ImGui::SliderFloat("##tearThreshold", &tearThreshold, 0.1f, 10.0f, "%.2f")) {
            m_SelectedSoftBody->SetTearThreshold(tearThreshold);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine(); HelpMarker("Stress threshold for tearing (higher = more resistant)");
    }
    
    ImGui::Spacing();
    
    // Anisotropic material
    if (ImGui::CollapsingHeader("Anisotropic Material")) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Anisotropic material settings");
        ImGui::Text("Coming soon...");
    }
}

// ============================================================================
// Tearing Panel
// ============================================================================

void SoftBodyEditor::RenderTearingPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    // Healing section
    if (ImGui::CollapsingHeader("Healing", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool healingEnabled = false; // TODO: Add getter
        if (ImGui::Checkbox("Enable Healing", &healingEnabled)) {
            m_SelectedSoftBody->SetHealingEnabled(healingEnabled);
        }
        ImGui::SameLine(); HelpMarker("Allow tears to heal over time");
        
        if (healingEnabled) {
            float healingRate = 0.1f; // TODO: Add getter
            ImGui::Text("Healing Rate:");
            ImGui::SameLine(150);
            ImGui::PushItemWidth(200);
            if (ImGui::SliderFloat("##healRate", &healingRate, 0.01f, 1.0f, "%.2f")) {
                m_SelectedSoftBody->SetHealingRate(healingRate);
            }
            ImGui::PopItemWidth();
            ImGui::SameLine(); HelpMarker("Rate of resistance recovery");
            
            float healingDelay = 1.0f; // TODO: Add getter
            ImGui::Text("Healing Delay:");
            ImGui::SameLine(150);
            ImGui::PushItemWidth(200);
            if (ImGui::SliderFloat("##healDelay", &healingDelay, 0.0f, 10.0f, "%.1f s")) {
                m_SelectedSoftBody->SetHealingDelay(healingDelay);
            }
            ImGui::PopItemWidth();
            ImGui::SameLine(); HelpMarker("Delay before healing starts");
        }
    }
    
    ImGui::Spacing();
    
    // Plasticity section
    if (ImGui::CollapsingHeader("Plasticity", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool plasticityEnabled = false; // TODO: Add getter
        if (ImGui::Checkbox("Enable Plasticity", &plasticityEnabled)) {
            m_SelectedSoftBody->SetPlasticityEnabled(plasticityEnabled);
        }
        ImGui::SameLine(); HelpMarker("Allow permanent deformation");
        
        if (plasticityEnabled) {
            float plasticThreshold = 1.5f; // TODO: Add getter
            ImGui::Text("Plastic Threshold:");
            ImGui::SameLine(150);
            ImGui::PushItemWidth(200);
            if (ImGui::SliderFloat("##plasticThresh", &plasticThreshold, 0.5f, 5.0f, "%.2f")) {
                m_SelectedSoftBody->SetPlasticThreshold(plasticThreshold);
            }
            ImGui::PopItemWidth();
            ImGui::SameLine(); HelpMarker("Stress threshold for plastic deformation");
            
            float plasticityRate = 0.1f; // TODO: Add getter
            ImGui::Text("Plasticity Rate:");
            ImGui::SameLine(150);
            ImGui::PushItemWidth(200);
            if (ImGui::SliderFloat("##plasticRate", &plasticityRate, 0.01f, 1.0f, "%.2f")) {
                m_SelectedSoftBody->SetPlasticityRate(plasticityRate);
            }
            ImGui::PopItemWidth();
            ImGui::SameLine(); HelpMarker("Rate of plastic deformation");
        }
    }
    
    ImGui::Spacing();
    
    // Manual tear controls
    if (ImGui::CollapsingHeader("Manual Tear Controls")) {
        ImGui::Text("Tear Type:");
        ImGui::SameLine(150);
        
        const char* tearTypes[] = {"Point", "Path", "Region"};
        static int tearType = 0;
        if (ImGui::Combo("##tearType", &tearType, tearTypes, 3)) {
            m_ManualTearTrigger->SetType(static_cast<ManualTearTrigger::TriggerType>(tearType));
        }
        ImGui::SameLine(); HelpMarker("Type of manual tear to apply");
        
        ImGui::Text("Intensity:");
        ImGui::SameLine(150);
        float intensity = m_ManualTearTrigger->GetIntensity();
        ImGui::PushItemWidth(200);
        if (ImGui::SliderFloat("##intensity", &intensity, 0.0f, 1.0f, "%.2f")) {
            m_ManualTearTrigger->SetIntensity(intensity);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine(); HelpMarker("Tear intensity (0=no tear, 1=complete)");
        
        ImGui::Spacing();
        
        // Tear mode toggle
        if (!m_TearMode) {
            if (ImGui::Button("Enter Tear Mode", ImVec2(150, 0))) {
                m_TearMode = true;
                m_SelectedVertices.clear();
            }
            ImGui::SameLine(); HelpMarker("Enter mode to select vertices for tearing");
        } else {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Tear Mode Active");
            ImGui::Text("Selected Vertices: %d", static_cast<int>(m_SelectedVertices.size()));
            ImGui::Text("(Click vertices in 3D view to select)");
            
            ImGui::Spacing();
            
            if (ImGui::Button("Apply Tear", ImVec2(120, 0))) {
                if (!m_SelectedVertices.empty()) {
                    m_ManualTearTrigger->SetAffectedVertices(m_SelectedVertices);
                    m_ManualTearTrigger->Apply(m_SelectedSoftBody);
                    
                    // Record in history
                    TearHistory::TearAction action;
                    action.affectedVertices = m_SelectedVertices;
                    action.tearThresholdBefore = m_SelectedSoftBody->GetTearThreshold();
                    m_TearHistory->RecordTear(action);
                    
                    m_TearMode = false;
                    m_SelectedVertices.clear();
                }
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                m_TearMode = false;
                m_SelectedVertices.clear();
            }
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Undo/Redo controls
        ImGui::Text("History:");
        
        bool canUndo = m_TearHistory->CanUndo();
        bool canRedo = m_TearHistory->CanRedo();
        
        if (!canUndo) ImGui::BeginDisabled();
        if (ImGui::Button("Undo", ImVec2(80, 0))) {
            TearHistory::TearAction action = m_TearHistory->Undo();
            // TODO: Apply undo to soft body
            std::cout << "Undo tear" << std::endl;
        }
        if (!canUndo) ImGui::EndDisabled();
        
        ImGui::SameLine();
        
        if (!canRedo) ImGui::BeginDisabled();
        if (ImGui::Button("Redo", ImVec2(80, 0))) {
            TearHistory::TearAction action = m_TearHistory->Redo();
            // TODO: Apply redo to soft body
            std::cout << "Redo tear" << std::endl;
        }
        if (!canRedo) ImGui::EndDisabled();
        
        ImGui::SameLine();
        
        if (ImGui::Button("Clear History", ImVec2(120, 0))) {
            m_TearHistory->Clear();
        }
        
        ImGui::Text("History Size: %d / %d", 
                    m_TearHistory->GetCurrentIndex() + 1,
                    m_TearHistory->GetHistorySize());
    }
    
    ImGui::Spacing();
    
    // Tear preview
    if (ImGui::CollapsingHeader("Tear Preview")) {
        bool previewEnabled = m_TearPreview->IsEnabled();
        if (ImGui::Checkbox("Enable Preview", &previewEnabled)) {
            m_TearPreview->SetEnabled(previewEnabled);
        }
        ImGui::SameLine(); HelpMarker("Show preview of tear effects");
        
        if (previewEnabled) {
            ImGui::Spacing();
            
            if (ImGui::Button("Calculate Preview", ImVec2(150, 0))) {
                m_TearPreview->SetTearPath(m_SelectedVertices);
                m_TearPreview->Calculate(m_SelectedSoftBody);
            }
            
            ImGui::Spacing();
            
            const auto& info = m_TearPreview->GetPreviewInfo();
            if (info.isValid) {
                ImGui::Text("Preview Results:");
                ImGui::Text("  Affected Vertices: %d", static_cast<int>(info.affectedVertices.size()));
                ImGui::Text("  Affected Tetrahedra: %d", static_cast<int>(info.affectedTetrahedra.size()));
                ImGui::Text("  Estimated New Vertices: %d", info.estimatedNewVertices);
                ImGui::Text("  Estimated Stress: %.2f", info.estimatedStress);
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No preview calculated");
            }
        }
    }
}

// ============================================================================
// Visualization Panel
// ============================================================================

void SoftBodyEditor::RenderVisualizationPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Debug Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool debugHull = false; // TODO: Add getter
        if (ImGui::Checkbox("Show Convex Hull", &debugHull)) {
            m_SelectedSoftBody->SetDebugDrawHull(debugHull);
        }
        ImGui::SameLine(); HelpMarker("Visualize collision hull");
        
        bool debugSurface = false; // TODO: Add getter
        if (ImGui::Checkbox("Show Surface Mesh", &debugSurface)) {
            m_SelectedSoftBody->SetDebugDrawSurface(debugSurface);
        }
        ImGui::SameLine(); HelpMarker("Visualize surface triangles");
        
        bool debugSpheres = false; // TODO: Add getter
        if (ImGui::Checkbox("Show Collision Spheres", &debugSpheres)) {
            m_SelectedSoftBody->SetDebugDrawCollisionSpheres(debugSpheres);
        }
        ImGui::SameLine(); HelpMarker("Visualize collision spheres");
    }
}

// ============================================================================
// Serialization Panel
// ============================================================================

void SoftBodyEditor::RenderSerializationPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Save/Load", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Save section
        ImGui::Text("Save Configuration:");
        ImGui::InputText("##savePath", &m_SaveFilePath[0], m_SaveFilePath.capacity());
        if (ImGui::Button("Save", ImVec2(100, 0))) {
            if (m_SelectedSoftBody->SaveToFile(m_SaveFilePath)) {
                std::cout << "Saved soft body configuration to " << m_SaveFilePath << std::endl;
            } else {
                std::cerr << "Failed to save soft body configuration" << std::endl;
            }
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Load section
        ImGui::Text("Load Configuration:");
        ImGui::InputText("##loadPath", &m_LoadFilePath[0], m_LoadFilePath.capacity());
        if (ImGui::Button("Load", ImVec2(100, 0))) {
            if (m_SelectedSoftBody->LoadFromFile(m_LoadFilePath)) {
                std::cout << "Loaded soft body configuration from " << m_LoadFilePath << std::endl;
            } else {
                std::cerr << "Failed to load soft body configuration" << std::endl;
            }
        }
    }
}

// ============================================================================
// Statistics Panel
// ============================================================================

void SoftBodyEditor::RenderStatisticsPanel() {
    if (!m_SelectedSoftBody) return;
    
    ImGui::Spacing();
    
    // Mesh information
    if (ImGui::CollapsingHeader("Mesh Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Vertex Count:");
        ImGui::SameLine(150);
        ImGui::Text("%d", m_SelectedSoftBody->GetVertexCount());
        
        ImGui::Text("Total Mass:");
        ImGui::SameLine(150);
        ImGui::Text("%.2f kg", m_SelectedSoftBody->GetTotalMass());
    }
    
    ImGui::Spacing();
    
    // Performance statistics
    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        const SoftBodyStats& stats = m_SelectedSoftBody->GetStats();
        PerformanceGraph(stats);
        
        ImGui::Spacing();
        ImGui::Text("Total Update Time: %.2f ms", stats.updateTimeMs);
    }
}

// ============================================================================
// Preset Panel
// ============================================================================

void SoftBodyEditor::RenderPresetPanel() {
    if (!m_SelectedSoftBody || !m_PresetLibrary) return;
    
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Current Preset:");
        ImGui::SameLine(150);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s", m_CurrentPreset.c_str());
        
        ImGui::Spacing();
        
        // Preset selector
        ImGui::Text("Apply Preset:");
        ImGui::SameLine(150);
        ImGui::PushItemWidth(200);
        
        if (ImGui::BeginCombo("##preset", "Select...")) {
            // Group by category
            for (const auto& category : m_PresetLibrary->GetCategories()) {
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "%s", category.c_str());
                ImGui::Separator();
                
                for (const auto& presetName : m_PresetLibrary->GetPresetsInCategory(category)) {
                    if (ImGui::Selectable(presetName.c_str())) {
                        const SoftBodyPreset* preset = m_PresetLibrary->GetPreset(presetName);
                        if (preset) {
                            preset->ApplyToSoftBody(m_SelectedSoftBody);
                            m_CurrentPreset = presetName;
                            std::cout << "Applied preset: " << presetName << std::endl;
                        }
                    }
                    
                    // Show description on hover
                    if (ImGui::IsItemHovered()) {
                        const SoftBodyPreset* preset = m_PresetLibrary->GetPreset(presetName);
                        if (preset && !preset->description.empty()) {
                            ImGui::SetTooltip("%s", preset->description.c_str());
                        }
                    }
                }
                
                ImGui::Spacing();
            }
            
            ImGui::EndCombo();
        }
        
        ImGui::PopItemWidth();
        ImGui::SameLine(); HelpMarker("Apply a predefined configuration");
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Save current as preset
        ImGui::Text("Save Current as Preset:");
        
        static char presetName[128] = "";
        static char presetDesc[256] = "";
        
        ImGui::InputText("Name", presetName, 128);
        ImGui::InputText("Description", presetDesc, 256);
        
        if (ImGui::Button("Save Preset", ImVec2(120, 0))) {
            if (strlen(presetName) > 0) {
                SoftBodyPreset newPreset = SoftBodyPreset::CreateFromSoftBody(
                    m_SelectedSoftBody,
                    presetName,
                    presetDesc
                );
                
                m_PresetLibrary->AddPreset(newPreset);
                m_CurrentPreset = presetName;
                
                // Optionally save to file
                std::string filename = std::string("presets/") + presetName + ".preset";
                newPreset.SaveToFile(filename);
                
                std::cout << "Saved preset: " << presetName << std::endl;
                
                // Clear inputs
                presetName[0] = '\0';
                presetDesc[0] = '\0';
            }
        }
    }
}

// ============================================================================
// Stress Visualization Panel
// ============================================================================

void SoftBodyEditor::RenderStressVisualizationPanel() {
    if (!m_SelectedSoftBody || !m_StressVisualizer) return;
    
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Stress Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool stressEnabled = m_StressVisualizer->IsEnabled();
        if (ImGui::Checkbox("Enable Stress Visualization", &stressEnabled)) {
            m_StressVisualizer->SetEnabled(stressEnabled);
        }
        ImGui::SameLine(); HelpMarker("Show stress heatmap on soft body");
        
        if (!stressEnabled) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Stress visualization is disabled");
            return;
        }
        
        ImGui::Spacing();
        
        // Stress calculation mode
        ImGui::Text("Calculation Mode:");
        ImGui::SameLine(150);
        
        const char* modes[] = {"Displacement", "Volume Change", "Combined"};
        int currentMode = static_cast<int>(m_StressVisualizer->GetStressMode());
        
        if (ImGui::Combo("##stressMode", &currentMode, modes, 3)) {
            m_StressVisualizer->SetStressMode(static_cast<StressVisualizer::StressMode>(currentMode));
        }
        ImGui::SameLine(); HelpMarker("Method for calculating stress values");
        
        ImGui::Spacing();
        
        // Calculate stress
        if (ImGui::Button("Calculate Stress", ImVec2(150, 0))) {
            m_StressVisualizer->CalculateStress(m_SelectedSoftBody);
        }
        ImGui::SameLine(); HelpMarker("Update stress values for current deformation");
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Stress statistics
        ImGui::Text("Stress Range:");
        ImGui::Text("Min:");
        ImGui::SameLine(150);
        ImGui::Text("%.3f", m_StressVisualizer->GetMinStress());
        
        ImGui::Text("Max:");
        ImGui::SameLine(150);
        ImGui::Text("%.3f", m_StressVisualizer->GetMaxStress());
        
        ImGui::Spacing();
        
        // Color legend
        ImGui::Text("Color Legend:");
        ImGui::SameLine();
        
        // Draw color gradient
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float width = 200.0f;
        float height = 20.0f;
        
        for (int i = 0; i < 100; ++i) {
            float t = i / 99.0f;
            ImU32 color;
            
            if (t < 0.25f) {
                // Blue to Cyan
                float local_t = t / 0.25f;
                color = ImGui::ColorConvertFloat4ToU32(ImVec4(0, local_t, 1, 1));
            } else if (t < 0.5f) {
                // Cyan to Green
                float local_t = (t - 0.25f) / 0.25f;
                color = ImGui::ColorConvertFloat4ToU32(ImVec4(0, 1, 1 - local_t, 1));
            } else if (t < 0.75f) {
                // Green to Yellow
                float local_t = (t - 0.5f) / 0.25f;
                color = ImGui::ColorConvertFloat4ToU32(ImVec4(local_t, 1, 0, 1));
            } else {
                // Yellow to Red
                float local_t = (t - 0.75f) / 0.25f;
                color = ImGui::ColorConvertFloat4ToU32(ImVec4(1, 1 - local_t, 0, 1));
            }
            
            drawList->AddRectFilled(
                ImVec2(p.x + i * width / 100, p.y),
                ImVec2(p.x + (i + 1) * width / 100, p.y + height),
                color
            );
        }
        
        ImGui::Dummy(ImVec2(width, height));
        ImGui::Text("Low Stress");
        ImGui::SameLine(width - 70);
        ImGui::Text("High Stress");
    }
}

// ============================================================================
// Vertex Selection Helpers
// ============================================================================

void SoftBodyEditor::AddVertex(int vertexIndex) {
    if (!IsVertexSelected(vertexIndex)) {
        m_SelectedVertices.push_back(vertexIndex);
        m_VertexHighlighter->AddSelectedVertex(vertexIndex);
    }
}

void SoftBodyEditor::RemoveVertex(int vertexIndex) {
    auto it = std::find(m_SelectedVertices.begin(), m_SelectedVertices.end(), vertexIndex);
    if (it != m_SelectedVertices.end()) {
        m_SelectedVertices.erase(it);
        m_VertexHighlighter->RemoveSelectedVertex(vertexIndex);
    }
}

bool SoftBodyEditor::IsVertexSelected(int vertexIndex) const {
    return std::find(m_SelectedVertices.begin(), m_SelectedVertices.end(), vertexIndex) 
           != m_SelectedVertices.end();
}

// Note: HandleMouseInput would be called from the main application's input system
// This is a placeholder showing how it would work:
/*
void SoftBodyEditor::HandleMouseInput(const Mouse& mouse, const Camera& camera, 
                                      int screenWidth, int screenHeight) {
    if (!m_TearMode || !m_SelectedSoftBody) return;
    
    // Create ray from camera through mouse position
    Ray ray = camera.ScreenPointToRay(mouse.x, mouse.y, screenWidth, screenHeight);
    
    // Update hover
    auto hoverResult = m_VertexPicker->PickVertex(ray, m_SelectedSoftBody);
    if (hoverResult.hit) {
        m_VertexHighlighter->SetHoveredVertex(hoverResult.vertexIndex);
    } else {
        m_VertexHighlighter->SetHoveredVertex(-1);
    }
    
    // Handle click
    if (mouse.leftButtonPressed) {
        if (hoverResult.hit) {
            // Toggle selection
            if (IsVertexSelected(hoverResult.vertexIndex)) {
                RemoveVertex(hoverResult.vertexIndex);
            } else {
                AddVertex(hoverResult.vertexIndex);
            }
        }
    }
    
    // Render highlights
    m_VertexHighlighter->Render(m_SelectedSoftBody);
}
*/
