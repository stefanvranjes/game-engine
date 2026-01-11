#pragma once

#include <memory>
#include <string>
#include <vector>

class PhysXSoftBody;
class StressVisualizer;
class SoftBodyPresetLibrary;
class ManualTearTrigger;
class TearHistory;
class TearPreview;
class VertexPicker;
class VertexHighlighter;
class SoftBodyRecordingPanel;
class Camera;
struct Mouse;

/**
 * @brief ImGui-based editor for PhysX soft body properties
 * 
 * Provides a comprehensive interface for editing all soft body parameters
 * including physics properties, LOD configuration, material settings,
 * tearing system, and visualization options.
 */
class SoftBodyEditor {
public:
    SoftBodyEditor();
    ~SoftBodyEditor();
    
    /**
     * @brief Render the editor window
     * @param softBody Soft body to edit (nullptr to show empty state)
     */
    void Render(PhysXSoftBody* softBody);
    
    /**
     * @brief Set the selected soft body
     * @param softBody Soft body to edit
     */
    void SetSelectedSoftBody(PhysXSoftBody* softBody);
    
    /**
     * @brief Get the selected soft body
     */
    PhysXSoftBody* GetSelectedSoftBody() const { return m_SelectedSoftBody; }
    
    /**
     * @brief Show/hide the editor window
     */
    void SetVisible(bool visible) { m_Visible = visible; }
    bool IsVisible() const { return m_Visible; }
    
private:
    PhysXSoftBody* m_SelectedSoftBody;
    bool m_Visible;
    
    // UI state
    int m_CurrentTab;
    std::string m_SaveFilePath;
    std::string m_LoadFilePath;
    
    // Enhancement systems
    std::unique_ptr<StressVisualizer> m_StressVisualizer;
    std::unique_ptr<SoftBodyPresetLibrary> m_PresetLibrary;
    std::unique_ptr<ManualTearTrigger> m_ManualTearTrigger;
    std::unique_ptr<TearHistory> m_TearHistory;
    std::unique_ptr<TearPreview> m_TearPreview;
    std::unique_ptr<VertexPicker> m_VertexPicker;
    std::unique_ptr<VertexHighlighter> m_VertexHighlighter;
    std::unique_ptr<SoftBodyRecordingPanel> m_RecordingPanel;
    std::string m_CurrentPreset;
    
    // Tear mode state
    bool m_TearMode;
    std::vector<int> m_SelectedVertices;
    
    // Render different panels
    void RenderBasicPropertiesPanel();
    void RenderLODPanel();
    void RenderMaterialPanel();
    void RenderTearingPanel();
    void RenderVisualizationPanel();
    void RenderSerializationPanel();
    void RenderStatisticsPanel();
    void RenderPresetPanel();
    void RenderStressVisualizationPanel();
    void RenderRecordingPanel();
    
    // Input handling
    void HandleMouseInput(const Mouse& mouse, const Camera& camera, int screenWidth, int screenHeight);
    
    // Helper methods for vertex selection
    void AddVertex(int vertexIndex);
    void RemoveVertex(int vertexIndex);
    bool IsVertexSelected(int vertexIndex) const;
    
    // Helper methods
    void RenderEmptyState();
    void RenderTabBar();
};
