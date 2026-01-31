#include "../include/VisualScriptEditor/VisualScriptEditor.h"
#include "../include/VisualScriptEditor/CodeGenerator.h"
#include "../include/VisualScriptEditor/NodeRegistry.h"
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <fstream>
#include <sstream>

VisualScriptEditor::VisualScriptEditor() 
    : m_SelectedNodeId(0), m_HoveredNodeId(0), m_HoveredPortId(0),
      m_IsConnecting(false), m_CanvasScale(1.0f) {
}

VisualScriptEditor::~VisualScriptEditor() {
}

void VisualScriptEditor::NewGraph(const std::string& name, GraphType type) {
    switch (type) {
        case GraphType::BehaviorTree:
            m_CurrentGraph = std::make_shared<BehaviorTreeGraph>(name);
            break;
        case GraphType::StateMachine:
            m_CurrentGraph = std::make_shared<StateMachineGraph>(name);
            break;
        case GraphType::LogicGraph:
            m_CurrentGraph = std::make_shared<LogicGraph>(name);
            break;
        case GraphType::AnimationGraph:
            m_CurrentGraph = std::make_shared<AnimationGraph>(name);
            break;
    }
    
    m_CurrentFilePath = "";
    m_IsModified = false;
    m_SelectedNodeId = 0;
    m_ValidationErrors.clear();
}

void VisualScriptEditor::OpenGraph(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        m_ValidationErrors.push_back("Failed to open file: " + filePath);
        return;
    }

    try {
        json data;
        file >> data;
        
        GraphType type = static_cast<GraphType>(data["type"].get<int>());
        std::string name = data["name"];
        
        NewGraph(name, type);
        
        if (m_CurrentGraph->Deserialize(data)) {
            m_CurrentFilePath = filePath;
            m_IsModified = false;
            AddRecentFile(filePath);
        } else {
            m_ValidationErrors.push_back("Failed to deserialize graph");
        }
    } catch (const std::exception& e) {
        m_ValidationErrors.push_back("Error loading graph: " + std::string(e.what()));
    }
    
    file.close();
}

void VisualScriptEditor::SaveGraph(const std::string& filePath) {
    if (!m_CurrentGraph) {
        m_ValidationErrors.push_back("No graph loaded");
        return;
    }

    std::string path = filePath.empty() ? m_CurrentFilePath : filePath;
    if (path.empty()) {
        m_ValidationErrors.push_back("No save path specified");
        return;
    }

    try {
        json data = m_CurrentGraph->Serialize();
        
        std::ofstream file(path);
        file << data.dump(2);
        file.close();
        
        m_CurrentFilePath = path;
        m_IsModified = false;
        AddRecentFile(path);
    } catch (const std::exception& e) {
        m_ValidationErrors.push_back("Error saving graph: " + std::string(e.what()));
    }
}

void VisualScriptEditor::CloseGraph() {
    if (m_IsModified) {
        // In a real implementation, prompt user to save
    }
    m_CurrentGraph = nullptr;
    m_CurrentFilePath = "";
    m_SelectedNodeId = 0;
}

void VisualScriptEditor::RenderUI() {
    if (!m_CurrentGraph) {
        ImGui::Begin("Visual Script Editor");
        ImGui::TextColored({0.8f, 0.2f, 0.2f, 1.0f}, "No graph loaded");
        ImGui::End();
        return;
    }

    RenderMenuBar();
    
    // Main window
    ImGui::SetNextWindowPos({0, 25}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize - ImVec2(0, 50), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Visual Script Editor", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize)) {
        ImGui::Columns(3);
        
        // Left panel: Node library
        if (m_ShowNodeLibrary) {
            RenderNodeLibrary();
        }
        ImGui::NextColumn();
        
        // Center panel: Canvas
        RenderGraphCanvas();
        ImGui::NextColumn();
        
        // Right panel: Properties
        if (m_ShowPropertiesPanel) {
            RenderPropertiesPanel();
        }
        
        ImGui::Columns(1);
    }
    ImGui::End();

    RenderStatusBar();
    
    // Additional panels
    if (m_ShowGeneratedCode) {
        ImGui::Begin("Generated Code", &m_ShowGeneratedCode);
        ImGui::TextUnformatted(m_GeneratedCode.c_str());
        ImGui::End();
    }

    if (m_ShowValidationErrors && !m_ValidationErrors.empty()) {
        ImGui::Begin("Validation Errors", &m_ShowValidationErrors);
        for (const auto& error : m_ValidationErrors) {
            ImGui::BulletText("%s", error.c_str());
        }
        ImGui::End();
    }
}

void VisualScriptEditor::RenderMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New##Graph")) {
                // Would show dialog to select graph type
                NewGraph("NewGraph", GraphType::BehaviorTree);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Open")) {
                // Would show file dialog
            }
            if (ImGui::MenuItem("Save", "Ctrl+S")) {
                SaveGraph();
            }
            if (ImGui::MenuItem("Save As")) {
                // Would show file dialog
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Close")) {
                CloseGraph();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, CanUndo())) {
                Undo();
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, CanRedo())) {
                Redo();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Delete Node", "Delete", false, m_SelectedNodeId != 0)) {
                DeleteNode(m_SelectedNodeId);
                m_SelectedNodeId = 0;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tools")) {
            if (ImGui::MenuItem("Generate Code")) {
                GenerateCode();
            }
            if (ImGui::MenuItem("Compile & Reload")) {
                CompileAndReload();
            }
            if (ImGui::MenuItem("Execute Graph")) {
                ExecuteGraph();
            }
            ImGui::Separator();
            ImGui::MenuItem("Show Node IDs", nullptr, &m_ShowNodeIds);
            ImGui::MenuItem("Auto-Reload", nullptr, &m_AutoReload);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Node Library", nullptr, &m_ShowNodeLibrary);
            ImGui::MenuItem("Properties", nullptr, &m_ShowPropertiesPanel);
            ImGui::MenuItem("Generated Code", nullptr, &m_ShowGeneratedCode);
            ImGui::MenuItem("Validation Errors", nullptr, &m_ShowValidationErrors);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void VisualScriptEditor::RenderNodeLibrary() {
    ImGui::Text("Node Library");
    ImGui::Separator();
    
    for (const auto& category : NodeRegistry::Get().GetCategories()) {
        if (ImGui::TreeNode(category.c_str())) {
            // Get definitions for this category
            for (const auto& [type, def] : NodeRegistry::Get().GetAllDefinitions()) {
                if (def.category == category) {
                    if (ImGui::Selectable(def.name.c_str())) {
                        // Start dragging a node
                        ImGui::SetDragDropPayload("NODE_TYPE", &type, sizeof(NodeType));
                    }
                }
            }
            ImGui::TreePop();
        }
    }
}

void VisualScriptEditor::RenderGraphCanvas() {
    ImGui::Text("Graph: %s", m_CurrentGraph->GetName().c_str());
    if (m_IsModified) ImGui::SameLine();
    ImGui::SameLine();
    ImGui::TextColored({1.0f, 1.0f, 0.0f, 1.0f}, m_IsModified ? "[Modified]" : "");
    
    ImGui::Separator();

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();

    // Draw background
    drawList->AddRectFilled(canvasPos, {canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y}, 0xFF1a1a1a);

    // Draw grid
    const float GRID_STEP = 20.0f;
    for (float x = 0; x < canvasSize.x; x += GRID_STEP) {
        drawList->AddLine({canvasPos.x + x, canvasPos.y}, {canvasPos.x + x, canvasPos.y + canvasSize.y}, 0xFF333333);
    }
    for (float y = 0; y < canvasSize.y; y += GRID_STEP) {
        drawList->AddLine({canvasPos.x, canvasPos.y + y}, {canvasPos.x + canvasSize.x, canvasPos.y + y}, 0xFF333333);
    }

    // Render all nodes
    for (const auto& [id, node] : m_CurrentGraph->GetAllNodes()) {
        RenderNode(id);
    }

    // Render connections
    for (const auto& conn : m_CurrentGraph->GetAllConnections()) {
        auto sourceNode = m_CurrentGraph->GetNode(conn.sourceNodeId);
        auto targetNode = m_CurrentGraph->GetNode(conn.targetNodeId);
        if (sourceNode && targetNode) {
            Vec2 from = sourceNode->GetPosition() + Vec2(100, 40);  // Approximate center
            Vec2 to = targetNode->GetPosition();
            RenderConnectionLine(from, to);
        }
    }

    // Render current connection being drawn
    if (m_IsConnecting) {
        if (auto node = m_CurrentGraph->GetNode(m_ConnectionSourceNodeId)) {
            Vec2 from = node->GetPosition() + Vec2(100, 40);
            RenderConnectionLine(from, m_ConnectionCurrentPos);
        }
    }

    ImGui::InvisibleButton("canvas", canvasSize);
    
    // Handle canvas interaction
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            m_CanvasOffset.x += ImGui::GetIO().MouseDelta.x;
            m_CanvasOffset.y += ImGui::GetIO().MouseDelta.y;
        }
        
        if (ImGui::IsMouseScrolling()) {
            m_CanvasScale += ImGui::GetIO().MouseWheel * 0.1f;
            m_CanvasScale = std::clamp(m_CanvasScale, 0.1f, 3.0f);
        }
    }

    // Context menu
    if (ImGui::BeginPopupContextItem("canvas_context")) {
        RenderCanvasContextMenu();
        ImGui::EndPopup();
    }
}

void VisualScriptEditor::RenderPropertiesPanel() {
    ImGui::Text("Properties");
    ImGui::Separator();
    
    if (m_SelectedNodeId == 0) {
        ImGui::TextDisabled("(Select a node to edit)");
        return;
    }

    auto node = m_CurrentGraph->GetNode(m_SelectedNodeId);
    if (!node) {
        ImGui::TextDisabled("(Invalid node)");
        return;
    }

    ImGui::Text("Node: %s", node->GetName().c_str());
    ImGui::Text("ID: %u", m_SelectedNodeId);
    ImGui::Text("Type: %d", static_cast<int>(node->GetNodeType()));

    ImGui::Separator();
    ImGui::Text("Properties:");
    
    for (const auto& [key, value] : node->GetAllProperties()) {
        ImGui::Text("%s: ", key.c_str());
        ImGui::SameLine();
        if (value.is_string()) {
            ImGui::TextWrapped("%s", value.get<std::string>().c_str());
        } else if (value.is_number()) {
            ImGui::Text("%f", value.get<double>());
        } else if (value.is_boolean()) {
            ImGui::Text("%s", value.get<bool>() ? "true" : "false");
        } else {
            ImGui::Text("%s", value.dump().c_str());
        }
    }

    ImGui::Separator();
    ImGui::Text("Ports:");
    
    if (!node->GetInputPorts().empty()) {
        ImGui::Text("Inputs:");
        for (const auto& port : node->GetInputPorts()) {
            ImGui::BulletText("%s (%s)", port.name.c_str(), port.dataType.c_str());
        }
    }

    if (!node->GetOutputPorts().empty()) {
        ImGui::Text("Outputs:");
        for (const auto& port : node->GetOutputPorts()) {
            ImGui::BulletText("%s (%s)", port.name.c_str(), port.dataType.c_str());
        }
    }
}

void VisualScriptEditor::RenderStatusBar() {
    ImGui::Begin("Status Bar", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos({0, ImGui::GetIO().DisplaySize.y - 25});
    ImGui::SetWindowSize({ImGui::GetIO().DisplaySize.x, 25});
    
    ImGui::Text("Nodes: %zu | Connections: %zu | Scale: %.2f%%", 
                m_CurrentGraph->GetNodeCount(), 
                m_CurrentGraph->GetConnectionCount(),
                m_CanvasScale * 100);
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    
    if (!m_ValidationErrors.empty()) {
        ImGui::TextColored({1.0f, 0.0f, 0.0f, 1.0f}, "Errors: %zu", m_ValidationErrors.size());
    }
    
    ImGui::End();
}

void VisualScriptEditor::RenderNode(uint32_t nodeId) {
    auto node = m_CurrentGraph->GetNode(nodeId);
    if (!node) return;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    Vec2 pos = node->GetPosition();
    float width = node->GetWidth();
    float height = node->GetHeight();

    // Node rectangle
    uint32_t bgColor = node->GetColor();
    if (m_SelectedNodeId == nodeId) {
        bgColor = 0xFFFFFF00;  // Yellow highlight
    }

    ImVec2 min = {pos.x, pos.y};
    ImVec2 max = {pos.x + width, pos.y + height};
    drawList->AddRectFilled(min, max, bgColor);
    drawList->AddRect(min, max, 0xFFFFFFFF, 0.0f, ImDrawCornerFlags_None, 2.0f);

    // Node title
    drawList->AddText({pos.x + 5, pos.y + 5}, 0xFF000000, node->GetName().c_str());
    
    if (m_ShowNodeIds) {
        char idStr[32];
        snprintf(idStr, sizeof(idStr), "#%u", nodeId);
        ImVec2 textSize = ImGui::CalcTextSize(idStr);
        drawList->AddText({pos.x + width - textSize.x - 5, pos.y + height - 15}, 0xFF000000, idStr);
    }

    // Render ports
    float portY = pos.y + 25;
    for (const auto& port : node->GetInputPorts()) {
        RenderNodePort(port, {pos.x, portY}, true);
        portY += 15;
    }

    portY = pos.y + 25;
    for (const auto& port : node->GetOutputPorts()) {
        RenderNodePort(port, {pos.x + width, portY}, false);
        portY += 15;
    }
}

void VisualScriptEditor::RenderNodePort(const Port& port, const Vec2& nodePos, bool isInput) {
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    float portRadius = 5.0f;
    
    Vec2 portPos = nodePos + Vec2(isInput ? -portRadius : portRadius, 0);
    uint32_t portColor = port.isConnected ? 0xFF00FF00 : 0xFF888888;
    
    drawList->AddCircleFilled({portPos.x, portPos.y}, portRadius, portColor);
}

void VisualScriptEditor::RenderConnectionLine(const Vec2& from, const Vec2& to) {
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    
    // Bezier curve
    Vec2 cp1 = from + Vec2((to.x - from.x) * 0.5f, 0);
    Vec2 cp2 = to + Vec2((from.x - to.x) * 0.5f, 0);
    
    drawList->AddBezierCurve(
        {from.x, from.y}, {cp1.x, cp1.y}, {cp2.x, cp2.y}, {to.x, to.y},
        0xFF00FF00, 2.0f, 8
    );
}

void VisualScriptEditor::RenderCanvasContextMenu() {
    // Add node options
}

void VisualScriptEditor::RenderNodeContextMenu() {
    if (ImGui::MenuItem("Delete")) {
        DeleteNode(m_SelectedNodeId);
    }
    if (ImGui::MenuItem("Duplicate")) {
        // Implement duplication
    }
}

void VisualScriptEditor::DeleteNode(uint32_t nodeId) {
    m_CurrentGraph->RemoveNode(nodeId);
    MarkGraphModified();
}

void VisualScriptEditor::DeleteConnection(const Connection& conn) {
    m_CurrentGraph->Disconnect(conn.sourceNodeId, conn.sourcePortId,
                               conn.targetNodeId, conn.targetPortId);
    MarkGraphModified();
}

void VisualScriptEditor::StartConnection(uint32_t nodeId, uint32_t portId) {
    m_IsConnecting = true;
    m_ConnectionSourceNodeId = nodeId;
    m_ConnectionSourcePortId = portId;
}

void VisualScriptEditor::EndConnection(uint32_t nodeId, uint32_t portId) {
    if (m_IsConnecting) {
        m_CurrentGraph->Connect(m_ConnectionSourceNodeId, m_ConnectionSourcePortId,
                               nodeId, portId);
        MarkGraphModified();
    }
    CancelConnection();
}

void VisualScriptEditor::CancelConnection() {
    m_IsConnecting = false;
    m_ConnectionSourceNodeId = 0;
    m_ConnectionSourcePortId = 0;
}

void VisualScriptEditor::GenerateCode() {
    if (!m_CurrentGraph) return;

    auto generator = CodeGeneratorFactory::CreateGenerator(m_CurrentGraph->GetType());
    if (generator) {
        m_GeneratedCode = generator->GenerateCode(m_CurrentGraph);
        m_ShowGeneratedCode = true;
    }
}

void VisualScriptEditor::CompileAndReload() {
    GenerateCode();
    // In a real implementation, this would trigger the hot-reload system
}

void VisualScriptEditor::ExecuteGraph() {
    if (m_CurrentGraph) {
        m_IsExecuting = true;
        m_CurrentGraph->Execute();
        m_IsExecuting = false;
    }
}

void VisualScriptEditor::PauseExecution() {
    m_IsExecuting = false;
}

void VisualScriptEditor::StopExecution() {
    m_IsExecuting = false;
    m_CurrentGraph->Reset();
}

void VisualScriptEditor::ValidateGraph() {
    m_ValidationErrors.clear();
    if (m_CurrentGraph) {
        m_ValidationErrors = m_CurrentGraph->GetValidationErrors();
        m_ShowValidationErrors = !m_ValidationErrors.empty();
    }
}

void VisualScriptEditor::Undo() {
    if (CanUndo()) {
        m_RedoStack.push_back(m_CurrentGraph->Serialize());
        auto state = m_UndoStack.back();
        m_UndoStack.pop_back();
        m_CurrentGraph->Deserialize(state);
    }
}

void VisualScriptEditor::Redo() {
    if (CanRedo()) {
        m_UndoStack.push_back(m_CurrentGraph->Serialize());
        auto state = m_RedoStack.back();
        m_RedoStack.pop_back();
        m_CurrentGraph->Deserialize(state);
    }
}

bool VisualScriptEditor::CanUndo() const {
    return !m_UndoStack.empty();
}

bool VisualScriptEditor::CanRedo() const {
    return !m_RedoStack.empty();
}

void VisualScriptEditor::AddRecentFile(const std::string& path) {
    auto it = std::find(m_RecentFiles.begin(), m_RecentFiles.end(), path);
    if (it != m_RecentFiles.end()) {
        m_RecentFiles.erase(it);
    }
    m_RecentFiles.insert(m_RecentFiles.begin(), path);
    if (m_RecentFiles.size() > MAX_RECENT_FILES) {
        m_RecentFiles.pop_back();
    }
}

Vec2 VisualScriptEditor::ConvertScreenToGraph(const Vec2& screenPos) const {
    return (screenPos - m_CanvasOffset) / m_CanvasScale;
}

Vec2 VisualScriptEditor::ConvertGraphToScreen(const Vec2& graphPos) const {
    return graphPos * m_CanvasScale + m_CanvasOffset;
}
