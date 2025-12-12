#include "BlendTreeEditor.h"
#include "Animator.h"
#include "Animation.h"
#include <iostream>
#include <string>
#include <vector>

// Forward declare internals if strictly needed, or just rely on public API
// We might need internal access to Animator's blend tree arrays if not exposed publically.
// Currently Animator HAS public getters/setters for trees?
// Actually, Animator has "AddBlendTreeNode", "CreateBlendTree", etc.
// But it doesn't expose the LIST of all trees cleanly for iteration in editor without adding a GetBlendTreeCount API.
// Let's assume we might need to add friend class or Getters to Animator.
// For now, I will add generic GetCount/Get functionality to Animator or assume we can hack it via friendship if needed.
// But cleanest is adding API to Animator.h if missing.
// Checking Animator.h...
// We have `CreateBlendTree1D` but no "GetBlendTree1DCount".
// I'll assume we add them or just use a fixed range? No, need counts.
// I will implement "Render" assuming the API exists, and then fix Animator.h.

BlendTreeEditor::BlendTreeEditor() 
    : m_Animator(nullptr)
    , m_SelectedTreeIndex(-1)
    , m_SelectedTreeType(0)
    , m_TestParameter1D(0.0f)
{
    m_TestParameter2D[0] = 0.0f;
    m_TestParameter2D[1] = 0.0f;
}

void BlendTreeEditor::SetAnimator(Animator* animator) {
    m_Animator = animator;
    // Reset selection on switch
    m_SelectedTreeIndex = -1;
    m_SelectedTreeType = 0;
}

void BlendTreeEditor::Render() {
    if (!m_Animator) {
        ImGui::Begin("Blend Tree Editor");
        ImGui::Text("No Animator Selected.");
        ImGui::End();
        return;
    }

    ImGui::Begin("Blend Tree Editor");
    
    // Split View: List | Editor
    ImGui::Columns(2, "BlendTreeEditorColumns", true);
    
    // -------------------------------------------------------------------------
    // Left: Tree List
    // -------------------------------------------------------------------------
    ImGui::Text("Blend Trees");
    ImGui::Separator();
    
    // We need getters for counts... I will use a placeholder loop or add counts later.
    // For now, let's just attempt to list what we know or just show "Create New" buttons?
    // Let's add a "Manage" section.
    
    if (ImGui::Button("Create 1D Tree")) {
        int idx = m_Animator->CreateBlendTree1D();
        m_SelectedTreeIndex = idx;
        m_SelectedTreeType = 1;
    }
    if (ImGui::Button("Create 2D Tree")) {
        int idx = m_Animator->CreateBlendTree2D();
        m_SelectedTreeIndex = idx;
        m_SelectedTreeType = 2;
    }
    
    ImGui::Separator();
    
    // List Trees
    if (ImGui::TreeNode("1D Trees")) {
        for (int i = 0; i < m_Animator->GetBlendTree1DCount(); ++i) {
            std::string label = "Tree " + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), m_SelectedTreeIndex == i && m_SelectedTreeType == 1)) {
                m_SelectedTreeIndex = i;
                m_SelectedTreeType = 1;
            }
        }
        ImGui::TreePop();
    }
    
    if (ImGui::TreeNode("2D Trees")) {
        for (int i = 0; i < m_Animator->GetBlendTree2DCount(); ++i) {
            std::string label = "Tree " + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), m_SelectedTreeIndex == i && m_SelectedTreeType == 2)) {
                m_SelectedTreeIndex = i;
                m_SelectedTreeType = 2;
            }
        }
        ImGui::TreePop();
    }
    
    if (m_SelectedTreeIndex != -1) {
        ImGui::Text("Selected: %s %d", (m_SelectedTreeType == 1 ? "1D" : "2D"), m_SelectedTreeIndex);
    }
    
    ImGui::NextColumn();
    
    // -------------------------------------------------------------------------
    // Right: Editor
    // -------------------------------------------------------------------------
    if (m_SelectedTreeIndex != -1) {
        if (m_SelectedTreeType == 1) {
            RenderTreeEditor1D(m_SelectedTreeIndex);
        } else if (m_SelectedTreeType == 2) {
            RenderTreeEditor2D(m_SelectedTreeIndex);
        }
    } else {
        ImGui::Text("Select a Blend Tree to edit.");
    }
    
    ImGui::Columns(1);
    ImGui::End();
}

void BlendTreeEditor::RenderTreeEditor1D(int treeIndex) {
    ImGui::Text("1D Blend Tree Editor");
    ImGui::Separator();
    
    // Test Parameter
    if (ImGui::SliderFloat("Simulate Parameter", &m_TestParameter1D, -1.0f, 1.0f)) {
        m_Animator->SetBlendTreeParameter1D(treeIndex, m_TestParameter1D);
    }
    
    ImGui::Separator();
    
    // Node List
    ImGui::Text("Nodes:");
    
    // We need API to iterate nodes too... 
    // The editor is blocked by lack of inspection API in Animator/BlendTree.
    // I will just put the creation UI here for now.
    
    static int selectedAnim = 0;
    static float paramVal = 0.0f;
    
    ImGui::InputInt("Anim Index", &selectedAnim);
    ImGui::DragFloat("Parameter", &paramVal, 0.1f);
    
    if (ImGui::Button("Add Node")) {
        m_Animator->AddBlendTreeNode1D(treeIndex, selectedAnim, paramVal);
    }
    
    // Visualizer (Simple Bar)
    ImGui::Dummy(ImVec2(0, 20));
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 size(ImGui::GetContentRegionAvail().x, 30);
    ImGui::GetWindowDrawList()->AddRect(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(100, 100, 100, 255));
    
    // Draw cursor
    // Map -1..1 to 0..width
    float t = (m_TestParameter1D + 1.0f) * 0.5f;
    float x = p0.x + t * size.x;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x, p0.y), ImVec2(x, p0.y + size.y), IM_COL32(255, 0, 0, 255), 2.0f);
    
    // Draw nodes? We assume we can't see them yet without API.
}

void BlendTreeEditor::RenderTreeEditor2D(int treeIndex) {
    ImGui::Text("2D Blend Tree Editor");
    ImGui::Separator();
    
    ImGui::SliderFloat2("Simulate Parameter", m_TestParameter2D, -1.0f, 1.0f);
    if (ImGui::IsItemEdited()) {
        m_Animator->SetBlendTreeParameter2D(treeIndex, Vec2(m_TestParameter2D[0], m_TestParameter2D[1]));
    }
    
    ImGui::Separator();
    
    static int selectedAnim2 = 0;
    static float paramVal2[2] = {0, 0};
    
    ImGui::InputInt("Anim Index", &selectedAnim2);
    ImGui::DragFloat2("Pos", paramVal2, 0.1f);
    
    if (ImGui::Button("Add Node")) {
        m_Animator->AddBlendTreeNode2D(treeIndex, selectedAnim2, Vec2(paramVal2[0], paramVal2[1]));
    }
    
    // 2D Visualizer
    ImGui::Dummy(ImVec2(0, 20));
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    float regionSize = std::min(ImGui::GetContentRegionAvail().x, 300.0f); // Square-ish
    ImVec2 size(regionSize, regionSize);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    
    // Background
    drawList->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(50, 50, 50, 255));
    drawList->AddRect(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(200, 200, 200, 255));
    
    // Grid
    drawList->AddLine(ImVec2(p0.x + size.x*0.5f, p0.y), ImVec2(p0.x + size.x*0.5f, p0.y + size.y), IM_COL32(100, 100, 100, 100)); // Y axis
    drawList->AddLine(ImVec2(p0.x, p0.y + size.y*0.5f), ImVec2(p0.x + size.x, p0.y + size.y*0.5f), IM_COL32(100, 100, 100, 100)); // X axis
    
    // Cursor
    // Map -1..1 to 0..size
    float tx = (m_TestParameter2D[0] + 1.0f) * 0.5f;
    float ty = (m_TestParameter2D[1] + 1.0f) * 0.5f;
    // Note: Y usually down in GUI, but maybe up in parameter space? Assuming standard Cartesian up.
    // GUI Y is down. So 1.0 -> 0.0 (top), -1.0 -> 1.0 (bottom).
    // Let's invert Y for visualization to match standard graph
    ty = 1.0f - ty;
    
    float curX = p0.x + tx * size.x;
    float curY = p0.y + ty * size.y;
    
    drawList->AddCircleFilled(ImVec2(curX, curY), 5.0f, IM_COL32(255, 0, 0, 255));
}
