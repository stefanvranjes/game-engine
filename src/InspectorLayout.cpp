#include "InspectorLayout.h"
#include "ColorScheme.h"
#include <imgui_internal.h>

// Static member initialization
std::map<std::string, bool> InspectorLayout::s_SectionStates;
std::map<std::string, float> InspectorLayout::s_ColumnWidths;
float InspectorLayout::s_LabelWidth = 120.0f;
InspectorLayout::SectionConfig InspectorLayout::s_SectionConfig;
int InspectorLayout::s_IndentLevel = 0;
const float InspectorLayout::SECTION_HEADER_HEIGHT = 28.0f;

bool InspectorLayout::BeginSection(const std::string& id, const std::string& label, 
                                    bool& expanded, const ImVec4& headerColor) {
    // Load or initialize expansion state from persistent storage
    if (s_SectionStates.find(id) == s_SectionStates.end()) {
        s_SectionStates[id] = expanded;
    }
    expanded = s_SectionStates[id];

    ImGui::PushID(id.c_str());

    // Render header
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 headerPos = ImGui::GetCursorScreenPos();
    ImVec2 headerSize(ImGui::GetContentRegionAvail().x, SECTION_HEADER_HEIGHT);

    // Draw header background
    ImU32 headerColor32 = ImGui::GetColorU32(headerColor);
    draw_list->AddRectFilled(headerPos, ImVec2(headerPos.x + headerSize.x, headerPos.y + headerSize.y),
                            headerColor32);

    // Draw border
    draw_list->AddRect(headerPos, ImVec2(headerPos.x + headerSize.x, headerPos.y + headerSize.y),
                      ImGui::GetColorU32(ImVec4(0.6f, 0.6f, 0.6f, 1.0f)), 0.0f, 0, 1.0f);

    ImGui::SetCursorScreenPos(ImVec2(headerPos.x + 8, headerPos.y + 4));

    // Render expand/collapse button
    const char* icon = expanded ? "▼" : "▶";
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.1f));

    if (ImGui::Button(icon, ImVec2(20, 20))) {
        expanded = !expanded;
        s_SectionStates[id] = expanded;
    }

    ImGui::PopStyleColor(2);

    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", label.c_str());

    ImGui::NewLine();

    ImGui::PopID();

    if (expanded) {
        ImGui::Indent(12.0f);
        return true;
    }
    return false;
}

void InspectorLayout::EndSection() {
    ImGui::Unindent(12.0f);
}

void InspectorLayout::BeginPropertyRow(const std::string& label, float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    ImGui::PushID(label.c_str());

    // Calculate actual label width based on available space
    float contentRegionWidth = ImGui::GetContentRegionAvail().x;
    float actualLabelWidth = labelWidth > contentRegionWidth ? contentRegionWidth * 0.4f : labelWidth;

    // Render label
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", label.c_str());

    // Move to value column, accounting for indentation
    float indentSize = s_IndentLevel * 15.0f;
    ImGui::SameLine(actualLabelWidth + indentSize, 8.0f);
}

void InspectorLayout::EndPropertyRow() {
    ImGui::PopID();
}

void InspectorLayout::BeginTwoColumnLayout(float labelWidth) {
    s_LabelWidth = labelWidth;
    ImGui::Columns(2, nullptr, false);
    ImGui::SetColumnWidth(0, labelWidth);
}

void InspectorLayout::NextColumn() {
    ImGui::NextColumn();
}

void InspectorLayout::EndTwoColumnLayout() {
    ImGui::Columns(1);
}

void InspectorLayout::PropertyVector3(const std::string& label, float* values, 
                                     float resetValue, float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    BeginPropertyRow(label, labelWidth);

    ImGui::SetNextItemWidth(-1);
    ImGui::DragFloat3(("##" + label).c_str(), values, 0.1f);

    ImGui::SameLine();
    ImGui::PushID(("reset" + label).c_str());
    if (ImGui::Button("R", ImVec2(25, 0))) {
        values[0] = resetValue;
        values[1] = resetValue;
        values[2] = resetValue;
    }
    ImGui::PopID();

    EndPropertyRow();
}

void InspectorLayout::PropertyColor(const std::string& label, float* color, float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    BeginPropertyRow(label, labelWidth);
    ImGui::SetNextItemWidth(-1);
    ImGui::ColorEdit4(("##" + label).c_str(), color);
    EndPropertyRow();
}

void InspectorLayout::PropertySlider(const std::string& label, float* value, 
                                    float min, float max, float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    BeginPropertyRow(label, labelWidth);
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat(("##" + label).c_str(), value, min, max);
    EndPropertyRow();
}

void InspectorLayout::PropertyCheckbox(const std::string& label, bool* value, float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    BeginPropertyRow(label, labelWidth);
    ImGui::Checkbox(("##" + label).c_str(), value);
    EndPropertyRow();
}

void InspectorLayout::PropertyText(const std::string& label, char* buffer, size_t bufferSize, 
                                  float labelWidth) {
    if (labelWidth == 0.0f) {
        labelWidth = s_LabelWidth;
    }

    BeginPropertyRow(label, labelWidth);
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText(("##" + label).c_str(), buffer, bufferSize);
    EndPropertyRow();
}

void InspectorLayout::Separator() {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
}

void InspectorLayout::PushIndent(float amount) {
    ImGui::Indent(amount);
    s_IndentLevel++;
}

void InspectorLayout::PopIndent() {
    ImGui::Unindent(15.0f);
    if (s_IndentLevel > 0) {
        s_IndentLevel--;
    }
}

bool InspectorLayout::ColumnSplitter(const std::string& id, float& columnPos, 
                                    float minWidth, float maxWidth) {
    // Get or initialize column width
    if (s_ColumnWidths.find(id) == s_ColumnWidths.end()) {
        s_ColumnWidths[id] = columnPos;
    }

    float& currentWidth = s_ColumnWidths[id];

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 size = ImGui::GetWindowSize();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    // Calculate splitter position
    float splitterX = pos.x + currentWidth;

    ImVec2 splitterPos1(splitterX - 2, pos.y);
    ImVec2 splitterPos2(splitterX + 2, pos.y + size.y);

    ImGui::PushID(id.c_str());

    // Render splitter line
    draw_list->AddLine(ImVec2(splitterX, pos.y), ImVec2(splitterX, pos.y + size.y),
                      ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.5f, 1.0f)), 1.0f);

    ImGui::SetCursorScreenPos(splitterPos1);
    bool hovered = ImGui::InvisibleButton("##Splitter", ImVec2(4, size.y),
                                          ImGuiButtonFlags_FlattenChildren);

    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    if (ImGui::IsItemActive()) {
        float delta = ImGui::GetIO().MouseDelta.x;
        currentWidth = std::max(minWidth, std::min(maxWidth, currentWidth + delta));
        columnPos = currentWidth;
    }

    ImGui::PopID();

    return hovered;
}

float InspectorLayout::GetColumnWidth(const std::string& id) {
    auto it = s_ColumnWidths.find(id);
    if (it != s_ColumnWidths.end()) {
        return it->second;
    }
    return s_LabelWidth;
}

void InspectorLayout::SetColumnWidth(const std::string& id, float width) {
    s_ColumnWidths[id] = width;
}

void InspectorLayout::ResetLayout() {
    s_SectionStates.clear();
    s_ColumnWidths.clear();
    s_IndentLevel = 0;
}
