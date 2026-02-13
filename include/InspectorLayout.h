#pragma once

#include <imgui.h>
#include <string>
#include <map>
#include <vector>

/**
 * @brief Utility for creating collapsible inspector sections and resizable columns
 * 
 * Provides UI helpers for:
 * - Collapsible component sections with headers
 * - Property groups with visual separation
 * - Resizable column layouts
 * - Compound control layouts with labels and values
 */
class InspectorLayout {
public:
    /**
     * @brief Configuration for collapsible sections
     */
    struct SectionConfig {
        float labelWidth = 120.0f;         // Width of property labels
        float spacing = 4.0f;              // Vertical spacing between properties
        bool showComponentCount = true;    // Show component count in header
        bool allowRemove = true;           // Show remove button for components
        bool showIcon = true;              // Display component type icon
    };

    /**
     * @brief Begin a collapsible section (like a group header)
     * @param id Unique ID for the section (for state persistence)
     * @param label Display label/name
     * @param expanded Output: whether section is expanded
     * @param headerColor Color of the header (optional)
     * @return true if the section is expanded and content should be rendered
     * 
     * Usage:
     * bool expanded = true;
     * if (InspectorLayout::BeginSection("transform", "Transform", expanded)) {
     *     // Render properties here
     *     ImGui::DragFloat("X", &x);
     *     InspectorLayout::EndSection();
     * }
     */
    static bool BeginSection(const std::string& id, const std::string& label, 
                            bool& expanded, const ImVec4& headerColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

    /**
     * @brief End a collapsible section
     */
    static void EndSection();

    /**
     * @brief Render a property row with label and control
     * Automatically handles label column alignment
     * @param label Property name
     */
    static void BeginPropertyRow(const std::string& label, float labelWidth = 0.0f);

    /**
     * @brief End a property row
     */
    static void EndPropertyRow();

    /**
     * @brief Begin a two-column layout for inspector sections
     * @param labelWidth Width of the left (label) column
     */
    static void BeginTwoColumnLayout(float labelWidth);

    /**
     * @brief Move to the next column in layout
     */
    static void NextColumn();

    /**
     * @brief End the two-column layout
     */
    static void EndTwoColumnLayout();

    /**
     * @brief Helper for vector3 properties with label
     */
    static void PropertyVector3(const std::string& label, float* values, 
                               float resetValue = 0.0f, float labelWidth = 0.0f);

    /**
     * @brief Helper for color properties with label
     */
    static void PropertyColor(const std::string& label, float* color, 
                             float labelWidth = 0.0f);

    /**
     * @brief Helper for float slider property
     */
    static void PropertySlider(const std::string& label, float* value, 
                              float min, float max, float labelWidth = 0.0f);

    /**
     * @brief Helper for checkbox property
     */
    static void PropertyCheckbox(const std::string& label, bool* value, 
                                float labelWidth = 0.0f);

    /**
     * @brief Helper for text input property
     */
    static void PropertyText(const std::string& label, char* buffer, size_t bufferSize, 
                            float labelWidth = 0.0f);

    /**
     * @brief Draw a separator between sections
     */
    static void Separator();

    /**
     * @brief Push property indentation
     * Used for nested properties (indent from left)
     */
    static void PushIndent(float amount = 15.0f);

    /**
     * @brief Pop property indentation
     */
    static void PopIndent();

    /**
     * @brief Get the current label width being used
     */
    static float GetLabelWidth() { return s_LabelWidth; }

    /**
     * @brief Set the global label width for all properties
     */
    static void SetLabelWidth(float width) { s_LabelWidth = width; }

    /**
     * @brief Create a resizable column splitter
     * @param id Unique column ID
     * @param columnPos Current column position (modified by this function)
     * @param minWidth Minimum width for left column
     * @param maxWidth Maximum width for left column
     * @param width Reference to stored width
     */
    static bool ColumnSplitter(const std::string& id, float& columnPos, 
                               float minWidth, float maxWidth);

    /**
     * @brief Get stored column width
     */
    static float GetColumnWidth(const std::string& id);

    /**
     * @brief Set stored column width
     */
    static void SetColumnWidth(const std::string& id, float width);

    /**
     * @brief Reset all column widths and section states
     */
    static void ResetLayout();

    /**
     * @brief Set the configuration for sections
     */
    static void SetSectionConfig(const SectionConfig& config) { s_SectionConfig = config; }

    /**
     * @brief Get current section configuration
     */
    static const SectionConfig& GetSectionConfig() { return s_SectionConfig; }

private:
    static std::map<std::string, bool> s_SectionStates;      // Expanded/collapsed states
    static std::map<std::string, float> s_ColumnWidths;      // Resizable column widths
    static float s_LabelWidth;
    static SectionConfig s_SectionConfig;
    static int s_IndentLevel;

    static const float SECTION_HEADER_HEIGHT;
};
