#pragma once

#include "Math/Vec3.h"
#include <imgui.h>

class SoftBodyLODConfig;
struct SoftBodyStats;

/**
 * @brief Custom ImGui widgets for soft body editor
 */
namespace SoftBodyEditorWidgets {
    
    /**
     * @brief Vec3 input with individual X, Y, Z fields
     * @param label Widget label
     * @param value Vec3 to edit
     * @param speed Drag speed
     * @return True if value changed
     */
    inline bool Vec3Input(const char* label, Vec3& value, float speed = 0.1f) {
        ImGui::PushID(label);
        ImGui::Text("%s", label);
        ImGui::SameLine(150);
        
        bool changed = false;
        ImGui::PushItemWidth(60);
        changed |= ImGui::DragFloat("##X", &value.x, speed);
        ImGui::SameLine();
        changed |= ImGui::DragFloat("##Y", &value.y, speed);
        ImGui::SameLine();
        changed |= ImGui::DragFloat("##Z", &value.z, speed);
        ImGui::PopItemWidth();
        
        ImGui::PopID();
        return changed;
    }
    
    /**
     * @brief Stiffness slider with visual feedback
     * @param label Widget label
     * @param value Stiffness value (0.0 - 1.0)
     * @return True if value changed
     */
    inline bool StiffnessSlider(const char* label, float& value) {
        ImGui::PushID(label);
        ImGui::Text("%s", label);
        ImGui::SameLine(150);
        
        // Color based on value
        ImVec4 color;
        if (value < 0.3f) {
            color = ImVec4(0.8f, 0.3f, 0.3f, 1.0f); // Red - soft
        } else if (value < 0.7f) {
            color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f); // Yellow - medium
        } else {
            color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f); // Green - stiff
        }
        
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, color);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, color);
        
        ImGui::PushItemWidth(200);
        bool changed = ImGui::SliderFloat("##slider", &value, 0.0f, 1.0f, "%.2f");
        ImGui::PopItemWidth();
        
        ImGui::PopStyleColor(2);
        ImGui::PopID();
        
        return changed;
    }
    
    /**
     * @brief LOD level indicator with color coding
     * @param currentLOD Current LOD level
     * @param maxLOD Maximum LOD level
     */
    inline void LODIndicator(int currentLOD, int maxLOD) {
        ImGui::Text("Current LOD:");
        ImGui::SameLine();
        
        for (int i = 0; i <= maxLOD; ++i) {
            if (i > 0) ImGui::SameLine();
            
            if (i == currentLOD) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
                ImGui::Button("●");
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
                ImGui::Button("○");
                ImGui::PopStyleColor();
            }
            
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("LOD %d", i);
            }
        }
        
        ImGui::SameLine();
        ImGui::Text("(Level %d)", currentLOD);
    }
    
    /**
     * @brief Performance graph for statistics
     * @param stats Soft body statistics
     */
    inline void PerformanceGraph(const SoftBodyStats& stats) {
        float values[] = {
            static_cast<float>(stats.updateTimeMs),
            static_cast<float>(stats.collisionGenTimeMs),
            static_cast<float>(stats.tearCheckTimeMs)
        };
        
        const char* labels[] = {"Update", "Collision", "Tear Check"};
        
        ImGui::Text("Performance (ms):");
        
        for (int i = 0; i < 3; ++i) {
            ImGui::Text("%s:", labels[i]);
            ImGui::SameLine(120);
            
            // Color based on performance
            ImVec4 color;
            if (values[i] < 1.0f) {
                color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f); // Green - good
            } else if (values[i] < 5.0f) {
                color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f); // Yellow - ok
            } else {
                color = ImVec4(0.8f, 0.3f, 0.3f, 1.0f); // Red - slow
            }
            
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
            ImGui::ProgressBar(values[i] / 10.0f, ImVec2(150, 0), "");
            ImGui::PopStyleColor();
            
            ImGui::SameLine();
            ImGui::Text("%.2f ms", values[i]);
        }
    }
    
    /**
     * @brief Help marker with tooltip
     * @param desc Description text
     */
    inline void HelpMarker(const char* desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}
