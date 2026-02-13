#include "ColorScheme.h"
#include <algorithm>
#include <cctype>

// Static member initialization
std::map<ColorScheme::ColorCategory, ImVec4> ColorScheme::s_Colors;
bool ColorScheme::s_DarkTheme = true;
bool ColorScheme::s_Initialized = false;

void ColorScheme::Initialize() {
    if (s_Initialized) return;
    InitializeDefaultColors();
    s_Initialized = true;
}

void ColorScheme::InitializeDefaultColors() {
    // Object type colors
    s_Colors[ColorCategory::ObjectDefault] = ImVec4(0.7f, 0.7f, 0.7f, 0.2f);
    s_Colors[ColorCategory::ObjectMesh] = ImVec4(0.2f, 0.6f, 1.0f, 0.2f);      // Blue
    s_Colors[ColorCategory::ObjectLight] = ImVec4(1.0f, 0.9f, 0.2f, 0.2f);     // Yellow
    s_Colors[ColorCategory::ObjectCamera] = ImVec4(0.4f, 0.8f, 1.0f, 0.2f);    // Cyan
    s_Colors[ColorCategory::ObjectParticle] = ImVec4(1.0f, 0.6f, 0.2f, 0.2f);  // Orange
    s_Colors[ColorCategory::ObjectAudio] = ImVec4(0.5f, 0.8f, 0.5f, 0.2f);     // Green
    s_Colors[ColorCategory::ObjectPhysics] = ImVec4(0.9f, 0.3f, 0.3f, 0.2f);   // Red
    s_Colors[ColorCategory::ObjectAnimation] = ImVec4(0.8f, 0.4f, 0.8f, 0.2f); // Purple

    // UI State colors
    s_Colors[ColorCategory::StateNormal] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    s_Colors[ColorCategory::StateHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.8f);
    s_Colors[ColorCategory::StateSelected] = ImVec4(0.2f, 0.6f, 1.0f, 0.8f);
    s_Colors[ColorCategory::StateDisabled] = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);

    // Component category colors
    s_Colors[ColorCategory::ComponentTransform] = ImVec4(1.0f, 0.5f, 0.5f, 0.1f);    // Red
    s_Colors[ColorCategory::ComponentMaterial] = ImVec4(0.2f, 0.8f, 0.2f, 0.1f);     // Green
    s_Colors[ColorCategory::ComponentPhysics] = ImVec4(0.5f, 0.2f, 0.8f, 0.1f);      // Purple
    s_Colors[ColorCategory::ComponentAudio] = ImVec4(0.0f, 0.8f, 0.8f, 0.1f);        // Cyan
    s_Colors[ColorCategory::ComponentAnimation] = ImVec4(0.8f, 0.6f, 0.0f, 0.1f);    // Orange
    s_Colors[ColorCategory::ComponentLogic] = ImVec4(0.3f, 0.6f, 0.3f, 0.1f);        // Dark Green

    // UI element colors
    s_Colors[ColorCategory::ElementBackground] = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
    s_Colors[ColorCategory::ElementBorder] = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
    s_Colors[ColorCategory::ElementHighlight] = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);
    s_Colors[ColorCategory::ElementShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.5f);
}

ImVec4 ColorScheme::GetColor(ColorCategory category) {
    if (!s_Initialized) Initialize();

    auto it = s_Colors.find(category);
    if (it != s_Colors.end()) {
        return it->second;
    }
    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
}

ImU32 ColorScheme::GetColorU32(ColorCategory category) {
    return ImGui::GetColorU32(GetColor(category));
}

void ColorScheme::SetColor(ColorCategory category, const ImVec4& color) {
    if (!s_Initialized) Initialize();
    s_Colors[category] = color;
}

void ColorScheme::SetColor(ColorCategory category, float r, float g, float b, float a) {
    SetColor(category, ImVec4(r, g, b, a));
}

ImVec4 ColorScheme::GetObjectTypeColor(const std::string& objectTypeName) {
    return GetColor(CategorizeObjectType(objectTypeName));
}

ImVec4 ColorScheme::GetComponentTypeColor(const std::string& componentTypeName) {
    return GetColor(CategorizeComponentType(componentTypeName));
}

ImVec4 ColorScheme::Dim(const ImVec4& color, float factor) {
    return ImVec4(
        color.x * factor,
        color.y * factor,
        color.z * factor,
        color.w
    );
}

ImVec4 ColorScheme::Brighten(const ImVec4& color, float factor) {
    return ImVec4(
        std::min(1.0f, color.x * factor),
        std::min(1.0f, color.y * factor),
        std::min(1.0f, color.z * factor),
        color.w
    );
}

ImVec4 ColorScheme::WithAlpha(const ImVec4& color, float alpha) {
    return ImVec4(color.x, color.y, color.z, alpha);
}

ImVec4 ColorScheme::GetContrast(const ImVec4& color) {
    float luminance = GetLuminance(color);
    return luminance > 0.5f ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);
}

ColorScheme::ColorCategory ColorScheme::CategorizeObjectType(const std::string& typeName) {
    std::string lower = typeName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("mesh") != std::string::npos || lower.find("renderer") != std::string::npos) {
        return ColorCategory::ObjectMesh;
    }
    if (lower.find("light") != std::string::npos) {
        return ColorCategory::ObjectLight;
    }
    if (lower.find("camera") != std::string::npos) {
        return ColorCategory::ObjectCamera;
    }
    if (lower.find("particle") != std::string::npos || lower.find("emitter") != std::string::npos) {
        return ColorCategory::ObjectParticle;
    }
    if (lower.find("audio") != std::string::npos) {
        return ColorCategory::ObjectAudio;
    }
    if (lower.find("rigidbody") != std::string::npos || lower.find("collider") != std::string::npos) {
        return ColorCategory::ObjectPhysics;
    }
    if (lower.find("animator") != std::string::npos || lower.find("animation") != std::string::npos) {
        return ColorCategory::ObjectAnimation;
    }

    return ColorCategory::ObjectDefault;
}

ColorScheme::ColorCategory ColorScheme::CategorizeComponentType(const std::string& typeName) {
    std::string lower = typeName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("transform") != std::string::npos) {
        return ColorCategory::ComponentTransform;
    }
    if (lower.find("material") != std::string::npos || lower.find("shader") != std::string::npos) {
        return ColorCategory::ComponentMaterial;
    }
    if (lower.find("rigidbody") != std::string::npos || lower.find("collider") != std::string::npos) {
        return ColorCategory::ComponentPhysics;
    }
    if (lower.find("audio") != std::string::npos) {
        return ColorCategory::ComponentAudio;
    }
    if (lower.find("animator") != std::string::npos || lower.find("animation") != std::string::npos) {
        return ColorCategory::ComponentAnimation;
    }
    if (lower.find("script") != std::string::npos || lower.find("behavior") != std::string::npos) {
        return ColorCategory::ComponentLogic;
    }

    return ColorCategory::ComponentTransform;
}

void ColorScheme::ResetToDefaults() {
    s_Colors.clear();
    InitializeDefaultColors();
}

void ColorScheme::RenderColoredText(const std::string& text, ColorCategory category) {
    ImGui::PushStyleColor(ImGuiCol_Text, GetColor(category));
    ImGui::Text("%s", text.c_str());
    ImGui::PopStyleColor();
}

void ColorScheme::RenderColoredBadge(const std::string& text, ColorCategory category, 
                                     float size, float rounding) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImVec4 color = GetColor(category);
    ImU32 col = ImGui::GetColorU32(color);

    // Draw rounded rectangle
    draw_list->AddRectFilled(pos, ImVec2(pos.x + size + 8, pos.y + size + 4), col, rounding);

    // Draw text in the middle
    ImGui::Text("%s", text.c_str());

    ImGui::Dummy(ImVec2(size, size));
}

float ColorScheme::GetLuminance(const ImVec4& color) {
    // Standard luminance calculation for sRGB colors
    // L = 0.299*R + 0.587*G + 0.114*B
    return 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
}
