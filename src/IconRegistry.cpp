#include "IconRegistry.h"
#include <cctype>
#include <algorithm>

// Static member initialization
std::map<IconRegistry::IconType, std::string> IconRegistry::s_IconGlyphs;
std::map<IconRegistry::IconType, std::string> IconRegistry::s_IconNames;
std::map<IconRegistry::IconType, ImVec4> IconRegistry::s_IconColors;
bool IconRegistry::s_ColoredIconsEnabled = true;
bool IconRegistry::s_Initialized = false;

void IconRegistry::Initialize() {
    if (s_Initialized) return;

    InitializeGlyphs();
    InitializeColors();
    s_Initialized = true;
}

void IconRegistry::InitializeGlyphs() {
    // Object types
    s_IconGlyphs[IconType::GameObject] = "•";
    s_IconGlyphs[IconType::Mesh] = "M";
    s_IconGlyphs[IconType::Light] = "☀";
    s_IconGlyphs[IconType::PointLight] = "●";
    s_IconGlyphs[IconType::DirectionalLight] = "➤";
    s_IconGlyphs[IconType::SpotLight] = "◆";
    s_IconGlyphs[IconType::Camera] = "📷";
    s_IconGlyphs[IconType::Particle] = "*";
    s_IconGlyphs[IconType::Emitter] = "✶";
    s_IconGlyphs[IconType::AudioSource] = "♪";
    s_IconGlyphs[IconType::Animator] = "▶";
    s_IconGlyphs[IconType::RigidBody] = "⬜";
    s_IconGlyphs[IconType::Collider] = "⚫";
    s_IconGlyphs[IconType::Sprite] = "◻";
    s_IconGlyphs[IconType::SoftBody] = "≈";
    s_IconGlyphs[IconType::Cloth] = "∿";
    s_IconGlyphs[IconType::Decal] = "✕";
    s_IconGlyphs[IconType::Terrain] = "▲";
    s_IconGlyphs[IconType::Water] = "≋";

    // Actions
    s_IconGlyphs[IconType::Add] = "+";
    s_IconGlyphs[IconType::Remove] = "-";
    s_IconGlyphs[IconType::Play] = "▶";
    s_IconGlyphs[IconType::Pause] = "⏸";
    s_IconGlyphs[IconType::Stop] = "⏹";
    s_IconGlyphs[IconType::Settings] = "⚙";
    s_IconGlyphs[IconType::Expand] = "△";
    s_IconGlyphs[IconType::Collapse] = "▽";
    s_IconGlyphs[IconType::Visible] = "👁";
    s_IconGlyphs[IconType::Hidden] = "⊘";
    s_IconGlyphs[IconType::Locked] = "🔒";
    s_IconGlyphs[IconType::Unlocked] = "🔓";
    s_IconGlyphs[IconType::Link] = "🔗";
    s_IconGlyphs[IconType::Unlink] = "⊘";

    // Files and assets
    s_IconGlyphs[IconType::Folder] = "📁";
    s_IconGlyphs[IconType::File] = "📄";
    s_IconGlyphs[IconType::Asset] = "◈";
    s_IconGlyphs[IconType::Script] = "⟨⟩";
    s_IconGlyphs[IconType::Material] = "◉";
    s_IconGlyphs[IconType::Texture] = "▦";
    s_IconGlyphs[IconType::Sound] = "🔊";
    s_IconGlyphs[IconType::Animation] = "⟲";

    // Unknown
    s_IconGlyphs[IconType::Unknown] = "?";

    // Store icon names
    s_IconNames[IconType::GameObject] = "GameObject";
    s_IconNames[IconType::Mesh] = "Mesh";
    s_IconNames[IconType::Light] = "Light";
    s_IconNames[IconType::PointLight] = "PointLight";
    s_IconNames[IconType::DirectionalLight] = "DirectionalLight";
    s_IconNames[IconType::SpotLight] = "SpotLight";
    s_IconNames[IconType::Camera] = "Camera";
    s_IconNames[IconType::Particle] = "Particle";
    s_IconNames[IconType::Emitter] = "Emitter";
    s_IconNames[IconType::AudioSource] = "AudioSource";
    s_IconNames[IconType::Animator] = "Animator";
    s_IconNames[IconType::RigidBody] = "RigidBody";
    s_IconNames[IconType::Collider] = "Collider";
    s_IconNames[IconType::Sprite] = "Sprite";
    s_IconNames[IconType::SoftBody] = "SoftBody";
    s_IconNames[IconType::Cloth] = "Cloth";
    s_IconNames[IconType::Decal] = "Decal";
    s_IconNames[IconType::Terrain] = "Terrain";
    s_IconNames[IconType::Water] = "Water";
    s_IconNames[IconType::Unknown] = "Unknown";
}

void IconRegistry::InitializeColors() {
    // Object type colors (RGBA, normalized)
    s_IconColors[IconType::GameObject] = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);   // Gray
    s_IconColors[IconType::Mesh] = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);          // Blue
    s_IconColors[IconType::Light] = ImVec4(1.0f, 0.9f, 0.2f, 1.0f);         // Yellow
    s_IconColors[IconType::PointLight] = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);    // Orange
    s_IconColors[IconType::DirectionalLight] = ImVec4(1.0f, 1.0f, 0.5f, 1.0f);  // Bright yellow
    s_IconColors[IconType::SpotLight] = ImVec4(1.0f, 0.7f, 0.0f, 1.0f);     // Dark orange
    s_IconColors[IconType::Camera] = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);        // Cyan
    s_IconColors[IconType::Particle] = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);      // Orange-red
    s_IconColors[IconType::Emitter] = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);       // Red-orange
    s_IconColors[IconType::AudioSource] = ImVec4(0.5f, 0.8f, 0.5f, 1.0f);   // Green
    s_IconColors[IconType::Animator] = ImVec4(0.8f, 0.4f, 0.8f, 1.0f);      // Purple
    s_IconColors[IconType::RigidBody] = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);     // Red
    s_IconColors[IconType::Collider] = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);      // Dark red
    s_IconColors[IconType::Sprite] = ImVec4(0.2f, 0.9f, 0.9f, 1.0f);        // Cyan-green
    s_IconColors[IconType::SoftBody] = ImVec4(0.9f, 0.6f, 0.1f, 1.0f);      // Yellow-orange
    s_IconColors[IconType::Cloth] = ImVec4(0.9f, 0.7f, 0.5f, 1.0f);         // Tan
    s_IconColors[IconType::Decal] = ImVec4(0.5f, 0.5f, 0.8f, 1.0f);         // Blue-purple
    s_IconColors[IconType::Terrain] = ImVec4(0.6f, 0.5f, 0.3f, 1.0f);       // Brown
    s_IconColors[IconType::Water] = ImVec4(0.2f, 0.7f, 1.0f, 1.0f);         // Water blue

    // Action colors
    s_IconColors[IconType::Add] = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);           // Bright green
    s_IconColors[IconType::Remove] = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);        // Bright red
    s_IconColors[IconType::Play] = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);          // Green
    s_IconColors[IconType::Pause] = ImVec4(1.0f, 0.9f, 0.3f, 1.0f);         // Yellow
    s_IconColors[IconType::Stop] = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);          // Red
    s_IconColors[IconType::Settings] = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);      // Gray
    s_IconColors[IconType::Visible] = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);       // Green
    s_IconColors[IconType::Hidden] = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);        // Gray
    s_IconColors[IconType::Locked] = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);        // Orange
    s_IconColors[IconType::Unlocked] = ImVec4(0.5f, 1.0f, 0.5f, 1.0f);      // Green
    s_IconColors[IconType::Unknown] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);       // Light gray
}

const char* IconRegistry::GetIcon(IconType type) {
    if (!s_Initialized) Initialize();
    
    auto it = s_IconGlyphs.find(type);
    if (it != s_IconGlyphs.end()) {
        return it->second.c_str();
    }
    return s_IconGlyphs[IconType::Unknown].c_str();
}

std::string IconRegistry::GetIconedLabel(IconType type, const std::string& label) {
    if (!s_Initialized) Initialize();
    
    std::string icon = GetIcon(type);
    return std::string(icon) + " " + label;
}

void IconRegistry::RenderIconSquare(IconType type, float size) {
    if (!s_Initialized) Initialize();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImVec4 color = GetIconColor(type);
    ImU32 col = ImGui::GetColorU32(color);

    if (s_ColoredIconsEnabled) {
        draw_list->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), col, 2.0f);
        draw_list->AddRect(pos, ImVec2(pos.x + size, pos.y + size), 
                          ImGui::GetColorU32(ImVec4(1, 1, 1, 0.5f)), 2.0f, 0, 1.0f);
    } else {
        draw_list->AddRect(pos, ImVec2(pos.x + size, pos.y + size), col, 2.0f, 0, 1.5f);
    }

    ImGui::Dummy(ImVec2(size, size));
}

void IconRegistry::RenderIcon(IconType type, const std::string& label, bool sameLine) {
    if (!s_Initialized) Initialize();

    ImVec4 color = GetIconColor(type);
    
    if (s_ColoredIconsEnabled) {
        ImGui::PushStyleColor(ImGuiCol_Text, color);
    }

    ImGui::Text("%s", GetIcon(type));

    if (s_ColoredIconsEnabled) {
        ImGui::PopStyleColor();
    }

    if (!label.empty()) {
        if (sameLine) ImGui::SameLine();
        ImGui::Text("%s", label.c_str());
    }
}

IconRegistry::IconType IconRegistry::GetIconTypeFromString(const std::string& typeName) {
    if (!s_Initialized) Initialize();

    for (const auto& pair : s_IconNames) {
        if (pair.second == typeName) {
            return pair.first;
        }
    }
    return IconType::Unknown;
}

const char* IconRegistry::GetIconTypeName(IconType type) {
    if (!s_Initialized) Initialize();

    auto it = s_IconNames.find(type);
    if (it != s_IconNames.end()) {
        return it->second.c_str();
    }
    return "Unknown";
}

IconRegistry::IconType IconRegistry::DetectIconType(const std::string& objectName) {
    if (!s_Initialized) Initialize();

    std::string lower = objectName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Check for common patterns
    if (lower.find("mesh") != std::string::npos || lower.find("renderer") != std::string::npos) {
        return IconType::Mesh;
    }
    if (lower.find("light") != std::string::npos) {
        if (lower.find("spot") != std::string::npos) return IconType::SpotLight;
        if (lower.find("directional") != std::string::npos) return IconType::DirectionalLight;
        if (lower.find("point") != std::string::npos) return IconType::PointLight;
        return IconType::Light;
    }
    if (lower.find("camera") != std::string::npos) {
        return IconType::Camera;
    }
    if (lower.find("particle") != std::string::npos || lower.find("emitter") != std::string::npos) {
        return lower.find("emitter") != std::string::npos ? IconType::Emitter : IconType::Particle;
    }
    if (lower.find("audio") != std::string::npos) {
        return IconType::AudioSource;
    }
    if (lower.find("animator") != std::string::npos || lower.find("animation") != std::string::npos) {
        return IconType::Animator;
    }
    if (lower.find("rigidbody") != std::string::npos) {
        return IconType::RigidBody;
    }
    if (lower.find("collider") != std::string::npos) {
        return IconType::Collider;
    }
    if (lower.find("sprite") != std::string::npos) {
        return IconType::Sprite;
    }
    if (lower.find("cloth") != std::string::npos) {
        return IconType::Cloth;
    }
    if (lower.find("softbody") != std::string::npos) {
        return IconType::SoftBody;
    }
    if (lower.find("decal") != std::string::npos) {
        return IconType::Decal;
    }
    if (lower.find("terrain") != std::string::npos) {
        return IconType::Terrain;
    }
    if (lower.find("water") != std::string::npos) {
        return IconType::Water;
    }

    return IconType::GameObject;
}

void IconRegistry::RenderIconBadge(IconType type, float padding) {
    if (!s_Initialized) Initialize();

    ImVec4 color = GetIconColor(type);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", GetIcon(type));
    ImGui::PopStyleColor();

    ImGui::SameLine(0.0f, padding);
}

ImVec4 IconRegistry::GetIconColor(IconType type) {
    if (!s_Initialized) Initialize();

    if (!s_ColoredIconsEnabled) {
        return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
    }

    auto it = s_IconColors.find(type);
    if (it != s_IconColors.end()) {
        return it->second;
    }
    return GetDefaultColor();
}

ImVec4 IconRegistry::GetDefaultColor() {
    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
}
