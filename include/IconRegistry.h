#pragma once

#include <string>
#include <map>
#include <memory>
#include <imgui.h>

/**
 * @brief Icon and glyph registry for consistent editor visuals
 * 
 * Provides a centralized system for managing editor icons including:
 * - Object type icons (mesh, light, camera, particle, etc.)
 * - Component icons
 * - Action icons (add, remove, play, pause, etc.)
 * - Glyph families (FontAwesome Unicode equivalents)
 */
class IconRegistry {
public:
    /**
     * @brief Predefined object/component types
     */
    enum class IconType {
        // Object types
        GameObject,
        Mesh,
        Light,
        PointLight,
        DirectionalLight,
        SpotLight,
        Camera,
        Particle,
        Emitter,
        AudioSource,
        Animator,
        RigidBody,
        Collider,
        Sprite,
        SoftBody,
        Cloth,
        Decal,
        Terrain,
        Water,
        
        // Actions
        Add,
        Remove,
        Play,
        Pause,
        Stop,
        Settings,
        Expand,
        Collapse,
        Visible,
        Hidden,
        Locked,
        Unlocked,
        Link,
        Unlink,
        
        // Other
        Folder,
        File,
        Asset,
        Script,
        Material,
        Texture,
        Sound,
        Animation,
        
        Unknown
    };

    /**
     * @brief Initialize the icon registry
     * ImGui must already be initialized before calling this
     */
    static void Initialize();

    /**
     * @brief Get icon string/glyph for a given type
     * Returns UTF-8 icon glyphs suitable for ImGui rendering
     */
    static const char* GetIcon(IconType type);

    /**
     * @brief Get icon as string with label
     * Formats as "icon label"
     */
    static std::string GetIconedLabel(IconType type, const std::string& label);

    /**
     * @brief Get icon as colored square (ImGui rendering)
     * @param type Icon type
     * @param size Size of the icon in pixels
     */
    static void RenderIconSquare(IconType type, float size = 20.0f);

    /**
     * @brief Render an icon with label in ImGui
     * @param type Icon type
     * @param label Text label
     * @param sameLineLabel Whether to keep on same line
     */
    static void RenderIcon(IconType type, const std::string& label = "", bool sameLine = false);

    /**
     * @brief Get icon type from string (for serialization)
     * @param typeName Type name string
     * @return Corresponding IconType, or IconType::Unknown if not found
     */
    static IconType GetIconTypeFromString(const std::string& typeName);

    /**
     * @brief Get string representation of icon type
     */
    static const char* GetIconTypeName(IconType type);

    /**
     * @brief Detect icon type from object type name
     * Automatically infers icon type from class/component name
     * e.g., "MeshRenderer" -> IconType::Mesh
     */
    static IconType DetectIconType(const std::string& objectName);

    /**
     * @brief Get the preferred icon size for hierarchy/inspector display
     */
    static float GetIconSize(bool isCompact = false) {
        return isCompact ? 16.0f : 20.0f;
    }

    /**
     * @brief Render a small icon badge (for displays like hierarchy)
     * @param type Icon type
     * @param padding Padding around icon
     */
    static void RenderIconBadge(IconType type, float padding = 2.0f);

    /**
     * @brief Enable/disable colored icons
     * When disabled, all icons render as text only
     */
    static void SetColoredIconsEnabled(bool enabled) { s_ColoredIconsEnabled = enabled; }

    /**
     * @brief Check if colored icons are enabled
     */
    static bool AreColoredIconsEnabled() { return s_ColoredIconsEnabled; }

    /**
     * @brief Get default icon color for a type (normalized 0-1 RGBA)
     */
    static ImVec4 GetIconColor(IconType type);

private:
    static std::map<IconType, std::string> s_IconGlyphs;
    static std::map<IconType, std::string> s_IconNames;
    static std::map<IconType, ImVec4> s_IconColors;
    static bool s_ColoredIconsEnabled;
    static bool s_Initialized;

    static void InitializeGlyphs();
    static void InitializeColors();
    static ImVec4 GetDefaultColor();
};
