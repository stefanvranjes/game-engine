#pragma once

#include <imgui.h>
#include <glm/glm.hpp>
#include <string>
#include <map>

/**
 * @brief Color scheme system for consistent editor theming
 * 
 * Manages colors for:
 * - Object types in hierarchy (color-coded backgrounds/text)
 * - Inspector sections (collapsible section headers)
 * - UI states (selected, hovered, disabled)
 * - Component badges and labels
 */
class ColorScheme {
public:
    /**
     * @brief Predefined color categories
     */
    enum class ColorCategory {
        // Object types
        ObjectDefault,
        ObjectMesh,
        ObjectLight,
        ObjectCamera,
        ObjectParticle,
        ObjectAudio,
        ObjectPhysics,
        ObjectAnimation,
        
        // UI states
        StateNormal,
        StateHovered,
        StateSelected,
        StateDisabled,
        
        // Component categories
        ComponentTransform,
        ComponentMaterial,
        ComponentPhysics,
        ComponentAudio,
        ComponentAnimation,
        ComponentLogic,
        
        // UI elements
        ElementBackground,
        ElementBorder,
        ElementHighlight,
        ElementShadow
    };

    /**
     * @brief Initialize the color scheme with default colors
     */
    static void Initialize();

    /**
     * @brief Get color for a category (normalized RGBA 0-1)
     */
    static ImVec4 GetColor(ColorCategory category);

    /**
     * @brief Get color as ImU32 (suitable for ImGui drawing)
     */
    static ImU32 GetColorU32(ColorCategory category);

    /**
     * @brief Set color for a category
     */
    static void SetColor(ColorCategory category, const ImVec4& color);
    static void SetColor(ColorCategory category, float r, float g, float b, float a = 1.0f);

    /**
     * @brief Get color for an object type name
     * Automatically categorizes object types
     */
    static ImVec4 GetObjectTypeColor(const std::string& objectTypeName);

    /**
     * @brief Get color for a component type name
     */
    static ImVec4 GetComponentTypeColor(const std::string& componentTypeName);

    /**
     * @brief Get a dimmed version of a color
     * @param color Base color
     * @param factor Dimming factor (0.5 = 50% darker)
     */
    static ImVec4 Dim(const ImVec4& color, float factor = 0.5f);

    /**
     * @brief Get a brightened version of a color
     * @param color Base color
     * @param factor Brightening factor (1.5 = 50% brighter)
     */
    static ImVec4 Brighten(const ImVec4& color, float factor = 1.5f);

    /**
     * @brief Get a color with adjusted alpha
     */
    static ImVec4 WithAlpha(const ImVec4& color, float alpha);

    /**
     * @brief Get complementary/contrasting color (good for text)
     * Returns black or white depending on luminance
     */
    static ImVec4 GetContrast(const ImVec4& color);

    /**
     * @brief Enable/disable dark theme color adjustments
     */
    static void SetDarkTheme(bool darkTheme) { s_DarkTheme = darkTheme; }

    /**
     * @brief Check if dark theme is enabled
     */
    static bool IsDarkTheme() { return s_DarkTheme; }

    /**
     * @brief Get the category for an object type
     */
    static ColorCategory CategorizeObjectType(const std::string& typeName);

    /**
     * @brief Get the category for a component type
     */
    static ColorCategory CategorizeComponentType(const std::string& typeName);

    /**
     * @brief Reset all colors to defaults
     */
    static void ResetToDefaults();

    /**
     * @brief Helper for hierarchy object name rendering with color
     * Renders the name with appropriate color coding
     */
    static void RenderColoredText(const std::string& text, ColorCategory category);

    /**
     * @brief Helper for colored badges (small colored boxes with text)
     * @param text Text to display
     * @param category Color category
     * @param size Size of the badge
     */
    static void RenderColoredBadge(const std::string& text, ColorCategory category, 
                                  float size = 20.0f, float rounding = 4.0f);

private:
    static std::map<ColorCategory, ImVec4> s_Colors;
    static bool s_DarkTheme;
    static bool s_Initialized;

    static void InitializeDefaultColors();
    static float GetLuminance(const ImVec4& color);
};
