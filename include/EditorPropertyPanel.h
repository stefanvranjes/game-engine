#pragma once

#include <string>
#include <memory>
#include <vector>
#include "GameObject.h"
#include "InspectorLayout.h"
#include "IconRegistry.h"
#include "ColorScheme.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

/**
 * @brief Enhanced property inspector panel supporting multiple components
 * 
 * Allows editing of:
 * - Transform (position, rotation, scale)
 * - Material (colors, textures, PBR properties)
 * - RigidBody (mass, drag, constraints)
 * - Collider (shape, size, layer)
 * - AudioSource (volume, pitch, range, 3D settings)
 * - Animator (animation parameters, current state)
 * - Sprite (atlas settings, animation)
 */
class EditorPropertyPanel {
public:
    EditorPropertyPanel();
    ~EditorPropertyPanel();

    /**
     * @brief Render the property inspector panel
     * @param object The GameObject to inspect
     */
    void Render(std::shared_ptr<GameObject> object);

    /**
     * @brief Check if there are any unsaved changes
     */
    bool HasChanges() const { return m_HasChanges; }

    /**
     * @brief Mark all changes as saved
     */
    void ClearChanges() { m_HasChanges = false; }

    /**
     * @brief Enable/disable icons display in inspector
     */
    void SetShowComponentIcons(bool show) { m_ShowComponentIcons = show; }
    bool IsShowingComponentIcons() const { return m_ShowComponentIcons; }

    /**
     * @brief Enable/disable colored component sections
     */
    void SetColoredSectionsEnabled(bool enabled) { m_ColoredSectionsEnabled = enabled; }
    bool AreColoredSectionsEnabled() const { return m_ColoredSectionsEnabled; }

    /**
     * @brief Set the label width for properties
     */
    void SetLabelWidth(float width) { InspectorLayout::SetLabelWidth(width); }

    /**
     * @brief Reset all section expansion states
     */
    void ResetLayout() { InspectorLayout::ResetLayout(); }

private:
    // Component rendering functions
    void RenderTransformProperties(std::shared_ptr<GameObject> object);
    void RenderMaterialProperties(std::shared_ptr<GameObject> object);
    void RenderRigidBodyProperties(std::shared_ptr<GameObject> object);
    void RenderColliderProperties(std::shared_ptr<GameObject> object);
    void RenderAudioSourceProperties(std::shared_ptr<GameObject> object);
    void RenderAnimatorProperties(std::shared_ptr<GameObject> object);
    void RenderSpriteProperties(std::shared_ptr<GameObject> object);
    void RenderCustomComponents(std::shared_ptr<GameObject> object);

    // Helper methods
    void RenderVector3Control(const char* label, float* value, float resetValue = 0.0f, float columnWidth = 100.0f);
    void RenderQuaternionControl(const char* label, float* value, float resetValue = 0.0f);
    void RenderColorControl(const char* label, float* color);
    void RenderSliderControl(const char* label, float* value, float min, float max);
    bool RenderComponentHeader(const char* componentName, bool& expanded, bool canRemove = true);

    // Property change callbacks
    void OnTransformChanged();
    void OnMaterialChanged();

    // State
    std::shared_ptr<GameObject> m_CurrentObject;
    bool m_HasChanges;

    // Expansion states for component sections
    bool m_ShowTransform;
    bool m_ShowMaterial;
    bool m_ShowRigidBody;
    bool m_ShowCollider;
    bool m_ShowAudioSource;
    bool m_ShowAnimator;
    bool m_ShowSprite;
    bool m_ShowCustomComponents;

    // Temporary editing values to allow cancellation
    struct TransformCache {
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 scale;
    } m_TransformCache;

    // Color picker state
    int m_ColorPickerTarget;
    float m_ColorPickerValue[4];
    bool m_ShowColorPicker;

    // Component list
    struct ComponentInfo {
        std::string typeName;
        bool isRemovable;
    };
    std::vector<ComponentInfo> m_Components;

    // Last render object ID (to detect selection changes)
    void* m_LastObjectPtr;

    // Helper constant for column width
    static constexpr float LABEL_COLUMN_WIDTH = 120.0f;

    // Visual configuration
    bool m_ShowComponentIcons = true;
    bool m_ColoredSectionsEnabled = true;
};
