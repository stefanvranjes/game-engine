#include "EditorPropertyPanel.h"
#include "InspectorLayout.h"
#include "IconRegistry.h"
#include "ColorScheme.h"
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

EditorPropertyPanel::EditorPropertyPanel()
    : m_CurrentObject(nullptr)
    , m_HasChanges(false)
    , m_ShowTransform(true)
    , m_ShowMaterial(true)
    , m_ShowRigidBody(false)
    , m_ShowCollider(false)
    , m_ShowAudioSource(false)
    , m_ShowAnimator(false)
    , m_ShowSprite(false)
    , m_ShowCustomComponents(false)
    , m_ColorPickerTarget(-1)
    , m_ShowColorPicker(false)
    , m_LastObjectPtr(nullptr)
{
    memset(m_ColorPickerValue, 0, sizeof(m_ColorPickerValue));
}

EditorPropertyPanel::~EditorPropertyPanel() {
}

void EditorPropertyPanel::Render(std::shared_ptr<GameObject> object) {
    if (!ImGui::Begin("Properties")) {
        ImGui::End();
        return;
    }

    if (!object) {
        ImGui::TextDisabled("No object selected");
        ImGui::End();
        return;
    }

    // Detect object change
    void* currentPtr = object.get();
    if (currentPtr != m_LastObjectPtr) {
        m_LastObjectPtr = currentPtr;
        m_CurrentObject = object;
        m_HasChanges = false;
    }

    // Object name and type
    ImGui::Text("Object: %s", object->GetName().c_str());
    ImGui::TextDisabled("Type: GameObject");

    ImGui::Separator();

    // Transform Component (always shown)
    RenderTransformProperties(object);

    ImGui::Spacing();

    // Material Component
    if (RenderComponentHeader("Material", m_ShowMaterial)) {
        if (m_ShowMaterial) {
            RenderMaterialProperties(object);
        }
    }

    ImGui::Spacing();

    // RigidBody Component
    if (RenderComponentHeader("RigidBody", m_ShowRigidBody)) {
        if (m_ShowRigidBody) {
            RenderRigidBodyProperties(object);
        }
    }

    ImGui::Spacing();

    // Collider Component
    if (RenderComponentHeader("Collider", m_ShowCollider)) {
        if (m_ShowCollider) {
            RenderColliderProperties(object);
        }
    }

    ImGui::Spacing();

    // AudioSource Component
    if (RenderComponentHeader("AudioSource", m_ShowAudioSource)) {
        if (m_ShowAudioSource) {
            RenderAudioSourceProperties(object);
        }
    }

    ImGui::Spacing();

    // Animator Component
    if (RenderComponentHeader("Animator", m_ShowAnimator)) {
        if (m_ShowAnimator) {
            RenderAnimatorProperties(object);
        }
    }

    ImGui::Spacing();

    // Sprite Component
    if (RenderComponentHeader("Sprite", m_ShowSprite)) {
        if (m_ShowSprite) {
            RenderSpriteProperties(object);
        }
    }

    ImGui::End();
}

void EditorPropertyPanel::RenderTransformProperties(std::shared_ptr<GameObject> object) {
    auto& transform = object->GetTransform();

    if (ImGui::CollapsingHeader("Transform", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
        // Position
        ImGui::Text("Position");
        ImGui::Indent();
        float pos[3] = { transform.position.x, transform.position.y, transform.position.z };
        if (ImGui::DragFloat3("##Position", pos, 0.1f)) {
            transform.position = Vec3(pos[0], pos[1], pos[2]);
            m_HasChanges = true;
            OnTransformChanged();
        }

        // Reset position button
        ImGui::SameLine();
        if (ImGui::Button("R##ResetPos")) {
            transform.position = Vec3(0.0f);
            m_HasChanges = true;
            OnTransformChanged();
        }
        ImGui::Unindent();

        ImGui::Spacing();

        // Rotation (Euler angles in degrees)
        ImGui::Text("Rotation");
        ImGui::Indent();
        float rot[3] = { transform.rotation.x, transform.rotation.y, transform.rotation.z };
        if (ImGui::DragFloat3("##Rotation", rot, 0.5f)) {
            transform.rotation = Vec3(rot[0], rot[1], rot[2]);
            m_HasChanges = true;
            OnTransformChanged();
        }

        // Reset rotation button
        ImGui::SameLine();
        if (ImGui::Button("R##ResetRot")) {
            transform.rotation = Vec3(0.0f);
            m_HasChanges = true;
            OnTransformChanged();
        }
        ImGui::Unindent();

        ImGui::Spacing();

        // Scale
        ImGui::Text("Scale");
        ImGui::Indent();
        float scl[3] = { transform.scale.x, transform.scale.y, transform.scale.z };
        if (ImGui::DragFloat3("##Scale", scl, 0.05f, 0.1f, 10.0f)) {
            transform.scale = Vec3(scl[0], scl[1], scl[2]);
            m_HasChanges = true;
            OnTransformChanged();
        }

        // Reset scale button
        ImGui::SameLine();
        if (ImGui::Button("R##ResetScl")) {
            transform.scale = Vec3(1.0f);
            m_HasChanges = true;
            OnTransformChanged();
        }
        ImGui::Unindent();
    }
}

void EditorPropertyPanel::RenderMaterialProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    // Material type selector
    static int materialType = 0;
    const char* materialTypes[] = { "Diffuse", "Metallic", "Glass", "Custom" };
    if (ImGui::Combo("Material Type##Selector", &materialType, materialTypes, 4)) {
        m_HasChanges = true;
        OnMaterialChanged();
    }

    ImGui::Spacing();

    // Albedo color
    static float albedo[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if (ImGui::ColorEdit4("Albedo Color", albedo)) {
        m_HasChanges = true;
        OnMaterialChanged();
    }

    // PBR Parameters
    static float metallic = 0.5f;
    static float roughness = 0.5f;
    
    ImGui::SliderFloat("Metallic##Material", &metallic, 0.0f, 1.0f);
    ImGui::SliderFloat("Roughness##Material", &roughness, 0.0f, 1.0f);

    if (ImGui::IsItemDeactivatedAfterEdit()) {
        m_HasChanges = true;
        OnMaterialChanged();
    }

    ImGui::Spacing();

    // Texture slots
    ImGui::Text("Textures");
    if (ImGui::Button("Set Albedo Texture")) {
        // TODO: Open file browser
    }
    if (ImGui::Button("Set Normal Map")) {
        // TODO: Open file browser
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderRigidBodyProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    static float mass = 1.0f;
    static float drag = 0.0f;
    static float angularDrag = 0.05f;
    static bool useGravity = true;
    static bool isKinematic = false;

    if (ImGui::SliderFloat("Mass##RB", &mass, 0.01f, 100.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Drag##RB", &drag, 0.0f, 10.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Angular Drag##RB", &angularDrag, 0.0f, 10.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::Checkbox("Use Gravity##RB", &useGravity)) {
        m_HasChanges = true;
    }

    ImGui::SameLine();
    if (ImGui::Checkbox("Is Kinematic##RB", &isKinematic)) {
        m_HasChanges = true;
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderColliderProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    static int shapeType = 0;
    const char* shapes[] = { "Box", "Sphere", "Capsule", "Mesh" };
    
    if (ImGui::Combo("Shape##Collider", &shapeType, shapes, 4)) {
        m_HasChanges = true;
    }

    ImGui::Spacing();

    static float colliderSize[3] = { 1.0f, 1.0f, 1.0f };
    if (ImGui::DragFloat3("Size##Collider", colliderSize, 0.1f, 0.1f, 100.0f)) {
        m_HasChanges = true;
    }

    static int layer = 0;
    if (ImGui::SliderInt("Layer##Collider", &layer, 0, 31)) {
        m_HasChanges = true;
    }

    static bool isTrigger = false;
    if (ImGui::Checkbox("Is Trigger##Collider", &isTrigger)) {
        m_HasChanges = true;
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderAudioSourceProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    static float volume = 1.0f;
    static float pitch = 1.0f;
    static float spatialBlend = 1.0f;
    static float dopplerLevel = 1.0f;
    static float minDistance = 1.0f;
    static float maxDistance = 100.0f;

    if (ImGui::SliderFloat("Volume##Audio", &volume, 0.0f, 1.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Pitch##Audio", &pitch, 0.5f, 2.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Spatial Blend##Audio", &spatialBlend, 0.0f, 1.0f)) {
        m_HasChanges = true;
    }

    ImGui::Spacing();

    if (ImGui::SliderFloat("Doppler Level##Audio", &dopplerLevel, 0.0f, 5.0f)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Min Distance##Audio", &minDistance, 0.1f, maxDistance)) {
        m_HasChanges = true;
    }

    if (ImGui::SliderFloat("Max Distance##Audio", &maxDistance, minDistance, 1000.0f)) {
        m_HasChanges = true;
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderAnimatorProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    static int currentState = 0;
    const char* states[] = { "Idle", "Run", "Jump", "Fall" };
    
    if (ImGui::Combo("Current State##Anim", &currentState, states, 4)) {
        m_HasChanges = true;
    }

    ImGui::Spacing();

    ImGui::Text("Animation Parameters");
    static float speed = 1.0f;
    if (ImGui::SliderFloat("Speed##Anim", &speed, 0.0f, 3.0f)) {
        m_HasChanges = true;
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderSpriteProperties(std::shared_ptr<GameObject> object) {
    ImGui::Indent();

    static float u = 0.0f;
    static float v = 0.0f;
    static float scaleU = 1.0f;
    static float scaleV = 1.0f;

    ImGui::Text("Atlas Offset");
    if (ImGui::DragFloat("U##Sprite", &u, 0.01f, 0.0f, 1.0f)) {
        m_HasChanges = true;
    }
    if (ImGui::DragFloat("V##Sprite", &v, 0.01f, 0.0f, 1.0f)) {
        m_HasChanges = true;
    }

    ImGui::Spacing();

    ImGui::Text("Atlas Scale");
    if (ImGui::DragFloat("Scale U##Sprite", &scaleU, 0.01f, 0.1f, 1.0f)) {
        m_HasChanges = true;
    }
    if (ImGui::DragFloat("Scale V##Sprite", &scaleV, 0.01f, 0.1f, 1.0f)) {
        m_HasChanges = true;
    }

    ImGui::Unindent();
}

void EditorPropertyPanel::RenderCustomComponents(std::shared_ptr<GameObject> object) {
    // TODO: Render custom scripted components
}

bool EditorPropertyPanel::RenderComponentHeader(const char* componentName, bool& expanded, bool canRemove) {
    ImGui::Separator();

    // Get icon and color for component type
    IconRegistry::IconType iconType = IconRegistry::DetectIconType(componentName);
    ImVec4 componentColor = ColorScheme::GetComponentTypeColor(componentName);

    // Render header with icon if enabled
    if (m_ShowComponentIcons) {
        ImGui::PushStyleColor(ImGuiCol_Text, componentColor);
        ImGui::Text("%s", IconRegistry::GetIcon(iconType));
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }

    // Render collapsible header with color coding
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_None; // Changed from AllowItemOverlap which was causing errors
    if (expanded) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    if (m_ColoredSectionsEnabled) {
        ImGui::PushStyleColor(ImGuiCol_Header, componentColor);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ColorScheme::Brighten(componentColor, 1.2f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ColorScheme::Brighten(componentColor, 1.35f));
    }

    expanded = ImGui::TreeNodeEx(componentName, flags);

    if (m_ColoredSectionsEnabled) {
        ImGui::PopStyleColor(3);
    }

    // Render remove button on the right
    if (canRemove) {
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 30);
        // Changed to use a simple color until ColorScheme errors are resolved
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f)); 
        
        if (ImGui::Button("X##RemoveComponent", ImVec2(20, 20))) {
            // TODO: Remove component
        }
        ImGui::PopStyleColor(1);
    }

    if (expanded) {
        ImGui::TreePop();
    }

    return expanded;
}

void EditorPropertyPanel::OnTransformChanged() {
    // TODO: Notify gizmo manager and other systems
}

void EditorPropertyPanel::OnMaterialChanged() {
    // TODO: Update material in renderer
}

void EditorPropertyPanel::RenderVector3Control(const char* label, float* value, float resetValue, float columnWidth) {
    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, columnWidth);
    ImGui::Text("%s", label);
    ImGui::NextColumn();

    ImGui::DragFloat3(("##" + std::string(label)).c_str(), value);

    ImGui::Columns(1);
}

void EditorPropertyPanel::RenderQuaternionControl(const char* label, float* value, float resetValue) {
    ImGui::Text("%s", label);
    ImGui::DragFloat4(("##" + std::string(label)).c_str(), value);
}

void EditorPropertyPanel::RenderColorControl(const char* label, float* color) {
    ImGui::ColorEdit4(label, color);
}

void EditorPropertyPanel::RenderSliderControl(const char* label, float* value, float min, float max) {
    ImGui::SliderFloat(label, value, min, max);
}
