#include "EditorHierarchy.h"
#include "FuzzyMatcher.h"
#include "IconRegistry.h"
#include "ColorScheme.h"
#include <imgui.h>
#include <algorithm>
#include <chrono>

int EditorHierarchy::s_NodeCounter = 0;

EditorHierarchy::EditorHierarchy()
    : m_SelectedObject(nullptr)
    , m_SearchFilter("")
    , m_DraggedObject(nullptr)
    , m_ShowDragDropIndicator(false)
    , m_DragDropInsertIndex(-1)
    , m_ContextMenuObject(nullptr)
    , m_ShowContextMenu(false)
    , m_IsRenaming(false)
    , m_RenamingObject(nullptr)
    , m_SortOrder(SortOrder::CreationOrder)
    , m_LastClickedObject(nullptr)
    , m_LastClickTime(0.0)
{
    memset(m_SearchBuffer, 0, sizeof(m_SearchBuffer));
}

EditorHierarchy::~EditorHierarchy() {
}

int EditorHierarchy::Render(std::shared_ptr<GameObject> rootObject) {
    if (!ImGui::Begin("Scene Hierarchy")) {
        ImGui::End();
        return -1;
    }

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##HierarchySearch", "Search objects...", m_SearchBuffer, sizeof(m_SearchBuffer))) {
        SetSearchFilter(m_SearchBuffer);
    }

    ImGui::Separator();

    // Buttons for expand/collapse all
    if (ImGui::Button("Expand All##Hierarchy")) {
        ExpandAll();
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All##Hierarchy")) {
        CollapseAll();
    }

    ImGui::Separator();

    // Tree view
    ImGui::BeginChild("HierarchyTree");
    if (rootObject) {
        for (auto& child : rootObject->GetChildren()) {
            RenderNode(child, 0);
        }
    }
    ImGui::EndChild();

    // Context menu
    if (m_ShowContextMenu && m_ContextMenuObject) {
        RenderContextMenu(m_ContextMenuObject);
    }

    ImGui::End();

    return m_SelectedObject ? 0 : -1;
}

void EditorHierarchy::RenderNode(std::shared_ptr<GameObject> object, int depth) {
    if (!MatchesFilter(object->GetName())) {
        return;
    }

    // Create unique ID for this node
    std::string nodeId = object->GetName() + "##" + std::to_string(reinterpret_cast<uintptr_t>(object.get()));
    
    // Check if node should be expanded
    bool isExpanded = std::find(m_ExpandedNodes.begin(), m_ExpandedNodes.end(), object) != m_ExpandedNodes.end();
    
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
    if (m_SelectedObject == object) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (object->GetChildren().empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }

    ImGui::Indent(depth * 10.0f);

    // Visibility toggle
    bool isVisible = !std::count(m_HiddenObjects.begin(), m_HiddenObjects.end(), object);
    bool isLocked = std::count(m_LockedObjects.begin(), m_LockedObjects.end(), object) > 0;

    ImGui::PushID(object.get());

    // Eye icon for visibility
    const char* visibilityIcon = isVisible ? "👁" : "🚫";
    if (ImGui::Button(visibilityIcon, ImVec2(20, 0))) {
        SetObjectVisible(object, !isVisible);
    }
    ImGui::SameLine();

    // Lock icon
    const char* lockIcon = isLocked ? "🔒" : "🔓";
    if (ImGui::Button(lockIcon, ImVec2(20, 0))) {
        SetObjectLocked(object, !isLocked);
    }
    ImGui::SameLine();

    // Get icon type and color for this object
    IconRegistry::IconType iconType = IconRegistry::DetectIconType(object->GetName());
    ImVec4 objectColor = ColorScheme::GetObjectTypeColor(object->GetName());

    // Render object type icon if enabled
    if (m_ShowObjectTypeIcons) {
        if (m_ColoredIconsEnabled) {
            ImGui::PushStyleColor(ImGuiCol_Text, ColorScheme::GetObjectTypeColor(object->GetName()));
        }
        ImGui::Text("%s", IconRegistry::GetIcon(iconType));
        if (m_ColoredIconsEnabled) {
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();
    }

    // Tree node with object name or rename input
    bool nodeOpen = false;
    if (m_IsRenaming && m_RenamingObject == object) {
        ImGui::SetKeyboardFocusHere();
        char buf[256];
        strncpy_s(buf, sizeof(buf), m_RenameBuffer.c_str(), _TRUNCATE);
        if (ImGui::InputText("##Rename", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
            object->SetName(buf);
            m_IsRenaming = false;
            m_RenamingObject = nullptr;
        }
        if (!ImGui::IsItemActive() && (ImGui::IsMouseClicked(0) || ImGui::IsKeyDown(ImGuiKey_Escape))) {
            m_IsRenaming = false;
            m_RenamingObject = nullptr;
        }
    } else {
        std::string displayName = GetDisplayName(object);

        // Apply background color if object type coloring is enabled
        if (m_ObjectTypeColoringEnabled) {
            ImGui::PushStyleColor(ImGuiCol_Header, objectColor);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ColorScheme::Brighten(objectColor, 1.2f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ColorScheme::Brighten(objectColor, 1.35f));
        }

        nodeOpen = ImGui::TreeNodeEx(displayName.c_str(), flags);

        if (m_ObjectTypeColoringEnabled) {
            ImGui::PopStyleColor(3);
        }
    }


    // Handle selection
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        m_SelectedObject = object;
        if (m_OnObjectSelected) {
            m_OnObjectSelected(object);
        }

        // Double-click detection
        auto now = std::chrono::high_resolution_clock::now();
        if (m_LastClickedObject == object && 
            std::chrono::duration<double>(now.time_since_epoch()).count() - m_LastClickTime < DOUBLE_CLICK_TIME) {
            if (m_OnObjectDoubleClicked) {
                m_OnObjectDoubleClicked(object);
            }
        }
        m_LastClickedObject = object;
        m_LastClickTime = std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    // Context menu
    if (ImGui::IsItemClicked(1)) {  // Right-click
        m_ContextMenuObject = object;
        m_ShowContextMenu = true;
    }

    // Drag and drop source
    if (ImGui::BeginDragDropSource()) {
        m_DraggedObject = object;
        ImGui::SetDragDropPayload("GAMEOBJECT", &object, sizeof(object));
        ImGui::Text("%s", object->GetName().c_str());
        ImGui::EndDragDropSource();
    }

    // Drag and drop target
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GAMEOBJECT")) {
            std::shared_ptr<GameObject>* droppedObj = (std::shared_ptr<GameObject>*)payload->Data;
            if (droppedObj && *droppedObj != object) {
                // Reparent the dropped object
                if (m_OnObjectReparented) {
                    m_OnObjectReparented(*droppedObj, object);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }

    // Render children
    if (nodeOpen) {
        auto children = object->GetChildren();
        
        // Sort children based on sort order
        switch (m_SortOrder) {
            case SortOrder::Alphabetical:
                std::sort(children.begin(), children.end(),
                    [](const auto& a, const auto& b) { return a->GetName() < b->GetName(); });
                break;
            case SortOrder::Reverse:
                std::sort(children.begin(), children.end(),
                    [](const auto& a, const auto& b) { return a->GetName() > b->GetName(); });
                break;
            default:
                break;
        }

        for (auto& child : children) {
            RenderNode(child, depth + 1);
        }

        ImGui::TreePop();
    }

    ImGui::PopID();
    ImGui::Unindent(depth * 10.0f);
}

void EditorHierarchy::RenderContextMenu(std::shared_ptr<GameObject> object) {
    if (ImGui::BeginPopupContextVoid("HierarchyContextMenu", 1)) {
        if (ImGui::MenuItem("Rename")) {
            m_IsRenaming = true;
            m_RenamingObject = object;
            m_RenameBuffer = object->GetName();
        }

        if (ImGui::MenuItem("Duplicate")) {
            if (m_OnObjectDuplicated) {
                m_OnObjectDuplicated(object);
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Delete")) {
            if (m_OnObjectDeleted) {
                m_OnObjectDeleted(object);
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Copy", "Ctrl+C")) {
            // TODO: Copy to clipboard
        }

        if (ImGui::MenuItem("Paste", "Ctrl+V")) {
            // TODO: Paste from clipboard
        }

        ImGui::EndPopup();
    } else {
        m_ShowContextMenu = false;
    }

    if (m_ShowContextMenu) {
        ImGui::OpenPopup("HierarchyContextMenu");
    }
}

void EditorHierarchy::SetSearchFilter(const std::string& filter) {
    m_SearchFilter = filter;
}

bool EditorHierarchy::MatchesFilter(const std::string& name) const {
    if (m_SearchFilter.empty()) {
        return true;
    }

    switch (m_SearchMode) {
        case SearchMode::Exact: {
            // Exact case-insensitive substring matching
            std::string lowerName = name;
            std::string lowerFilter = m_SearchFilter;
            
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);
            
            return lowerName.find(lowerFilter) != std::string::npos;
        }
        
        case SearchMode::Fuzzy: {
            // Fuzzy matching with score threshold
            float score = FuzzyMatcher::GetScore(name, m_SearchFilter, m_CaseSensitive);
            return score >= m_FuzzyThreshold;
        }
        
        case SearchMode::Regex:
        default:
            // For now, fall back to exact matching for regex mode
            // Production code could use <regex> library
            {
                std::string lowerName = name;
                std::string lowerFilter = m_SearchFilter;
                
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);
                
                return lowerName.find(lowerFilter) != std::string::npos;
            }
    }
}

const char* EditorHierarchy::GetObjectTypeIcon(std::shared_ptr<GameObject> obj) const {
    // Return appropriate icon based on object type/components
    // This is simplified; production code would check actual component types
    return "🔷";
}

std::string EditorHierarchy::GetDisplayName(std::shared_ptr<GameObject> obj) const {
    std::string name = obj->GetName();
    
    // Add component count badge
    int componentCount = 0;  // Simplified; would actually count components
    
    if (componentCount > 0) {
        name += " [" + std::to_string(componentCount) + "]";
    }

    return name;
}

bool EditorHierarchy::IsObjectVisible(std::shared_ptr<GameObject> obj) const {
    return std::find(m_HiddenObjects.begin(), m_HiddenObjects.end(), obj) == m_HiddenObjects.end();
}

void EditorHierarchy::SetObjectVisible(std::shared_ptr<GameObject> obj, bool visible) {
    if (visible) {
        auto it = std::find(m_HiddenObjects.begin(), m_HiddenObjects.end(), obj);
        if (it != m_HiddenObjects.end()) {
            m_HiddenObjects.erase(it);
        }
    } else {
        if (std::find(m_HiddenObjects.begin(), m_HiddenObjects.end(), obj) == m_HiddenObjects.end()) {
            m_HiddenObjects.push_back(obj);
        }
    }
}

bool EditorHierarchy::IsObjectLocked(std::shared_ptr<GameObject> obj) const {
    return std::find(m_LockedObjects.begin(), m_LockedObjects.end(), obj) != m_LockedObjects.end();
}

void EditorHierarchy::SetObjectLocked(std::shared_ptr<GameObject> obj, bool locked) {
    if (locked) {
        if (std::find(m_LockedObjects.begin(), m_LockedObjects.end(), obj) == m_LockedObjects.end()) {
            m_LockedObjects.push_back(obj);
        }
    } else {
        auto it = std::find(m_LockedObjects.begin(), m_LockedObjects.end(), obj);
        if (it != m_LockedObjects.end()) {
            m_LockedObjects.erase(it);
        }
    }
}

void EditorHierarchy::ExpandNode(std::shared_ptr<GameObject> obj) {
    if (std::find(m_ExpandedNodes.begin(), m_ExpandedNodes.end(), obj) == m_ExpandedNodes.end()) {
        m_ExpandedNodes.push_back(obj);
    }
}

void EditorHierarchy::CollapseNode(std::shared_ptr<GameObject> obj) {
    auto it = std::find(m_ExpandedNodes.begin(), m_ExpandedNodes.end(), obj);
    if (it != m_ExpandedNodes.end()) {
        m_ExpandedNodes.erase(it);
    }
}

void EditorHierarchy::ExpandAll() {
    // TODO: Recursively expand all nodes
}

void EditorHierarchy::CollapseAll() {
    m_ExpandedNodes.clear();
}
