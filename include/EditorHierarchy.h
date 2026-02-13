#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "GameObject.h"

/**
 * @brief Enhanced scene hierarchy panel with search, visibility toggles, and tree navigation
 * 
 * Provides a professional scene tree view with:
 * - Expandable/collapsible nodes
 * - Search filtering
 * - Visibility and lock toggles
 * - Drag-and-drop reparenting
 * - Context menus for object operations
 */
class EditorHierarchy {
public:
    EditorHierarchy();
    ~EditorHierarchy();

    /**
     * @brief Render the hierarchy panel
     * @param rootObject The root GameObject of the scene
     * @return Selected GameObject ID, or -1 if none selected
     */
    int Render(std::shared_ptr<GameObject> rootObject);

    /**
     * @brief Get the currently selected object
     */
    std::shared_ptr<GameObject> GetSelectedObject() const { return m_SelectedObject; }

    /**
     * @brief Set the selected object
     */
    void SetSelectedObject(std::shared_ptr<GameObject> obj) { m_SelectedObject = obj; }

    /**
     * @brief Clear all selections
     */
    void ClearSelection() { m_SelectedObject = nullptr; }

    // Callbacks
    void SetOnObjectSelected(std::function<void(std::shared_ptr<GameObject>)> callback) {
        m_OnObjectSelected = callback;
    }
    void SetOnObjectDoubleClicked(std::function<void(std::shared_ptr<GameObject>)> callback) {
        m_OnObjectDoubleClicked = callback;
    }
    void SetOnObjectDeleted(std::function<void(std::shared_ptr<GameObject>)> callback) {
        m_OnObjectDeleted = callback;
    }
    void SetOnObjectDuplicated(std::function<void(std::shared_ptr<GameObject>)> callback) {
        m_OnObjectDuplicated = callback;
    }
    void SetOnObjectRenamed(std::function<void(std::shared_ptr<GameObject>, const std::string&)> callback) {
        m_OnObjectRenamed = callback;
    }
    void SetOnObjectReparented(std::function<void(std::shared_ptr<GameObject>, std::shared_ptr<GameObject>)> callback) {
        m_OnObjectReparented = callback;
    }

    // Visibility management
    bool IsObjectVisible(std::shared_ptr<GameObject> obj) const;
    void SetObjectVisible(std::shared_ptr<GameObject> obj, bool visible);
    bool IsObjectLocked(std::shared_ptr<GameObject> obj) const;
    void SetObjectLocked(std::shared_ptr<GameObject> obj, bool locked);

    // Search and filtering
    void SetSearchFilter(const std::string& filter);
    const std::string& GetSearchFilter() const { return m_SearchFilter; }

    // Node expansion state
    void ExpandNode(std::shared_ptr<GameObject> obj);
    void CollapseNode(std::shared_ptr<GameObject> obj);
    void ExpandAll();
    void CollapseAll();

    // Sorting
    enum class SortOrder {
        CreationOrder,
        Alphabetical,
        ByType,
        Reverse
    };
    void SetSortOrder(SortOrder order) { m_SortOrder = order; }

private:
    /**
     * @brief Render a single tree node and its children
     */
    void RenderNode(std::shared_ptr<GameObject> object, int depth = 0);

    /**
     * @brief Render context menu for selected object
     */
    void RenderContextMenu(std::shared_ptr<GameObject> object);

    /**
     * @brief Check if object matches current search filter
     */
    bool MatchesFilter(const std::string& name) const;

    /**
     * @brief Get icon character for object type
     */
    const char* GetObjectTypeIcon(std::shared_ptr<GameObject> obj) const;

    /**
     * @brief Get display name with component count badge
     */
    std::string GetDisplayName(std::shared_ptr<GameObject> obj) const;

    // State
    std::shared_ptr<GameObject> m_SelectedObject;
    std::vector<std::shared_ptr<GameObject>> m_ExpandedNodes;
    std::string m_SearchFilter;
    char m_SearchBuffer[256];
    
    // Visibility and lock tracking
    std::vector<std::shared_ptr<GameObject>> m_HiddenObjects;
    std::vector<std::shared_ptr<GameObject>> m_LockedObjects;

    // Drag and drop state
    std::shared_ptr<GameObject> m_DraggedObject;
    bool m_ShowDragDropIndicator;
    int m_DragDropInsertIndex;

    // Context menu state
    std::shared_ptr<GameObject> m_ContextMenuObject;
    bool m_ShowContextMenu;
    std::string m_RenameBuffer;
    bool m_IsRenaming;
    std::shared_ptr<GameObject> m_RenamingObject;

    // Sorting
    SortOrder m_SortOrder;

    // Double-click detection
    std::shared_ptr<GameObject> m_LastClickedObject;
    double m_LastClickTime;
    const double DOUBLE_CLICK_TIME = 0.3;

    // Callbacks
    std::function<void(std::shared_ptr<GameObject>)> m_OnObjectSelected;
    std::function<void(std::shared_ptr<GameObject>)> m_OnObjectDoubleClicked;
    std::function<void(std::shared_ptr<GameObject>)> m_OnObjectDeleted;
    std::function<void(std::shared_ptr<GameObject>)> m_OnObjectDuplicated;
    std::function<void(std::shared_ptr<GameObject>, const std::string&)> m_OnObjectRenamed;
    std::function<void(std::shared_ptr<GameObject>, std::shared_ptr<GameObject>)> m_OnObjectReparented;

    // Helper for tree node rendering
    static int s_NodeCounter;
};
