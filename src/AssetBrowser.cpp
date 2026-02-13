#include "AssetBrowser.h"
#include "AssetThumbnailGenerator.h"
#include "FuzzyMatcher.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <sstream>
#include <iomanip>

// Helper function for C++17 compatibility
static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

AssetBrowser::AssetBrowser()
    : m_Database(nullptr)
    , m_IsVisible(true)
    , m_ViewMode(ViewMode::Grid)
    , m_SortOrder(SortOrder::NameAsc)
    , m_LastClickedIndex(0)
    , m_IsDragging(false)
    , m_ThumbnailScale(1.0f)
    , m_ShowContextMenu(false)
{
    std::memset(m_SearchBuffer, 0, sizeof(m_SearchBuffer));
}

AssetBrowser::~AssetBrowser() {
    Shutdown();
}

bool AssetBrowser::Initialize(const Config& config, AssetDatabase* database) {
    m_Config = config;
    m_Database = database;
    m_CurrentDirectory = "";

    // Create thumbnail cache directory if it doesn't exist
    std::filesystem::create_directories(m_Config.thumbnailCacheDir);

    // Load default icons
    LoadDefaultIcons();

    // Initial scan
    Refresh();

    return true;
}

void AssetBrowser::Shutdown() {
    m_CurrentAssets.clear();
    m_SelectedIndices.clear();
    m_DirectoryHistory.clear();
    m_RecentAssets.clear();
}

void AssetBrowser::Render() {
    if (!m_IsVisible) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Asset Browser", &m_IsVisible)) {
        RenderToolbar();
        
        ImGui::BeginChild("Content", ImVec2(0, -30), true);
        
        // Split view: folder tree on left, assets on right
        ImGui::Columns(2, "BrowserColumns", true);
        static bool firstTime = true;
        if (firstTime) {
            ImGui::SetColumnWidth(0, 200);
            firstTime = false;
        }

        // Left panel: Folder tree
        ImGui::BeginChild("FolderTree");
        RenderFolderTree();
        ImGui::EndChild();

        ImGui::NextColumn();

        // Right panel: Asset grid or list
        ImGui::BeginChild("AssetView");
        if (m_ViewMode == ViewMode::Grid) {
            RenderAssetGrid();
        } else {
            RenderAssetList();
        }
        ImGui::EndChild();

        ImGui::Columns(1);
        ImGui::EndChild();

        RenderStatusBar();

        // Context menu
        if (m_ShowContextMenu) {
            RenderContextMenu();
        }
    }
    ImGui::End();
}

void AssetBrowser::RenderToolbar() {
    // Navigation buttons
    if (ImGui::Button("<")) {
        NavigateUp();
    }
    ImGui::SameLine();

    // Current path
    ImGui::Text("%s", m_CurrentDirectory.empty() ? "/" : m_CurrentDirectory.c_str());
    ImGui::SameLine();

    // Search box
    ImGui::SetNextItemWidth(200);
    if (ImGui::InputText("##Search", m_SearchBuffer, sizeof(m_SearchBuffer))) {
        SetSearchFilter(m_SearchBuffer);
    }
    ImGui::SameLine();

    // View mode toggle
    if (ImGui::Button(m_ViewMode == ViewMode::Grid ? "Grid" : "List")) {
        m_ViewMode = (m_ViewMode == ViewMode::Grid) ? ViewMode::List : ViewMode::Grid;
    }
    ImGui::SameLine();

    // Refresh button
    if (ImGui::Button("Refresh")) {
        Refresh();
    }
}

void AssetBrowser::RenderFolderTree() {
    // Root folder
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
    if (m_CurrentDirectory.empty()) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    bool rootOpen = ImGui::TreeNodeEx("Assets", flags);
    if (ImGui::IsItemClicked()) {
        SetCurrentDirectory("");
    }

    if (rootOpen) {
        // Scan for subdirectories
        std::filesystem::path rootPath(m_Config.assetRootDir);
        try {
            for (const auto& entry : std::filesystem::directory_iterator(rootPath)) {
                if (entry.is_directory()) {
                    std::string dirName = entry.path().filename().string();
                    
                    // Skip hidden directories
                    if (!m_Config.showHiddenFiles && dirName[0] == '.') continue;

                    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                    std::string relativePath = dirName;
                    
                    if (m_CurrentDirectory == relativePath) {
                        nodeFlags |= ImGuiTreeNodeFlags_Selected;
                    }

                    bool nodeOpen = ImGui::TreeNodeEx(dirName.c_str(), nodeFlags);
                    if (ImGui::IsItemClicked()) {
                        SetCurrentDirectory(relativePath);
                    }

                    if (nodeOpen) {
                        // TODO: Recursively render subdirectories
                        ImGui::TreePop();
                    }
                }
            }
        } catch (const std::exception& e) {
            ImGui::Text("Error scanning directory");
        }

        ImGui::TreePop();
    }
}

void AssetBrowser::RenderAssetGrid() {
    float thumbnailSize = m_Config.thumbnailSize * m_ThumbnailScale;
    float cellSize = thumbnailSize + 20; // Padding
    float panelWidth = ImGui::GetContentRegionAvail().x;
    int columns = std::max(1, (int)(panelWidth / cellSize));

    for (size_t i = 0; i < m_CurrentAssets.size(); ++i) {
        const auto& item = m_CurrentAssets[i];
        
        if (!MatchesFilter(item)) continue;

        ImGui::BeginGroup();
        
        // Thumbnail or icon
        unsigned int texID = item.thumbnailLoaded ? item.thumbnailID : 0;
        if (texID == 0) {
            texID = m_DefaultIcons[item.type];
        }

        ImVec2 uv0(0, 0);
        ImVec2 uv1(1, 1);
        ImVec4 tintCol(1, 1, 1, 1);
        ImVec4 borderCol = IsSelected(i) ? ImVec4(0.2f, 0.5f, 1.0f, 1.0f) : ImVec4(0, 0, 0, 0);

        std::string buttonID = "##asset_" + std::to_string(i);
        if (ImGui::ImageButton(buttonID.c_str(), (ImTextureID)(intptr_t)texID, ImVec2(thumbnailSize, thumbnailSize), uv0, uv1, borderCol, tintCol)) {
            bool ctrlPressed = ImGui::GetIO().KeyCtrl;
            bool shiftPressed = ImGui::GetIO().KeyShift;
            SelectAsset(i, ctrlPressed || shiftPressed);
        }

        // Double-click handling
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
            if (item.isDirectory) {
                NavigateToDirectory(item.path);
            } else if (m_DoubleClickCallback) {
                m_DoubleClickCallback(item.path);
            }
        }

        // Drag-and-drop source
        if (m_Config.enableDragDrop && !item.isDirectory && ImGui::BeginDragDropSource()) {
            m_IsDragging = true;
            m_DraggedAsset = item.path;
            ImGui::SetDragDropPayload("ASSET_PATH", item.path.c_str(), item.path.size() + 1);
            ImGui::Text("%s", item.name.c_str());
            ImGui::EndDragDropSource();
        } else if (m_IsDragging && !ImGui::IsMouseDragging(0)) {
            m_IsDragging = false;
        }

        // Context menu
        if (ImGui::IsItemClicked(1)) {
            m_ShowContextMenu = true;
            m_ContextMenuAsset = item.path;
        }

        // Asset name
        ImGui::TextWrapped("%s", item.name.c_str());

        ImGui::EndGroup();

        // Layout columns
        if ((i + 1) % columns != 0) {
            ImGui::SameLine();
        }
    }
}

void AssetBrowser::RenderAssetList() {
    if (ImGui::BeginTable("AssetListTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Size");
        ImGui::TableSetupColumn("Modified");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < m_CurrentAssets.size(); ++i) {
            const auto& item = m_CurrentAssets[i];
            
            if (!MatchesFilter(item)) continue;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            // Selectable row
            bool isSelected = IsSelected(i);
            if (ImGui::Selectable(item.name.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns)) {
                bool ctrlPressed = ImGui::GetIO().KeyCtrl;
                bool shiftPressed = ImGui::GetIO().KeyShift;
                SelectAsset(i, ctrlPressed || shiftPressed);
            }

            // Double-click
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                if (item.isDirectory) {
                    NavigateToDirectory(item.path);
                } else if (m_DoubleClickCallback) {
                    m_DoubleClickCallback(item.path);
                }
            }

            // Context menu
            if (ImGui::IsItemClicked(1)) {
                m_ShowContextMenu = true;
                m_ContextMenuAsset = item.path;
            }

            ImGui::TableNextColumn();
            ImGui::Text("%s", item.type.c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%s", item.isDirectory ? "-" : FormatFileSize(item.fileSize).c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%s", item.lastModified.c_str());
        }

        ImGui::EndTable();
    }
}

void AssetBrowser::RenderStatusBar() {
    ImGui::Separator();
    
    size_t selectedCount = m_SelectedIndices.size();
    size_t totalCount = m_CurrentAssets.size();
    
    if (selectedCount > 0) {
        std::string selectedAsset = GetSelectedAsset();
        if (!selectedAsset.empty()) {
            // Find the asset item
            for (const auto& item : m_CurrentAssets) {
                if (item.path == selectedAsset) {
                    ImGui::Text("Selected: %s | %s | %s", 
                        item.name.c_str(), 
                        item.type.c_str(),
                        item.isDirectory ? "Folder" : FormatFileSize(item.fileSize).c_str());
                    break;
                }
            }
        }
    } else {
        ImGui::Text("%zu items", totalCount);
    }
}

void AssetBrowser::RenderContextMenu() {
    if (ImGui::BeginPopupContextVoid("AssetContextMenu")) {
        if (ImGui::MenuItem("Open")) {
            // TODO: Open asset
        }
        if (ImGui::MenuItem("Reimport")) {
            // TODO: Reimport asset
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Show in Explorer")) {
            // TODO: Show in file explorer
        }
        if (ImGui::MenuItem("Copy Path")) {
            ImGui::SetClipboardText(m_ContextMenuAsset.c_str());
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete")) {
            // TODO: Delete asset with confirmation
        }
        if (ImGui::MenuItem("Properties")) {
            // TODO: Show properties dialog
        }

        ImGui::EndPopup();
    } else {
        m_ShowContextMenu = false;
    }

    if (m_ShowContextMenu) {
        ImGui::OpenPopup("AssetContextMenu");
    }
}

void AssetBrowser::Refresh() {
    ScanCurrentDirectory();
    SortAssets();
}

void AssetBrowser::SetCurrentDirectory(const std::string& path) {
    m_CurrentDirectory = path;
    ClearSelection();
    Refresh();
}

void AssetBrowser::SetSortOrder(SortOrder order) {
    m_SortOrder = order;
    SortAssets();
}

std::string AssetBrowser::GetSelectedAsset() const {
    if (m_SelectedIndices.empty()) return "";
    if (m_SelectedIndices[0] >= m_CurrentAssets.size()) return "";
    return m_CurrentAssets[m_SelectedIndices[0]].path;
}

std::vector<std::string> AssetBrowser::GetSelectedAssets() const {
    std::vector<std::string> result;
    for (size_t idx : m_SelectedIndices) {
        if (idx < m_CurrentAssets.size()) {
            result.push_back(m_CurrentAssets[idx].path);
        }
    }
    return result;
}

void AssetBrowser::SetSearchFilter(const std::string& filter) {
    m_SearchFilter = filter;
}

void AssetBrowser::SetTypeFilter(const std::string& type) {
    m_TypeFilter = type;
}

void AssetBrowser::ScanCurrentDirectory() {
    m_CurrentAssets.clear();

    std::filesystem::path currentPath = m_Config.assetRootDir;
    if (!m_CurrentDirectory.empty()) {
        currentPath /= m_CurrentDirectory;
    }

    try {
        for (const auto& entry : std::filesystem::directory_iterator(currentPath)) {
            AssetItem item;
            item.path = GetRelativePath(entry.path().string());
            item.name = entry.path().filename().string();
            item.isDirectory = entry.is_directory();
            
            // Skip hidden files if configured
            if (!m_Config.showHiddenFiles && item.name[0] == '.') continue;

            if (item.isDirectory) {
                item.type = "folder";
                item.fileSize = 0;
            } else {
                item.extension = entry.path().extension().string();
                item.fileSize = std::filesystem::file_size(entry.path());
                
                // Detect asset type
                if (item.extension == ".png" || item.extension == ".jpg" || item.extension == ".jpeg" || 
                    item.extension == ".tga" || item.extension == ".bmp" || item.extension == ".hdr") {
                    item.type = "texture";
                } else if (item.extension == ".obj" || item.extension == ".fbx" || item.extension == ".gltf" || item.extension == ".glb") {
                    item.type = "model";
                } else if (item.extension == ".vert" || item.extension == ".frag" || item.extension == ".geom" || item.extension == ".comp") {
                    item.type = "shader";
                } else if (item.extension == ".prefab") {
                    item.type = "prefab";
                } else if (item.extension == ".scene") {
                    item.type = "scene";
                } else if (item.extension == ".wav" || item.extension == ".mp3" || item.extension == ".ogg") {
                    item.type = "audio";
                } else {
                    item.type = "file";
                }
            }

            // Get last modified time
            auto ftime = std::filesystem::last_write_time(entry.path());
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
            std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&cftime), "%Y-%m-%d %H:%M");
            item.lastModified = ss.str();

            item.thumbnailLoaded = false;
            item.thumbnailID = 0;

            m_CurrentAssets.push_back(item);
        }
    } catch (const std::exception& e) {
        // Error scanning directory
    }
}

void AssetBrowser::SortAssets() {
    std::sort(m_CurrentAssets.begin(), m_CurrentAssets.end(), [this](const AssetItem& a, const AssetItem& b) {
        // Directories always first
        if (a.isDirectory != b.isDirectory) {
            return a.isDirectory;
        }

        switch (m_SortOrder) {
            case SortOrder::NameAsc:
                return a.name < b.name;
            case SortOrder::NameDesc:
                return a.name > b.name;
            case SortOrder::SizeAsc:
                return a.fileSize < b.fileSize;
            case SortOrder::SizeDesc:
                return a.fileSize > b.fileSize;
            case SortOrder::TypeAsc:
                return a.type < b.type;
            case SortOrder::TypeDesc:
                return a.type > b.type;
            default:
                return a.name < b.name;
        }
    });
}

bool AssetBrowser::MatchesFilter(const AssetItem& item) const {
    // Type filter
    if (!m_AdvancedFilters.typeFilter.empty() && item.type != m_AdvancedFilters.typeFilter) {
        return false;
    }

    // Legacy type filter fallback
    if (!m_TypeFilter.empty() && item.type != m_TypeFilter) {
        return false;
    }

    // File size filter
    if (item.fileSize < m_AdvancedFilters.minFileSize || 
        item.fileSize > m_AdvancedFilters.maxFileSize) {
        return false;
    }

    // Name/search filter
    if (!m_AdvancedFilters.namePattern.empty() || !m_SearchFilter.empty()) {
        std::string searchPattern = !m_AdvancedFilters.namePattern.empty() ? 
                                   m_AdvancedFilters.namePattern : m_SearchFilter;
        
        if (!searchPattern.empty()) {
            if (m_AdvancedFilters.fuzzyMatch) {
                // Fuzzy matching with score threshold
                float score = FuzzyMatcher::GetScore(item.name, searchPattern, false);
                if (score < m_AdvancedFilters.minFuzzyScore) {
                    return false;
                }
            } else {
                // Exact substring matching
                std::string lowerName = item.name;
                std::string lowerFilter = searchPattern;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);
                
                if (lowerName.find(lowerFilter) == std::string::npos) {
                    return false;
                }
            }
        }
    }

    // Labels filter (if implemented)
    if (m_AdvancedFilters.labelsEnabled && !m_AdvancedFilters.labels.empty()) {
        // TODO: Implement label filtering when label system is available
        // For now, skip label filtering
    }

    return true;
}

void AssetBrowser::SelectAsset(size_t index, bool addToSelection) {
    if (index >= m_CurrentAssets.size()) return;

    if (!addToSelection) {
        m_SelectedIndices.clear();
    }

    // Toggle selection if already selected
    auto it = std::find(m_SelectedIndices.begin(), m_SelectedIndices.end(), index);
    if (it != m_SelectedIndices.end()) {
        if (addToSelection) {
            m_SelectedIndices.erase(it);
        }
    } else {
        m_SelectedIndices.push_back(index);
    }

    m_LastClickedIndex = index;

    // Callback
    if (m_SelectionCallback && !m_SelectedIndices.empty()) {
        m_SelectionCallback(m_CurrentAssets[m_SelectedIndices[0]].path);
    }
}

void AssetBrowser::ClearSelection() {
    m_SelectedIndices.clear();
}

bool AssetBrowser::IsSelected(size_t index) const {
    return std::find(m_SelectedIndices.begin(), m_SelectedIndices.end(), index) != m_SelectedIndices.end();
}

void AssetBrowser::NavigateUp() {
    if (m_CurrentDirectory.empty()) return;

    size_t lastSlash = m_CurrentDirectory.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        SetCurrentDirectory("");
    } else {
        SetCurrentDirectory(m_CurrentDirectory.substr(0, lastSlash));
    }
}

void AssetBrowser::NavigateToDirectory(const std::string& path) {
    SetCurrentDirectory(path);
}

void AssetBrowser::AddToRecent(const std::string& path) {
    // Remove if already in list
    auto it = std::find(m_RecentAssets.begin(), m_RecentAssets.end(), path);
    if (it != m_RecentAssets.end()) {
        m_RecentAssets.erase(it);
    }

    // Add to front
    m_RecentAssets.insert(m_RecentAssets.begin(), path);

    // Limit size
    if (m_RecentAssets.size() > m_Config.maxRecentAssets) {
        m_RecentAssets.pop_back();
    }
}

std::string AssetBrowser::GetAssetIcon(const std::string& type) const {
    // Return icon name based on type
    return type;
}

std::string AssetBrowser::FormatFileSize(size_t bytes) const {
    const char* units[] = { "B", "KB", "MB", "GB" };
    int unitIndex = 0;
    double size = (double)bytes;

    while (size >= 1024.0 && unitIndex < 3) {
        size /= 1024.0;
        unitIndex++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
    return ss.str();
}

void AssetBrowser::ResetFilters() {
    m_SearchFilter.clear();
    m_TypeFilter.clear();
    m_AdvancedFilters = SearchFilters();
    m_AdvancedFilters.fuzzyMatch = true;
    m_AdvancedFilters.minFuzzyScore = 0.3f;
}

void AssetBrowser::SetFileSizeFilter(size_t minBytes, size_t maxBytes) {
    m_AdvancedFilters.minFileSize = minBytes;
    m_AdvancedFilters.maxFileSize = maxBytes;
}

std::string AssetBrowser::GetRelativePath(const std::string& fullPath) const {
    std::filesystem::path full(fullPath);
    std::filesystem::path root(m_Config.assetRootDir);
    
    try {
        std::filesystem::path relative = std::filesystem::relative(full, root);
        return relative.string();
    } catch (...) {
        return fullPath;
    }
}

void AssetBrowser::RequestThumbnail(AssetItem& item) {
    // TODO: Integrate with AssetThumbnailGenerator
}

void AssetBrowser::LoadDefaultIcons() {
    // TODO: Load default icons for different asset types
    // For now, use placeholder texture IDs
    m_DefaultIcons["folder"] = 0;
    m_DefaultIcons["texture"] = 0;
    m_DefaultIcons["model"] = 0;
    m_DefaultIcons["shader"] = 0;
    m_DefaultIcons["prefab"] = 0;
    m_DefaultIcons["scene"] = 0;
    m_DefaultIcons["audio"] = 0;
    m_DefaultIcons["file"] = 0;
}
