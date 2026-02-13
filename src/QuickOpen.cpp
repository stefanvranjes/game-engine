#include "QuickOpen.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

QuickOpen::QuickOpen() {
    memset(m_SearchBuffer, 0, sizeof(m_SearchBuffer));
}

QuickOpen::~QuickOpen() {
}

void QuickOpen::Initialize(const Config& config) {
    m_Config = config;
}

void QuickOpen::Show() {
    m_IsVisible = true;
    m_FocusSearchInput = true;
    m_HasSearched = false;
    m_SelectedResultIndex = 0;
    memset(m_SearchBuffer, 0, sizeof(m_SearchBuffer));
    m_CurrentResults.clear();
    m_LastOpenTime = ImGui::GetTime();
}

void QuickOpen::Hide() {
    m_IsVisible = false;
}

bool QuickOpen::Render() {
    if (!m_IsVisible) {
        return false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.3f),
                           ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    if (!ImGui::BeginPopupModal("Quick Open##QuickOpenDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return true;
    }

    // Search input
    RenderSearchInput();

    ImGui::Separator();

    // Results or hints
    if (!m_HasSearched) {
        RenderRecentItems();
        if (m_Config.showSearchHint) {
            RenderSearchHints();
        }
    } else {
        RenderResultsList();
    }

    ImGui::Separator();

    // Info footer
    ImGui::Text("Results: %zu", m_CurrentResults.size());
    ImGui::SameLine();
    ImGui::Text("(Use ↑↓ to navigate, Enter to select, ESC to cancel)");

    ImGui::EndPopupModal();

    return m_IsVisible;
}

void QuickOpen::RenderSearchInput() {
    ImGui::SetNextItemWidth(-1);
    if (m_FocusSearchInput) {
        ImGui::SetKeyboardFocusHere();
        m_FocusSearchInput = false;
    }

    if (ImGui::InputText("##QuickOpenSearch", m_SearchBuffer, sizeof(m_SearchBuffer),
                        ImGuiInputTextFlags_AutoSelectAll)) {
        std::string query = m_SearchBuffer;
        if (!query.empty()) {
            PerformSearch(query);
            m_HasSearched = true;
        } else {
            m_HasSearched = false;
            m_CurrentResults.clear();
        }
    }

    HandleKeyInput();
}

void QuickOpen::RenderResultsList() {
    ImGui::BeginChild("QuickOpenResults", ImVec2(0, 250), true, ImGuiWindowFlags_VerticalScrollbar);

    for (size_t i = 0; i < m_CurrentResults.size(); ++i) {
        const auto& result = m_CurrentResults[i];
        bool isSelected = (i == m_SelectedResultIndex);

        std::string displayLabel = GetResultIcon(result) + " " + result.name;
        std::string desc = GetResultDescription(result);
        if (!desc.empty()) {
            displayLabel += " (" + desc + ")";
        }

        if (ImGui::Selectable(displayLabel.c_str(), isSelected, ImGuiSelectableFlags_SelectOnClick)) {
            m_SelectedResultIndex = i;
            SelectResult(result);
        }

        if (isSelected) {
            ImGui::SetItemDefaultFocus();
        }
    }

    ImGui::EndChild();
}

void QuickOpen::RenderRecentItems() {
    if (m_RecentResults.empty()) {
        ImGui::TextDisabled("No recent items. Start typing to search...");
        return;
    }

    ImGui::TextDisabled("Recent items:");
    ImGui::BeginChild("QuickOpenRecent", ImVec2(0, 150), true);

    for (size_t i = 0; i < m_RecentResults.size() && i < (size_t)m_Config.recentItemsCount; ++i) {
        const auto& result = m_RecentResults[i];
        std::string displayLabel = GetResultIcon(result) + " " + result.name;

        if (ImGui::Selectable(displayLabel.c_str())) {
            SelectResult(result);
        }
    }

    ImGui::EndChild();
}

void QuickOpen::RenderSearchHints() {
    ImGui::TextDisabled("Search hints:");
    ImGui::BulletText("Type asset names to search assets");
    ImGui::BulletText("Type @ to search scene objects");
    ImGui::BulletText("Use fuzzy matching: 'SR' matches 'SpriteRenderer'");
}

void QuickOpen::PerformSearch(const std::string& query) {
    m_CurrentResults.clear();
    m_SelectedResultIndex = 0;

    std::string searchQuery = query;
    std::string searchType = ExtractSearchType(searchQuery);

    // Search based on type
    if (searchType.empty() || searchType == "*") {
        // Search both
        if (m_Config.showAssets) {
            SearchAssets(searchQuery, m_CurrentResults);
        }
        if (m_Config.showObjects) {
            SearchGameObjects(searchQuery, m_CurrentResults);
        }
    } else if (searchType == "@") {
        // Search only objects
        if (m_Config.showObjects) {
            SearchGameObjects(searchQuery, m_CurrentResults);
        }
    } else if (searchType == "#") {
        // Search only assets
        if (m_Config.showAssets) {
            SearchAssets(searchQuery, m_CurrentResults);
        }
    }

    // Sort results by score (descending)
    std::sort(m_CurrentResults.begin(), m_CurrentResults.end(),
             [](const Result& a, const Result& b) {
                 return a.matchScore > b.matchScore;
             });

    // Limit results
    if (m_CurrentResults.size() > (size_t)m_Config.maxResults) {
        m_CurrentResults.resize(m_Config.maxResults);
    }
}

void QuickOpen::SearchAssets(const std::string& pattern, std::vector<Result>& results) {
    if (!fs::exists(m_AssetRoot)) {
        return;
    }

    try {
        for (const auto& entry : fs::recursive_directory_iterator(m_AssetRoot)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            float score = FuzzyMatcher::GetScore(filename, pattern, false);

            if (score >= m_Config.minFuzzyScore) {
                Result result;
                result.resultType = Result::Type::Asset;
                result.path = entry.path().string();
                result.name = filename;
                result.matchScore = score;

                // Determine asset type
                std::string ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".bmp") {
                    result.name = result.name + " [Texture]";
                } else if (ext == ".fbx" || ext == ".obj" || ext == ".gltf") {
                    result.name = result.name + " [Model]";
                } else if (ext == ".wav" || ext == ".mp3" || ext == ".ogg") {
                    result.name = result.name + " [Audio]";
                }

                results.push_back(result);
            }
        }
    } catch (const std::exception&) {
        // Silently ignore filesystem errors
    }
}

void QuickOpen::SearchGameObjects(const std::string& pattern, std::vector<Result>& results) {
    if (!m_SceneRoot) {
        return;
    }

    SearchGameObjectsRecursive(m_SceneRoot, pattern, results);
}

void QuickOpen::SearchGameObjectsRecursive(std::shared_ptr<GameObject> object, 
                                           const std::string& pattern, 
                                           std::vector<Result>& results) {
    if (!object) return;

    std::string name = object->GetName();
    float score = FuzzyMatcher::GetScore(name, pattern, false);

    if (score >= m_Config.minFuzzyScore) {
        Result result;
        result.resultType = Result::Type::GameObject;
        result.name = name;
        result.object = object;
        result.matchScore = score;
        results.push_back(result);
    }

    // Search children
    for (auto& child : object->GetChildren()) {
        SearchGameObjectsRecursive(child, pattern, results);
    }
}

void QuickOpen::HandleKeyInput() {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        Hide();
        if (m_OnCancellation) {
            m_OnCancellation();
        }
        return;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (!m_CurrentResults.empty() && m_SelectedResultIndex < m_CurrentResults.size()) {
            SelectResult(m_CurrentResults[m_SelectedResultIndex]);
        }
        return;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        if (m_SelectedResultIndex > 0) {
            m_SelectedResultIndex--;
        } else if (!m_CurrentResults.empty()) {
            m_SelectedResultIndex = m_CurrentResults.size() - 1;
        }
        ImGui::SetScrollHereY(0.0f);
        return;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        if (m_SelectedResultIndex < m_CurrentResults.size() - 1) {
            m_SelectedResultIndex++;
        } else {
            m_SelectedResultIndex = 0;
        }
        ImGui::SetScrollHereY(1.0f);
        return;
    }
}

void QuickOpen::SelectResult(const Result& result) {
    m_LastResult = result;
    AddToRecent(result);
    Hide();

    if (m_OnSelection) {
        m_OnSelection(result);
    }
}

void QuickOpen::AddToRecent(const Result& result) {
    // Check if already in recent
    auto it = std::find_if(m_RecentResults.begin(), m_RecentResults.end(),
                          [&result](const Result& r) {
                              return r.resultType == result.resultType && r.name == result.name;
                          });

    if (it != m_RecentResults.end()) {
        // Move to front
        std::rotate(m_RecentResults.begin(), it, it + 1);
    } else {
        // Add to front
        m_RecentResults.insert(m_RecentResults.begin(), result);

        // Limit recent items
        if (m_RecentResults.size() > (size_t)m_Config.recentItemsCount) {
            m_RecentResults.resize(m_Config.recentItemsCount);
        }
    }
}

void QuickOpen::ClearHistory() {
    m_RecentResults.clear();
    m_SearchHistory.clear();
}

std::string QuickOpen::ExtractSearchType(std::string& query) {
    if (query.empty()) return "";

    char first = query[0];
    if (first == '@' || first == '#' || first == '*') {
        std::string type(1, first);
        query = query.substr(1);
        return type;
    }
    return "";
}

float QuickOpen::CalculateResultScore(const Result& result, const std::string& query) {
    float score = FuzzyMatcher::GetScore(result.name, query, false);
    
    // Boost score for exact prefix matches
    if (result.name.find(query) == 0) {
        score = std::min(1.0f, score + 0.2f);
    }

    return score;
}

std::string QuickOpen::GetResultIcon(const Result& result) const {
    switch (result.resultType) {
        case Result::Type::Asset:
            return "[A]";
        case Result::Type::GameObject:
            return "[O]";
        case Result::Type::Command:
            return "[>]";
        default:
            return "[ ]";
    }
}

std::string QuickOpen::GetResultDescription(const Result& result) const {
    switch (result.resultType) {
        case Result::Type::Asset:
            return result.path;  // Show path for assets
        case Result::Type::GameObject:
            if (result.object) {
                // Show object path in hierarchy
                std::string path = result.name;
                auto parent = result.object->GetParent();
                while (parent) {
                    path = parent->GetName() + "/" + path;
                    parent = parent->GetParent();
                }
                return path;
            }
            return "";
        case Result::Type::Command:
            return "Command";
        default:
            return "";
    }
}
