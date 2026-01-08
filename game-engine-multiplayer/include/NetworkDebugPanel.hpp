#pragma once

#include "MultiplayerManager.hpp"
#include "../../include/imgui/imgui.h"
#include <string>
#include <vector>
#include <deque>

/**
 * @brief ImGui debug panel for network statistics and monitoring
 * 
 * Provides visual insight into:
 * - Connection status and ping
 * - Bandwidth usage
 * - Entity replication status
 * - Network event log
 */
class NetworkDebugPanel {
public:
    NetworkDebugPanel() = default;
    ~NetworkDebugPanel() = default;
    
    /**
     * @brief Set the multiplayer manager to monitor
     */
    void SetMultiplayerManager(MultiplayerManager* manager) { m_Manager = manager; }
    
    /**
     * @brief Render the debug panel
     * Call this each frame within ImGui context
     */
    void Render() {
        if (!m_ShowPanel) return;
        
        ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Network Debug", &m_ShowPanel)) {
            ImGui::End();
            return;
        }
        
        RenderConnectionStatus();
        ImGui::Separator();
        RenderPlayerList();
        ImGui::Separator();
        RenderStatistics();
        ImGui::Separator();
        RenderEntityInfo();
        ImGui::Separator();
        RenderEventLog();
        
        ImGui::End();
    }
    
    /**
     * @brief Toggle panel visibility
     */
    void Toggle() { m_ShowPanel = !m_ShowPanel; }
    bool IsVisible() const { return m_ShowPanel; }
    void SetVisible(bool visible) { m_ShowPanel = visible; }
    
    /**
     * @brief Log a network event
     */
    void LogEvent(const std::string& event) {
        m_EventLog.push_front(event);
        if (m_EventLog.size() > 100) {
            m_EventLog.pop_back();
        }
    }
    
    /**
     * @brief Update statistics (call every frame)
     */
    void Update(float deltaTime) {
        m_SampleTimer += deltaTime;
        
        if (m_SampleTimer >= 0.5f) {  // Sample every 500ms
            m_SampleTimer = 0.0f;
            
            // Record ping history
            if (m_Manager) {
                m_PingHistory.push_back(static_cast<float>(m_Manager->GetPing()));
                if (m_PingHistory.size() > 60) {
                    m_PingHistory.erase(m_PingHistory.begin());
                }
            }
            
            // Record bandwidth (placeholder)
            m_OutgoingBandwidth.push_back(m_OutgoingBytes);
            m_IncomingBandwidth.push_back(m_IncomingBytes);
            if (m_OutgoingBandwidth.size() > 60) {
                m_OutgoingBandwidth.erase(m_OutgoingBandwidth.begin());
                m_IncomingBandwidth.erase(m_IncomingBandwidth.begin());
            }
            
            m_OutgoingBytes = 0;
            m_IncomingBytes = 0;
        }
    }
    
    // Bandwidth tracking
    void AddOutgoingBytes(size_t bytes) { m_OutgoingBytes += bytes; }
    void AddIncomingBytes(size_t bytes) { m_IncomingBytes += bytes; }
    
private:
    MultiplayerManager* m_Manager = nullptr;
    bool m_ShowPanel = false;
    
    // Statistics
    std::vector<float> m_PingHistory;
    std::vector<float> m_OutgoingBandwidth;
    std::vector<float> m_IncomingBandwidth;
    float m_SampleTimer = 0.0f;
    size_t m_OutgoingBytes = 0;
    size_t m_IncomingBytes = 0;
    
    // Event log
    std::deque<std::string> m_EventLog;
    
    void RenderConnectionStatus() {
        ImGui::Text("Connection Status");
        
        if (!m_Manager) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No manager set");
            return;
        }
        
        auto state = m_Manager->GetState();
        switch (state) {
            case MultiplayerManager::State::Disconnected:
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "Disconnected");
                break;
            case MultiplayerManager::State::Connecting:
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Connecting...");
                break;
            case MultiplayerManager::State::Connected:
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Connected (Client)");
                break;
            case MultiplayerManager::State::Hosting:
                ImGui::TextColored(ImVec4(0, 1, 1, 1), "Hosting (Server)");
                break;
        }
        
        if (m_Manager->IsConnected()) {
            ImGui::Text("Player ID: %u", m_Manager->GetLocalPlayerId());
            ImGui::Text("Ping: %u ms", m_Manager->GetPing());
        }
    }
    
    void RenderPlayerList() {
        if (!m_Manager || !m_Manager->IsConnected()) return;
        
        ImGui::Text("Players (%u)", m_Manager->GetPlayerCount());
        
        if (ImGui::BeginTable("players", 4, ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("ID");
            ImGui::TableSetupColumn("Name");
            ImGui::TableSetupColumn("Ping");
            ImGui::TableSetupColumn("Status");
            ImGui::TableHeadersRow();
            
            for (const auto& player : m_Manager->GetConnectedPlayers()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%u", player.playerId);
                ImGui::TableNextColumn();
                ImGui::Text("%s", player.name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%u ms", player.ping);
                ImGui::TableNextColumn();
                if (player.isHost) {
                    ImGui::TextColored(ImVec4(0, 1, 1, 1), "Host");
                } else if (player.isLocal) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Local");
                } else {
                    ImGui::Text("Remote");
                }
            }
            
            ImGui::EndTable();
        }
    }
    
    void RenderStatistics() {
        ImGui::Text("Network Statistics");
        
        // Ping graph
        if (!m_PingHistory.empty()) {
            ImGui::PlotLines("Ping", m_PingHistory.data(), 
                           static_cast<int>(m_PingHistory.size()),
                           0, nullptr, 0.0f, 200.0f, ImVec2(0, 50));
        }
        
        // Bandwidth
        if (!m_OutgoingBandwidth.empty()) {
            float maxOut = *std::max_element(m_OutgoingBandwidth.begin(), m_OutgoingBandwidth.end());
            ImGui::PlotLines("Out (B/s)", m_OutgoingBandwidth.data(),
                           static_cast<int>(m_OutgoingBandwidth.size()),
                           0, nullptr, 0.0f, maxOut * 1.2f, ImVec2(0, 30));
        }
        
        if (!m_IncomingBandwidth.empty()) {
            float maxIn = *std::max_element(m_IncomingBandwidth.begin(), m_IncomingBandwidth.end());
            ImGui::PlotLines("In (B/s)", m_IncomingBandwidth.data(),
                           static_cast<int>(m_IncomingBandwidth.size()),
                           0, nullptr, 0.0f, maxIn * 1.2f, ImVec2(0, 30));
        }
    }
    
    void RenderEntityInfo() {
        if (!m_Manager) return;
        
        auto& repl = const_cast<MultiplayerManager*>(m_Manager)->GetReplicationManager();
        
        ImGui::Text("Replicated Entities");
        ImGui::Text("Total: %zu", repl.GetEntityCount());
        ImGui::Text("Dirty: %zu", repl.GetDirtyEntityCount());
    }
    
    void RenderEventLog() {
        ImGui::Text("Event Log");
        
        if (ImGui::BeginChild("EventLog", ImVec2(0, 100), true)) {
            for (const auto& event : m_EventLog) {
                ImGui::TextWrapped("%s", event.c_str());
            }
        }
        ImGui::EndChild();
        
        if (ImGui::Button("Clear Log")) {
            m_EventLog.clear();
        }
    }
};
