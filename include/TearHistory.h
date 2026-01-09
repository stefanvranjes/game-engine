#pragma once

#include "Math/Vec3.h"
#include <vector>
#include <chrono>

/**
 * @brief Records and manages tear history for undo/redo
 * 
 * Maintains a history of tear operations with support for
 * undoing and redoing tears, including vertex state restoration.
 */
class TearHistory {
public:
    struct TearAction {
        std::vector<int> tornTetrahedra;
        std::vector<int> affectedVertices;
        std::vector<Vec3> vertexPositionsBefore;
        std::vector<Vec3> vertexPositionsAfter;
        float tearThresholdBefore;
        float tearThresholdAfter;
        std::chrono::steady_clock::time_point timestamp;
        
        TearAction() 
            : tearThresholdBefore(0.0f)
            , tearThresholdAfter(0.0f)
            , timestamp(std::chrono::steady_clock::now())
        {}
    };
    
    TearHistory();
    
    /**
     * @brief Record a tear action
     */
    void RecordTear(const TearAction& action);
    
    /**
     * @brief Check if undo is available
     */
    bool CanUndo() const;
    
    /**
     * @brief Check if redo is available
     */
    bool CanRedo() const;
    
    /**
     * @brief Undo last tear
     * @return The undone action
     */
    TearAction Undo();
    
    /**
     * @brief Redo last undone tear
     * @return The redone action
     */
    TearAction Redo();
    
    /**
     * @brief Clear all history
     */
    void Clear();
    
    /**
     * @brief Get number of actions in history
     */
    int GetHistorySize() const { return static_cast<int>(m_History.size()); }
    
    /**
     * @brief Get current position in history
     */
    int GetCurrentIndex() const { return m_CurrentIndex; }
    
private:
    std::vector<TearAction> m_History;
    int m_CurrentIndex;
    static constexpr int MAX_HISTORY = 50;
};
