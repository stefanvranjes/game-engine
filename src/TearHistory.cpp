#include "TearHistory.h"
#include <iostream>

TearHistory::TearHistory()
    : m_CurrentIndex(-1)
{
}

void TearHistory::RecordTear(const TearAction& action) {
    // Remove any actions after current index (redo history)
    if (m_CurrentIndex < static_cast<int>(m_History.size()) - 1) {
        m_History.erase(m_History.begin() + m_CurrentIndex + 1, m_History.end());
    }
    
    // Add new action
    m_History.push_back(action);
    m_CurrentIndex++;
    
    // Limit history size
    if (m_History.size() > MAX_HISTORY) {
        m_History.erase(m_History.begin());
        m_CurrentIndex--;
    }
    
    std::cout << "Recorded tear action (history size: " << m_History.size() 
              << ", index: " << m_CurrentIndex << ")" << std::endl;
}

bool TearHistory::CanUndo() const {
    return m_CurrentIndex >= 0 && !m_History.empty();
}

bool TearHistory::CanRedo() const {
    return m_CurrentIndex < static_cast<int>(m_History.size()) - 1;
}

TearHistory::TearAction TearHistory::Undo() {
    if (!CanUndo()) {
        std::cerr << "Cannot undo: no actions in history" << std::endl;
        return TearAction();
    }
    
    TearAction action = m_History[m_CurrentIndex];
    m_CurrentIndex--;
    
    std::cout << "Undid tear action (index now: " << m_CurrentIndex << ")" << std::endl;
    
    return action;
}

TearHistory::TearAction TearHistory::Redo() {
    if (!CanRedo()) {
        std::cerr << "Cannot redo: no actions to redo" << std::endl;
        return TearAction();
    }
    
    m_CurrentIndex++;
    TearAction action = m_History[m_CurrentIndex];
    
    std::cout << "Redid tear action (index now: " << m_CurrentIndex << ")" << std::endl;
    
    return action;
}

void TearHistory::Clear() {
    m_History.clear();
    m_CurrentIndex = -1;
    
    std::cout << "Cleared tear history" << std::endl;
}
