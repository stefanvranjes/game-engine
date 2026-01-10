#pragma once

#include <deque>
#include <mutex>
#include <optional>

/**
 * @brief Work-stealing deque for thread pool
 * 
 * Each worker thread has its own deque. The owning thread pushes/pops from one end,
 * while other threads can steal from the other end.
 */
template<typename T>
class WorkStealingQueue {
public:
    WorkStealingQueue() = default;
    
    /**
     * @brief Push item to the queue (called by owner thread)
     */
    void Push(T item) {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Deque.push_back(std::move(item));
    }
    
    /**
     * @brief Pop item from the queue (called by owner thread)
     * @return Item if available, nullopt otherwise
     */
    std::optional<T> Pop() {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (m_Deque.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(m_Deque.back());
        m_Deque.pop_back();
        return item;
    }
    
    /**
     * @brief Steal item from the queue (called by other threads)
     * @return Item if available, nullopt otherwise
     */
    std::optional<T> Steal() {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (m_Deque.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(m_Deque.front());
        m_Deque.pop_front();
        return item;
    }
    
    /**
     * @brief Check if queue is empty
     */
    bool Empty() const {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_Deque.empty();
    }
    
    /**
     * @brief Get queue size
     */
    size_t Size() const {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_Deque.size();
    }

private:
    std::deque<T> m_Deque;
    mutable std::mutex m_Mutex;
};
