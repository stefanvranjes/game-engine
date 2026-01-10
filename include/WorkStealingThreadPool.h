#pragma once

#include "WorkStealingQueue.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <random>

/**
 * @brief Thread pool with work stealing for better load balancing
 */
class WorkStealingThreadPool {
public:
    /**
     * @brief Constructor
     * @param numThreads Number of worker threads (default: hardware concurrency)
     */
    explicit WorkStealingThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~WorkStealingThreadPool();
    
    /**
     * @brief Submit a task for execution
     * @param f Function to execute
     * @param args Arguments for the function
     * @return Future for the result
     */
    template<typename F, typename... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;
    
    /**
     * @brief Wait for all pending tasks to complete
     */
    void WaitForAll();
    
    /**
     * @brief Get number of worker threads
     */
    size_t GetThreadCount() const { return m_Workers.size(); }
    
    /**
     * @brief Get number of active jobs
     */
    size_t GetActiveJobCount() const { return m_ActiveJobs.load(); }
    
    /**
     * @brief Get total pending jobs across all queues
     */
    size_t GetPendingJobCount() const;
    
    /**
     * @brief Enable thread affinity (pin threads to cores)
     * @param enable True to enable, false to disable
     */
    void EnableThreadAffinity(bool enable);
    
    /**
     * @brief Check if thread affinity is enabled
     */
    bool IsThreadAffinityEnabled() const { return m_UseThreadAffinity; }

private:
    using Task = std::function<void()>;
    
    std::vector<std::thread> m_Workers;
    std::vector<std::unique_ptr<WorkStealingQueue<Task>>> m_Queues;
    
    std::condition_variable m_Condition;
    std::condition_variable m_CompletionCondition;
    std::mutex m_WaitMutex;
    
    std::atomic<bool> m_Stop;
    std::atomic<size_t> m_ActiveJobs;
    std::atomic<size_t> m_SubmitIndex;  // Round-robin submission
    bool m_UseThreadAffinity;
    
    /**
     * @brief Worker thread function
     * @param threadIndex Index of this worker thread
     */
    void WorkerThread(size_t threadIndex);
    
    /**
     * @brief Try to steal work from other threads
     * @param threadIndex Index of the thread trying to steal
     * @return Task if stolen, nullopt otherwise
     */
    std::optional<Task> TrySteal(size_t threadIndex);
};

// Template implementation
template<typename F, typename... Args>
auto WorkStealingThreadPool::Submit(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type> 
{
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    if (m_Stop) {
        throw std::runtime_error("Cannot submit task to stopped WorkStealingThreadPool");
    }
    
    // Round-robin distribution to worker queues
    size_t queueIndex = m_SubmitIndex.fetch_add(1) % m_Queues.size();
    m_Queues[queueIndex]->Push([task]() { (*task)(); });
    
    m_ActiveJobs++;
    m_Condition.notify_one();
    
    return result;
}
