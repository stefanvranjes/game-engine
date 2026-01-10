#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

/**
 * @brief Thread pool for parallel task execution
 */
class ThreadPool {
public:
    /**
     * @brief Constructor
     * @param numThreads Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool();
    
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
     * @brief Get number of pending jobs
     */
    size_t GetPendingJobCount() const;

private:
    std::vector<std::thread> m_Workers;
    std::queue<std::function<void()>> m_Jobs;
    
    std::mutex m_QueueMutex;
    std::condition_variable m_Condition;
    std::condition_variable m_CompletionCondition;
    
    std::atomic<bool> m_Stop;
    std::atomic<size_t> m_ActiveJobs;
    
    /**
     * @brief Worker thread function
     */
    void WorkerThread();
};

// Template implementation
template<typename F, typename... Args>
auto ThreadPool::Submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(m_QueueMutex);
        
        if (m_Stop) {
            throw std::runtime_error("Cannot submit task to stopped ThreadPool");
        }
        
        m_Jobs.emplace([task]() { (*task)(); });
        m_ActiveJobs++;
    }
    
    m_Condition.notify_one();
    
    return result;
}
