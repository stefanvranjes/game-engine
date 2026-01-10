#include "ThreadPool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads)
    : m_Stop(false)
    , m_ActiveJobs(0)
{
    // Ensure at least one thread
    numThreads = std::max(size_t(1), numThreads);
    
    // Create worker threads
    m_Workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        m_Workers.emplace_back(&ThreadPool::WorkerThread, this);
    }
    
    std::cout << "ThreadPool created with " << numThreads << " threads" << std::endl;
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(m_QueueMutex);
        m_Stop = true;
    }
    
    m_Condition.notify_all();
    
    // Wait for all threads to finish
    for (std::thread& worker : m_Workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    std::cout << "ThreadPool destroyed" << std::endl;
}

void ThreadPool::WaitForAll() {
    std::unique_lock<std::mutex> lock(m_QueueMutex);
    m_CompletionCondition.wait(lock, [this] {
        return m_Jobs.empty() && m_ActiveJobs == 0;
    });
}

size_t ThreadPool::GetPendingJobCount() const {
    std::unique_lock<std::mutex> lock(m_QueueMutex);
    return m_Jobs.size();
}

void ThreadPool::WorkerThread() {
    while (true) {
        std::function<void()> job;
        
        {
            std::unique_lock<std::mutex> lock(m_QueueMutex);
            
            // Wait for a job or stop signal
            m_Condition.wait(lock, [this] {
                return m_Stop || !m_Jobs.empty();
            });
            
            // Exit if stopped and no jobs remaining
            if (m_Stop && m_Jobs.empty()) {
                return;
            }
            
            // Get next job
            if (!m_Jobs.empty()) {
                job = std::move(m_Jobs.front());
                m_Jobs.pop();
            }
        }
        
        // Execute job
        if (job) {
            try {
                job();
            }
            catch (const std::exception& e) {
                std::cerr << "ThreadPool: Exception in worker thread: " << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "ThreadPool: Unknown exception in worker thread" << std::endl;
            }
            
            // Decrement active job count
            m_ActiveJobs--;
            
            // Notify completion if all jobs are done
            if (m_ActiveJobs == 0 && m_Jobs.empty()) {
                m_CompletionCondition.notify_all();
            }
        }
    }
}
