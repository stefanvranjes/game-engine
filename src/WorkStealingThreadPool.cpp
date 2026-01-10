#include "WorkStealingThreadPool.h"
#include "ThreadAffinity.h"
#include <iostream>
#include <algorithm>

WorkStealingThreadPool::WorkStealingThreadPool(size_t numThreads)
    : m_Stop(false)
    , m_ActiveJobs(0)
    , m_SubmitIndex(0)
    , m_UseThreadAffinity(false)
{
    // Ensure at least one thread
    numThreads = std::max(size_t(1), numThreads);
    
    // Create work-stealing queues (one per thread)
    m_Queues.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        m_Queues.push_back(std::make_unique<WorkStealingQueue<Task>>());
    }
    
    // Create worker threads
    m_Workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        m_Workers.emplace_back(&WorkStealingThreadPool::WorkerThread, this, i);
        
        // Set thread affinity if enabled
        if (m_UseThreadAffinity && ThreadAffinity::IsSupported()) {
            ThreadAffinity::SetAffinity(m_Workers.back(), i % ThreadAffinity::GetCoreCount());
        }
    }
    
    std::cout << "WorkStealingThreadPool created with " << numThreads << " threads";
    if (m_UseThreadAffinity) {
        std::cout << " (thread affinity enabled)";
    }
    std::cout << std::endl;
}

WorkStealingThreadPool::~WorkStealingThreadPool() {
    m_Stop.store(true);
    m_Condition.notify_all();
    
    // Wait for all threads to finish
    for (std::thread& worker : m_Workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    std::cout << "WorkStealingThreadPool destroyed" << std::endl;
}

void WorkStealingThreadPool::WaitForAll() {
    std::unique_lock<std::mutex> lock(m_WaitMutex);
    m_CompletionCondition.wait(lock, [this] {
        // Check if all queues are empty and no active jobs
        if (m_ActiveJobs > 0) return false;
        
        for (const auto& queue : m_Queues) {
            if (!queue->Empty()) return false;
        }
        
        return true;
    });
}

size_t WorkStealingThreadPool::GetPendingJobCount() const {
    size_t total = 0;
    for (const auto& queue : m_Queues) {
        total += queue->Size();
    }
    return total;
}

void WorkStealingThreadPool::WorkerThread(size_t threadIndex) {
    while (true) {
        std::optional<Task> task;
        
        // Try to get work from own queue first
        task = m_Queues[threadIndex]->Pop();
        
        // If no work in own queue, try to steal from others
        if (!task) {
            task = TrySteal(threadIndex);
        }
        
        // If still no work, wait for notification
        if (!task) {
            std::unique_lock<std::mutex> lock(m_WaitMutex);
            m_Condition.wait_for(lock, std::chrono::milliseconds(1), [this, threadIndex] {
                return m_Stop.load() || !m_Queues[threadIndex]->Empty();
            });
            
            // Exit if stopped and no work remaining
            if (m_Stop.load()) {
                // Check one more time for remaining work
                task = m_Queues[threadIndex]->Pop();
                if (!task) {
                    task = TrySteal(threadIndex);
                }
                if (!task) {
                    return;
                }
            } else {
                continue;
            }
        }
        
        // Execute task
        if (task) {
            try {
                (*task)();
            }
            catch (const std::exception& e) {
                std::cerr << "WorkStealingThreadPool: Exception in worker " << threadIndex 
                          << ": " << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "WorkStealingThreadPool: Unknown exception in worker " << threadIndex << std::endl;
            }
            
            // Decrement active job count
            m_ActiveJobs--;
            
            // Notify if all work is done
            if (m_ActiveJobs == 0 && GetPendingJobCount() == 0) {
                m_CompletionCondition.notify_all();
            }
        }
    }
}

std::optional<WorkStealingThreadPool::Task> WorkStealingThreadPool::TrySteal(size_t threadIndex) {
    // Try to steal from other threads in random order
    size_t numThreads = m_Queues.size();
    
    // Create random starting point to avoid always stealing from same thread
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, numThreads - 1);
    size_t startIndex = dist(rng);
    
    // Try each queue once
    for (size_t i = 0; i < numThreads; ++i) {
        size_t victimIndex = (startIndex + i) % numThreads;
        
        // Don't steal from self
        if (victimIndex == threadIndex) {
            continue;
        }
        
        // Try to steal from this queue
        auto task = m_Queues[victimIndex]->Steal();
        if (task) {
            return task;
        }
    }
    
    return std::nullopt;
}

void WorkStealingThreadPool::EnableThreadAffinity(bool enable) {
    if (enable && !ThreadAffinity::IsSupported()) {
        std::cerr << "Thread affinity not supported on this platform" << std::endl;
        return;
    }
    
    m_UseThreadAffinity = enable;
    
    if (enable) {
        // Apply affinity to existing threads
        for (size_t i = 0; i < m_Workers.size(); ++i) {
            ThreadAffinity::SetAffinity(m_Workers[i], i % ThreadAffinity::GetCoreCount());
        }
        std::cout << "Thread affinity enabled" << std::endl;
    } else {
        std::cout << "Thread affinity disabled" << std::endl;
    }
}

