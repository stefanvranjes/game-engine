#include "JobScheduler.h"
#include <iostream>
#include <thread>
#include <chrono>

JobScheduler::JobScheduler(ThreadPool& threadPool)
    : m_ThreadPool(threadPool)
{
}

void JobScheduler::ScheduleJob(std::shared_ptr<Job> job) {
    if (!job) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_JobMutex);
    m_PendingJobs.push({job});
    
    // Process jobs to submit ready ones
    ProcessPendingJobs();
}

void JobScheduler::ScheduleJobs(const std::vector<std::shared_ptr<Job>>& jobs) {
    std::lock_guard<std::mutex> lock(m_JobMutex);
    
    for (const auto& job : jobs) {
        if (job) {
            m_PendingJobs.push({job});
        }
    }
    
    // Process jobs to submit ready ones
    ProcessPendingJobs();
}

void JobScheduler::WaitForAll() {
    // Wait until all jobs are complete
    while (true) {
        {
            std::lock_guard<std::mutex> lock(m_JobMutex);
            if (m_PendingJobs.empty() && m_RunningJobs == 0) {
                break;
            }
        }
        
        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // Try to process pending jobs
        {
            std::lock_guard<std::mutex> lock(m_JobMutex);
            ProcessPendingJobs();
        }
    }
}

size_t JobScheduler::GetPendingJobCount() const {
    std::lock_guard<std::mutex> lock(m_JobMutex);
    return m_PendingJobs.size();
}

void JobScheduler::ProcessPendingJobs() {
    // Note: Caller must hold m_JobMutex
    
    std::vector<JobEntry> stillPending;
    
    // Check all pending jobs
    while (!m_PendingJobs.empty()) {
        JobEntry entry = m_PendingJobs.top();
        m_PendingJobs.pop();
        
        if (entry.job->IsReady()) {
            // Job is ready, submit to thread pool
            m_RunningJobs++;
            
            m_ThreadPool.Submit([this, job = entry.job]() {
                ExecuteJob(job);
            });
        } else {
            // Job not ready yet, keep it pending
            stillPending.push_back(entry);
        }
    }
    
    // Re-add jobs that aren't ready yet
    for (const auto& entry : stillPending) {
        m_PendingJobs.push(entry);
    }
}

void JobScheduler::ExecuteJob(std::shared_ptr<Job> job) {
    try {
        // Execute the job
        job->Execute();
        
        // Mark as complete
        job->MarkComplete();
    }
    catch (const std::exception& e) {
        std::cerr << "JobScheduler: Exception in job '" << job->GetName() 
                  << "': " << e.what() << std::endl;
        job->MarkComplete();  // Mark complete even on error
    }
    catch (...) {
        std::cerr << "JobScheduler: Unknown exception in job '" << job->GetName() << "'" << std::endl;
        job->MarkComplete();  // Mark complete even on error
    }
    
    // Decrement running count
    m_RunningJobs--;
    
    // Try to process more pending jobs
    {
        std::lock_guard<std::mutex> lock(m_JobMutex);
        ProcessPendingJobs();
    }
}
