#pragma once

#include "Job.h"
#include "ThreadPool.h"
#include <queue>
#include <mutex>
#include <vector>
#include <memory>

/**
 * @brief Job scheduler with dependency resolution
 */
class JobScheduler {
public:
    explicit JobScheduler(ThreadPool& threadPool);
    
    /**
     * @brief Schedule a job for execution
     * @param job Job to schedule
     */
    void ScheduleJob(std::shared_ptr<Job> job);
    
    /**
     * @brief Schedule multiple jobs
     */
    void ScheduleJobs(const std::vector<std::shared_ptr<Job>>& jobs);
    
    /**
     * @brief Wait for all scheduled jobs to complete
     */
    void WaitForAll();
    
    /**
     * @brief Get number of pending jobs
     */
    size_t GetPendingJobCount() const;
    
    /**
     * @brief Get number of running jobs
     */
    size_t GetRunningJobCount() const { return m_RunningJobs.load(); }

private:
    struct JobEntry {
        std::shared_ptr<Job> job;
        
        bool operator<(const JobEntry& other) const {
            // Higher priority first (reverse comparison for priority queue)
            return static_cast<int>(job->GetPriority()) < static_cast<int>(other.job->GetPriority());
        }
    };
    
    ThreadPool& m_ThreadPool;
    std::priority_queue<JobEntry> m_PendingJobs;
    mutable std::mutex m_JobMutex;
    std::atomic<size_t> m_RunningJobs{0};
    
    /**
     * @brief Process pending jobs and submit ready ones
     */
    void ProcessPendingJobs();
    
    /**
     * @brief Execute a job and mark it complete
     */
    void ExecuteJob(std::shared_ptr<Job> job);
};
