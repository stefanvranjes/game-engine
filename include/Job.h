#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <functional>
#include <string>

/**
 * @brief Job priority levels
 */
enum class JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

/**
 * @brief Base class for jobs with dependency management
 */
class Job : public std::enable_shared_from_this<Job> {
public:
    using CompletionCallback = std::function<void(Job*)>;
    
    virtual ~Job() = default;
    
    /**
     * @brief Execute the job
     */
    virtual void Execute() = 0;
    
    /**
     * @brief Get job name for debugging
     */
    virtual std::string GetName() const { return "Job"; }
    
    /**
     * @brief Add a dependency (this job waits for dependency to complete)
     */
    void AddDependency(std::shared_ptr<Job> dependency);
    
    /**
     * @brief Check if all dependencies are complete
     */
    bool IsReady() const;
    
    /**
     * @brief Mark job as complete
     */
    void MarkComplete();
    
    /**
     * @brief Check if job is complete
     */
    bool IsComplete() const { return m_Complete.load(); }
    
    /**
     * @brief Set completion callback
     */
    void SetCompletionCallback(CompletionCallback callback) { m_CompletionCallback = callback; }
    
    /**
     * @brief Set job priority
     */
    void SetPriority(JobPriority priority) { m_Priority = priority; }
    
    /**
     * @brief Get job priority
     */
    JobPriority GetPriority() const { return m_Priority; }
    
    /**
     * @brief Get dependency count
     */
    size_t GetDependencyCount() const { return m_Dependencies.size(); }

protected:
    std::vector<std::weak_ptr<Job>> m_Dependencies;
    std::atomic<bool> m_Complete{false};
    CompletionCallback m_CompletionCallback;
    JobPriority m_Priority = JobPriority::Normal;
};

/**
 * @brief Lambda job - wraps a lambda function
 */
class LambdaJob : public Job {
public:
    using JobFunction = std::function<void()>;
    
    explicit LambdaJob(JobFunction func, const std::string& name = "LambdaJob")
        : m_Function(std::move(func))
        , m_Name(name)
    {}
    
    void Execute() override {
        if (m_Function) {
            m_Function();
        }
    }
    
    std::string GetName() const override { return m_Name; }

private:
    JobFunction m_Function;
    std::string m_Name;
};
