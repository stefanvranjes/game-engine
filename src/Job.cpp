#include "Job.h"
#include <algorithm>

void Job::AddDependency(std::shared_ptr<Job> dependency) {
    if (dependency) {
        m_Dependencies.push_back(dependency);
    }
}

bool Job::IsReady() const {
    // Check if all dependencies are complete
    for (const auto& weakDep : m_Dependencies) {
        auto dep = weakDep.lock();
        if (dep && !dep->IsComplete()) {
            return false;
        }
    }
    return true;
}

void Job::MarkComplete() {
    m_Complete.store(true);
    
    // Call completion callback if set
    if (m_CompletionCallback) {
        m_CompletionCallback(this);
    }
}
