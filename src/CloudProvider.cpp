#include "CloudProvider.h"
#include <iostream>
#include <sstream>

StubCloudProvider::StubCloudProvider(CloudProviderType type) 
    : m_Type(type), m_NextInstanceId(1) {
    std::cout << "Initialized stub cloud provider: " << (int)type << std::endl;
    std::cout << "NOTE: This is a stub implementation for testing." << std::endl;
    std::cout << "Integrate actual cloud SDKs for production deployment." << std::endl;
}

std::string StubCloudProvider::LaunchInstance(const std::string& instanceType,
                                               const std::string& region) {
    std::stringstream ss;
    ss << "instance-" << m_NextInstanceId++;
    std::string instanceId = ss.str();
    
    CloudInstance instance;
    instance.instanceId = instanceId;
    instance.publicIp = "192.168.1." + std::to_string(m_NextInstanceId);
    instance.privateIp = "10.0.0." + std::to_string(m_NextInstanceId);
    instance.provider = m_Type;
    instance.instanceType = instanceType;
    instance.gpuCount = 1;  // Assume 1 GPU
    instance.memoryMB = 61440;  // 60 GB
    instance.region = region;
    instance.availabilityZone = region + "a";
    instance.isRunning = true;
    instance.costPerHour = 3.06;  // p3.2xlarge price
    
    m_Instances.push_back(instance);
    
    std::cout << "Launched instance: " << instanceId 
              << " (" << instanceType << " in " << region << ")" << std::endl;
    
    return instanceId;
}

bool StubCloudProvider::TerminateInstance(const std::string& instanceId) {
    for (auto it = m_Instances.begin(); it != m_Instances.end(); ++it) {
        if (it->instanceId == instanceId) {
            std::cout << "Terminated instance: " << instanceId << std::endl;
            m_Instances.erase(it);
            return true;
        }
    }
    
    std::cerr << "Instance not found: " << instanceId << std::endl;
    return false;
}

CloudInstance StubCloudProvider::GetInstanceInfo(const std::string& instanceId) {
    for (const auto& instance : m_Instances) {
        if (instance.instanceId == instanceId) {
            return instance;
        }
    }
    
    return CloudInstance{};
}

std::vector<CloudInstance> StubCloudProvider::ListInstances() {
    return m_Instances;
}

bool StubCloudProvider::SetAutoScaling(int minInstances, int maxInstances,
                                        const std::string& instanceType) {
    std::cout << "Set auto-scaling: min=" << minInstances 
              << ", max=" << maxInstances 
              << ", type=" << instanceType << std::endl;
    return true;
}

int StubCloudProvider::GetCurrentInstanceCount() const {
    return static_cast<int>(m_Instances.size());
}

double StubCloudProvider::GetCurrentCost() const {
    double totalCost = 0.0;
    for (const auto& instance : m_Instances) {
        if (instance.isRunning) {
            totalCost += instance.costPerHour;
        }
    }
    return totalCost;
}

double StubCloudProvider::EstimateCost(int instanceCount, int hours) const {
    // Assume p3.2xlarge pricing
    double costPerHour = 3.06;
    return instanceCount * hours * costPerHour;
}
