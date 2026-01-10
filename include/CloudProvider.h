#pragma once

#include <string>
#include <vector>
#include <memory>

/**
 * @brief Cloud provider types
 */
enum class CloudProviderType {
    AWS,
    AZURE,
    GCP,
    ON_PREMISE
};

/**
 * @brief Cloud instance information
 */
struct CloudInstance {
    std::string instanceId;
    std::string publicIp;
    std::string privateIp;
    CloudProviderType provider;
    std::string instanceType;
    int gpuCount;
    uint64_t memoryMB;
    std::string region;
    std::string availabilityZone;
    bool isRunning;
    double costPerHour;
};

/**
 * @brief Cloud provider interface
 * 
 * Stub implementation for cloud integration.
 * Integrate actual cloud SDKs (AWS SDK, Azure SDK, GCP SDK) when deploying to cloud.
 */
class ICloudProvider {
public:
    virtual ~ICloudProvider() = default;
    
    /**
     * @brief Launch cloud instance
     * @param instanceType Instance type (e.g., "p3.2xlarge")
     * @param region Cloud region
     * @return Instance ID
     */
    virtual std::string LaunchInstance(const std::string& instanceType,
                                      const std::string& region) = 0;
    
    /**
     * @brief Terminate cloud instance
     * @param instanceId Instance ID
     * @return True if successful
     */
    virtual bool TerminateInstance(const std::string& instanceId) = 0;
    
    /**
     * @brief Get instance information
     * @param instanceId Instance ID
     * @return Instance info
     */
    virtual CloudInstance GetInstanceInfo(const std::string& instanceId) = 0;
    
    /**
     * @brief List all instances
     * @return List of instances
     */
    virtual std::vector<CloudInstance> ListInstances() = 0;
    
    /**
     * @brief Set auto-scaling configuration
     * @param minInstances Minimum instances
     * @param maxInstances Maximum instances
     * @param instanceType Instance type
     * @return True if successful
     */
    virtual bool SetAutoScaling(int minInstances, int maxInstances,
                               const std::string& instanceType) = 0;
    
    /**
     * @brief Get current instance count
     * @return Instance count
     */
    virtual int GetCurrentInstanceCount() const = 0;
    
    /**
     * @brief Get current cost
     * @return Cost per hour
     */
    virtual double GetCurrentCost() const = 0;
    
    /**
     * @brief Estimate cost
     * @param instanceCount Number of instances
     * @param hours Hours to run
     * @return Estimated cost
     */
    virtual double EstimateCost(int instanceCount, int hours) const = 0;
};

/**
 * @brief Stub cloud provider (for testing/development)
 */
class StubCloudProvider : public ICloudProvider {
public:
    StubCloudProvider(CloudProviderType type);
    
    std::string LaunchInstance(const std::string& instanceType,
                              const std::string& region) override;
    bool TerminateInstance(const std::string& instanceId) override;
    CloudInstance GetInstanceInfo(const std::string& instanceId) override;
    std::vector<CloudInstance> ListInstances() override;
    bool SetAutoScaling(int minInstances, int maxInstances,
                       const std::string& instanceType) override;
    int GetCurrentInstanceCount() const override;
    double GetCurrentCost() const override;
    double EstimateCost(int instanceCount, int hours) const override;
    
private:
    CloudProviderType m_Type;
    std::vector<CloudInstance> m_Instances;
    int m_NextInstanceId;
};
