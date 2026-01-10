# Cloud SDK Integration Guide

## AWS SDK Integration

### Prerequisites

```bash
# Install AWS SDK for C++
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp
cd aws-sdk-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_ONLY="ec2;autoscaling;cloudwatch"
make -j$(nproc)
sudo make install
```

### CMake Integration

```cmake
# Find AWS SDK
find_package(AWSSDK REQUIRED COMPONENTS ec2 autoscaling cloudwatch)

# Link libraries
target_link_libraries(GameEngine
    ${AWSSDK_LINK_LIBRARIES}
)
```

### Implementation

```cpp
// AwsProvider.h
#include <aws/core/Aws.h>
#include <aws/ec2/EC2Client.h>
#include <aws/autoscaling/AutoScalingClient.h>

class AwsProvider : public ICloudProvider {
public:
    AwsProvider(const std::string& accessKey, 
                const std::string& secretKey,
                const std::string& region) {
        
        Aws::SDKOptions options;
        Aws::InitAPI(options);
        
        Aws::Client::ClientConfiguration config;
        config.region = region;
        
        Aws::Auth::AWSCredentials credentials(accessKey, secretKey);
        
        m_Ec2Client = std::make_unique<Aws::EC2::EC2Client>(credentials, config);
        m_AutoScalingClient = std::make_unique<Aws::AutoScaling::AutoScalingClient>(
            credentials, config);
    }
    
    std::string LaunchInstance(const std::string& instanceType,
                              const std::string& region) override {
        Aws::EC2::Model::RunInstancesRequest request;
        request.SetImageId("ami-xxxxx");  // Your AMI
        request.SetInstanceType(
            Aws::EC2::Model::InstanceTypeMapper::GetInstanceTypeForName(instanceType));
        request.SetMinCount(1);
        request.SetMaxCount(1);
        
        auto outcome = m_Ec2Client->RunInstances(request);
        
        if (outcome.IsSuccess()) {
            return outcome.GetResult().GetInstances()[0].GetInstanceId();
        }
        
        return "";
    }
    
    bool SetAutoScaling(int minInstances, int maxInstances,
                       const std::string& instanceType) override {
        Aws::AutoScaling::Model::UpdateAutoScalingGroupRequest request;
        request.SetAutoScalingGroupName("physics-workers");
        request.SetMinSize(minInstances);
        request.SetMaxSize(maxInstances);
        
        auto outcome = m_AutoScalingClient->UpdateAutoScalingGroup(request);
        return outcome.IsSuccess();
    }
    
private:
    std::unique_ptr<Aws::EC2::EC2Client> m_Ec2Client;
    std::unique_ptr<Aws::AutoScaling::AutoScalingClient> m_AutoScalingClient;
};
```

---

## Azure SDK Integration

### Prerequisites

```bash
# Install Azure SDK for C++
git clone https://github.com/Azure/azure-sdk-for-cpp.git
cd azure-sdk-for-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### CMake Integration

```cmake
# Find Azure SDK
find_package(azure-identity-cpp CONFIG REQUIRED)
find_package(azure-compute CONFIG REQUIRED)

# Link libraries
target_link_libraries(GameEngine
    Azure::azure-identity
    Azure::azure-compute
)
```

### Implementation

```cpp
// AzureProvider.h
#include <azure/identity.hpp>
#include <azure/compute.hpp>

class AzureProvider : public ICloudProvider {
public:
    AzureProvider(const std::string& subscriptionId,
                  const std::string& clientId,
                  const std::string& clientSecret,
                  const std::string& tenantId) {
        
        auto credential = std::make_shared<Azure::Identity::ClientSecretCredential>(
            tenantId, clientId, clientSecret);
        
        m_ComputeClient = std::make_unique<Azure::Compute::ComputeManagementClient>(
            credential, subscriptionId);
    }
    
    std::string LaunchInstance(const std::string& instanceType,
                              const std::string& region) override {
        Azure::Compute::Models::VirtualMachine vm;
        vm.Location = region;
        vm.HardwareProfile.VmSize = instanceType;
        
        auto result = m_ComputeClient->VirtualMachines.CreateOrUpdate(
            "physics-rg", "worker-vm", vm);
        
        return result.Value.Id;
    }
    
private:
    std::unique_ptr<Azure::Compute::ComputeManagementClient> m_ComputeClient;
};
```

---

## GCP SDK Integration

### Prerequisites

```bash
# Install Google Cloud C++ SDK
git clone https://github.com/googleapis/google-cloud-cpp.git
cd google-cloud-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target google-cloud-cpp
sudo cmake --build build --target install
```

### CMake Integration

```cmake
# Find GCP SDK
find_package(google_cloud_cpp_compute REQUIRED)

# Link libraries
target_link_libraries(GameEngine
    google-cloud-cpp::compute
)
```

### Implementation

```cpp
// GcpProvider.h
#include <google/cloud/compute/v1/instances_client.h>

namespace gc = ::google::cloud;
namespace compute = ::google::cloud::compute_v1;

class GcpProvider : public ICloudProvider {
public:
    GcpProvider(const std::string& projectId,
                const std::string& serviceAccountKey) {
        
        auto credentials = gc::MakeServiceAccountCredentials(serviceAccountKey);
        
        m_InstancesClient = std::make_unique<compute::InstancesClient>(
            compute::MakeInstancesConnection(
                gc::Options{}.set<gc::UnifiedCredentialsOption>(credentials)));
        
        m_ProjectId = projectId;
    }
    
    std::string LaunchInstance(const std::string& instanceType,
                              const std::string& region) override {
        compute::Instance instance;
        instance.set_name("physics-worker");
        instance.set_machine_type(instanceType);
        
        auto result = m_InstancesClient->InsertInstance(
            m_ProjectId, region, instance);
        
        if (result.ok()) {
            return result->name();
        }
        
        return "";
    }
    
private:
    std::unique_ptr<compute::InstancesClient> m_InstancesClient;
    std::string m_ProjectId;
};
```

---

## Multi-Cloud Factory

```cpp
// CloudProviderFactory.h
class CloudProviderFactory {
public:
    static std::unique_ptr<ICloudProvider> Create(CloudProviderType type) {
        switch (type) {
            case CloudProviderType::AWS:
                return std::make_unique<AwsProvider>(
                    std::getenv("AWS_ACCESS_KEY_ID"),
                    std::getenv("AWS_SECRET_ACCESS_KEY"),
                    std::getenv("AWS_REGION"));
            
            case CloudProviderType::AZURE:
                return std::make_unique<AzureProvider>(
                    std::getenv("AZURE_SUBSCRIPTION_ID"),
                    std::getenv("AZURE_CLIENT_ID"),
                    std::getenv("AZURE_CLIENT_SECRET"),
                    std::getenv("AZURE_TENANT_ID"));
            
            case CloudProviderType::GCP:
                return std::make_unique<GcpProvider>(
                    std::getenv("GCP_PROJECT_ID"),
                    std::getenv("GOOGLE_APPLICATION_CREDENTIALS"));
            
            default:
                return std::make_unique<StubCloudProvider>(type);
        }
    }
};
```

---

## Usage Example

```cpp
// Create cloud provider
auto provider = CloudProviderFactory::Create(CloudProviderType::AWS);

// Set up auto-scaling
provider->SetAutoScaling(2, 20, "p3.2xlarge");

// Integrate with distributed manager
DistributedBatchManager manager;
manager.InitializeAsGlobalMaster(8080);
manager.SetCloudProvider(provider.get());

// Launch workers on demand
if (manager.GetStatistics().pendingBatches > 10) {
    provider->LaunchInstance("p3.2xlarge", "us-west-2");
}
```

---

## Environment Variables

```bash
# AWS
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-west-2"

# Azure
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_TENANT_ID="your-tenant-id"

# GCP
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

---

## Testing

```cpp
// Test with stub provider first
auto provider = std::make_unique<StubCloudProvider>(CloudProviderType::AWS);

// Launch test instance
std::string instanceId = provider->LaunchInstance("p3.2xlarge", "us-west-2");
std::cout << "Launched: " << instanceId << std::endl;

// Check cost
double cost = provider->GetCurrentCost();
std::cout << "Current cost: $" << cost << "/hour" << std::endl;

// Terminate
provider->TerminateInstance(instanceId);
```

---

## Production Checklist

- [ ] Install cloud SDK
- [ ] Configure credentials
- [ ] Test with stub provider
- [ ] Integrate real provider
- [ ] Set cost limits
- [ ] Enable monitoring
- [ ] Test auto-scaling
- [ ] Deploy to production
