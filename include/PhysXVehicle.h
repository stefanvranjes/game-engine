#pragma once

#include "GameObject.h"
#include <PxPhysicsAPI.h>

class PhysXVehicle {
public:
    PhysXVehicle(class PhysXBackend* backend, const physx::PxVec3& position);
    virtual ~PhysXVehicle();

    void Update(float deltaTime);
    void SyncTransform();
    
    // Controls
    void SetInputs(bool accelerate, bool brake, bool steerLeft, bool steerRight);
    
    // Accessor for the backend
    physx::PxVehicleWheels* GetPxVehicle() const { return m_VehicleDrive; }

    void SetChassisTransform(const physx::PxTransform& transform);

private:
    class PhysXBackend* m_Backend;
    physx::PxVehicleDrive4W* m_VehicleDrive;
    physx::PxVehicleDrive4WRawInputData m_InputData;
    
    // Helpers to create vehicle components
    void CreateVehicle4W(const physx::PxVec3& position);
    physx::PxVehicleChassisData CreateChassisData();
    physx::PxVehicleWheelData* CreateWheelData();
    physx::PxVehicleTireData* CreateTireData();
    physx::PxVehicleSuspensionData* CreateSuspensionData();
    void ComputeWheelCenterActorOffsets4W(float wheelFrontZ, float wheelRearZ, const physx::PxVec3& chassisDims, float wheelWidth, float wheelRadius, uint32_t numWheels, physx::PxVec3* wheelCentreOffsets);
    void SetupWheelsSimulationData(const float wheelMass, const float wheelMOI, const float wheelRadius, const float wheelWidth, const uint32_t numWheels, const physx::PxVec3* wheelCenterActorOffsets, const physx::PxVec3& chassisCMOffset, const float chassisMass, physx::PxVehicleWheelsSimData* wheelsSimData);
};
