#include "PhysXVehicle.h"
#include "PhysXBackend.h"
#include <iostream>

using namespace physx;

PhysXVehicle::PhysXVehicle(PhysXBackend* backend, const PxVec3& position)
    : m_Backend(backend)
    , m_VehicleDrive(nullptr)
{
    CreateVehicle4W(position);
    
    if (m_Backend) {
        m_Backend->RegisterVehicle(this);
    }
}

PhysXVehicle::~PhysXVehicle() {
    if (m_Backend) {
        m_Backend->UnregisterVehicle(this);
    }

    if (m_VehicleDrive) {
        m_VehicleDrive->free();
        m_VehicleDrive = nullptr;
    }
}

void PhysXVehicle::Update(float deltaTime) {
    // Inputs are applied via PxVehicleConcurrentUpdateData and PxVehicleDrive4WRawInputData
    // In this basic version, we prepare inputs for the next separate update call in PhysXBackend
}

void PhysXVehicle::SetInputs(bool accelerate, bool brake, bool steerLeft, bool steerRight) {
    if (!m_VehicleDrive) return;

    m_InputData.setDigitalAccel(accelerate);
    m_InputData.setDigitalBrake(brake);
    float steer = 0.0f;
    if (steerLeft) steer += 1.0f;
    if (steerRight) steer -= 1.0f;
    m_InputData.setAnalogSteer(steer);
    
    // Smooth inputs
    PxVehicleDrive4WSmoothDigitalRawInputsAndSetAnalogInputs(
        PxVehiclePadSmoothingData(), 
        PxVehicleDrive4W().mDriveSimData.getSteerVsForwardSpeedTable(), 
        m_InputData, 
        1.0f/60.0f, // TODO: Use real timestep 
        false, 
        *m_VehicleDrive
    );
}

void PhysXVehicle::SyncTransform() {
    if (!m_VehicleDrive) return;
    
    // Here we would typically update the associated GameObject's transform
    // For now, we just let it be simulated
}

void PhysXVehicle::CreateVehicle4W(const PxVec3& position) {
    PxPhysics* physics = m_Backend->GetPhysics();
    PxMaterial* material = m_Backend->GetDefaultMaterial();
    
    // 1. Create chassis
    PxVehicleChassisData chassisData = CreateChassisData();
    
    // 2. Create wheels, tires, suspension
    PxVehicleWheelData* wheels = CreateWheelData();
    PxVehicleTireData* tires = CreateTireData();
    PxVehicleSuspensionData* suspensions = CreateSuspensionData();
    
    // 3. Create wheels sim data
    PxVehicleWheelsSimData* wheelsSimData = PxVehicleWheelsSimData::allocate(4);
    
    PxVec3 wheelCentreOffsets[4];
    PxVec3 chassisDims(2.5f, 2.0f, 5.0f);
    float wheelWidth = 0.4f;
    float wheelRadius = 0.5f;
    
    ComputeWheelCenterActorOffsets4W(
        chassisDims.z * 0.5f, -chassisDims.z * 0.5f, 
        chassisDims, wheelWidth, wheelRadius, 4, wheelCentreOffsets
    );
    
    SetupWheelsSimulationData(
        20.0f, 0.5f * 20.0f * wheelRadius * wheelRadius, 
        wheelRadius, wheelWidth, 4, wheelCentreOffsets, 
        PxVec3(0.0f, -chassisDims.y * 0.5f + 0.65f, 0.25f), 
        1500.0f, wheelsSimData
    );
    
    for(int i = 0; i < 4; i++) {
        wheelsSimData->setWheelData(i, wheels[i]);
        wheelsSimData->setTireData(i, tires[i]);
        wheelsSimData->setSuspensionData(i, suspensions[i]);
    }

    // 4. Create Drive Sim Data
    PxVehicleDriveSimData4W driveSimData;
    
    // Diff
    PxVehicleDifferential4WData diff;
    diff.mType = PxVehicleDifferential4WData::eDIFF_TYPE_LS_4WD;
    driveSimData.setDiffData(diff);
    
    // Engine
    PxVehicleEngineData engine;
    engine.mPeakTorque = 500.0f;
    engine.mMaxOmega = 600.0f;
    driveSimData.setEngineData(engine);
    
    // Gears
    PxVehicleGearsData gears;
    gears.mSwitchTime = 0.5f;
    driveSimData.setGearsData(gears);
    
    // Clutch
    PxVehicleClutchData clutch;
    clutch.mStrength = 10.0f;
    driveSimData.setClutchData(clutch);
    
    // Auto-box
    PxVehicleAutoBoxData autobox;
    driveSimData.setAutoBoxData(autobox);
    
    // 5. Create Vehicle Actor
    PxRigidDynamic* vehActor = physics->createRigidDynamic(PxTransform(position));
    
    // Add shapes (Simplified box for chassis)
    PxShape* chassisShape = PxRigidActorExt::createExclusiveShape(*vehActor, PxBoxGeometry(chassisDims.x*0.5f, chassisDims.y*0.5f, chassisDims.z*0.5f), *material);
    
    // Add shapes for wheels
    for(int i=0; i<4; i++) {
        PxShape* wheelShape = PxRigidActorExt::createExclusiveShape(*vehActor, PxBoxGeometry(wheelWidth/2, wheelRadius, wheelRadius), *material);
        // Note: Real implementation needs local pose relative to chassis
    }
    
    vehActor->setMass(1500.0f);
    vehActor->setMassSpaceInertiaTensor(PxVec3(2500, 2500, 2500)); // simplify
    vehActor->setCMassLocalPose(PxTransform(PxVec3(0,0,0), PxQuat(PxIdentity)));

    // 6. Instantiate vehicle
    m_VehicleDrive = PxVehicleDrive4W::allocate(4);
    m_VehicleDrive->setup(physics, vehActor, *wheelsSimData, driveSimData, 4 - 4); // 4 wheels, 0 non-driven wheels
    
    if (m_VehicleDrive) {
        // Add to scene
        m_Backend->GetScene()->addActor(*m_VehicleDrive->getRigidDynamicActor());
        
        // Input setup
        m_VehicleDrive->setToRestState();
        m_VehicleDrive->mDriveDynData.forceGearChange(PxVehicleGearsData::eFIRST);
        m_VehicleDrive->mDriveDynData.setUseAutoGears(true);
    }
    
    wheelsSimData->free();
    delete[] wheels;
    delete[] tires;
    delete[] suspensions;
}

PxVehicleChassisData PhysXVehicle::CreateChassisData() {
    PxVehicleChassisData chassis;
    chassis.mMOI = PxVec3(2500, 2500, 2500);
    chassis.mMass = 1500.0f;
    chassis.mCMOffset = PxVec3(0,0,0);
    return chassis;
}

PxVehicleWheelData* PhysXVehicle::CreateWheelData() {
    PxVehicleWheelData* wheels = new PxVehicleWheelData[4];
    for(int i=0; i<4; i++) {
        wheels[i].mMass = 20.0f;
        wheels[i].mMOI = 0.5f * 20.0f * 0.5f * 0.5f;
        wheels[i].mRadius = 0.5f;
        wheels[i].mWidth = 0.4f;
        wheels[i].mMaxBrakeTorque = 1500.0f;
        wheels[i].mMaxHandBrakeTorque = 4000.0f;
        wheels[i].mMaxSteer = PxPi * 0.3333f;
    }
    return wheels;
}

PxVehicleTireData* PhysXVehicle::CreateTireData() {
    PxVehicleTireData* tires = new PxVehicleTireData[4];
    for(int i=0; i<4; i++) {
        tires[i].mType = 0; // Default material type
    }
    return tires;
}

PxVehicleSuspensionData* PhysXVehicle::CreateSuspensionData() {
    PxVehicleSuspensionData* susps = new PxVehicleSuspensionData[4];
    for(int i=0; i<4; i++) {
        susps[i].mMaxCompression = 0.3f;
        susps[i].mMaxDroop = 0.1f;
        susps[i].mSpringStrength = 35000.0f;
        susps[i].mSpringDamperRate = 4500.0f;
    }
    return susps;
}

void PhysXVehicle::ComputeWheelCenterActorOffsets4W(float wheelFrontZ, float wheelRearZ, const PxVec3& chassisDims, float wheelWidth, float wheelRadius, uint32_t numWheels, PxVec3* wheelCentreOffsets) {
    // 4W setup: FL, FR, RL, RR
    wheelCentreOffsets[PxVehicleDrive4WWheelOrder::eFRONT_LEFT] = PxVec3((-chassisDims.x + wheelWidth)*0.5f, -chassisDims.y*0.5f + wheelRadius, wheelFrontZ);
    wheelCentreOffsets[PxVehicleDrive4WWheelOrder::eFRONT_RIGHT] = PxVec3((chassisDims.x - wheelWidth)*0.5f, -chassisDims.y*0.5f + wheelRadius, wheelFrontZ);
    wheelCentreOffsets[PxVehicleDrive4WWheelOrder::eREAR_LEFT] = PxVec3((-chassisDims.x + wheelWidth)*0.5f, -chassisDims.y*0.5f + wheelRadius, wheelRearZ);
    wheelCentreOffsets[PxVehicleDrive4WWheelOrder::eREAR_RIGHT] = PxVec3((chassisDims.x - wheelWidth)*0.5f, -chassisDims.y*0.5f + wheelRadius, wheelRearZ);
}

void PhysXVehicle::SetupWheelsSimulationData(const float wheelMass, const float wheelMOI, const float wheelRadius, const float wheelWidth, const uint32_t numWheels, const PxVec3* wheelCenterActorOffsets, const PxVec3& chassisCMOffset, const float chassisMass, PxVehicleWheelsSimData* wheelsSimData) {
    PxVehicleWheelData wheels[4]; // Default
    for(uint32_t i = 0; i < numWheels; i++) {
        wheelsSimData->setWheelData(i, wheels[i]);
        wheelsSimData->setTireData(i, PxVehicleTireData());
        wheelsSimData->setSuspensionData(i, PxVehicleSuspensionData());
        wheelsSimData->setSuspTravelDirection(i, PxVec3(0,-1,0));
        wheelsSimData->setWheelCentreOffset(i, wheelCenterActorOffsets[i] - chassisCMOffset);
        wheelsSimData->setSuspForceAppPointOffset(i, wheelCenterActorOffsets[i] - chassisCMOffset);
        wheelsSimData->setTireForceAppPointOffset(i, wheelCenterActorOffsets[i] - chassisCMOffset);
    }
}
