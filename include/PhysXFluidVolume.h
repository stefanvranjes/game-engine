#pragma once

#include "Math/Vec3.h"

class PhysXRigidBody;

/**
 * @brief Represents a fluid volume in PhysX world.
 * Attached to a trigger RigidBody to define fluid properties.
 */
class PhysXFluidVolume {
public:
    PhysXFluidVolume(PhysXRigidBody* triggerBody);
    ~PhysXFluidVolume();

    // Fluid Properties
    float Density = 1000.0f;        // kg/m^3 (Water = 1000)
    float LinearDrag = 0.5f;        // Drag coefficient for linear motion
    float AngularDrag = 0.5f;       // Drag coefficient for angular motion
    Vec3 FlowDirection = Vec3(0,0,0); // Flow velocity vector
    float FlowSpeed = 0.0f;         // Speed of flow if direction is normalized
    
    // Wave parameters (optional connection to visual water)
    bool UseWaveHeight = false;
    float BaseHeight = 0.0f;        // Y-level of fluid surface if flat

    // Accessor
    PhysXRigidBody* GetTriggerBody() const { return m_TriggerBody; }

    // Helper to calculate submerged volume fraction and center of buoyancy
    // Returns fraction [0,1]
    float CalculateSubmergedVolume(PhysXRigidBody* body, Vec3& outBuoyancyCenter, Vec3& outFlowVelocity);

private:
    PhysXRigidBody* m_TriggerBody;
};
