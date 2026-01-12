#include "PythonScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include "Transform.h"
#include "Math/Vec3.h"
#include "PhysXRagdoll.h"
#include "PhysXArticulationLink.h" // Already included in Ragdoll header but good for clarity

void PythonScriptSystem::RegisterTypes() {
    py::module_ m = py::module_::import("__main__");

    // Bind Vec3
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
        
    // Bind Transform
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def_property("position", &Transform::GetPosition, &Transform::SetPosition)
        .def_property("rotation", &Transform::GetRotation, &Transform::SetRotation) // Wrap Quat?
        .def_property("scale", &Transform::GetScale, &Transform::SetScale);

    // Bind GameObject (Simplified)
    // Note: Shared_ptr handling in pybind11 requires care.
    py::class_<GameObject, std::shared_ptr<GameObject>>(m, "GameObject")
        .def(py::init<std::string>())
        .def("get_name", &GameObject::GetName)
        .def("get_transform", &GameObject::GetTransform, py::return_value_policy::reference);

    // --- PhysX ML Integration ---
    
    // Bind RagdollState
    py::enum_<RagdollState>(m, "RagdollState")
        .value("Kinematic", RagdollState::Kinematic)
        .value("Dynamic", RagdollState::Dynamic)
        .value("Active", RagdollState::Active)
        .export_values();

    // Bind RagdollBoneConfig
    py::class_<RagdollBoneConfig>(m, "RagdollBoneConfig")
        .def(py::init<>())
        .def_readwrite("bone_name", &RagdollBoneConfig::boneName)
        .def_readwrite("mass", &RagdollBoneConfig::mass)
        .def_readwrite("drive_stiffness", &RagdollBoneConfig::driveStiffness)
        .def_readwrite("drive_damping", &RagdollBoneConfig::driveDamping);
        // Add more fields as needed

    // Bind PhysXArticulationLink
    // Note: We don't construct these in Python, only access them
    py::class_<PhysXArticulationLink>(m, "PhysXArticulationLink")
        .def("get_position", [](PhysXArticulationLink* self) { 
            Vec3 pos; Quat rot; self->SyncTransformFromPhysics(pos, rot); return pos; 
        })
        .def("get_rotation", [](PhysXArticulationLink* self) { 
            Vec3 pos; Quat rot; self->SyncTransformFromPhysics(pos, rot); return rot; 
        })
        .def("get_linear_velocity", &PhysXArticulationLink::GetLinearVelocity)
        .def("get_angular_velocity", &PhysXArticulationLink::GetAngularVelocity)
        .def("add_torque", &PhysXArticulationLink::ApplyTorque)
        .def("add_force", &PhysXArticulationLink::ApplyForce);

    // Bind PhysXRagdoll
    py::class_<PhysXRagdoll>(m, "PhysXRagdoll")
        .def("set_state", &PhysXRagdoll::SetState)
        .def("get_state", &PhysXRagdoll::GetState)
        .def("get_link", &PhysXRagdoll::GetLink, py::return_value_policy::reference)
        // .def("add_bone", &PhysXRagdoll::AddBone) // Requires wrapping IPhysicsShape? For now assume setup in C++
        .def("update", &PhysXRagdoll::Update);

}
