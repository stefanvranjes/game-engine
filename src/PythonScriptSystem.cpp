#include "PythonScriptSystem.h"
#include "GameObject.h"
#include "Transform.h"
#include "Math/Vec3.h"

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
}
