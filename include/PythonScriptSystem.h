#pragma once

#include "IScriptSystem.h"
#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

class PythonScriptSystem : public IScriptSystem {
public:
    static PythonScriptSystem& GetInstance() {
        static PythonScriptSystem instance;
        return instance;
    }

    void Init() override {
        // Initialize Python interpreter
        try {
            py::initialize_interpreter();
            RegisterTypes();
            std::cout << "PythonScriptSystem Initialized" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize Python: " << e.what() << std::endl;
        }
    }

    void Shutdown() override {
        py::finalize_interpreter();
    }

    void Update(float deltaTime) override {
        // Global updates if needed
    }

    bool RunScript(const std::string& filepath) override {
        try {
            py::eval_file(filepath);
            return true;
        } catch (const py::error_already_set& e) {
            std::cerr << "Python Error: " << e.what() << std::endl;
            return false;
        }
    }

    void RegisterTypes(); // Implemented in cpp to avoid header clutter

private:
    PythonScriptSystem() = default;
    ~PythonScriptSystem() = default;
};
