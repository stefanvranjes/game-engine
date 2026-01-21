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

    bool ExecuteString(const std::string& source) override {
        try {
            py::exec(source);
            return true;
        } catch (const py::error_already_set& e) {
            std::cerr << "Python Error: " << e.what() << std::endl;
            return false;
        }
    }

    // Language metadata
    ScriptLanguage GetLanguage() const override { return ScriptLanguage::Python; }
    ScriptExecutionMode GetExecutionMode() const override { return ScriptExecutionMode::Interpreted; }
    std::string GetLanguageName() const override { return "Python"; }
    std::string GetFileExtension() const override { return ".py"; }

    void RegisterTypes(); // Implemented in cpp to avoid header clutter

private:
    PythonScriptSystem() = default;
    ~PythonScriptSystem() = default;
};
