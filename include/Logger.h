#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

class Logger {
public:
    enum class LogLevel {
        INFO,
        WARN,
        ERR // Renamed to ERR to avoid conflict with windows.h ERROR
    };

    static void Init();
    static void Shutdown();

    static void Log(LogLevel level, const std::string& message, const char* file, const char* function, int line);

private:
    static long UnhandledExceptionFilter(_EXCEPTION_POINTERS* ExceptionInfo);
    static void SignalHandler(int signal);

    static std::ofstream m_LogFile;
    static std::streambuf* m_CoutBuffer;
    static std::streambuf* m_CerrBuffer;
};

#define LOG_INFO(message) Logger::Log(Logger::LogLevel::INFO, message, __FILE__, __FUNCTION__, __LINE__)
#define LOG_WARN(message) Logger::Log(Logger::LogLevel::WARN, message, __FILE__, __FUNCTION__, __LINE__)
#define LOG_ERROR(message) Logger::Log(Logger::LogLevel::ERR, message, __FILE__, __FUNCTION__, __LINE__)
