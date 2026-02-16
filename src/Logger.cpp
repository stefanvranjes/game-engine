#include "Logger.h"
#include <windows.h>
#include <csignal>
#include <ctime>
#include <iomanip>

std::ofstream Logger::m_LogFile;
std::streambuf* Logger::m_CoutBuffer = nullptr;
std::streambuf* Logger::m_CerrBuffer = nullptr;

void Logger::Init() {
    m_LogFile.open("engine.log", std::ios::out | std::ios::app);
    if (m_LogFile.is_open()) {
        // Redirection
        m_CoutBuffer = std::cout.rdbuf();
        m_CerrBuffer = std::cerr.rdbuf();
        std::cout.rdbuf(m_LogFile.rdbuf());
        std::cerr.rdbuf(m_LogFile.rdbuf());

        auto now = std::time(nullptr);
        std::cout << "\n--- Engine Session Started: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << " ---\n" << std::endl;
    }

    // Set crash handler (Windows)
    SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)UnhandledExceptionFilter);

    // Set signal handlers
    std::signal(SIGSEGV, SignalHandler);
    std::signal(SIGFPE, SignalHandler);
    std::signal(SIGILL, SignalHandler);
    std::signal(SIGABRT, SignalHandler);
}

void Logger::Shutdown() {
    if (m_LogFile.is_open()) {
        auto now = std::time(nullptr);
        std::cout << "\n--- Engine Session Ended: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << " ---\n" << std::endl;
        
        // Restore buffers
        std::cout.rdbuf(m_CoutBuffer);
        std::cerr.rdbuf(m_CerrBuffer);
        m_LogFile.close();
    }
}

void Logger::Log(LogLevel level, const std::string& message, const char* file, const char* function, int line) {
    if (m_LogFile.is_open()) {
        auto now = std::time(nullptr);
        auto tm = std::localtime(&now);
        
        std::string levelStr;
        switch (level) {
            case LogLevel::INFO: levelStr = "INFO"; break;
            case LogLevel::WARN: levelStr = "WARN"; break;
            case LogLevel::ERR: levelStr = "ERROR"; break;
        }

        // Extract filename from path
        std::string filename = file;
        size_t lastSlash = filename.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            filename = filename.substr(lastSlash + 1);
        }

        std::ostream& out = (level == LogLevel::ERR) ? std::cerr : std::cout;
        
        // We need to write to the original buffers if we want it on console, 
        // BUT Logger::Init redirects cout/cerr to the file.
        // So writing to cout/cerr here will go to the file.
        // To avoid infinite recursion if we were using std::cout/cerr to log, we are using the new Log function manually.
        // However, the user request says to "Implement debugging symbols and function names".
        // Use a formatted string.
        
        out << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] "
            << "[" << levelStr << "] "
            << "[" << filename << ":" << line << " (" << function << ")] "
            << message << std::endl;
    }
}

long Logger::UnhandledExceptionFilter(_EXCEPTION_POINTERS* ExceptionInfo) {
    std::cerr << "CRITICAL ERROR: Unhandled Exception detected!" << std::endl;
    std::cerr << "Exception Code: 0x" << std::hex << ExceptionInfo->ExceptionRecord->ExceptionCode << std::dec << std::endl;
    std::cerr << "Address: 0x" << std::hex << ExceptionInfo->ExceptionRecord->ExceptionAddress << std::dec << std::endl;
    
    Logger::Shutdown(); // Ensure log is flushed
    return EXCEPTION_EXECUTE_HANDLER;
}

void Logger::SignalHandler(int signal) {
    std::cerr << "CRITICAL ERROR: Received signal " << signal << std::endl;
    Logger::Shutdown();
    std::exit(signal);
}
