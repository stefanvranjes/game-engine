#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

class Logger {
public:
    static void Init();
    static void Shutdown();

private:
    static long UnhandledExceptionFilter(_EXCEPTION_POINTERS* ExceptionInfo);
    static void SignalHandler(int signal);

    static std::ofstream m_LogFile;
    static std::streambuf* m_CoutBuffer;
    static std::streambuf* m_CerrBuffer;
};
