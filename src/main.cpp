#include "Application.h"
#include "Logger.h"

int main() {
    Logger::Init();
    
    Application app;
    if (app.Init()) {
        app.Run();
    }
    
    Logger::Shutdown();
    return 0;
}

