@echo off
REM Build script with sanitizer and test options for Game Engine

setlocal enabledelayedexpansion

if "%1%"=="" (
    echo Usage: build_with_options.bat [option]
    echo.
    echo Options:
    echo   release      - Build Release configuration
    echo   debug        - Build Debug configuration
    echo   test         - Build with unit tests
    echo   asan         - Build with AddressSanitizer (Linux only)
    echo   ubsan        - Build with UndefinedBehaviorSanitizer (Linux only)
    echo   tidy         - Run clang-tidy static analysis
    echo   clean        - Clean build directory
    echo   all          - Full rebuild with tests
    echo.
    echo Examples:
    echo   build_with_options.bat release
    echo   build_with_options.bat test
    echo   build_with_options.bat clean all
    exit /b 0
)

set "BUILD_DIR=build"
set "CONFIG=Release"
set "BUILD_TESTS=OFF"
set "ENABLE_TIDY=OFF"

:parse_args
if "%1%"=="" goto build_start

if /i "%1%"=="release" (
    set "CONFIG=Release"
) else if /i "%1%"=="debug" (
    set "CONFIG=Debug"
) else if /i "%1%"=="test" (
    set "BUILD_TESTS=ON"
) else if /i "%1%"=="tidy" (
    set "ENABLE_TIDY=ON"
) else if /i "%1%"=="clean" (
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    mkdir "%BUILD_DIR%"
) else if /i "%1%"=="asan" (
    echo AddressSanitizer is only available on Linux with Clang
    exit /b 1
) else if /i "%1%"=="ubsan" (
    echo UndefinedBehaviorSanitizer is only available on Linux with Clang
    exit /b 1
) else if /i "%1%"=="all" (
    set "CONFIG=Release"
    set "BUILD_TESTS=ON"
    set "ENABLE_TIDY=ON"
)

shift
goto parse_args

:build_start
echo.
echo =====================================
echo Game Engine Build Configuration
echo =====================================
echo Configuration: %CONFIG%
echo Build Tests: %BUILD_TESTS%
echo Enable Clang-Tidy: %ENABLE_TIDY%
echo.

if not exist "%BUILD_DIR%" (
    echo Creating build directory...
    mkdir "%BUILD_DIR%"
)

cd %BUILD_DIR%

echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" ^
    -DCMAKE_BUILD_TYPE=%CONFIG% ^
    -DBUILD_TESTS=%BUILD_TESTS% ^
    -DENABLE_CLANG_TIDY=%ENABLE_TIDY%

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

echo.
echo Building...
cmake --build . --config %CONFIG% -j4

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

if "%BUILD_TESTS%"=="ON" (
    echo.
    echo Running tests...
    ctest --output-on-failure --build-config %CONFIG%
    
    if errorlevel 1 (
        echo Some tests failed!
        exit /b 1
    )
)

echo.
echo =====================================
echo Build completed successfully!
echo =====================================
echo.

if "%BUILD_TESTS%"=="ON" (
    echo Test executable: %BUILD_DIR%\bin\tests.exe
    echo Run all tests: %BUILD_DIR%\bin\tests.exe
    echo Run specific test: %BUILD_DIR%\bin\tests.exe --gtest_filter=TestName.*
    echo.
)

echo Game executable: %BUILD_DIR%\bin\GameEngine.exe
cd ..

exit /b 0
