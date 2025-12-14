#!/usr/bin/env pwsh
# PowerShell build script with sanitizer and test options

param(
    [Parameter(Position = 0)]
    [ValidateSet('release', 'debug', 'test', 'asan', 'ubsan', 'tsan', 'msan', 'tidy', 'clean', 'all', 'help')]
    [string]$Option = 'help',
    
    [switch]$Parallel = $true,
    [int]$Jobs = 4,
    [switch]$Verbose = $false
)

$BuildDir = 'build'
$Config = 'Release'
$BuildTests = $false
$EnableTidy = $false
$UseAsan = $false
$UseUbsan = $false
$UseTsan = $false
$UseMsan = $false

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host $Message -ForegroundColor Green
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host ""
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Header "Game Engine Build Tool - Help"
    Write-Host "Usage: .\build.ps1 [option]`n"
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  release    - Build Release configuration"
    Write-Host "  debug      - Build Debug configuration"
    Write-Host "  test       - Build with unit tests"
    Write-Host "  tidy       - Run clang-tidy static analysis"
    Write-Host "  clean      - Clean build directory"
    Write-Host "  all        - Full rebuild with tests and analysis"
    Write-Host "  help       - Show this help message`n"
    Write-Host "Sanitizers (Linux with Clang only):" -ForegroundColor Yellow
    Write-Host "  asan       - AddressSanitizer (memory errors)"
    Write-Host "  ubsan      - UndefinedBehaviorSanitizer (UB detection)"
    Write-Host "  tsan       - ThreadSanitizer (data races)"
    Write-Host "  msan       - MemorySanitizer (uninitialized memory)`n"
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1 release"
    Write-Host "  .\build.ps1 test"
    Write-Host "  .\build.ps1 all"
    Write-Host "  .\build.ps1 clean release`n"
}

function Parse-Options {
    param([string[]]$Options)
    
    foreach ($opt in $Options) {
        switch ($opt.ToLower()) {
            'release' { $Script:Config = 'Release' }
            'debug' { $Script:Config = 'Debug' }
            'test' { $Script:BuildTests = $true }
            'tidy' { $Script:EnableTidy = $true }
            'asan' { $Script:UseAsan = $true }
            'ubsan' { $Script:UseUbsan = $true }
            'tsan' { $Script:UseTsan = $true }
            'msan' { $Script:UseMsan = $true }
            'all' {
                $Script:Config = 'Release'
                $Script:BuildTests = $true
                $Script:EnableTidy = $true
            }
            'clean' {
                Write-Section "Cleaning build directory..."
                if (Test-Path $BuildDir) {
                    Remove-Item -Path $BuildDir -Recurse -Force
                }
                New-Item -ItemType Directory -Path $BuildDir -Force > $null
            }
            'help' { Show-Help; exit 0 }
            default { Write-Host "Unknown option: $_" -ForegroundColor Red }
        }
    }
}

function Check-Prerequisites {
    Write-Section "Checking prerequisites..."
    
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmake) {
        Write-Host "ERROR: CMake not found. Please install CMake." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ CMake found: $($cmake.Source)" -ForegroundColor Green
    
    if ($EnableTidy) {
        $clangtidy = Get-Command clang-tidy -ErrorAction SilentlyContinue
        if (-not $clangtidy) {
            Write-Host "WARNING: clang-tidy not found. Skipping static analysis." -ForegroundColor Yellow
            $Script:EnableTidy = $false
        } else {
            Write-Host "✓ Clang-tidy found: $($clangtidy.Source)" -ForegroundColor Green
        }
    }
}

function Configure-CMake {
    Write-Section "Configuring CMake..."
    
    $cmakeArgs = @(
        '..',
        '-G', '"Visual Studio 17 2022"',
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DBUILD_TESTS=$(if ($BuildTests) { 'ON' } else { 'OFF' })",
        "-DENABLE_CLANG_TIDY=$(if ($EnableTidy) { 'ON' } else { 'OFF' })"
    )
    
    if ($UseAsan) { $cmakeArgs += '-DUSE_ASAN=ON' }
    if ($UseUbsan) { $cmakeArgs += '-DUSE_UBSAN=ON' }
    if ($UseTsan) { $cmakeArgs += '-DUSE_TSAN=ON' }
    if ($UseMsan) { $cmakeArgs += '-DUSE_MSAN=ON' }
    
    Write-Host "CMake arguments: $cmakeArgs`n" -ForegroundColor Gray
    
    Push-Location $BuildDir
    
    $output = & cmake @cmakeArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        Write-Host $output
        Pop-Location
        exit 1
    }
    
    if ($Verbose) {
        Write-Host $output -ForegroundColor Gray
    }
    
    Pop-Location
}

function Build-Project {
    Write-Section "Building project..."
    
    Write-Host "Configuration: $Config"
    Write-Host "Build Tests: $BuildTests"
    Write-Host "Enable Clang-Tidy: $EnableTidy"
    Write-Host ""
    
    Push-Location $BuildDir
    
    $buildArgs = @(
        '--build', '.',
        '--config', $Config,
        '-j', $Jobs.ToString()
    )
    
    if ($Verbose) {
        $buildArgs += '--verbose'
    }
    
    $output = & cmake @buildArgs 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        Write-Host $output
        Pop-Location
        exit 1
    }
    
    Write-Host "Build completed successfully!" -ForegroundColor Green
    
    if ($Verbose) {
        Write-Host $output -ForegroundColor Gray
    }
    
    Pop-Location
}

function Run-Tests {
    Write-Section "Running tests..."
    
    Push-Location $BuildDir
    
    $ctestArgs = @(
        '--output-on-failure',
        "--build-config", $Config
    )
    
    if ($Verbose) {
        $ctestArgs += '--verbose'
    }
    
    & ctest @ctestArgs
    
    $testResult = $LASTEXITCODE
    
    Pop-Location
    
    if ($testResult -ne 0) {
        Write-Host "Some tests failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "All tests passed!" -ForegroundColor Green
}

function Show-Summary {
    Write-Header "Build Summary"
    
    $exePath = Join-Path $BuildDir "bin" "GameEngine.exe"
    
    Write-Host "Build Configuration:" -ForegroundColor Yellow
    Write-Host "  Configuration: $Config"
    Write-Host "  Build Directory: $BuildDir"
    Write-Host "  Game Executable: $exePath`n"
    
    if ($BuildTests) {
        $testExePath = Join-Path $BuildDir "bin" "tests.exe"
        Write-Host "Test Information:" -ForegroundColor Yellow
        Write-Host "  Test Executable: $testExePath"
        Write-Host "  Run all tests: $testExePath"
        Write-Host "  Run specific test: $testExePath --gtest_filter=TestName.*"
        Write-Host "  List all tests: $testExePath --gtest_list_tests`n"
    }
    
    if ($Verbose) {
        Write-Host "Build completed at: $(Get-Date)" -ForegroundColor Gray
    }
}

# Main execution
Write-Header "Game Engine Build Tool"

if ($Option -eq 'help') {
    Show-Help
    exit 0
}

Parse-Options -Options $Option
Check-Prerequisites

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force > $null
}

Configure-CMake
Build-Project

if ($BuildTests) {
    Run-Tests
}

Show-Summary

Write-Host "✓ Build process completed successfully!" -ForegroundColor Green
