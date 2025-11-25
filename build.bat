@echo off
set CMAKE_PATH="C:\Program Files\CMake\bin\cmake.exe"
if not exist build mkdir build
cd build
%CMAKE_PATH% ..
%CMAKE_PATH% --build . --config Debug
cd ..
