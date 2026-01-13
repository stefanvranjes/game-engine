#pragma once

// GLAD wrapper for the engine's built-in GLExtensions system
#include "../GLExtensions.h"

// Define GLAD specific flags that the code expects
#ifndef GLAD_GL_KHR_debug
#define GLAD_GL_KHR_debug GLAD_GL_KHR_debug
#endif

// Some files might expect GLAD to define common OpenGL constants if they don't include GL/gl.h
// GLExtensions.h already includes GL/GL.h via windows.h or directly.
