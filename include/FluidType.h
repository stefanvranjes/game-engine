#pragma once

#include "Math/Vec3.h"
#include <string>

/**
 * @brief Fluid type definition for multi-fluid simulation
 * 
 * Defines physical properties and appearance for different fluid types
 */
struct FluidType {
    std::string name;           // Fluid type name
    
    // Physical properties
    float restDensity;          // Rest density (kg/m^3)
    float viscosity;            // Viscosity coefficient
    float surfaceTension;       // Surface tension coefficient
    float particleMass;         // Mass per particle
    
    // Appearance
    Vec3 color;                 // Base color
    float transparency;         // 0 = opaque, 1 = fully transparent
    float refractiveIndex;      // Index of refraction (1.0 = air, 1.33 = water)
    
    // Interaction parameters
    float adhesion;             // Adhesion to boundaries (0-1)
    float cohesion;             // Cohesion with same fluid type (0-1)
    
    // Default constructor (water)
    FluidType()
        : name("Water")
        , restDensity(1000.0f)      // 1000 kg/m^3
        , viscosity(0.01f)
        , surfaceTension(0.0728f)
        , particleMass(0.02f)
        , color(0.2f, 0.5f, 1.0f)   // Blue
        , transparency(0.7f)
        , refractiveIndex(1.33f)
        , adhesion(0.5f)
        , cohesion(0.8f)
    {}
    
    FluidType(const std::string& fluidName)
        : name(fluidName)
        , restDensity(1000.0f)
        , viscosity(0.01f)
        , surfaceTension(0.0728f)
        , particleMass(0.02f)
        , color(0.2f, 0.5f, 1.0f)
        , transparency(0.7f)
        , refractiveIndex(1.33f)
        , adhesion(0.5f)
        , cohesion(0.8f)
    {}
    
    // Preset fluid types
    static FluidType Water() {
        FluidType fluid("Water");
        fluid.restDensity = 1000.0f;
        fluid.viscosity = 0.01f;
        fluid.surfaceTension = 0.0728f;
        fluid.color = Vec3(0.2f, 0.5f, 1.0f);
        fluid.transparency = 0.7f;
        fluid.refractiveIndex = 1.33f;
        return fluid;
    }
    
    static FluidType Oil() {
        FluidType fluid("Oil");
        fluid.restDensity = 900.0f;
        fluid.viscosity = 0.05f;
        fluid.surfaceTension = 0.032f;
        fluid.color = Vec3(0.8f, 0.6f, 0.2f);
        fluid.transparency = 0.5f;
        fluid.refractiveIndex = 1.47f;
        return fluid;
    }
    
    static FluidType Honey() {
        FluidType fluid("Honey");
        fluid.restDensity = 1420.0f;
        fluid.viscosity = 2.0f;
        fluid.surfaceTension = 0.065f;
        fluid.color = Vec3(1.0f, 0.7f, 0.1f);
        fluid.transparency = 0.4f;
        fluid.refractiveIndex = 1.49f;
        return fluid;
    }
    
    static FluidType Blood() {
        FluidType fluid("Blood");
        fluid.restDensity = 1060.0f;
        fluid.viscosity = 0.04f;
        fluid.surfaceTension = 0.058f;
        fluid.color = Vec3(0.8f, 0.1f, 0.1f);
        fluid.transparency = 0.3f;
        fluid.refractiveIndex = 1.35f;
        return fluid;
    }
    
    static FluidType Lava() {
        FluidType fluid("Lava");
        fluid.restDensity = 3100.0f;
        fluid.viscosity = 100.0f;
        fluid.surfaceTension = 0.4f;
        fluid.color = Vec3(1.0f, 0.3f, 0.0f);
        fluid.transparency = 0.1f;
        fluid.refractiveIndex = 1.6f;
        return fluid;
    }
};
