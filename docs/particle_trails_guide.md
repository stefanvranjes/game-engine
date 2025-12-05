# Particle Trails System - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Trail Configuration](#trail-configuration)
4. [Trail Color Modes](#trail-color-modes)
5. [Usage Examples](#usage-examples)
6. [Preset Trail Effects](#preset-trail-effects)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Introduction

The particle trails system creates visual motion effects by rendering ribbons behind moving particles. Trails are perfect for rockets, magic spells, energy beams, and motion blur effects.

### Features
- ✅ **Ribbon geometry**: Camera-facing trails with dynamic width
- ✅ **Distance-based sampling**: Prevents gaps and clustering
- ✅ **Age-based fading**: Automatic alpha fade over time
- ✅ **Multiple color modes**: Particle color, fade to transparent, gradient, custom
- ✅ **Optional textures**: Support for textured trails
- ✅ **Configurable length**: 10-100+ points per trail

## Quick Start

### Basic Trail Setup

```cpp
// Create a particle emitter
auto emitter = ParticleEmitter::CreateSparks(Vec3(0, 5, 0));

// Enable trails
emitter->SetEnableTrails(true);
emitter->SetTrailLength(30);           // Max 30 points per trail
emitter->SetTrailLifetime(0.5f);       // Points live for 0.5 seconds
emitter->SetTrailWidth(0.3f);          // Trail width multiplier
emitter->SetTrailMinDistance(0.1f);    // Min distance between points

// Add to particle system
particleSystem->AddEmitter(emitter);
```

## Trail Configuration

### SetEnableTrails(bool enable)
Enable or disable trails for this emitter.

```cpp
emitter->SetEnableTrails(true);  // Enable trails
emitter->SetEnableTrails(false); // Disable trails
```

### SetTrailLength(int maxPoints)
Maximum number of points per trail (circular buffer).

- **Default**: 50
- **Range**: 5-200
- **Effect**: Longer trails = more points = smoother but more expensive

```cpp
emitter->SetTrailLength(20);   // Short trail
emitter->SetTrailLength(50);   // Medium trail (default)
emitter->SetTrailLength(100);  // Long trail
```

### SetTrailLifetime(float lifetime)
How long each trail point lives before fading out.

- **Default**: 1.0 seconds
- **Range**: 0.1-5.0 seconds
- **Effect**: Longer lifetime = longer visible trails

```cpp
emitter->SetTrailLifetime(0.2f);  // Quick fade
emitter->SetTrailLifetime(1.0f);  // Standard fade
emitter->SetTrailLifetime(2.0f);  // Slow fade
```

### SetTrailWidth(float width)
Trail width multiplier (multiplied by particle size).

- **Default**: 0.5
- **Range**: 0.1-2.0
- **Effect**: Larger values = wider trails

```cpp
emitter->SetTrailWidth(0.2f);  // Thin trail
emitter->SetTrailWidth(0.5f);  // Medium trail
emitter->SetTrailWidth(1.0f);  // Wide trail
```

### SetTrailMinDistance(float distance)
Minimum distance before adding a new trail point.

- **Default**: 0.1
- **Range**: 0.05-0.5
- **Effect**: Smaller values = more points = smoother trails

```cpp
emitter->SetTrailMinDistance(0.05f);  // Very smooth (more points)
emitter->SetTrailMinDistance(0.1f);   // Balanced
emitter->SetTrailMinDistance(0.3f);   // Coarse (fewer points)
```

### SetTrailTexture(Texture texture)
Optional texture for trails.

```cpp
auto trailTexture = std::make_shared<Texture>();
trailTexture->LoadFromFile("textures/trail_gradient.png");
emitter->SetTrailTexture(trailTexture);
```

### SetTrailColorMode(TrailColorMode mode)
How trail colors are determined.

```cpp
emitter->SetTrailColorMode(TrailColorMode::ParticleColor);
emitter->SetTrailColorMode(TrailColorMode::FadeToTransparent);
emitter->SetTrailColorMode(TrailColorMode::GradientToEnd);
emitter->SetTrailColorMode(TrailColorMode::Custom);
```

### SetTrailColor(Vec4 color)
Custom trail color (used with `TrailColorMode::Custom`).

```cpp
emitter->SetTrailColor(Vec4(1.0f, 0.5f, 0.0f, 1.0f)); // Orange
```

## Trail Color Modes

### ParticleColor
Trail uses the particle's current color at each point.

**Best for**: Effects where trail should match particle exactly

```cpp
emitter->SetTrailColorMode(TrailColorMode::ParticleColor);
```

### FadeToTransparent (Default)
Trail fades from particle color to transparent based on point age.

**Best for**: Most effects - natural looking fade

```cpp
emitter->SetTrailColorMode(TrailColorMode::FadeToTransparent);
// Newer points: full alpha
// Older points: faded alpha
```

### GradientToEnd
Trail creates a gradient from start color to end color.

**Best for**: Rainbow trails, color transitions

```cpp
emitter->SetTrailColorMode(TrailColorMode::GradientToEnd);
emitter->SetColorRange(
    Vec4(1.0f, 0.0f, 0.0f, 1.0f),  // Red start
    Vec4(0.0f, 0.0f, 1.0f, 1.0f)   // Blue end
);
```

### Custom
Trail uses a fixed custom color.

**Best for**: Uniform colored trails

```cpp
emitter->SetTrailColorMode(TrailColorMode::Custom);
emitter->SetTrailColor(Vec4(0.0f, 1.0f, 0.0f, 0.8f)); // Green
```

## Usage Examples

### Example 1: Rocket Trail

```cpp
auto rocket = std::make_shared<ParticleEmitter>(Vec3(0, 5, 0), 300);
rocket->SetSpawnRate(100.0f);
rocket->SetParticleLifetime(1.5f);
rocket->SetVelocityRange(Vec3(-0.5f, -2.0f, -0.5f), Vec3(0.5f, -1.0f, 0.5f));
rocket->SetColorRange(Vec4(1.0f, 0.8f, 0.2f, 1.0f), Vec4(1.0f, 0.3f, 0.0f, 0.0f));
rocket->SetSizeRange(0.4f, 0.2f);
rocket->SetBlendMode(BlendMode::Additive);

// Long, bright trails
rocket->SetEnableTrails(true);
rocket->SetTrailLength(60);
rocket->SetTrailLifetime(1.0f);
rocket->SetTrailWidth(0.6f);
rocket->SetTrailMinDistance(0.1f);
rocket->SetTrailColorMode(TrailColorMode::FadeToTransparent);

particleSystem->AddEmitter(rocket);
```

### Example 2: Magic Spell

```cpp
auto magic = ParticleEmitter::CreateMagic(Vec3(0, 2, 0));

// Colorful, swirling trails
magic->SetEnableTrails(true);
magic->SetTrailLength(40);
magic->SetTrailLifetime(0.8f);
magic->SetTrailWidth(0.4f);
magic->SetTrailColorMode(TrailColorMode::FadeToTransparent);

particleSystem->AddEmitter(magic);
```

### Example 3: Energy Beam

```cpp
auto beam = std::make_shared<ParticleEmitter>(Vec3(-5, 3, 0), 500);
beam->SetSpawnRate(200.0f);
beam->SetParticleLifetime(0.5f);
beam->SetVelocityRange(Vec3(10.0f, -0.2f, -0.2f), Vec3(12.0f, 0.2f, 0.2f));
beam->SetColorRange(Vec4(0.2f, 0.8f, 1.0f, 1.0f), Vec4(0.5f, 0.5f, 1.0f, 0.0f));
beam->SetSizeRange(0.3f, 0.2f);
beam->SetGravity(Vec3(0, 0, 0)); // No gravity
beam->SetBlendMode(BlendMode::Additive);

// Continuous beam effect
beam->SetEnableTrails(true);
beam->SetTrailLength(80);
beam->SetTrailLifetime(0.3f);
beam->SetTrailWidth(0.5f);
beam->SetTrailMinDistance(0.05f); // Very smooth
beam->SetTrailColorMode(TrailColorMode::FadeToTransparent);

particleSystem->AddEmitter(beam);
```

### Example 4: Motion Blur

```cpp
auto blur = std::make_shared<ParticleEmitter>(Vec3(0, 5, 0), 100);
blur->SetSpawnRate(30.0f);
blur->SetParticleLifetime(2.0f);
blur->SetVelocityRange(Vec3(-5.0f, -1.0f, -1.0f), Vec3(5.0f, 1.0f, 1.0f));
blur->SetColorRange(Vec4(1.0f, 1.0f, 1.0f, 0.8f), Vec4(1.0f, 1.0f, 1.0f, 0.0f));
blur->SetSizeRange(0.5f, 0.3f);
blur->SetBlendMode(BlendMode::Alpha);

// Short trails for motion blur
blur->SetEnableTrails(true);
blur->SetTrailLength(15);
blur->SetTrailLifetime(0.2f);
blur->SetTrailWidth(0.4f);
blur->SetTrailMinDistance(0.15f);
blur->SetTrailColorMode(TrailColorMode::FadeToTransparent);

particleSystem->AddEmitter(blur);
```

### Example 5: Textured Trail

```cpp
auto textured = std::make_shared<ParticleEmitter>(Vec3(0, 5, 0), 200);
textured->SetSpawnRate(50.0f);
textured->SetParticleLifetime(2.0f);
textured->SetVelocityRange(Vec3(-2.0f, 0.0f, -2.0f), Vec3(2.0f, 3.0f, 2.0f));
textured->SetColorRange(Vec4(1.0f, 1.0f, 1.0f, 1.0f), Vec4(1.0f, 1.0f, 1.0f, 0.0f));
textured->SetBlendMode(BlendMode::Additive);

// Load trail texture
auto trailTex = std::make_shared<Texture>();
trailTex->LoadFromFile("textures/trail_gradient.png");

textured->SetEnableTrails(true);
textured->SetTrailLength(30);
textured->SetTrailLifetime(0.6f);
textured->SetTrailWidth(0.5f);
textured->SetTrailTexture(trailTex);
textured->SetTrailColorMode(TrailColorMode::ParticleColor);

particleSystem->AddEmitter(textured);
```

## Preset Trail Effects

All preset emitters now include optimized trail configurations:

### CreateFire()
- **Trail Length**: 20 points
- **Lifetime**: 0.3 seconds
- **Width**: 0.3
- **Effect**: Short, flickering fire trails

### CreateSmoke()
- **Trail Length**: 15 points
- **Lifetime**: 0.5 seconds
- **Width**: 0.8
- **Effect**: Subtle, wispy smoke trails

### CreateSparks()
- **Trail Length**: 10 points
- **Lifetime**: 0.2 seconds
- **Width**: 0.2
- **Effect**: Short, bright spark streaks

### CreateMagic()
- **Trail Length**: 40 points
- **Lifetime**: 0.8 seconds
- **Width**: 0.4
- **Effect**: Long, colorful magic trails

## Performance Optimization

### Trail Length vs Performance

| Trail Length | Performance | Use Case |
|--------------|-------------|----------|
| 10-20 | ✅ Excellent | Sparks, quick effects |
| 30-50 | ✅ Good | Standard trails |
| 60-80 | ⚠️ Moderate | Long trails, beams |
| 100+ | ❌ Expensive | Special effects only |

### Optimization Tips

1. **Adjust MinDistance**: Larger values = fewer points = better performance
   ```cpp
   emitter->SetTrailMinDistance(0.2f); // Fewer points
   ```

2. **Reduce Trail Lifetime**: Shorter lifetime = points die faster
   ```cpp
   emitter->SetTrailLifetime(0.3f); // Quick fade
   ```

3. **Limit Particle Count**: Fewer particles = fewer trails
   ```cpp
   auto emitter = std::make_shared<ParticleEmitter>(position, 200); // Limit to 200
   ```

4. **Use Trails Selectively**: Enable only for key effects
   ```cpp
   mainEffect->SetEnableTrails(true);   // Important effect
   ambient->SetEnableTrails(false);     // Background effect
   ```

### Recommended Settings

| Particle Count | Trail Length | Notes |
|----------------|--------------|-------|
| < 100 | 50-80 | Safe for long trails |
| 100-300 | 30-50 | Balanced |
| 300-500 | 20-30 | Keep trails short |
| > 500 | 10-20 | Minimal trails only |

## Troubleshooting

### Trails Not Visible

**Problem**: Trails are enabled but not rendering.

**Solutions**:
1. Check trail lifetime isn't too short:
   ```cpp
   emitter->SetTrailLifetime(0.5f); // Increase if too short
   ```

2. Verify trail width isn't too small:
   ```cpp
   emitter->SetTrailWidth(0.5f); // Increase width
   ```

3. Ensure particles are moving:
   ```cpp
   // Trails need movement to be visible
   emitter->SetVelocityRange(Vec3(-2, 0, -2), Vec3(2, 3, 2));
   ```

### Trails Look Choppy

**Problem**: Trails have visible gaps or segments.

**Solutions**:
1. Decrease MinDistance:
   ```cpp
   emitter->SetTrailMinDistance(0.05f); // Smoother
   ```

2. Increase trail length:
   ```cpp
   emitter->SetTrailLength(50); // More points
   ```

3. Ensure particles move fast enough:
   ```cpp
   emitter->SetVelocityRange(Vec3(-3, -1, -3), Vec3(3, 3, 3));
   ```

### Trails Too Thick/Thin

**Problem**: Trail width doesn't look right.

**Solution**: Adjust trail width multiplier:
```cpp
emitter->SetTrailWidth(0.3f);  // Thinner
emitter->SetTrailWidth(0.8f);  // Thicker
```

### Performance Issues

**Problem**: FPS drops with trails enabled.

**Solutions**:
1. Reduce trail length:
   ```cpp
   emitter->SetTrailLength(20); // Shorter trails
   ```

2. Increase MinDistance:
   ```cpp
   emitter->SetTrailMinDistance(0.2f); // Fewer points
   ```

3. Reduce particle count:
   ```cpp
   auto emitter = std::make_shared<ParticleEmitter>(position, 150);
   ```

4. Shorten trail lifetime:
   ```cpp
   emitter->SetTrailLifetime(0.3f); // Faster fade
   ```

### Trails Wrong Color

**Problem**: Trail colors don't match expectations.

**Solutions**:
1. Check color mode:
   ```cpp
   emitter->SetTrailColorMode(TrailColorMode::FadeToTransparent);
   ```

2. For custom colors:
   ```cpp
   emitter->SetTrailColorMode(TrailColorMode::Custom);
   emitter->SetTrailColor(Vec4(1.0f, 0.5f, 0.0f, 1.0f));
   ```

3. Verify particle colors:
   ```cpp
   emitter->SetColorRange(
       Vec4(1.0f, 1.0f, 1.0f, 1.0f),  // Start
       Vec4(1.0f, 1.0f, 1.0f, 0.0f)   // End
   );
   ```

## API Quick Reference

### Trail Configuration
```cpp
void SetEnableTrails(bool enable);
void SetTrailLength(int maxPoints);
void SetTrailLifetime(float lifetime);
void SetTrailWidth(float width);
void SetTrailMinDistance(float distance);
void SetTrailTexture(std::shared_ptr<Texture> texture);
void SetTrailColorMode(TrailColorMode mode);
void SetTrailColor(const Vec4& color);
```

### Trail Color Modes
```cpp
enum class TrailColorMode {
    ParticleColor,      // Use particle's current color
    FadeToTransparent,  // Fade from particle color to transparent
    GradientToEnd,      // Gradient from start to end color
    Custom              // Use custom trail color
};
```

---

**Need Help?** Check the implementation details in:
- [ParticleTrail.h](../include/ParticleTrail.h) - Trail point management
- [ParticleEmitter.h](../include/ParticleEmitter.h) - Trail configuration
- [ParticleSystem.cpp](../src/ParticleSystem.cpp) - Trail rendering
