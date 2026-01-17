// Utility functions and helpers for Wren gameplay scripts
// Provides common functions used across different gameplay scripts

// Vector utilities
class VectorUtils {
    static distance(a, b) {
        return (a - b).magnitude
    }
    
    static direction(from, to) {
        var diff = to - from
        if (diff.magnitude == 0) {
            return Vec3.new(0, 0, 0)
        }
        return diff.normalized
    }
    
    static lerp(a, b, t) {
        return Vec3.new(
            Mathf.lerp(a.x, b.x, t),
            Mathf.lerp(a.y, b.y, t),
            Mathf.lerp(a.z, b.z, t)
        )
    }
    
    static clamp(vec, min, max) {
        return Vec3.new(
            Mathf.clamp(vec.x, min, max),
            Mathf.clamp(vec.y, min, max),
            Mathf.clamp(vec.z, min, max)
        )
    }
}

// Math utilities
class MathUtils {
    static random() {
        return System.random()
    }
    
    static randomRange(min, max) {
        return min + System.random() * (max - min)
    }
    
    static randomInt(min, max) {
        return (min + (System.random() * (max - min))).floor
    }
    
    static sign(value) {
        if (value > 0) return 1
        if (value < 0) return -1
        return 0
    }
    
    static pingPong(value, length) {
        if (value < 0) {
            return -value % (length * 2)
        }
        return value % (length * 2)
    }
    
    static repeat(value, length) {
        return value % length
    }
}

// Physics utilities
class PhysicsUtils {
    static raycast(origin, direction, maxDistance) {
        // Wrapper for physics raycast
        // Would be implemented by physics system
        return null
    }
    
    static overlapSphere(center, radius) {
        // Wrapper for overlap query
        // Would be implemented by physics system
        return []
    }
    
    static overlapBox(center, extents) {
        // Wrapper for box overlap
        return []
    }
    
    static groundCheck(origin, maxDistance) {
        // Check if there's ground below
        // Direction = -Y axis
        return raycast(origin, Vec3.new(0, -1, 0), maxDistance)
    }
}

// Animation utilities
class AnimationUtils {
    static blendToAnimation(animator, targetAnimation, blendTime) {
        // Play animation with blend-in effect
        animator.play(targetAnimation)
    }
    
    static crossfade(animator, fromAnimation, toAnimation, duration) {
        // Crossfade between two animations
        animator.play(toAnimation)
    }
    
    static setAnimationSpeed(animator, speed) {
        // Adjust playback speed of current animation
    }
    
    static isAnimationPlaying(animator, animationName) {
        // Check if specific animation is currently playing
        return true
    }
}

// Audio utilities
class AudioUtils {
    static playSoundAt(soundName, position, volume) {
        // Play 3D sound at position
        // AudioSystem.playAtPoint(soundName, position, volume)
    }
    
    static playSoundLoop(audioSource, soundName) {
        // Play looping sound
        audioSource.loop = true
        // audioSource.play(soundName)
    }
    
    static fadeAudio(audioSource, targetVolume, duration) {
        // Fade audio over time
        // Would require coroutine support
    }
    
    static stopAll() {
        // Stop all audio
        // AudioSystem.stopAll()
    }
}

// Particle utilities
class ParticleUtils {
    static burstParticles(particleSystem, count, position) {
        // Emit particle burst at position
        // particleSystem.emit(count)
    }
    
    static enableParticles(particleSystem) {
        // particleSystem.play()
    }
    
    static disableParticles(particleSystem) {
        // particleSystem.stop()
    }
}

// GameObject utilities
class GameObjectUtils {
    static findGameObjectByName(name) {
        // Find object by name
        // Would use scene graph or object registry
        return null
    }
    
    static findGameObjectsByTag(tag) {
        // Find all objects with tag
        return []
    }
    
    static findGameObjectsInRadius(center, radius) {
        // Find objects within radius
        return []
    }
    
    static instantiate(prefabName, position) {
        // Spawn instance of prefab
        // return SpawnManager.spawn(prefabName, position)
    }
    
    static destroy(gameObject, delay) {
        if (delay && delay > 0) {
            // Schedule destruction after delay
            // Timer.schedule(delay, { gameObject.destroy() })
        } else {
            gameObject.destroy()
        }
    }
}

// Debugging utilities
class DebugUtils {
    static Assert(condition, message) {
        if (!condition) {
            Debug.error("ASSERTION FAILED: %(message)")
        }
    }
    
    static logWarning(message) {
        Debug.warn(message)
    }
    
    static logError(message) {
        Debug.error(message)
    }
    
    static drawDebugInfo(pos, text) {
        Debug.log("%(text) at %(pos)")
    }
    
    static profileSection(name, func) {
        // Time the execution of a function
        var startTime = Time.time
        func.call()
        var elapsed = Time.time - startTime
        Debug.log("%(name) took %(elapsed) seconds")
    }
}

// Event system utilities
class EventUtils {
    static emit(eventName, data) {
        // Emit custom event
        System.print("Event: %(eventName)")
    }
    
    static on(eventName, callback) {
        // Subscribe to event
    }
    
    static off(eventName, callback) {
        // Unsubscribe from event
    }
}

// Pool utilities for object reuse
class ObjectPool {
    construct new(factory, initialSize) {
        _factory = factory
        _pool = []
        _available = []
        
        for (i in 0...initialSize) {
            var obj = _factory.call()
            _pool.add(obj)
            _available.add(obj)
        }
    }
    
    get() {
        if (_available.count > 0) {
            return _available.removeAt(0)
        }
        return _factory.call()
    }
    
    release(obj) {
        if (_available.count < _pool.count) {
            _available.add(obj)
        }
    }
    
    clear() {
        _pool.clear()
        _available.clear()
    }
}

// Timer/Coroutine utilities
class Timer {
    construct new(duration, callback) {
        _duration = duration
        _callback = callback
        _elapsed = 0.0
        _isFinished = false
    }
    
    update(dt) {
        if (_isFinished) return
        
        _elapsed = _elapsed + dt
        if (_elapsed >= _duration) {
            _isFinished = true
            if (_callback) {
                _callback.call()
            }
        }
    }
    
    isFinished { _isFinished }
    elapsed { _elapsed }
    remaining { (_duration - _elapsed).max(0) }
}

System.print("Utility functions loaded")
