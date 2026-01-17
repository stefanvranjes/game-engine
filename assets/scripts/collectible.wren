// Collectible item script for Wren game engine
// Demonstrates pickup mechanics, particles, and audio integration

class Collectible {
    construct new(gameObject, itemType) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _collider = gameObject.getComponent("Collider")
        _audioSource = gameObject.getComponent("AudioSource")
        _particleSystem = gameObject.getComponent("ParticleSystem")
        
        _itemType = itemType  // "coin", "health", "ammo", etc.
        _value = getValueForType(itemType)
        _isCollected = false
        
        // Bobbing animation
        _bobHeight = 0.2
        _bobSpeed = 2.0
        _originalY = _transform.position.y
        _bobTimer = 0.0
        
        // Rotation animation
        _rotationSpeed = 90.0  // degrees per second
        _currentRotation = 0.0
    }
    
    init() {
        _collider.setTrigger(true)
        _audioSource.spatialBlend = 1.0  // Full 3D audio
        System.print("Collectible spawned: type=%(_itemType) value=%(_value)")
    }
    
    update(dt) {
        if (_isCollected) return
        
        // Bobbing animation
        _bobTimer = _bobTimer + dt
        var bobOffset = Mathf.sin(_bobTimer * _bobSpeed * 3.14159) * _bobHeight
        
        var currentPos = _transform.position
        _transform.setPosition(
            currentPos.x,
            _originalY + bobOffset,
            currentPos.z
        )
        
        // Rotation animation
        _currentRotation = _currentRotation + _rotationSpeed * dt
        if (_currentRotation > 360) {
            _currentRotation = _currentRotation - 360
        }
        // _transform.setRotation would apply rotation here
    }
    
    onCollectionEnter(collector) {
        if (_isCollected) return
        
        if (collector.tag == "Player") {
            collect(collector)
        }
    }
    
    collect(collector) {
        _isCollected = true
        System.print("Collected %(itemType): +%(_value)")
        
        // Play collection sound
        _audioSource.playOneShotAtPoint(
            "collect_sound",
            _transform.position.x,
            _transform.position.y,
            _transform.position.z,
            1.0
        )
        
        // Emit particles
        if (_particleSystem) {
            _particleSystem.emit(20)
        }
        
        // Award points/items to player
        awardToPlayer(collector)
        
        // Destroy after a short delay (for animation)
        // _gameObject.destroy() would be called after particle animation
    }
    
    awardToPlayer(player) {
        // Apply effect based on item type
        if (_itemType == "coin") {
            player.addCoins(_value)
        } else if (_itemType == "health") {
            player.heal(_value)
        } else if (_itemType == "ammo") {
            player.addAmmo("default", _value)
        } else if (_itemType == "powerup") {
            player.addPowerup(_itemType, 10.0)
        }
    }
    
    getValueForType(itemType) {
        if (itemType == "coin") return 10
        if (itemType == "health") return 25
        if (itemType == "ammo") return 30
        if (itemType == "powerup") return 1
        return 1
    }
    
    destroy() {
        System.print("Collectible destroyed")
    }
}

// Global collectible instance
var _collectible = null
var _itemType = "coin"  // Can be set before init

// Lifecycle functions
construct init() {
    _collectible = Collectible.new(_gameObject, _itemType)
    _collectible.init()
}

construct update(dt) {
    _collectible.update(dt)
}

construct destroy() {
    _collectible.destroy()
}
