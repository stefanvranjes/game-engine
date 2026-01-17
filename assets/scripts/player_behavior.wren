// Player behavior script for Wren game engine
// Demonstrates basic player control, animation, and physics interaction

class Player {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _rigidBody = gameObject.getComponent("RigidBody")
        _animator = gameObject.getComponent("Animator")
        
        // Movement
        _moveSpeed = 5.0
        _jumpForce = 10.0
        _isGrounded = true
        _groundDrag = 0.5
        _airDrag = 0.1
        
        // Animation
        _currentAnimation = "idle"
        _isMoving = false
        _moveDirection = Vec3.new(0, 0, 0)
    }
    
    init() {
        System.print("Player initialized at %(getPosition())")
        _animator.play("idle")
    }
    
    update(dt) {
        handleInput(dt)
        updateMovement(dt)
        updateAnimation()
        updatePhysics()
    }
    
    handleInput(dt) {
        var moveX = 0.0
        var moveZ = 0.0
        
        // Get input axes
        if (Input.getKey("W")) moveZ = moveZ + 1.0
        if (Input.getKey("S")) moveZ = moveZ - 1.0
        if (Input.getKey("A")) moveX = moveX - 1.0
        if (Input.getKey("D")) moveX = moveX + 1.0
        
        // Normalize movement
        _moveDirection = Vec3.new(moveX, 0, moveZ)
        if (_moveDirection.magnitude > 0) {
            _moveDirection = _moveDirection.normalized * _moveSpeed
        }
        
        _isMoving = _moveDirection.magnitude > 0.1
        
        // Jump
        if (Input.getKeyDown("Space") && _isGrounded) {
            jump()
        }
    }
    
    updateMovement(dt) {
        if (_isMoving) {
            var currentVel = _rigidBody.velocity
            _rigidBody.setVelocity(
                _moveDirection.x,
                currentVel.y,
                _moveDirection.z
            )
        }
    }
    
    updateAnimation() {
        var targetAnim = _isMoving ? "run" : "idle"
        
        if (targetAnim != _currentAnimation) {
            _animator.play(targetAnim)
            _currentAnimation = targetAnim
        }
    }
    
    updatePhysics() {
        // Apply air resistance
        var dragCoeff = _isGrounded ? _groundDrag : _airDrag
        // Physics system would apply damping
    }
    
    jump() {
        if (_isGrounded) {
            _rigidBody.applyImpulse(0, _jumpForce, 0)
            _animator.play("jump")
            _isGrounded = false
            System.print("Jump!")
        }
    }
    
    getPosition() {
        return _transform.position
    }
    
    setPosition(x, y, z) {
        _transform.setPosition(x, y, z)
    }
    
    onCollisionEnter(collider) {
        System.print("Collision with: %(collider.gameObject.name)")
        if (collider.gameObject.tag == "Ground") {
            _isGrounded = true
        }
    }
    
    onTrigger(other) {
        System.print("Triggered: %(other.name)")
    }
    
    takeDamage(amount) {
        System.print("Taking damage: %(amount)")
    }
    
    destroy() {
        System.print("Player destroyed")
    }
}

// Global player instance
var _player = null

// Script lifecycle functions called by WrenScriptComponent
construct init() {
    _player = Player.new(_gameObject)
    _player.init()
}

construct update(dt) {
    _player.update(dt)
}

construct destroy() {
    _player.destroy()
}
