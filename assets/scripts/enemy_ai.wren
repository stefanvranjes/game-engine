// Enemy AI behavior script for Wren game engine
// Demonstrates state machine, patrol, chase, and attack logic

class EnemyAI {
    construct new(gameObject) {
        _gameObject = gameObject
        _transform = gameObject.transform
        _rigidBody = gameObject.getComponent("RigidBody")
        _animator = gameObject.getComponent("Animator")
        _audioSource = gameObject.getComponent("AudioSource")
        
        // AI states
        _state = "patrol"
        _patrolWaypoints = []
        _currentWaypoint = 0
        
        // Detection
        _detectRadius = 10.0
        _attackRadius = 2.0
        _target = null
        _hasLineOfSight = false
        
        // Movement
        _moveSpeed = 3.0
        _chaseSpeed = 5.0
        _turnSpeed = 5.0
        
        // Combat
        _health = 100.0
        _maxHealth = 100.0
        _attackCooldown = 1.5
        _attackTimer = 0.0
        _attackDamage = 15.0
        
        // Timers
        _stateTimer = 0.0
        _idleTimeout = 5.0
    }
    
    init() {
        setupPatrol()
        _animator.play("idle")
        System.print("Enemy spawned at %(getPosition())")
    }
    
    update(dt) {
        // Update timers
        _stateTimer = _stateTimer + dt
        _attackTimer = _attackTimer - dt
        
        // Detect player
        detectTarget()
        
        // State machine
        if (_state == "patrol") {
            updatePatrol(dt)
        } else if (_state == "chase") {
            updateChase(dt)
        } else if (_state == "attack") {
            updateAttack(dt)
        } else if (_state == "idle") {
            updateIdle(dt)
        }
    }
    
    setupPatrol() {
        // Initialize patrol waypoints
        // In a real scenario, these would come from the scene or level data
        _patrolWaypoints = []
        _patrolWaypoints.add(_transform.position)
        _patrolWaypoints.add(_transform.position + Vec3.new(5, 0, 0))
        _patrolWaypoints.add(_transform.position + Vec3.new(5, 0, 5))
        _patrolWaypoints.add(_transform.position + Vec3.new(0, 0, 5))
    }
    
    updatePatrol(dt) {
        if (_patrolWaypoints.count == 0) return
        
        var target = _patrolWaypoints[_currentWaypoint]
        var distance = (_transform.position - target).magnitude
        
        if (distance < 0.5) {
            _currentWaypoint = (_currentWaypoint + 1) % _patrolWaypoints.count
            _stateTimer = 0.0
        } else {
            moveTowards(target, _moveSpeed, dt)
            _animator.play("walk")
        }
    }
    
    updateChase(dt) {
        if (!_target) {
            _state = "idle"
            _stateTimer = 0.0
            return
        }
        
        var targetPos = _target.transform.position
        var distance = (_transform.position - targetPos).magnitude
        
        if (distance > _detectRadius * 1.5 || !hasLineOfSight()) {
            _state = "idle"
            _stateTimer = 0.0
            _target = null
        } else if (distance < _attackRadius) {
            _state = "attack"
            _stateTimer = 0.0
        } else {
            moveTowards(targetPos, _chaseSpeed, dt)
            _animator.play("run")
        }
    }
    
    updateAttack(dt) {
        if (!_target) {
            _state = "chase"
            return
        }
        
        var targetPos = _target.transform.position
        var distance = (_transform.position - targetPos).magnitude
        
        if (distance > _attackRadius * 1.5) {
            _state = "chase"
        } else {
            faceTarget(targetPos)
            _animator.play("attack")
            
            if (_attackTimer <= 0) {
                performAttack()
                _attackTimer = _attackCooldown
            }
        }
    }
    
    updateIdle(dt) {
        if (_stateTimer > _idleTimeout) {
            _state = "patrol"
            _stateTimer = 0.0
        }
    }
    
    detectTarget() {
        // In a real implementation, this would use proper spatial queries
        // For now, we'll set up the detection system
        var inRangeEnemies = []
        
        // Physics system would return objects in range
        // if (Physics.overlapSphere(getPosition(), _detectRadius, inRangeEnemies)) {
        //     for (enemy in inRangeEnemies) {
        //         if (enemy.tag == "Player") {
        //             _target = enemy
        //             _state = "chase"
        //             return
        //         }
        //     }
        // }
    }
    
    hasLineOfSight() {
        if (!_target) return false
        
        // In a real implementation, cast a ray to check for line of sight
        // var hit = Physics.raycast(getPosition(), (_target.transform.position - getPosition()).normalized)
        // return hit && hit.collider.gameObject == _target
        
        return true // Placeholder
    }
    
    moveTowards(targetPos, speed, dt) {
        var direction = (targetPos - _transform.position).normalized
        var newPos = _transform.position + direction * speed * dt
        _transform.setPosition(newPos.x, newPos.y, newPos.z)
        faceTarget(targetPos)
    }
    
    faceTarget(targetPos) {
        var direction = (targetPos - _transform.position).normalized
        var angle = System.atan2(direction.z, direction.x)
        // Smoothly rotate towards target
        // _transform.rotation would be updated here
    }
    
    performAttack() {
        System.print("Enemy attacking!")
        _audioSource.playOneShot("attack_sfx")
        
        if (_target) {
            // _target.takeDamage(_attackDamage)
            System.print("Hit target for %((_attackDamage)) damage")
        }
    }
    
    takeDamage(amount) {
        _health = _health - amount
        System.print("Enemy health: %(_health) / %(_maxHealth)")
        
        if (_health <= 0) {
            die()
        } else {
            _animator.play("hurt")
            // Temporary alert state
            _state = "chase"
        }
    }
    
    die() {
        System.print("Enemy defeated!")
        _animator.play("death")
        // Schedule destruction after animation
        _gameObject.destroy()
    }
    
    getPosition() {
        return _transform.position
    }
    
    destroy() {
        System.print("Enemy destroyed")
    }
}

// Global enemy instance
var _enemy = null

// Lifecycle functions
construct init() {
    _enemy = EnemyAI.new(_gameObject)
    _enemy.init()
}

construct update(dt) {
    _enemy.update(dt)
}

construct destroy() {
    _enemy.destroy()
}
