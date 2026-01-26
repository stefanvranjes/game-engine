# Basic enemy AI with state machine
extends Node

class_name EnemyAI

# Enemy properties
var health: float = 50.0
var max_health: float = 50.0
var damage: float = 10.0
var detection_range: float = 15.0
var attack_range: float = 2.0
var patrol_speed: float = 2.0
var chase_speed: float = 5.0
var attack_cooldown: float = 1.0
var time_since_attack: float = 0.0

# State machine
var current_state: String = "idle"
var target: Node = null
var patrol_points: Array = []
var current_patrol_point: int = 0

# Animation and visual
var animation_speed: float = 1.0

# Signals
signal state_changed(new_state: String)
signal health_changed(new_health: float)
signal enemy_died(position: Vector3)
signal target_detected(target: Node)
signal target_lost()
signal attack_executed(damage: float)

func _ready():
    """Initialize enemy AI"""
    print("EnemyAI initialized")
    print("  Health: ", health, "/", max_health)
    print("  Damage: ", damage)
    print("  Detection Range: ", detection_range)
    setup_patrol_points()

func _process(delta: float):
    """Called every frame"""
    # Update attack cooldown
    if time_since_attack > 0:
        time_since_attack -= delta
    
    # Update state machine
    update_state_machine(delta)
    
    # Execute current state
    match current_state:
        "idle":
            idle_state(delta)
        "patrol":
            patrol_state(delta)
        "chase":
            chase_state(delta)
        "attack":
            attack_state(delta)
        "dead":
            # Dead state - no updates
            pass

func update_state_machine(delta: float):
    """Update state transitions"""
    match current_state:
        "idle":
            # Check if target is detected
            if should_chase():
                change_state("chase")
            elif should_patrol():
                change_state("patrol")
        
        "patrol":
            # Check if target is detected
            if should_chase():
                change_state("chase")
            elif reached_patrol_point():
                next_patrol_point()
        
        "chase":
            # Check if target is lost
            if target == null or target_distance() > detection_range * 1.2:
                emit_signal("target_lost")
                change_state("idle")
            elif target_distance() <= attack_range:
                change_state("attack")
        
        "attack":
            # Check if target is still in range
            if target == null or target_distance() > attack_range * 1.5:
                change_state("chase")
            elif time_since_attack <= 0 and target_distance() <= attack_range:
                execute_attack()

func idle_state(delta: float):
    """Idle state - waiting"""
    # Just idle
    pass

func patrol_state(delta: float):
    """Patrol state - moving between points"""
    if patrol_points.is_empty():
        return
    
    var current_point = patrol_points[current_patrol_point]
    # Move towards patrol point with patrol_speed
    # This would use actual movement in real game

func chase_state(delta: float):
    """Chase state - following target"""
    if target == null:
        return
    
    # Move towards target with chase_speed
    var direction = (target.position - position).normalized()
    # Actual movement logic would go here

func attack_state(delta: float):
    """Attack state - attacking target"""
    if target == null:
        return
    
    # Face target
    # Play attack animation
    # Damage is dealt via execute_attack()

func execute_attack():
    """Execute an attack on target"""
    if target == null or time_since_attack > 0:
        return
    
    if has_method("take_damage"):
        target.take_damage(damage)
    
    time_since_attack = attack_cooldown
    emit_signal("attack_executed", damage)

func change_state(new_state: String):
    """Change current state"""
    if current_state == new_state:
        return
    
    current_state = new_state
    emit_signal("state_changed", new_state)
    print("Enemy state changed to: ", new_state)

func should_chase() -> bool:
    """Check if should enter chase state"""
    if target == null:
        target = find_nearby_target()
        if target != null:
            emit_signal("target_detected", target)
            return true
        return false
    
    var distance = target_distance()
    return distance <= detection_range

func should_patrol() -> bool:
    """Check if should patrol"""
    return target == null and not patrol_points.is_empty()

func setup_patrol_points():
    """Setup default patrol points"""
    # Would be set by game logic
    patrol_points = []

func next_patrol_point():
    """Move to next patrol point"""
    if patrol_points.is_empty():
        return
    
    current_patrol_point = (current_patrol_point + 1) % patrol_points.size()

func reached_patrol_point() -> bool:
    """Check if reached patrol point"""
    if patrol_points.is_empty():
        return false
    
    # Check distance to current patrol point
    return true

func find_nearby_target() -> Node:
    """Find a target within detection range"""
    # In real implementation, would search scene for nearby actors
    return null

func target_distance() -> float:
    """Get distance to current target"""
    if target == null:
        return 999999.0
    
    return (target.position - position).length()

func take_damage(amount: float):
    """Take damage"""
    health -= amount
    emit_signal("health_changed", health)
    
    print("Enemy took ", amount, " damage. Health: ", health)
    
    if health <= 0:
        die()

func die():
    """Die"""
    if current_state == "dead":
        return
    
    current_state = "dead"
    emit_signal("enemy_died", position)
    print("Enemy died at position: ", position)

func heal(amount: float):
    """Heal the enemy"""
    health = min(health + amount, max_health)
    emit_signal("health_changed", health)

func get_health_percentage() -> float:
    """Get health as percentage"""
    if max_health <= 0:
        return 0.0
    return health / max_health

func set_target(new_target: Node):
    """Manually set target"""
    target = new_target
    if target != null:
        emit_signal("target_detected", target)
        change_state("chase")

func clear_target():
    """Clear current target"""
    target = null
    emit_signal("target_lost")
    change_state("idle")
