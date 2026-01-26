# Simple player movement controller for GDScript integration example
extends Node

class_name PlayerController

# Player properties
var velocity: Vector3 = Vector3.ZERO
var acceleration: Vector3 = Vector3.ZERO
var move_speed: float = 5.0
var jump_force: float = 10.0
var gravity: float = 9.8
var is_grounded: bool = true

# Input state
var input_direction: Vector3 = Vector3.ZERO

# Events
signal player_moved(position: Vector3)
signal player_jumped()
signal velocity_changed(new_velocity: Vector3)

func _ready():
    """Called when the script is loaded"""
    print("PlayerController initialized")
    print("  Speed: ", move_speed)
    print("  Jump Force: ", jump_force)

func _process(delta: float):
    """Called every frame"""
    if delta <= 0:
        return
    
    # Update movement
    handle_input(delta)
    update_velocity(delta)
    update_position(delta)

func handle_input(delta: float):
    """Process player input"""
    # In a real game, this would use actual input system
    # For demo purposes, we'll simulate input
    input_direction = Vector3.ZERO
    
    # Simulate moving forward
    if randf() > 0.7:  # 30% chance each frame
        input_direction.z = -1
    
    # Simulate strafing
    if randf() > 0.8:  # 20% chance each frame
        input_direction.x = 1 if randf() > 0.5 else -1

func update_velocity(delta: float):
    """Update velocity based on acceleration and gravity"""
    # Apply gravity
    if not is_grounded:
        velocity.y -= gravity * delta
    
    # Apply movement
    var target_velocity = input_direction.normalized() * move_speed
    velocity.x = target_velocity.x
    velocity.z = target_velocity.z
    
    # Clamp vertical velocity
    velocity.y = clamp(velocity.y, -20.0, jump_force)
    
    # Emit signal on velocity change
    emit_signal("velocity_changed", velocity)

func update_position(delta: float):
    """Update position based on velocity"""
    # In real implementation, use physics engine
    # For now, just track position changes
    var new_position = Vector3.ZERO  # Would be actual position
    emit_signal("player_moved", new_position)

func jump():
    """Execute a jump"""
    if is_grounded:
        velocity.y = jump_force
        is_grounded = false
        emit_signal("player_jumped")
        print("Player jumped with force: ", jump_force)

func set_grounded(grounded: bool):
    """Set whether player is on ground"""
    if is_grounded != grounded:
        is_grounded = grounded
        if is_grounded:
            velocity.y = 0.0

func get_speed() -> float:
    """Get current movement speed"""
    return velocity.length()

func get_velocity() -> Vector3:
    """Get current velocity"""
    return velocity
