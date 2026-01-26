# Game manager for handling game state and lifecycle
extends Node

class_name GameManager

# Game state
var is_running: bool = false
var is_paused: bool = false
var current_level: int = 1
var time_elapsed: float = 0.0
var frame_count: int = 0

# References
var player: Node = null
var enemies: Array = []
var ui_manager: Node = null

# Game configuration
var max_levels: int = 10
var difficulty: float = 1.0

# Signals
signal game_started()
signal game_paused()
signal game_resumed()
signal level_changed(level: int)
signal time_updated(time: float)
signal frame_rendered(frame: int)

func _ready():
    """Initialize game manager"""
    print("GameManager initialized")
    print("  Max Levels: ", max_levels)
    print("  Difficulty: ", difficulty)
    setup_game()

func _process(delta: float):
    """Called every frame"""
    if not is_running or is_paused:
        return
    
    time_elapsed += delta
    frame_count += 1
    
    # Update game state
    update_game(delta)
    
    # Emit periodic signals
    if frame_count % 60 == 0:  # Every ~1 second at 60 FPS
        emit_signal("time_updated", time_elapsed)
        emit_signal("frame_rendered", frame_count)

func setup_game():
    """Initialize game setup"""
    print("Setting up game...")
    
    # Initialize systems
    is_running = false
    is_paused = false
    time_elapsed = 0.0
    frame_count = 0
    
    print("Game setup complete")

func start_game():
    """Start the game"""
    if is_running:
        return
    
    is_running = true
    is_paused = false
    
    print("Game started at level ", current_level)
    emit_signal("game_started")

func pause_game():
    """Pause the game"""
    if not is_running or is_paused:
        return
    
    is_paused = true
    print("Game paused")
    emit_signal("game_paused")

func resume_game():
    """Resume the game"""
    if not is_running or not is_paused:
        return
    
    is_paused = false
    print("Game resumed")
    emit_signal("game_resumed")

func update_game(delta: float):
    """Update game logic"""
    # Update player
    if player:
        # Player logic would go here
        pass
    
    # Update enemies
    for enemy in enemies:
        if enemy:
            # Enemy update logic
            pass

func next_level():
    """Advance to next level"""
    if current_level >= max_levels:
        end_game()
        return
    
    current_level += 1
    print("Moving to level ", current_level)
    
    # Reset time for new level
    time_elapsed = 0.0
    frame_count = 0
    
    emit_signal("level_changed", current_level)

func previous_level():
    """Go back to previous level"""
    if current_level <= 1:
        return
    
    current_level -= 1
    print("Moving to level ", current_level)
    
    # Reset level state
    time_elapsed = 0.0
    frame_count = 0
    
    emit_signal("level_changed", current_level)

func end_game():
    """End the game"""
    is_running = false
    print("Game ended. Final time: ", time_elapsed, " seconds")
    print("Total frames: ", frame_count)

func set_player(new_player: Node):
    """Set the player reference"""
    player = new_player
    print("Player set: ", new_player)

func add_enemy(enemy: Node):
    """Add an enemy to the game"""
    enemies.append(enemy)
    print("Enemy added. Total enemies: ", enemies.size())

func remove_enemy(enemy: Node):
    """Remove an enemy from the game"""
    enemies.erase(enemy)
    print("Enemy removed. Total enemies: ", enemies.size())

func set_difficulty(new_difficulty: float):
    """Set game difficulty"""
    difficulty = clamp(new_difficulty, 0.1, 10.0)
    print("Difficulty set to: ", difficulty)

func get_game_state() -> Dictionary:
    """Get current game state as dictionary"""
    return {
        "is_running": is_running,
        "is_paused": is_paused,
        "level": current_level,
        "time": time_elapsed,
        "frames": frame_count,
        "difficulty": difficulty,
        "enemies": enemies.size()
    }

func get_elapsed_time() -> float:
    """Get elapsed time in seconds"""
    return time_elapsed

func get_frame_count() -> int:
    """Get total frame count"""
    return frame_count
