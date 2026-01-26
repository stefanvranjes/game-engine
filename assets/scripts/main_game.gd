# Example main game script showing how to integrate C++ and GDScript
# This script demonstrates the typical flow and integration points
extends Node

class_name MainGame

# Sub-systems
var game_manager: GameManager = null
var player_controller: PlayerController = null
var ui_manager: UIManager = null
var enemy_manager: Node = null

# Game state
var is_initialized: bool = false

# Signals
signal initialization_complete()
signal game_loop_started()
signal game_state_changed(state: String)

func _ready():
    """Called when script loads"""
    print("=== MainGame GDScript Initialization ===")
    print("GDScript integration example")
    
    # Initialize subsystems
    initialize_systems()
    
    # Connect signals
    setup_signal_connections()
    
    # Start game loop
    start_game_loop()
    
    is_initialized = true
    emit_signal("initialization_complete")

func initialize_systems():
    """Initialize game subsystems"""
    print("\n[INIT] Initializing game systems...")
    
    # Create/initialize game manager
    game_manager = GameManager.new()
    add_child(game_manager)
    print("[INIT] Game manager created")
    
    # Create/initialize player controller
    player_controller = PlayerController.new()
    add_child(player_controller)
    game_manager.set_player(player_controller)
    print("[INIT] Player controller created")
    
    # Create/initialize UI manager
    ui_manager = UIManager.new()
    add_child(ui_manager)
    print("[INIT] UI manager created")
    
    print("[INIT] All systems initialized\n")

func setup_signal_connections():
    """Connect signals between systems"""
    print("[SIGNALS] Setting up signal connections...")
    
    # Connect game manager signals
    if game_manager:
        game_manager.connect("game_started", Callable(self, "_on_game_started"))
        game_manager.connect("game_paused", Callable(self, "_on_game_paused"))
        game_manager.connect("game_resumed", Callable(self, "_on_game_resumed"))
        game_manager.connect("level_changed", Callable(self, "_on_level_changed"))
        game_manager.connect("time_updated", Callable(self, "_on_time_updated"))
    
    # Connect player signals
    if player_controller:
        player_controller.connect("player_moved", Callable(self, "_on_player_moved"))
        player_controller.connect("player_jumped", Callable(self, "_on_player_jumped"))
        player_controller.connect("velocity_changed", Callable(self, "_on_velocity_changed"))
    
    # Connect UI signals
    if ui_manager:
        ui_manager.connect("health_updated", Callable(self, "_on_health_updated"))
        ui_manager.connect("score_updated", Callable(self, "_on_score_updated"))
    
    print("[SIGNALS] Signal connections complete\n")

func start_game_loop():
    """Start the main game loop"""
    print("[GAMELOOP] Starting game loop...")
    
    if game_manager:
        game_manager.start_game()
    
    emit_signal("game_loop_started")
    print("[GAMELOOP] Game loop started\n")

func _process(delta: float):
    """Main update loop - called every frame"""
    if not is_initialized:
        return
    
    # Update game manager (handles frame counting)
    if game_manager:
        # Game manager's _process will be called automatically
        pass
    
    # Perform per-frame checks
    check_game_conditions(delta)

func check_game_conditions(delta: float):
    """Check various game conditions"""
    # Check if player is alive
    if player_controller:
        var speed = player_controller.get_speed()
        if speed > 0.1:
            # Player is moving
            pass

func _on_game_started():
    """Called when game starts"""
    print("[EVENT] Game started")
    emit_signal("game_state_changed", "running")
    
    # Setup initial game state
    if ui_manager:
        ui_manager.show_hud()
        ui_manager.set_health(100.0)
        ui_manager.set_score(0)

func _on_game_paused():
    """Called when game is paused"""
    print("[EVENT] Game paused")
    emit_signal("game_state_changed", "paused")
    
    if ui_manager:
        ui_manager.show_pause_menu()

func _on_game_resumed():
    """Called when game is resumed"""
    print("[EVENT] Game resumed")
    emit_signal("game_state_changed", "running")
    
    if ui_manager:
        ui_manager.close_menu()

func _on_level_changed(level: int):
    """Called when level changes"""
    print("[EVENT] Level changed to: ", level)
    emit_signal("game_state_changed", "level_change")
    
    if ui_manager:
        ui_manager.set_level(level)
        ui_manager.show_notification("Level %d" % level, 2.0)

func _on_time_updated(time: float):
    """Called periodically with elapsed time"""
    # This is called every ~1 second
    var minutes = int(time) / 60
    var seconds = int(time) % 60
    # Update timer display if needed

func _on_player_moved(position: Vector3):
    """Called when player moves"""
    # Update player position in world
    pass

func _on_player_jumped():
    """Called when player jumps"""
    print("[PLAYER] Jump executed")
    if ui_manager:
        ui_manager.show_notification("Jump!", 0.5)

func _on_velocity_changed(velocity: Vector3):
    """Called when player velocity changes"""
    # Velocity changed - could affect animations
    pass

func _on_health_updated(health: float, max_health: float):
    """Called when player health changes"""
    var health_percent = (health / max_health) * 100.0
    print("[HEALTH] Health: %.1f%%" % health_percent)

func _on_score_updated(score: int):
    """Called when score changes"""
    print("[SCORE] Current score: ", score)

# ============================================================================
# Game Control Methods
# ============================================================================

func pause_game():
    """Pause the game"""
    if game_manager:
        game_manager.pause_game()

func resume_game():
    """Resume the game"""
    if game_manager:
        game_manager.resume_game()

func next_level():
    """Go to next level"""
    if game_manager:
        game_manager.next_level()

func restart_level():
    """Restart current level"""
    if game_manager:
        game_manager.setup_game()

func end_game():
    """End the game"""
    if game_manager:
        game_manager.end_game()
    
    print("[GAME] Game ended")
    emit_signal("game_state_changed", "ended")

# ============================================================================
# C++ Integration Points
# ============================================================================

func get_game_state() -> Dictionary:
    """Get current game state - can be called from C++"""
    if game_manager:
        return game_manager.get_game_state()
    return {}

func set_player_health(health: float):
    """Set player health from C++"""
    if ui_manager:
        ui_manager.set_health(health)

func add_score(amount: int):
    """Add score from C++"""
    if ui_manager:
        ui_manager.add_score(amount)

func damage_player(amount: float):
    """Damage player from C++"""
    if ui_manager:
        ui_manager.take_damage(amount)

func get_elapsed_time() -> float:
    """Get elapsed game time from C++"""
    if game_manager:
        return game_manager.get_elapsed_time()
    return 0.0

func get_frame_count() -> int:
    """Get frame count from C++"""
    if game_manager:
        return game_manager.get_frame_count()
    return 0

func get_hud_text() -> String:
    """Get HUD text display from C++"""
    if ui_manager:
        return ui_manager.get_full_hud_text()
    return ""

# ============================================================================
# Cleanup
# ============================================================================

func _exit_tree():
    """Called when node is removed from scene"""
    print("[CLEANUP] MainGame shutting down")
    
    if game_manager:
        game_manager.queue_free()
    
    if player_controller:
        player_controller.queue_free()
    
    if ui_manager:
        ui_manager.queue_free()
