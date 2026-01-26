# Simple UI manager for game HUD and menus
extends Node

class_name UIManager

# UI state
var is_menu_open: bool = false
var is_hud_visible: bool = true
var current_menu: String = ""

# HUD elements (would normally be Node references)
var health_display: String = ""
var ammo_display: String = ""
var score_display: String = ""
var level_display: String = ""
var fps_display: String = ""

# UI values
var current_health: float = 100.0
var max_health: float = 100.0
var current_ammo: int = 30
var max_ammo: int = 30
var current_score: int = 0
var current_level: int = 1
var current_fps: float = 60.0

# Signals
signal menu_opened(menu_name: String)
signal menu_closed()
signal hud_toggled(visible: bool)
signal health_updated(health: float, max_health: float)
signal ammo_updated(ammo: int, max_ammo: int)
signal score_updated(score: int)
signal level_updated(level: int)

func _ready():
    """Initialize UI manager"""
    print("UIManager initialized")
    update_all_displays()

func _process(delta: float):
    """Called every frame"""
    # Update FPS
    if delta > 0:
        current_fps = 1.0 / delta
    
    # Update displays
    update_hud_displays()

func update_all_displays():
    """Update all UI displays"""
    update_health_display()
    update_ammo_display()
    update_score_display()
    update_level_display()

func update_hud_displays():
    """Update HUD displays each frame"""
    update_fps_display()

func update_health_display():
    """Update health display"""
    health_display = "Health: %d/%d" % [int(current_health), int(max_health)]
    emit_signal("health_updated", current_health, max_health)

func update_ammo_display():
    """Update ammo display"""
    ammo_display = "Ammo: %d/%d" % [current_ammo, max_ammo]
    emit_signal("ammo_updated", current_ammo, max_ammo)

func update_score_display():
    """Update score display"""
    score_display = "Score: %d" % current_score
    emit_signal("score_updated", current_score)

func update_level_display():
    """Update level display"""
    level_display = "Level: %d" % current_level
    emit_signal("level_updated", current_level)

func update_fps_display():
    """Update FPS display"""
    fps_display = "FPS: %.1f" % current_fps

# ============================================================================
# Health Management
# ============================================================================

func set_health(health: float):
    """Set current health"""
    current_health = clamp(health, 0.0, max_health)
    update_health_display()

func take_damage(amount: float):
    """Take damage"""
    set_health(current_health - amount)
    if current_health <= 0:
        show_game_over()

func heal(amount: float):
    """Heal"""
    set_health(current_health + amount)

# ============================================================================
# Ammo Management
# ============================================================================

func set_ammo(ammo: int):
    """Set current ammo"""
    current_ammo = clamp(ammo, 0, max_ammo)
    update_ammo_display()

func use_ammo(amount: int):
    """Use ammo"""
    set_ammo(current_ammo - amount)

func reload_ammo():
    """Reload ammo"""
    set_ammo(max_ammo)

func add_ammo(amount: int):
    """Add ammo"""
    set_ammo(current_ammo + amount)

# ============================================================================
# Score Management
# ============================================================================

func add_score(amount: int):
    """Add to score"""
    current_score += amount
    update_score_display()

func set_score(score: int):
    """Set score"""
    current_score = max(score, 0)
    update_score_display()

# ============================================================================
# Level Management
# ============================================================================

func set_level(level: int):
    """Set current level"""
    current_level = max(level, 1)
    update_level_display()

# ============================================================================
# Menu Management
# ============================================================================

func open_menu(menu_name: String):
    """Open a menu"""
    if is_menu_open:
        close_menu()
    
    is_menu_open = true
    current_menu = menu_name
    
    print("Opening menu: ", menu_name)
    emit_signal("menu_opened", menu_name)

func close_menu():
    """Close current menu"""
    if not is_menu_open:
        return
    
    is_menu_open = false
    print("Closing menu: ", current_menu)
    
    emit_signal("menu_closed")
    current_menu = ""

func toggle_menu(menu_name: String):
    """Toggle a menu"""
    if is_menu_open and current_menu == menu_name:
        close_menu()
    else:
        open_menu(menu_name)

func show_pause_menu():
    """Show pause menu"""
    open_menu("pause")

func show_game_over():
    """Show game over screen"""
    open_menu("game_over")
    print("Game Over!")

func show_main_menu():
    """Show main menu"""
    open_menu("main_menu")

# ============================================================================
# HUD Visibility
# ============================================================================

func toggle_hud():
    """Toggle HUD visibility"""
    is_hud_visible = not is_hud_visible
    emit_signal("hud_toggled", is_hud_visible)
    print("HUD visibility: ", is_hud_visible)

func show_hud():
    """Show HUD"""
    if not is_hud_visible:
        is_hud_visible = true
        emit_signal("hud_toggled", true)

func hide_hud():
    """Hide HUD"""
    if is_hud_visible:
        is_hud_visible = false
        emit_signal("hud_toggled", false)

# ============================================================================
# Notifications
# ============================================================================

func show_notification(message: String, duration: float = 3.0):
    """Show a temporary notification"""
    print("[NOTIFICATION] ", message)

func show_error(message: String):
    """Show an error notification"""
    print("[ERROR] ", message)

func show_warning(message: String):
    """Show a warning notification"""
    print("[WARNING] ", message)

func show_success(message: String):
    """Show a success notification"""
    print("[SUCCESS] ", message)

# ============================================================================
# Display Getters
# ============================================================================

func get_health_text() -> String:
    """Get formatted health text"""
    return health_display

func get_ammo_text() -> String:
    """Get formatted ammo text"""
    return ammo_display

func get_score_text() -> String:
    """Get formatted score text"""
    return score_display

func get_level_text() -> String:
    """Get formatted level text"""
    return level_display

func get_fps_text() -> String:
    """Get formatted FPS text"""
    return fps_display

func get_full_hud_text() -> String:
    """Get full HUD as formatted text"""
    return "%s | %s | %s | %s | %s" % [
        health_display,
        ammo_display,
        score_display,
        level_display,
        fps_display
    ]
