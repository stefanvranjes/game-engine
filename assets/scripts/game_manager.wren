// Gameplay manager script for Wren game engine
// Demonstrates level management, game state, UI updates, and event handling

class GameManager {
    construct new() {
        _gameState = "playing"  // "playing", "paused", "gameOver", "won"
        _score = 0
        _time = 0.0
        _levelTime = 300.0  // 5 minutes per level
        _enemiesDefeated = 0
        _itemsCollected = 0
        
        _playerRef = null
        _uiManager = null
        _audioManager = null
        
        _isPaused = false
    }
    
    init() {
        System.print("GameManager initialized")
        loadLevel()
    }
    
    update(dt) {
        if (_gameState == "gameOver" || _gameState == "won") {
            return
        }
        
        if (Input.getKeyDown("Escape")) {
            togglePause()
        }
        
        if (!_isPaused) {
            _time = _time + dt
            updateGameState(dt)
        }
    }
    
    loadLevel() {
        System.print("Loading level...")
        
        // Spawn player
        // _playerRef = SpawnManager.spawn("player", Vec3.new(0, 1, 0))
        
        // Spawn enemies
        spawnEnemies(5)
        
        // Spawn collectibles
        spawnCollectibles(20)
        
        System.print("Level loaded")
    }
    
    spawnEnemies(count) {
        for (i in 0...count) {
            var x = (i % 3) * 4.0
            var z = (i / 3) * 4.0
            // var enemy = SpawnManager.spawn("enemy", Vec3.new(x, 0, z))
            System.print("Enemy %((i + 1)) spawned")
        }
    }
    
    spawnCollectibles(count) {
        for (i in 0...count) {
            var types = ["coin", "health", "ammo"]
            var type = types[i % 3]
            
            var x = (System.random() - 0.5) * 20
            var z = (System.random() - 0.5) * 20
            
            // var item = SpawnManager.spawn("collectible", Vec3.new(x, 0.5, z))
            // item.setItemType(type)
            
            System.print("Collectible spawned: %((i + 1))")
        }
    }
    
    updateGameState(dt) {
        // Check win condition
        if (_time > _levelTime) {
            endLevel(true)
        }
        
        // Check lose condition
        // if (_playerRef.health <= 0) {
        //     endGame(false)
        // }
        
        // Update UI
        updateUI()
    }
    
    updateUI() {
        // Update HUD with current game state
        // _uiManager.setScore(_score)
        // _uiManager.setTime(_levelTime - _time)
        // _uiManager.setHealth(_playerRef.health)
    }
    
    togglePause() {
        _isPaused = !_isPaused
        _gameState = _isPaused ? "paused" : "playing"
        
        if (_isPaused) {
            System.print("Game paused")
            // _uiManager.showPauseMenu()
        } else {
            System.print("Game resumed")
            // _uiManager.hidePauseMenu()
        }
    }
    
    addScore(amount) {
        _score = _score + amount
        System.print("Score: %(_score)")
    }
    
    onEnemyDefeated() {
        _enemiesDefeated = _enemiesDefeated + 1
        addScore(100)
        System.print("Enemies defeated: %(_enemiesDefeated)")
    }
    
    onItemCollected() {
        _itemsCollected = _itemsCollected + 1
        addScore(10)
    }
    
    endLevel(won) {
        _gameState = won ? "won" : "gameOver"
        System.print("Level %(_gameState ? "won" : "lost")")
        
        if (won) {
            addScore(_levelTime * 10.0)  // Time bonus
            // _uiManager.showWinScreen()
        } else {
            // _uiManager.showGameOverScreen()
        }
    }
    
    restartLevel() {
        System.print("Restarting level...")
        // Reload level script
    }
    
    loadNextLevel() {
        System.print("Loading next level...")
        // Load next level
    }
    
    destroy() {
        System.print("GameManager destroyed")
    }
}

// Global game manager instance (singleton pattern)
var _gameManager = null

// Lifecycle functions
construct init() {
    if (_gameManager == null) {
        _gameManager = GameManager.new()
        _gameManager.init()
    }
}

construct update(dt) {
    if (_gameManager) {
        _gameManager.update(dt)
    }
}

construct destroy() {
    if (_gameManager) {
        _gameManager.destroy()
        _gameManager = null
    }
}

// Helper function to access game manager from other scripts
static getGameManager() {
    return _gameManager
}

static addScore(amount) {
    if (_gameManager) {
        _gameManager.addScore(amount)
    }
}

static onEnemyDefeated() {
    if (_gameManager) {
        _gameManager.onEnemyDefeated()
    }
}

static onItemCollected() {
    if (_gameManager) {
        _gameManager.onItemCollected()
    }
}
