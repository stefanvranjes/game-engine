# WASM Support Implementation Examples

## Example 1: Simple Game Logic Module

### Rust Source (game_logic.rs)

```rust
use std::cell::RefCell;

thread_local! {
    static GAME_STATE: RefCell<GameState> = RefCell::new(GameState::new());
}

struct GameState {
    score: i32,
    level: i32,
    enemies_defeated: i32,
}

impl GameState {
    fn new() -> Self {
        GameState {
            score: 0,
            level: 1,
            enemies_defeated: 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn init() {
    // Initialize game state
    GAME_STATE.with(|state| {
        let mut s = state.borrow_mut();
        s.score = 0;
        s.level = 1;
    });
}

#[no_mangle]
pub extern "C" fn update(delta_time: f32) {
    // Update game logic
    GAME_STATE.with(|state| {
        let mut s = state.borrow_mut();
        // Game update logic here
    });
}

#[no_mangle]
pub extern "C" fn get_score() -> i32 {
    GAME_STATE.with(|state| {
        state.borrow().score
    })
}

#[no_mangle]
pub extern "C" fn add_score(points: i32) {
    GAME_STATE.with(|state| {
        let mut s = state.borrow_mut();
        s.score += points;
    });
}

#[no_mangle]
pub extern "C" fn shutdown() {
    // Cleanup
}
```

Compile:
```bash
rustc --target wasm32-unknown-unknown -O game_logic.rs --crate-type cdylib
```

### C++ Usage

```cpp
WasmScriptSystem& wasmSys = WasmScriptSystem::GetInstance();
wasmSys.LoadWasmModule("game_logic.wasm");

// Call functions
auto instance = wasmSys.GetModuleInstance("game_logic");
instance->Call("init");
instance->Call("add_score", {WasmValue::I32(100)});

// Get result
auto result = instance->Call("get_score");
int32_t score = std::any_cast<int32_t>(result.value);
std::cout << "Score: " << score << std::endl;
```

## Example 2: Enemy AI Module

### Rust Source (enemy_ai.rs)

```rust
extern "C" {
    fn debug_log(ptr: i32);
    fn physics_apply_force(obj_id: i32, x: f32, y: f32, z: f32);
    fn render_set_color(obj_id: i32, r: f32, g: f32, b: f32);
}

#[repr(C)]
struct Enemy {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    health: f32,
    target_x: f32,
    target_y: f32,
}

static mut ENEMIES: [Enemy; 10] = [
    Enemy {
        pos_x: 0.0,
        pos_y: 0.0,
        pos_z: 0.0,
        health: 100.0,
        target_x: 0.0,
        target_y: 0.0,
    }; 10
];

#[no_mangle]
pub extern "C" fn init() {
    // Initialize enemy positions
}

#[no_mangle]
pub extern "C" fn update(delta_time: f32) {
    unsafe {
        for enemy in &mut ENEMIES {
            if enemy.health <= 0.0 {
                continue;
            }

            // Simple AI: move towards target
            let dx = enemy.target_x - enemy.pos_x;
            let dy = enemy.target_y - enemy.pos_y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.1 {
                let speed = 5.0;
                let vx = (dx / dist) * speed;
                let vy = (dy / dist) * speed;

                physics_apply_force(0, vx, vy, 0.0);
                enemy.pos_x += vx * delta_time;
                enemy.pos_y += vy * delta_time;
            }

            // Adjust color based on health
            let health_ratio = enemy.health / 100.0;
            render_set_color(0, health_ratio, health_ratio, health_ratio);
        }
    }
}

#[no_mangle]
pub extern "C" fn take_damage(enemy_id: i32, damage: f32) {
    unsafe {
        if (enemy_id as usize) < ENEMIES.len() {
            ENEMIES[enemy_id as usize].health -= damage;
        }
    }
}

#[no_mangle]
pub extern "C" fn shutdown() {
    // Cleanup
}
```

### C++ Usage

```cpp
// Load enemy AI module
wasmSys.LoadWasmModule("enemy_ai.wasm", "enemy_ai");

auto enemyObj = std::make_shared<GameObject>("enemy");
wasmSys.BindGameObject("enemy_ai", enemyObj);

// In update loop
auto instance = wasmSys.GetModuleInstance("enemy_ai");
instance->Call("update", {WasmValue::F32(deltaTime)});

// Call damage function
instance->Call("take_damage", {WasmValue::I32(0), WasmValue::F32(25.0f)});
```

## Example 3: Interactive Script with Memory Access

### C Example (particle_sim.c)

```c
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float life;
} Particle;

#define MAX_PARTICLES 1000

static Particle particles[MAX_PARTICLES];
static int particle_count = 0;

void init(void) {
    particle_count = 0;
}

void update(float delta_time) {
    for (int i = 0; i < particle_count; ++i) {
        Particle* p = &particles[i];
        
        // Update position
        p->x += p->vx * delta_time;
        p->y += p->vy * delta_time;
        p->z += p->vz * delta_time;
        
        // Gravity
        p->vy -= 9.8f * delta_time;
        
        // Decay
        p->life -= delta_time;
        
        // Remove dead particles
        if (p->life <= 0.0f) {
            particles[i] = particles[--particle_count];
            --i;
        }
    }
}

int create_particle(float x, float y, float z, float vx, float vy, float vz) {
    if (particle_count >= MAX_PARTICLES) {
        return -1;
    }
    
    Particle* p = &particles[particle_count];
    p->x = x;
    p->y = y;
    p->z = z;
    p->vx = vx;
    p->vy = vy;
    p->vz = vz;
    p->life = 2.0f;
    
    return particle_count++;
}

float get_particle_x(int id) {
    if (id >= 0 && id < particle_count) {
        return particles[id].x;
    }
    return 0.0f;
}

int get_particle_count(void) {
    return particle_count;
}

void shutdown(void) {
    particle_count = 0;
}
```

Compile:
```bash
clang --target=wasm32 -O3 -nostdlib -Wl,--no-entry \
  -Wl,--export=init \
  -Wl,--export=update \
  -Wl,--export=create_particle \
  -Wl,--export=get_particle_x \
  -Wl,--export=get_particle_count \
  -Wl,--export=shutdown \
  particle_sim.c -o particle_sim.wasm
```

### C++ Usage

```cpp
wasmSys.LoadWasmModule("particle_sim.wasm", "particles");
auto instance = wasmSys.GetModuleInstance("particles");

instance->Call("init");

// Create some particles
instance->Call("create_particle", {
    WasmValue::F32(0.0f), WasmValue::F32(10.0f), WasmValue::F32(0.0f),
    WasmValue::F32(2.0f), WasmValue::F32(5.0f), WasmValue::F32(0.0f)
});

// Update
instance->Call("update", {WasmValue::F32(deltaTime)});

// Get particle count
auto countResult = instance->Call("get_particle_count");
int count = std::any_cast<int32_t>(countResult.value);

// Access particle data from WASM memory
// Particles are stored at fixed offset in WASM memory
uint32_t particleOffset = 0;
for (int i = 0; i < count; ++i) {
    auto data = instance->ReadMemory(
        particleOffset + i * sizeof(float) * 7,
        sizeof(float) * 7
    );
    
    const float* pdata = reinterpret_cast<const float*>(data.data());
    std::cout << "Particle " << i << ": (" 
              << pdata[0] << ", " << pdata[1] << ", " << pdata[2] << ")" << std::endl;
}
```

## Example 4: Custom Bindings

### C++ Engine Code

```cpp
// Register custom binding for custom game mechanic
WasmEngineBindings& bindings = WasmEngineBindings::GetInstance();
auto instance = wasmSys.GetModuleInstance("game_logic");

bindings.RegisterBinding(instance, "trigger_explosion",
    [&gameWorld](std::shared_ptr<WasmInstance> inst, 
                 const std::vector<WasmValue>& args) -> WasmValue {
        if (args.size() >= 3) {
            try {
                float x = std::any_cast<float>(args[0].value);
                float y = std::any_cast<float>(args[1].value);
                float z = std::any_cast<float>(args[2].value);
                float radius = args.size() > 3 ? 
                    std::any_cast<float>(args[3].value) : 10.0f;

                // Trigger explosion in game world
                gameWorld.CreateExplosion(glm::vec3(x, y, z), radius);
                
                return WasmValue::I32(1);  // Success
            } catch (...) {
                return WasmValue::I32(0);  // Error
            }
        }
        return WasmValue::I32(0);
    }
);

// Register another binding
bindings.RegisterBinding(instance, "spawn_enemy",
    [&entityManager](std::shared_ptr<WasmInstance> inst,
                     const std::vector<WasmValue>& args) -> WasmValue {
        if (!args.empty()) {
            try {
                int enemy_type = std::any_cast<int32_t>(args[0].value);
                float x = args.size() > 1 ? 
                    std::any_cast<float>(args[1].value) : 0.0f;
                float y = args.size() > 2 ? 
                    std::any_cast<float>(args[2].value) : 0.0f;
                float z = args.size() > 3 ? 
                    std::any_cast<float>(args[3].value) : 0.0f;

                auto entity = entityManager.CreateEnemy(
                    enemy_type,
                    glm::vec3(x, y, z)
                );
                
                return WasmValue::I32(entity.GetID());
            } catch (...) {
                return WasmValue::I32(-1);
            }
        }
        return WasmValue::I32(-1);
    }
);
```

### Rust WASM Code

```rust
extern "C" {
    fn trigger_explosion(x: f32, y: f32, z: f32, radius: f32) -> i32;
    fn spawn_enemy(enemy_type: i32, x: f32, y: f32, z: f32) -> i32;
}

#[no_mangle]
pub extern "C" fn handle_level_event(event_type: i32) {
    match event_type {
        1 => {
            // Player defeated boss - trigger explosion and spawn rewards
            unsafe {
                trigger_explosion(0.0, 0.0, 0.0, 20.0);
            }
        },
        2 => {
            // Wave event - spawn multiple enemies
            unsafe {
                for i in 0..5 {
                    spawn_enemy(1, -10.0 + i as f32 * 5.0, 5.0, 0.0);
                }
            }
        },
        _ => {},
    }
}
```

