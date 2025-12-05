# Animation Events API

## Overview

Animation Events allow you to trigger callbacks at specific frames during sprite animation playback. This is useful for synchronizing game logic with animations (e.g., footstep sounds, particle effects, damage triggers).

## Basic Usage

### 1. Create a Sprite with Animation

```cpp
auto sprite = std::make_shared<Sprite>("MySprite");
sprite->SetAtlas(4, 4);  // 4x4 sprite atlas (16 frames)
```

### 2. Define Animation Sequences

```cpp
// Add a walk animation (frames 0-7, 8 fps, looping)
sprite->AddSequence("walk", 0, 7, 8.0f, true);

// Add an attack animation (frames 8-11, 12 fps, non-looping)
sprite->AddSequence("attack", 8, 11, 12.0f, false);
```

### 3. Add Events to Sequences

```cpp
// Add footstep sound events to walk animation
sprite->AddEventToSequence("walk", 2, "footstep_left", [](const std::string& eventName) {
    AudioManager::PlaySound("footstep.wav");
});

sprite->AddEventToSequence("walk", 6, "footstep_right", [](const std::string& eventName) {
    AudioManager::PlaySound("footstep.wav");
});

// Add damage event to attack animation
sprite->AddEventToSequence("attack", 3, "deal_damage", [](const std::string& eventName) {
    // Apply damage to enemies in range
    DamageNearbyEnemies();
});
```

### 4. Play the Animation

```cpp
sprite->PlaySequence("walk");
```

## API Reference

### Adding Events

```cpp
void AddEventToSequence(
    const std::string& sequenceName,  // Name of the animation sequence
    int frame,                         // Frame number (within sequence range)
    const std::string& eventName,      // Event identifier
    EventCallback callback             // Function to call when event triggers
);
```

**Note**: Frame number must be within the sequence's `[startFrame, endFrame]` range.

### Clearing Events

```cpp
void ClearEventsForSequence(const std::string& sequenceName);
```

Removes all events from the specified animation sequence.

### Event Callback Signature

```cpp
using EventCallback = std::function<void(const std::string& eventName)>;
```

The callback receives the event name as a parameter, allowing you to identify which event triggered.

## Important Behaviors

### Event Triggering on Loops

Events trigger **every time** their frame is reached, including on loop iterations. This is ideal for repeating animations like walk cycles.

```cpp
// This will play a footstep sound on EVERY loop iteration
sprite->AddEventToSequence("walk", 2, "footstep", [](const std::string& name) {
    PlaySound("step.wav");
});
```

### Multiple Events Per Frame

You can add multiple events to the same frame:

```cpp
sprite->AddEventToSequence("attack", 3, "damage", DealDamage);
sprite->AddEventToSequence("attack", 3, "effect", SpawnEffect);
sprite->AddEventToSequence("attack", 3, "sound", PlaySound);
```

All events on the same frame will trigger in the order they were added.

### Per-Sequence Storage

Events are stored per-sequence, so different sequences can have different events at the same frame numbers:

```cpp
// Frame 2 in "walk" plays footstep
sprite->AddEventToSequence("walk", 2, "footstep", PlayFootstep);

// Frame 2 in "run" plays different sound
sprite->AddEventToSequence("run", 2, "run_step", PlayRunStep);
```

## Complete Example

```cpp
// Create and configure sprite
auto character = std::make_shared<Sprite>("Character");
character->SetAtlas(8, 8);  // 8x8 atlas (64 frames)

// Define animations
character->AddSequence("idle", 0, 3, 4.0f, true);
character->AddSequence("walk", 4, 11, 8.0f, true);
character->AddSequence("attack", 12, 17, 15.0f, false);

// Add walk events
character->AddEventToSequence("walk", 6, "footstep_left", [](const std::string& name) {
    AudioManager::PlaySound("footstep.wav");
});

character->AddEventToSequence("walk", 10, "footstep_right", [](const std::string& name) {
    AudioManager::PlaySound("footstep.wav");
});

// Add attack events
character->AddEventToSequence("attack", 14, "swing_sound", [](const std::string& name) {
    AudioManager::PlaySound("sword_swing.wav");
});

character->AddEventToSequence("attack", 15, "deal_damage", [](const std::string& name) {
    // Check for enemies in attack range and apply damage
    auto enemies = GetEnemiesInRange(character->GetPosition(), 2.0f);
    for (auto& enemy : enemies) {
        enemy->TakeDamage(10);
    }
});

// Set up completion callback for attack
character->SetOnComplete([]() {
    // Return to idle after attack finishes
    character->PlaySequence("idle");
});

// Start with idle animation
character->PlaySequence("idle");

// Later, trigger attack
character->PlaySequence("attack");
```

## Tips

- **Event Names**: Use descriptive names to make debugging easier
- **Lambda Captures**: Be careful with lambda captures to avoid dangling references
- **Frame Validation**: Events are validated when added - invalid frame numbers will print a warning
- **Performance**: Events are checked only when frames change, so there's minimal overhead

## See Also

- `Sprite::SetOnComplete()` - Callback when animation finishes (non-looping)
- `Sprite::SetOnLoop()` - Callback when animation loops
- `Sprite::SetOnFrameChange()` - Callback on every frame change
