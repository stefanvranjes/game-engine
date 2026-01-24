package main

import (
	"fmt"
	"time"
)

// ExampleNPCBehavior demonstrates a concurrent NPC behavior tree
// This function runs as an independent goroutine for each NPC
func ExampleNPCBehavior(npcID int32) {
	fmt.Printf("[NPC %d] Behavior tree started\n", npcID)

	// Create channels for inter-goroutine communication
	stateChanges := make(chan string, 10)
	animationUpdates := make(chan string, 10)
	done := make(chan bool)

	// Start concurrent behavior subsystems
	go patrolBehavior(npcID, stateChanges)
	go detectPlayerBehavior(npcID, stateChanges)
	go animationController(npcID, animationUpdates)
	go pathfindingSystem(npcID, stateChanges)

	// Main behavior state machine
	timeout := time.After(10 * time.Second)
	for {
		select {
		case state := <-stateChanges:
			fmt.Printf("[NPC %d] State changed: %s\n", npcID, state)
			animationUpdates <- state

		case anim := <-animationUpdates:
			fmt.Printf("[NPC %d] Animation updated: %s\n", npcID, anim)

		case <-timeout:
			fmt.Printf("[NPC %d] Behavior tree completed\n", npcID)
			done <- true
			return

		case <-done:
			return
		}
	}
}

// patrolBehavior runs the patrol state for an NPC
func patrolBehavior(npcID int32, stateChanges chan string) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("[NPC %d] Patrol panic: %v\n", npcID, r)
		}
	}()

	patrolWaypoints := []string{"waypoint_1", "waypoint_2", "waypoint_3"}

	for _, waypoint := range patrolWaypoints {
		stateChanges <- fmt.Sprintf("patrol_%s", waypoint)
		time.Sleep(500 * time.Millisecond)
	}
}

// detectPlayerBehavior monitors for player detection
func detectPlayerBehavior(npcID int32, stateChanges chan string) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	detectionRange := 50.0
	for range ticker.C {
		// Simulate player detection check
		if npcID%3 == 0 {
			stateChanges <- fmt.Sprintf("detect_player_range=%.1f", detectionRange)
		}
	}
}

// animationController handles animation state
func animationController(npcID int32, animations chan string) {
	animationMap := map[string]string{
		"patrol_waypoint_1": "walk",
		"patrol_waypoint_2": "walk",
		"patrol_waypoint_3": "walk",
		"detect_player_range=50.0": "look_around",
	}

	for anim := range animations {
		if mapped, exists := animationMap[anim]; exists {
			fmt.Printf("[NPC %d] Playing animation: %s\n", npcID, mapped)
		}
	}
}

// pathfindingSystem handles path calculation
func pathfindingSystem(npcID int32, stateChanges chan string) {
	tick := time.NewTicker(800 * time.Millisecond)
	defer tick.Stop()

	count := 0
	for range tick.C {
		if count < 5 {
			stateChanges <- fmt.Sprintf("path_updated_%d", count)
			count++
		} else {
			return
		}
	}
}

// ExampleParallelPhysicsUpdate demonstrates physics calculations in parallel
func ExampleParallelPhysicsUpdate(deltaTime float32, actorCount int32) {
	fmt.Printf("Physics update starting for %d actors with dt=%.3f\n", actorCount, deltaTime)

	physicsChannel := make(chan PhysicsActor, 100)
	resultChannel := make(chan PhysicsResult, 100)
	done := make(chan int)

	// Spawn physics worker goroutines (one per 8 actors, similar to thread pool)
	numWorkers := (actorCount + 7) / 8
	if numWorkers > 16 {
		numWorkers = 16
	}

	for w := int32(0); w < numWorkers; w++ {
		go physicsWorker(w, physicsChannel, resultChannel, done)
	}

	// Send physics actors to be processed
	go func() {
		for i := int32(0); i < actorCount; i++ {
			actor := PhysicsActor{
				ID:       i,
				Position: [3]float32{float32(i), 0, 0},
				Velocity: [3]float32{1.0, 0.5, 0},
			}
			physicsChannel <- actor
		}
		close(physicsChannel)
	}()

	// Collect results
	resultsReceived := 0
	for result := range resultChannel {
		if resultsReceived%100 == 0 {
			fmt.Printf("Physics update: processed %d actors\n", resultsReceived)
		}
		_ = result // Use result to update actor positions
		resultsReceived++

		if resultsReceived >= int(actorCount) {
			break
		}
	}

	fmt.Printf("Physics update completed: %d actors\n", resultsReceived)
}

type PhysicsActor struct {
	ID       int32
	Position [3]float32
	Velocity [3]float32
}

type PhysicsResult struct {
	ActorID      int32
	NewPosition  [3]float32
	NewVelocity  [3]float32
	Collisions   int32
}

// physicsWorker processes physics for multiple actors
func physicsWorker(id int32, in chan PhysicsActor, out chan PhysicsResult, done chan int) {
	defer func() {
		done <- int(id)
	}()

	processCount := 0
	for actor := range in {
		// Simulate physics calculation
		newPos := [3]float32{
			actor.Position[0] + actor.Velocity[0]*0.016,
			actor.Position[1] + actor.Velocity[1]*0.016,
			actor.Position[2] + actor.Velocity[2]*0.016,
		}

		result := PhysicsResult{
			ActorID:     actor.ID,
			NewPosition: newPos,
			NewVelocity: actor.Velocity,
			Collisions:  0,
		}

		out <- result
		processCount++
	}
}

// ExampleNetworkReplication simulates network state replication
func ExampleNetworkReplication(playerID int32) {
	fmt.Printf("[Network] Starting replication for player %d\n", playerID)

	commandChannel := make(chan PlayerCommand, 50)
	stateChannel := make(chan PlayerState, 50)
	done := make(chan bool)

	// Goroutines for sending/receiving
	go commandReceiver(playerID, commandChannel)
	go stateTransmitter(playerID, stateChannel)
	go commandProcessor(playerID, commandChannel, stateChannel)

	time.Sleep(5 * time.Second)
	done <- true
}

type PlayerCommand struct {
	Action string
	Value  float32
}

type PlayerState struct {
	Position   [3]float32
	Rotation   [4]float32
	Animation  string
	Health     int32
}

func commandReceiver(playerID int32, out chan PlayerCommand) {
	commands := []PlayerCommand{
		{"move_forward", 1.0},
		{"rotate", 90.0},
		{"jump", 0.0},
		{"attack", 0.0},
	}

	for _, cmd := range commands {
		fmt.Printf("[Network Player %d] Received command: %s\n", playerID, cmd.Action)
		out <- cmd
		time.Sleep(500 * time.Millisecond)
	}

	close(out)
}

func stateTransmitter(playerID int32, in chan PlayerState) {
	count := 0
	for state := range in {
		fmt.Printf("[Network Player %d] Transmitting state: pos=(%.1f,%.1f,%.1f)\n",
			playerID, state.Position[0], state.Position[1], state.Position[2])
		count++
	}
	fmt.Printf("[Network Player %d] Transmitted %d states\n", playerID, count)
}

func commandProcessor(playerID int32, commands chan PlayerCommand, states chan PlayerState) {
	pos := [3]float32{0, 0, 0}

	for cmd := range commands {
		switch cmd.Action {
		case "move_forward":
			pos[2] += 5.0
		case "rotate":
			// Rotation update
		case "jump":
			pos[1] += 2.0
		case "attack":
			// Attack logic
		}

		state := PlayerState{
			Position:  pos,
			Rotation:  [4]float32{0, 0, 0, 1},
			Animation: cmd.Action,
			Health:    100,
		}

		states <- state
	}

	close(states)
}

// ExampleAssetLoading demonstrates concurrent resource loading
func ExampleAssetLoading(assetCount int32) {
	fmt.Printf("Asset loading started for %d assets\n", assetCount)

	loadChannel := make(chan AssetRequest, 100)
	resultChannel := make(chan LoadedAsset, 100)

	// Spawn loader workers
	numLoaders := 4
	for i := 0; i < numLoaders; i++ {
		go assetLoader(i, loadChannel, resultChannel)
	}

	// Queue all assets
	go func() {
		for i := int32(0); i < assetCount; i++ {
			loadChannel <- AssetRequest{
				AssetID: i,
				Path:    fmt.Sprintf("assets/model_%d.gltf", i),
			}
		}
		close(loadChannel)
	}()

	// Collect results
	loaded := 0
	for asset := range resultChannel {
		fmt.Printf("Loaded asset %d: %s (size: %d bytes)\n",
			asset.AssetID, asset.Path, asset.MemorySize)
		loaded++

		if loaded >= int(assetCount) {
			break
		}
	}

	fmt.Printf("Asset loading complete: %d assets loaded\n", loaded)
}

type AssetRequest struct {
	AssetID int32
	Path    string
}

type LoadedAsset struct {
	AssetID    int32
	Path       string
	MemorySize int32
	Success    bool
}

func assetLoader(id int, in chan AssetRequest, out chan LoadedAsset) {
	for req := range in {
		// Simulate loading
		time.Sleep(100 * time.Millisecond)

		asset := LoadedAsset{
			AssetID:    req.AssetID,
			Path:       req.Path,
			MemorySize: int32((req.AssetID + 1) * 1024),
			Success:    true,
		}

		out <- asset
	}
}
